// Package dragonfly implements the Dragonfly Algorithm (DA), a swarm
// intelligence metaheuristic modeled on the static and dynamic swarming
// behavior of dragonflies.
//
// A run maintains a population of dragonflies, each carrying a position X and a
// step ΔX (the velocity analog). Every iteration each dragonfly combines five
// primitives -- separation, alignment, cohesion, attraction to the food source
// (the best position seen) and distraction from the enemy (the worst position
// seen) -- into a new step, and adds that step to its position. Adaptive weight
// schedules move the swarm from exploration to exploitation as the run
// progresses; a dragonfly that is isolated and out of reach of the food source
// performs a Lévy random walk instead.
//
// The entry points are Optimize and OptimizeContext. Configure a run with
// NewDefaultConfig (or one of the other factories in config.go) and set
// ObjectiveFunc, ProblemSize, LowerBound and UpperBound.
//
// Reference:
// Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization
// technique for solving single-objective, discrete, and multi-objective
// problems. Neural Computing and Applications, 27(4), 1053-1073.
// DOI: 10.1007/s00521-015-1920-1
//
// Go implementation by Christian-W. Budde.
package dragonfly

import (
	"context"
	"errors"
	"math"
	"math/rand"
)

// ErrNoFiniteObjective is returned when the initial population contains no
// finite objective value. In that case there is no real food source from which
// a meaningful DA or BDA step can be computed.
var ErrNoFiniteObjective = errors.New("objective produced no finite value")

// runState is the mutable state of one optimization run: the swarm itself, the
// two extreme positions every step is computed against, and the objective call
// counter.
//
// It exists so the phases of the loop -- initialize, evaluate, move -- can be
// separate functions without threading five parameters and four return values
// through each of them.
type runState struct {
	// evaluator is the single authority on what "better" means: it scores a
	// position, records its aggregate constraint violation, and ranks two
	// candidates under the configured constraint policy. Every objective call
	// in a run goes through it, so an unconstrained run and a constrained one
	// differ only in the policy it holds.
	evaluator *constraintEvaluator
	// pool is the bounded worker pool the objective calls fan out across, or
	// nil for a sequential run. It holds no goroutines between batches and
	// needs no shutdown.
	pool  *evaluationPool
	swarm []Dragonfly
	food  Best
	enemy Best
	// movementEnemy is the reference that MATLAB-compatible DA steps use.
	// DA.m updates it only from positions strictly inside the search box,
	// whereas enemy records the actual worst evaluated candidate for Result.
	movementEnemy Best
	// funcEvals counts every call to config.ObjectiveFunc, including the ones
	// made while initializing the swarm.
	funcEvals int
}

// Optimize runs the Dragonfly Algorithm with a background context.
func Optimize(config *Config) (*Result, error) {
	return OptimizeContext(context.Background(), config)
}

// OptimizeContext runs the Dragonfly Algorithm, honoring context cancellation
// and the supplied run options.
//
// Cancellation is checked at the top of every iteration. A canceled run
// returns a nil result and ctx.Err(); partial results are deliberately not
// reported, so a caller cannot mistake an aborted run for a completed one.
//
// Observers registered through WithProgressObserver and WithPopulationObserver
// receive deep copies and run synchronously on this goroutine. They must not
// draw random numbers or mutate what they are handed: a seeded run is required
// to be reproducible, and an observer that reaches back into the swarm would
// be a back door around that.
func OptimizeContext(ctx context.Context, config *Config, options ...RunOption) (*Result, error) {
	contextErr := requireContext(ctx)
	if contextErr != nil {
		return nil, contextErr
	}

	resolved, optionsErr := resolveRunOptions(options)
	if optionsErr != nil {
		return nil, optionsErr
	}

	optionMeaningErr := validateSingleObjectiveRunOptions(resolved)
	if optionMeaningErr != nil {
		return nil, optionMeaningErr
	}

	validationErr := validateConfig(config)
	if validationErr != nil {
		return nil, validationErr
	}

	// The initial population is checked only after the config is known good,
	// because the check is against ProblemSize, NPop and the bounds.
	populationErr := validateInitialPopulation(config, resolved)
	if populationErr != nil {
		return nil, populationErr
	}

	// A configured seed is reproducible. A directly supplied generator has no
	// introspectable seed and is marked unknown in the result.
	rng, seed, seedKnown := resolveRandomSource(config)

	state, initializationErr := initializeRun(ctx, config, resolved, rng)
	if initializationErr != nil {
		return nil, initializationErr
	}

	if effectiveFidelityMode(config) != FidelityMATLAB && !hasFiniteObjective(state.swarm) {
		return nil, ErrNoFiniteObjective
	}

	tracker := newConvergenceTracker(config.Convergence, state.food, state.evaluator)

	logOptimizationStarted(ctx, resolved.logger, config)

	curve := make([]float64, 0, config.MaxIterations)
	reason := TerminationMaxIterations
	completed := 0

	for t := range config.MaxIterations {
		ctxErr := ctx.Err()
		if ctxErr != nil {
			return nil, ctxErr
		}

		if effectiveFidelityMode(config) == FidelityMATLAB {
			stopReason, stop, iterationErr := runMATLABIteration(
				ctx, config, state, resolved, tracker, t+1, rng, &curve,
			)
			if iterationErr != nil {
				return nil, iterationErr
			}

			completed = t + 1

			if stop {
				reason = stopReason

				break
			}

			continue
		}

		weights := computeWeights(config, t+1, config.MaxIterations, rng)
		prepareSwarm(state, config, weights, rng)

		evaluationErr := state.evaluate(ctx)
		if evaluationErr != nil {
			return nil, evaluationErr
		}

		curve = append(curve, state.food.Cost)
		completed = t + 1

		notifyProgress(resolved.observer, completed, state.funcEvals, state.food)
		notifyPopulation(resolved.populationObserver, completed, state.funcEvals,
			state.food, state.enemy, state.swarm)
		logIterationCompleted(ctx, resolved.logger, completed, state.funcEvals, state.food)

		stopReason, stop := tracker.observe(completed, state.food)
		if stop {
			reason = stopReason

			break
		}
	}

	result := &Result{
		ConvergenceCurve:  curve,
		TerminationReason: reason,
		GlobalBest:        state.food,
		Worst:             state.enemy,
		FuncEvalCount:     state.funcEvals,
		IterationCount:    completed,
		Seed:              seed,
		SeedKnown:         seedKnown,
	}

	logOptimizationCompleted(ctx, resolved.logger, result)

	return result, nil
}

// runMATLABIteration reproduces DA.m's evaluate-before-move lifecycle. The
// evaluated population is copied before movement so observer costs always
// describe the positions delivered with them; notification and early stopping
// still happen after the generation's move, as promised by the Go API.
func runMATLABIteration(
	ctx context.Context,
	config *Config,
	state *runState,
	options runOptions,
	tracker *convergenceTracker,
	iteration int,
	rng *rand.Rand,
	curve *[]float64,
) (TerminationReason, bool, error) {
	weights := computeWeights(config, iteration, config.MaxIterations, rng)

	evaluationErr := state.evaluate(ctx)
	if evaluationErr != nil {
		return "", false, evaluationErr
	}

	if !hasFiniteObjective(state.swarm) {
		return "", false, ErrNoFiniteObjective
	}

	state.updateMATLABMovementEnemy(config.LowerBound, config.UpperBound)

	var evaluatedSwarm []Dragonfly
	if options.populationObserver != nil {
		evaluatedSwarm = cloneDragonflies(state.swarm)
	}

	*curve = append(*curve, state.food.Cost)
	prepareSwarm(state, config, weights, rng)

	contextErr := ctx.Err()
	if contextErr != nil {
		return "", false, contextErr
	}

	notifyProgress(options.observer, iteration, state.funcEvals, state.food)
	notifyPopulation(options.populationObserver, iteration, state.funcEvals,
		state.food, state.movementEnemy, evaluatedSwarm)
	logIterationCompleted(ctx, options.logger, iteration, state.funcEvals, state.food)

	reason, stop := tracker.observe(iteration, state.food)

	return reason, stop, nil
}

// initializeRun builds the starting swarm. Paper mode evaluates it immediately
// so that the food and enemy positions the first step is computed against
// already exist. MATLAB mode leaves it unscored because DA.m evaluates the
// current population at the start of each iteration.
//
// Following the reference DA.m, both the position and the step are drawn
// uniformly from [LowerBound, UpperBound]. Seeding ΔX from the whole search box
// rather than from zero is what gives the first few iterations their momentum;
// the step clamp reins it in from the first move onwards.
//
// A caller-supplied initial population replaces the drawn positions of the
// leading slots only. The step draw and every slot beyond the supplied
// positions are left alone, so seeding the first few dragonflies does not shift
// the random stream the rest of the swarm is built from.
//
// The evaluation pool, when the run is a parallel one, is built here and lives
// as long as the run. It returns an error only when the context is canceled
// before the initial population has been scored.
func initializeRun(
	ctx context.Context, config *Config, options runOptions, rng *rand.Rand,
) (*runState, error) {
	swarm := make([]Dragonfly, config.NPop)
	for i := range swarm {
		position := unifrndVec(config.LowerBound, config.UpperBound, config.ProblemSize, rng)
		step := unifrndVec(config.LowerBound, config.UpperBound, config.ProblemSize, rng)

		if i < len(options.initialPositions) {
			position = copyVec(options.initialPositions[i])
		}

		swarm[i] = Dragonfly{
			Position: position,
			Step:     step,
			Cost:     math.Inf(1),
		}
	}

	state := &runState{
		evaluator: newConstraintEvaluator(config.ObjectiveFunc, config.Constraints),
		swarm:     swarm,
		// The food starts maximally infeasible as well as maximally costly, so
		// that the first candidate displaces it under either constraint policy.
		// The enemy starts feasible and maximally cheap for the mirror-image
		// reason: under Deb's rules an infeasible candidate is worse than a
		// feasible one, so the first infeasible dragonfly seen becomes the
		// enemy, which is what the enemy term should be steering away from.
		// The position slices are allocated up front because
		// copyDragonflyToBest overwrites them in place rather than
		// reallocating on every improvement.
		food: Best{
			Position:            make([]float64, config.ProblemSize),
			Cost:                math.Inf(1),
			ConstraintViolation: math.Inf(1),
		},
		enemy: Best{
			Position: make([]float64, config.ProblemSize),
			Cost:     math.Inf(-1),
		},
		movementEnemy: Best{
			Position: make([]float64, config.ProblemSize),
			Cost:     math.Inf(-1),
		},
	}

	if config.EnableParallel {
		state.pool = newEvaluationPool(state.evaluator, effectiveMaxWorkers(config), config.NPop)
	}

	if effectiveFidelityMode(config) != FidelityMATLAB {
		evaluationErr := state.evaluate(ctx)
		if evaluationErr != nil {
			return nil, evaluationErr
		}
	}

	return state, nil
}

// updateMATLABMovementEnemy applies DA.m's strict-interior guard to the enemy
// used for movement. The public enemy is updated independently by evaluate and
// therefore remains the true worst evaluated candidate even when every point
// lies on a bound.
func (state *runState) updateMATLABMovementEnemy(lower, upper float64) {
	for i := range state.swarm {
		fly := &state.swarm[i]
		if !strictlyInsideBounds(fly.Position, lower, upper) {
			continue
		}

		if state.evaluator.better(
			evaluationFromBest(state.movementEnemy), evaluationFromDragonfly(fly),
		) {
			copyDragonflyToBest(&state.movementEnemy, fly)
		}
	}
}

func strictlyInsideBounds(position []float64, lower, upper float64) bool {
	for _, value := range position {
		if !(value > lower && value < upper) {
			return false
		}
	}

	return len(position) > 0
}

// evaluateSwarm scores every dragonfly and updates the food (best) and enemy
// (worst) incumbents.
//
// Scoring goes through the run's constraintEvaluator rather than calling the
// objective directly, which is what keeps Dragonfly.ConstraintViolation
// populated and guarantees exactly one constraint evaluation per objective
// evaluation. Every objective result passes through sanitizeCost, so a NaN or
// -Inf from a misbehaving objective becomes +Inf rather than an incumbent no
// later candidate could ever displace.
//
// The food and enemy are chosen by the evaluator's ranking, not by comparing
// Cost. Under a constrained run a raw-cost comparison would happily steer the
// swarm toward an infeasible optimum; the enemy uses the same ranking read
// backwards, so the worst candidate is the one the incumbent enemy still beats.
//
// All calls happen on the calling goroutine. The parallel path in
// evaluateParallelStep produces exactly the same state by exactly the same
// rules; see runState.evaluate.
func (state *runState) evaluateSwarm() {
	for i := range state.swarm {
		fly := &state.swarm[i]
		state.evaluator.evaluateDragonfly(fly, true)
		state.funcEvals++

		if state.evaluator.betterDragonflyThanBest(fly, state.food) {
			copyDragonflyToBest(&state.food, fly)
		}

		if state.evaluator.better(evaluationFromBest(state.enemy), evaluationFromDragonfly(fly)) {
			copyDragonflyToBest(&state.enemy, fly)
		}
	}
}

// evaluate scores the swarm and updates the food source and the enemy, through
// the worker pool when the run is parallel and inline when it is not.
//
// Both paths compute the same thing. Cancellation is checked before and after
// the sequential batch; the parallel path can also stop between objective
// calls without committing a partial batch.
func (state *runState) evaluate(ctx context.Context) error {
	if state.pool == nil {
		contextErr := ctx.Err()
		if contextErr != nil {
			return contextErr
		}

		state.evaluateSwarm()

		return ctx.Err()
	}

	count, err := evaluateParallelStep(ctx, state, state.pool)
	if err != nil {
		return err
	}

	state.funcEvals += count

	return nil
}

func hasFiniteObjective(swarm []Dragonfly) bool {
	for i := range swarm {
		if isFinite(swarm[i].Cost) {
			return true
		}
	}

	return false
}

// foodInRadius reports whether the food source lies inside the per-dimension
// radius around position, the reference implementation's all(dist2Food <= r).
//
// This deliberately does not reuse withinRadius. That helper rejects an
// all-zero distance so that a dragonfly is never its own neighbor, which is
// correct for the neighborhood scan and wrong here: a dragonfly sitting
// exactly on the food source is at distance zero in every dimension, and
// withinRadius would report the food as out of range, sending the closest
// dragonfly in the swarm down the Lévy-walk branch. DA.m's food test has no
// such exclusion, and neither does this one.
//
// A missing or mismatched food vector counts as out of range, and a NaN
// component fails the test because every comparison against NaN is false.
func referenceInRadius(position, reference []float64, radius float64) bool {
	if len(position) != len(reference) || len(position) == 0 {
		return false
	}

	for k := range position {
		// Written as a negated <= rather than a >, so that a NaN component
		// fails the test: every comparison against NaN is false.
		if !(math.Abs(position[k]-reference[k]) <= radius) {
			return false
		}
	}

	return true
}

func foodInRadius(position, food []float64, radius float64) bool {
	return referenceInRadius(position, food, radius)
}
