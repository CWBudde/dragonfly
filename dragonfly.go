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
	"math"
	"math/rand"
	"time"
)

// runState is the mutable state of one optimization run: the swarm itself, the
// two extreme positions every step is computed against, and the objective call
// counter.
//
// It exists so the phases of the loop -- initialize, evaluate, move -- can be
// separate functions without threading five parameters and four return values
// through each of them.
type runState struct {
	swarm []Dragonfly
	food  Best
	enemy Best
	// funcEvals counts every call to config.ObjectiveFunc, including the ones
	// made while initializing the swarm.
	funcEvals int
}

// Optimize runs the Dragonfly Algorithm with a background context.
func Optimize(config *Config) (*Result, error) {
	return OptimizeContext(context.Background(), config)
}

// OptimizeContext runs the Dragonfly Algorithm, honoring context cancellation.
//
// Cancellation is checked at the top of every iteration. A canceled run
// returns a nil result and ctx.Err(); partial results are deliberately not
// reported, so a caller cannot mistake an aborted run for a completed one.
func OptimizeContext(ctx context.Context, config *Config) (*Result, error) {
	validationErr := validateConfig(config)
	if validationErr != nil {
		return nil, validationErr
	}

	// The seed is drawn whether or not it is used, so Result.Seed is always
	// populated. When the caller supplied their own *rand.Rand it is that
	// generator, not this seed, that drives the run -- the recorded value is
	// then the unused fallback and reproducing the run means reusing the
	// caller's generator. This is Mayfly's convention.
	seed := time.Now().UnixNano()
	if config.Rand == nil {
		config.Rand = rand.New(rand.NewSource(seed))
	}

	rng := config.Rand
	state := initializeRun(config, rng)

	curve := make([]float64, 0, config.MaxIterations)

	for t := range config.MaxIterations {
		ctxErr := ctx.Err()
		if ctxErr != nil {
			return nil, ctxErr
		}

		weights := computeWeights(config, t, config.MaxIterations, rng)

		for i := range state.swarm {
			moveDragonfly(state, i, config, weights, rng)
		}

		state.evaluateSwarm(config)
		curve = append(curve, state.food.Cost)
	}

	return &Result{
		ConvergenceCurve: curve,
		// Only the iteration cap can end a run today; the target-cost and
		// stagnation criteria land with convergence.go in a later phase.
		TerminationReason: TerminationMaxIterations,
		GlobalBest:        state.food,
		Worst:             state.enemy,
		FuncEvalCount:     state.funcEvals,
		IterationCount:    config.MaxIterations,
		Seed:              seed,
	}, nil
}

// initializeRun builds the starting swarm and evaluates it, so that the food
// and enemy positions the first step is computed against already exist.
//
// Following the reference DA.m, both the position and the step are drawn
// uniformly from [LowerBound, UpperBound]. Seeding ΔX from the whole search box
// rather than from zero is what gives the first few iterations their momentum;
// the step clamp reins it in from the first move onwards.
func initializeRun(config *Config, rng *rand.Rand) *runState {
	swarm := make([]Dragonfly, config.NPop)
	for i := range swarm {
		swarm[i] = Dragonfly{
			Position: unifrndVec(config.LowerBound, config.UpperBound, config.ProblemSize, rng),
			Step:     unifrndVec(config.LowerBound, config.UpperBound, config.ProblemSize, rng),
			Cost:     math.Inf(1),
		}
	}

	state := &runState{
		swarm: swarm,
		food:  Best{Cost: math.Inf(1)},
		enemy: Best{Cost: math.Inf(-1)},
	}

	state.evaluateSwarm(config)

	return state
}

// evaluateSwarm scores every dragonfly and updates the food (best) and enemy
// (worst) incumbents.
//
// Every objective result passes through sanitizeCost, so a NaN or -Inf from a
// misbehaving objective becomes +Inf rather than an incumbent no later
// candidate could ever displace.
//
// All calls happen on the calling goroutine. Parallel evaluation arrives in a
// later phase and depends on every RNG draw staying here.
func (state *runState) evaluateSwarm(config *Config) {
	for i := range state.swarm {
		fly := &state.swarm[i]
		fly.Cost = sanitizeCost(config.ObjectiveFunc(fly.Position))
		state.funcEvals++

		if fly.Cost < state.food.Cost {
			state.food = Best{Position: copyVec(fly.Position), Cost: fly.Cost}
		}

		if fly.Cost > state.enemy.Cost {
			state.enemy = Best{Position: copyVec(fly.Position), Cost: fly.Cost}
		}
	}
}

// moveDragonfly advances one dragonfly by a single step, following PLAN.md §1.2.
//
// Boundary repair ordering: the reference DA.m repairs a dragonfly at the top
// of its per-dragonfly block, so the swarm it computes S, A and C against is
// partly repaired and partly not, and the positions left in the population when
// the loop ends are the unrepaired ones. This implementation repairs at the
// bottom instead, immediately after the position update. The deviation buys two
// properties worth more than bit-fidelity with the MATLAB source: every
// position handed to the objective function is inside [LowerBound, UpperBound],
// and so is every position in the returned Result. Because the repair still
// happens exactly once per dragonfly per iteration, the dynamics are otherwise
// the same -- only the point in the iteration at which the wrap becomes visible
// to the other dragonflies moves.
func moveDragonfly(state *runState, index int, config *Config, weights weightSchedule, rng *rand.Rand) {
	fly := &state.swarm[index]

	neighbors := findNeighbours(state.swarm, index, weights.Radius)
	separation := separationVector(state.swarm, index, neighbors)
	alignment := alignmentVector(state.swarm, index, neighbors)
	cohesion := cohesionVector(state.swarm, index, neighbors)
	food := foodVector(fly.Position, state.food.Position)
	enemy := enemyVector(fly.Position, state.enemy.Position)

	switch {
	case foodInRadius(fly.Position, state.food.Position, weights.Radius):
		applyFullStep(fly, weights, separation, alignment, cohesion, food, enemy)
	case len(neighbors) > 1:
		applySwarmingStep(fly, weights, separation, alignment, cohesion, rng)
	default:
		applyLevyWalk(fly, config, rng)
	}

	sanitizeVec(fly.Position, config.LowerBound, config.UpperBound, rng)
	applyBounds(fly.Position, fly.Step, config.LowerBound, config.UpperBound,
		effectiveBoundaryMethod(config), rng)
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
func foodInRadius(position, food []float64, radius float64) bool {
	if len(position) != len(food) || len(position) == 0 {
		return false
	}

	for k := range position {
		// Written as a negated <= rather than a >, so that a NaN component
		// fails the test: every comparison against NaN is false.
		if !(math.Abs(position[k]-food[k]) <= radius) {
			return false
		}
	}

	return true
}

// applyFullStep is the food-in-range branch: the five-factor step of the paper,
//
//	ΔX = (s·S + a·A + c·C + f·F + e·E) + w·ΔX
//
// with the weights taken from the schedule, one set per iteration rather than
// per dimension.
func applyFullStep(fly *Dragonfly, weights weightSchedule, separation, alignment, cohesion, food, enemy []float64) {
	for j := range fly.Step {
		fly.Step[j] = weights.Separation*separation[j] +
			weights.Alignment*alignment[j] +
			weights.Cohesion*cohesion[j] +
			weights.Food*food[j] +
			weights.Enemy*enemy[j] +
			weights.Inertia*fly.Step[j]
	}

	commitStep(fly, weights.MaxStep)
}

// applySwarmingStep is the food-out-of-range branch for a dragonfly with more
// than one neighbor: local swarming only, with no food and no enemy term,
//
//	ΔX_j = w·ΔX_j + rand·A_j + rand·C_j + rand·S_j
//
// The three rand factors are drawn per dimension, inside this loop, exactly as
// DA.m draws them -- not once per dragonfly. Drawing them once would make the
// three primitives share a scaling factor within a step and change the search
// behavior, not just the random stream. The draw order (alignment, cohesion,
// separation) is the reference implementation's evaluation order.
func applySwarmingStep(
	fly *Dragonfly, weights weightSchedule, separation, alignment, cohesion []float64, rng *rand.Rand,
) {
	for j := range fly.Step {
		randAlignment := unifrnd(0, 1, rng)
		randCohesion := unifrnd(0, 1, rng)
		randSeparation := unifrnd(0, 1, rng)

		fly.Step[j] = weights.Inertia*fly.Step[j] +
			randAlignment*alignment[j] +
			randCohesion*cohesion[j] +
			randSeparation*separation[j]
	}

	commitStep(fly, weights.MaxStep)
}

// applyLevyWalk is the food-out-of-range branch for an isolated dragonfly: the
// multiplicative Lévy random walk X += Levy(d) ⊙ X, after which ΔX is reset to
// zero because the walk replaces the step rather than contributing to it.
//
// With Config.UseLevyWalk disabled the dragonfly stays where it is for this
// iteration and only its step is reset, which is the documented alternative.
func applyLevyWalk(fly *Dragonfly, config *Config, rng *rand.Rand) {
	if config.UseLevyWalk {
		walk := levyVector(len(fly.Position), config.LevyBeta, config.LevyScale, rng)
		for j := range fly.Position {
			fly.Position[j] += walk[j] * fly.Position[j]
		}
	}

	for j := range fly.Step {
		fly.Step[j] = 0
	}
}

// commitStep clamps ΔX componentwise to ±maxStep and adds it to the position.
//
// The step is sanitized before the clamp because clamping cannot repair a NaN:
// every comparison against NaN is false, so a NaN component would survive the
// clamp and poison the position.
func commitStep(fly *Dragonfly, maxStep float64) {
	for j := range fly.Step {
		if math.IsNaN(fly.Step[j]) {
			fly.Step[j] = 0
		}
	}

	clampVec(fly.Step, -maxStep, maxStep)

	for j := range fly.Position {
		fly.Position[j] += fly.Step[j]
	}
}
