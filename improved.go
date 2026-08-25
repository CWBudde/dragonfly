// Shared lifecycle and candidate evaluation for the improved single-objective
// Dragonfly variants.

package dragonfly

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// ErrImprovedVariantMATLAB is returned when an improved variant is combined
// with FidelityMATLAB. MHDA, CDA and QGDA were published independently of the
// author's DA.m lifecycle, so mixing their operators with that compatibility
// mode would create an unnamed algorithm.
var ErrImprovedVariantMATLAB = errors.New(
	"improved DA variants support FidelityPaper only")

// ErrBinaryConfigOnImprovedContinuousVariant is returned when an improved
// continuous variant is handed a binary configuration. All three variants here are
// continuous algorithms.
var ErrBinaryConfigOnImprovedContinuousVariant = errors.New(
	"binary config cannot run through a continuous improved DA variant")

type improvedRun struct {
	state     *runState
	tracker   *convergenceTracker
	rng       *rand.Rand
	options   runOptions
	seed      int64
	seedKnown bool
}

func setupImprovedRun(
	ctx context.Context,
	config *Config,
	options []RunOption,
	validateVariant func(*Config) error,
) (*improvedRun, error) {
	contextErr := requireContext(ctx)
	if contextErr != nil {
		return nil, contextErr
	}

	resolved, err := resolveRunOptions(options)
	if err != nil {
		return nil, err
	}

	optionMeaningErr := validateSingleObjectiveRunOptions(resolved)
	if optionMeaningErr != nil {
		return nil, optionMeaningErr
	}

	configErr := validateConfig(config)
	if configErr != nil {
		return nil, configErr
	}

	if config.UseBinary {
		return nil, ErrBinaryConfigOnImprovedContinuousVariant
	}

	if effectiveFidelityMode(config) == FidelityMATLAB {
		return nil, ErrImprovedVariantMATLAB
	}

	variantErr := validateVariant(config)
	if variantErr != nil {
		return nil, variantErr
	}

	populationErr := validateInitialPopulation(config, resolved)
	if populationErr != nil {
		return nil, populationErr
	}

	rng, seed, seedKnown := resolveRandomSource(config)

	state, err := initializeRun(ctx, config, resolved, rng)
	if err != nil {
		return nil, err
	}

	if !hasFiniteObjective(state.swarm) {
		return nil, ErrNoFiniteObjective
	}

	return &improvedRun{
		state:     state,
		tracker:   newConvergenceTracker(config.Convergence, state.food, state.evaluator),
		rng:       rng,
		options:   resolved,
		seed:      seed,
		seedKnown: seedKnown,
	}, nil
}

// evaluateCandidates scores a fully prepared candidate population, updates
// the run-wide best/worst observations, and greedily keeps a candidate only
// when it improves the corresponding current dragonfly. RNG is never touched
// here, so parallel and sequential runs remain bit-identical.
func evaluateCandidates(
	ctx context.Context,
	state *runState,
	candidates []Dragonfly,
) error {
	if len(candidates) != len(state.swarm) {
		return fmt.Errorf("candidate population has %d members, want %d", len(candidates), len(state.swarm))
	}

	contextErr := ctx.Err()
	if contextErr != nil {
		return contextErr
	}

	if state.pool != nil {
		batch, err := state.pool.evaluate(ctx, candidates)
		if err != nil {
			return err
		}

		for i := range candidates {
			candidates[i].Cost = batch.scores[i].Cost
			candidates[i].ConstraintViolation = batch.scores[i].ConstraintViolation
		}

		mergeBest(&state.food, batch.best)
		mergeBest(&state.enemy, batch.worst)
	} else {
		for i := range candidates {
			state.evaluator.evaluateDragonfly(&candidates[i])
		}

		contextErr := ctx.Err()
		if contextErr != nil {
			return contextErr
		}

		for i := range candidates {
			candidate := &candidates[i]
			if state.evaluator.betterDragonflyThanBest(candidate, state.food) {
				copyDragonflyToBest(&state.food, candidate)
			}

			if state.evaluator.better(evaluationFromBest(state.enemy), evaluationFromDragonfly(candidate)) {
				copyDragonflyToBest(&state.enemy, candidate)
			}
		}
	}

	state.funcEvals += len(candidates)
	for i := range candidates {
		if state.evaluator.betterDragonfly(&candidates[i], &state.swarm[i]) {
			state.swarm[i] = candidates[i]
		}
	}

	return nil
}

func candidateFrom(fly Dragonfly) Dragonfly {
	return Dragonfly{
		Position:            copyVec(fly.Position),
		Step:                copyVec(fly.Step),
		Cost:                math.Inf(1),
		ConstraintViolation: math.Inf(1),
	}
}

func finishImprovedIteration(
	ctx context.Context,
	run *improvedRun,
	iteration int,
	curve *[]float64,
) (TerminationReason, bool) {
	state := run.state
	*curve = append(*curve, state.food.Cost)

	notifyProgress(run.options.observer, iteration, state.funcEvals, state.food)
	notifyPopulation(run.options.populationObserver, iteration, state.funcEvals,
		state.food, state.enemy, state.swarm)
	logIterationCompleted(ctx, run.options.logger, iteration, state.funcEvals, state.food)

	return run.tracker.observe(iteration, state.food)
}

func improvedResult(run *improvedRun, curve []float64, completed int, reason TerminationReason) *Result {
	return &Result{
		ConvergenceCurve:  curve,
		TerminationReason: reason,
		GlobalBest:        run.state.food,
		Worst:             run.state.enemy,
		FuncEvalCount:     run.state.funcEvals,
		IterationCount:    completed,
		Seed:              run.seed,
		SeedKnown:         run.seedKnown,
	}
}

func validPositiveFinite(name string, value float64) error {
	if !isFinite(value) || value <= 0 {
		return fmt.Errorf("%s must be a positive finite number, got %v", name, value)
	}

	return nil
}
