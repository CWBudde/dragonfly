// Memory-based Hybrid Dragonfly Algorithm (MHDA).

package dragonfly

import (
	"context"
	"math"
	"math/rand"
)

// OptimizeMemoryHybrid runs MHDA with a background context.
func OptimizeMemoryHybrid(config *Config, options ...RunOption) (*Result, error) {
	return OptimizeMemoryHybridContext(context.Background(), config, options...)
}

// OptimizeMemoryHybridContext runs Ranjini and Murugan's memory-based hybrid
// DA. Every iteration first performs the ordinary continuous DA movement, then
// updates each dragonfly's personal best and uses those memories to initialize
// a PSO exploitation movement. A complete run evaluates NPop*(2*T+1)
// candidates: initialization, one DA batch and one PSO batch per iteration.
func OptimizeMemoryHybridContext(
	ctx context.Context,
	config *Config,
	options ...RunOption,
) (*Result, error) {
	run, err := setupImprovedRun(ctx, config, options, validateMemoryHybridConfig)
	if err != nil {
		return nil, err
	}

	personalBest := cloneDragonflies(run.state.swarm)
	curve := make([]float64, 0, config.MaxIterations)
	reason := TerminationMaxIterations
	completed := 0

	logOptimizationStarted(ctx, run.options.logger, config)

	for t := range config.MaxIterations {
		contextErr := ctx.Err()
		if contextErr != nil {
			return nil, contextErr
		}

		weights := computeWeights(config, t+1, config.MaxIterations, run.rng)
		prepareSwarm(run.state, config, weights, run.rng)

		evaluationErr := run.state.evaluate(ctx)
		if evaluationErr != nil {
			return nil, evaluationErr
		}

		updatePersonalBest(run.state.evaluator, personalBest, run.state.swarm)
		candidates := preparePSOCandidates(config, run.state.swarm, personalBest,
			run.state.food, weights.Inertia, run.rng)

		candidateErr := evaluateCandidates(ctx, run.state, candidates)
		if candidateErr != nil {
			return nil, candidateErr
		}

		updatePersonalBest(run.state.evaluator, personalBest, run.state.swarm)

		completed = t + 1

		stopReason, stop := finishImprovedIteration(ctx, run, completed, &curve)
		if stop {
			reason = stopReason
			break
		}
	}

	result := improvedResult(run, curve, completed, reason)
	logOptimizationCompleted(ctx, run.options.logger, result)

	return result, nil
}

func validateMemoryHybridConfig(config *Config) error {
	cognitiveErr := validPositiveFinite("pso_cognitive_weight", config.PSOCognitiveWeight)
	if cognitiveErr != nil {
		return cognitiveErr
	}

	return validPositiveFinite("pso_social_weight", config.PSOSocialWeight)
}

// ValidateMemoryHybridConfig checks both the shared Config contract and the
// MHDA-only acceleration constants and fidelity restrictions.
func ValidateMemoryHybridConfig(config *Config) error {
	configErr := validateConfig(config)
	if configErr != nil {
		return configErr
	}

	if config.UseBinary {
		return ErrBinaryConfigOnImprovedContinuousVariant
	}

	if effectiveFidelityMode(config) == FidelityMATLAB {
		return ErrImprovedVariantMATLAB
	}

	return validateMemoryHybridConfig(config)
}

func updatePersonalBest(evaluator *constraintEvaluator, memory, swarm []Dragonfly) {
	for i := range swarm {
		if evaluator.betterDragonfly(&swarm[i], &memory[i]) {
			memory[i] = cloneDragonfly(swarm[i])
		}
	}
}

func preparePSOCandidates(
	config *Config,
	swarm, personalBest []Dragonfly,
	globalBest Best,
	inertia float64,
	rng *rand.Rand,
) []Dragonfly {
	candidates := make([]Dragonfly, len(swarm))
	for i := range swarm {
		fly := &swarm[i]
		memory := &personalBest[i]
		candidate := candidateFrom(*fly)

		dimensions := min(len(candidate.Position), len(candidate.Step), len(fly.Step),
			len(memory.Position), len(globalBest.Position))
		for j := range dimensions {
			r1 := rng.Float64()
			r2 := rng.Float64()

			candidate.Step[j] = inertia*fly.Step[j] +
				config.PSOCognitiveWeight*r1*(memory.Position[j]-fly.Position[j]) +
				config.PSOSocialWeight*r2*(globalBest.Position[j]-fly.Position[j])
			if math.IsNaN(candidate.Step[j]) {
				candidate.Step[j] = 0
			}

			candidate.Position[j] += candidate.Step[j]
		}

		sanitizeVec(candidate.Position, config.LowerBound, config.UpperBound, rng)
		applyBounds(candidate.Position, candidate.Step, config.LowerBound, config.UpperBound,
			effectiveBoundaryMethod(config), rng)
		candidates[i] = candidate
	}

	return candidates
}
