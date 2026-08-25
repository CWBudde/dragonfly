// Quantum-behaved and Gaussian mutational Dragonfly Algorithm (QGDA).

package dragonfly

import (
	"context"
	"math"
	"math/rand"
)

// OptimizeQuantum runs QGDA with a background context.
func OptimizeQuantum(config *Config, options ...RunOption) (*Result, error) {
	return OptimizeQuantumContext(context.Background(), config, options...)
}

// OptimizeQuantumContext runs Yu et al.'s QGDA. Each ordinary DA batch is
// followed by a greedy Gaussian mutation batch and a greedy quantum-rotation
// batch. A complete run therefore evaluates NPop*(3*T+1) candidates.
func OptimizeQuantumContext(
	ctx context.Context,
	config *Config,
	options ...RunOption,
) (*Result, error) {
	run, err := setupImprovedRun(ctx, config, options, validateQuantumConfig)
	if err != nil {
		return nil, err
	}

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

		mutants := prepareGaussianCandidates(config, run.state.swarm, run.rng)

		mutationErr := evaluateCandidates(ctx, run.state, mutants)
		if mutationErr != nil {
			return nil, mutationErr
		}

		rotated := prepareQuantumCandidates(config, run.state.swarm, run.state.food,
			run.state.evaluator)

		rotationErr := evaluateCandidates(ctx, run.state, rotated)
		if rotationErr != nil {
			return nil, rotationErr
		}

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

func validateQuantumConfig(config *Config) error {
	mutationErr := validPositiveFinite("gaussian_mutation_weight", config.GaussianMutationWeight)
	if mutationErr != nil {
		return mutationErr
	}

	return validPositiveFinite("quantum_rotation_angle", config.QuantumRotationAngle)
}

// ValidateQuantumConfig checks both the shared Config contract and QGDA's
// mutation/rotation parameters and fidelity restrictions.
func ValidateQuantumConfig(config *Config) error {
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

	return validateQuantumConfig(config)
}

func prepareGaussianCandidates(config *Config, swarm []Dragonfly, rng *rand.Rand) []Dragonfly {
	candidates := make([]Dragonfly, len(swarm))
	for i := range swarm {
		candidate := candidateFrom(swarm[i])

		factor := 1 + config.GaussianMutationWeight*rng.NormFloat64()
		for j := range candidate.Position {
			candidate.Position[j] *= factor
		}

		clampVec(candidate.Position, config.LowerBound, config.UpperBound)
		candidates[i] = candidate
	}

	return candidates
}

func prepareQuantumCandidates(
	config *Config,
	swarm []Dragonfly,
	best Best,
	evaluator *constraintEvaluator,
) []Dragonfly {
	candidates := make([]Dragonfly, len(swarm))
	for i := range swarm {
		candidate := candidateFrom(swarm[i])
		for j := range candidate.Position {
			direction := quantumRotationDirection(evaluator, &swarm[i], best,
				swarm[i].Position[j], best.Position[j])
			candidate.Position[j], _ = QuantumRotate(swarm[i].Position[j], best.Position[j],
				direction*config.QuantumRotationAngle)
		}

		clampVec(candidate.Position, config.LowerBound, config.UpperBound)
		candidates[i] = candidate
	}

	return candidates
}

// quantumRotationDirection implements Table 2's real-valued specialization of
// the quantum-gate direction table. Zero means the two fitness values tie and
// leaves the state unchanged.
func quantumRotationDirection(
	evaluator *constraintEvaluator,
	current *Dragonfly,
	best Best,
	alpha, beta float64,
) float64 {
	currentEvaluation := evaluationFromDragonfly(current)
	bestEvaluation := evaluationFromBest(best)
	currentBetter := evaluator.better(currentEvaluation, bestEvaluation)

	bestBetter := evaluator.better(bestEvaluation, currentEvaluation)
	if !currentBetter && !bestBetter {
		return 0
	}

	direction := 0.0

	switch {
	case alpha*beta > 0:
		direction = 1
	case alpha*beta < 0:
		direction = -1
	case alpha == 0 && beta != 0:
		direction = 1
	}

	if currentBetter {
		direction = -direction
	}

	return direction
}

// QuantumRotate applies the paper's two-state quantum rotation gate.
func QuantumRotate(alpha, beta, angle float64) (float64, float64) {
	cosTheta := math.Cos(angle)
	sinTheta := math.Sin(angle)

	return cosTheta*alpha - sinTheta*beta,
		sinTheta*alpha + cosTheta*beta
}
