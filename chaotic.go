// Chaotic Dragonfly Algorithm (CDA) and its ten published maps.

package dragonfly

import (
	"context"
	"fmt"
	"math"
)

// ChaosMap names a deterministic recurrence used by CDA to generate the
// movement coefficients that ordinary DA draws uniformly.
type ChaosMap string

const (
	ChaosChebyshev  ChaosMap = "chebyshev"
	ChaosCircle     ChaosMap = "circle"
	ChaosGauss      ChaosMap = "gauss"
	ChaosIterative  ChaosMap = "iterative"
	ChaosLogistic   ChaosMap = "logistic"
	ChaosPiecewise  ChaosMap = "piecewise"
	ChaosSine       ChaosMap = "sine"
	ChaosSinger     ChaosMap = "singer"
	ChaosSinusoidal ChaosMap = "sinusoidal"
	ChaosTent       ChaosMap = "tent"
)

var chaosMapOrder = []ChaosMap{
	ChaosChebyshev,
	ChaosCircle,
	ChaosGauss,
	ChaosIterative,
	ChaosLogistic,
	ChaosPiecewise,
	ChaosSine,
	ChaosSinger,
	ChaosSinusoidal,
	ChaosTent,
}

// ChaosMapNames returns the ten CDA maps in their published table order.
func ChaosMapNames() []ChaosMap {
	names := make([]ChaosMap, len(chaosMapOrder))
	copy(names, chaosMapOrder)

	return names
}

// ChaoticMapValue advances one chaotic map. iteration is one-based for the
// Chebyshev recurrence; other maps ignore it. The returned coefficient is
// in the map's published range: [-1,1] for Chebyshev and Iterative, [0,1]
// for the other eight maps.
func ChaoticMapValue(name ChaosMap, value float64, iteration int) (float64, error) {
	if !isFinite(value) {
		return 0, fmt.Errorf("chaotic map input must be finite, got %v", value)
	}

	x := value
	if name != ChaosChebyshev && name != ChaosIterative {
		x = unitInterval(x)
	} else {
		x = max(-1, min(1, x))
	}

	next, found := advanceChaosMap(name, x, iteration)
	if !found {
		return 0, fmt.Errorf("unknown chaos map %q (expected one of %v)", name, ChaosMapNames())
	}

	if !isFinite(next) {
		return 0, fmt.Errorf("chaos map %q produced non-finite value %v", name, next)
	}

	if name == ChaosChebyshev || name == ChaosIterative {
		return max(-1, min(1, next)), nil
	}

	return unitInterval(next), nil
}

func advanceChaosMap(name ChaosMap, x float64, iteration int) (float64, bool) {
	switch name {
	case ChaosChebyshev:
		return math.Cos(float64(max(iteration, 1)) * math.Acos(x)), true
	case ChaosCircle:
		return math.Mod(x+0.2-(0.5/(2*math.Pi))*math.Sin(2*math.Pi*x), 1), true
	case ChaosGauss:
		if x == 0 {
			return 0, true
		}

		return math.Mod(1/x, 1), true
	case ChaosIterative:
		if x == 0 {
			return 0, true
		}

		return math.Sin(0.7 * math.Pi / x), true
	case ChaosLogistic:
		return 4 * x * (1 - x), true
	case ChaosPiecewise:
		return piecewiseChaos(x, 0.4), true
	case ChaosSine:
		return math.Sin(math.Pi * x), true
	case ChaosSinger:
		return 1.07 * (7.86*x - 23.31*x*x + 28.75*x*x*x - 13.302875*x*x*x*x), true
	case ChaosSinusoidal:
		return 2.3 * x * x * math.Sin(math.Pi*x), true
	case ChaosTent:
		if x < 0.7 {
			return x / 0.7, true
		}

		return (1 - x) / (1 - 0.7), true
	}

	return 0, false
}

func unitInterval(value float64) float64 {
	value = math.Abs(value)
	if value > 1 && value-1 <= 1e-15 {
		return 1
	}

	if value > 1 {
		value = math.Mod(value, 1)
	}

	return min(1, max(0, value))
}

func piecewiseChaos(x, p float64) float64 {
	switch {
	case x <= p:
		return x / p
	case x <= 0.5:
		return (x - p) / (0.5 - p)
	case x <= 1-p:
		return (1 - p - x) / (0.5 - p)
	default:
		return (1 - x) / p
	}
}

type chaosSequence struct {
	name  ChaosMap
	value float64
	step  int
}

func (sequence *chaosSequence) next() float64 {
	sequence.step++

	value, err := ChaoticMapValue(sequence.name, sequence.value, sequence.step)
	if err != nil {
		// Configuration and the previous value are validated, so this is only a
		// numeric safety net for a hostile recurrence state.
		sequence.value = 0.7
		return sequence.value
	}

	sequence.value = value

	return value
}

func computeChaoticWeights(
	config *Config,
	iteration, maxIterations int,
	sequence *chaosSequence,
) weightSchedule {
	span := config.UpperBound - config.LowerBound

	chaos := sequence.next()

	return weightSchedule{
		Inertia:    chaos,
		Separation: resolveWeight(config.SeparationWeight, chaos),
		Alignment:  resolveWeight(config.AlignmentWeight, chaos),
		Cohesion:   resolveWeight(config.CohesionWeight, chaos),
		Food:       resolveWeight(config.FoodWeight, chaos),
		Enemy:      resolveWeight(config.EnemyWeight, chaos),
		Radius:     neighborhoodRadius(config, iteration, maxIterations),
		MaxStep:    span * config.MaxStepRatio,
	}
}

// OptimizeChaotic runs CDA with a background context.
func OptimizeChaotic(config *Config, options ...RunOption) (*Result, error) {
	return OptimizeChaoticContext(context.Background(), config, options...)
}

// OptimizeChaoticContext runs Sayed, Tharwat and Hassanien's CDA. It replaces
// the five DA movement weights and inertia with one chaotic-map value per
// iteration. Evaluation accounting remains NPop*(T+1).
func OptimizeChaoticContext(
	ctx context.Context,
	config *Config,
	options ...RunOption,
) (*Result, error) {
	run, err := setupImprovedRun(ctx, config, options, validateChaoticConfig)
	if err != nil {
		return nil, err
	}

	sequence := &chaosSequence{name: config.ChaosMap, value: config.ChaosSeed, step: 0}
	curve := make([]float64, 0, config.MaxIterations)
	reason := TerminationMaxIterations
	completed := 0

	logOptimizationStarted(ctx, run.options.logger, config)

	for t := range config.MaxIterations {
		contextErr := ctx.Err()
		if contextErr != nil {
			return nil, contextErr
		}

		weights := computeChaoticWeights(config, t+1, config.MaxIterations, sequence)
		prepareSwarm(run.state, config, weights, run.rng)

		evaluationErr := run.state.evaluate(ctx)
		if evaluationErr != nil {
			return nil, evaluationErr
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

func validateChaoticConfig(config *Config) error {
	if config.ChaosMap == "" {
		return fmt.Errorf("chaos_map must be set (expected one of %v)", ChaosMapNames())
	}

	if !isFinite(config.ChaosSeed) || config.ChaosSeed <= 0 || config.ChaosSeed >= 1 {
		return fmt.Errorf("chaos_seed must be finite and strictly between 0 and 1, got %v", config.ChaosSeed)
	}

	_, err := ChaoticMapValue(config.ChaosMap, config.ChaosSeed, 1)

	return err
}

// ValidateChaoticConfig checks the shared continuous Config contract plus
// CDA's selected chaotic map, initial condition and fidelity restriction.
func ValidateChaoticConfig(config *Config) error {
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

	return validateChaoticConfig(config)
}
