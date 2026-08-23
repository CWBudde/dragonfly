// Numeric helpers, effective-value resolvers and configuration validation.

package dragonfly

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// unifrnd generates a random float64 between lower and upper.
func unifrnd(lower, upper float64, rng *rand.Rand) float64 {
	if rng == nil {
		return lower + rand.Float64()*(upper-lower) //nolint:gosec // math/rand is the point of a metaheuristic
	}

	return lower + rng.Float64()*(upper-lower)
}

// unifrndVec generates a vector of random float64 values between lower and upper.
func unifrndVec(lower, upper float64, size int, rng *rand.Rand) []float64 {
	vec := make([]float64, size)
	for i := range vec {
		vec[i] = unifrnd(lower, upper, rng)
	}

	return vec
}

// randn generates a normally distributed random number.
func randn(rng *rand.Rand) float64 {
	if rng == nil {
		return rand.NormFloat64() //nolint:gosec // math/rand is the point of a metaheuristic
	}

	return rng.NormFloat64()
}

// maxVec raises every component of vec that falls below lower to lower, in
// place. It is the element-wise maximum of a vector and a scalar.
func maxVec(vec []float64, lower float64) {
	for i := range vec {
		if vec[i] < lower {
			vec[i] = lower
		}
	}
}

// minVec lowers every component of vec that exceeds upper to upper, in place.
// It is the element-wise minimum of a vector and a scalar.
func minVec(vec []float64, upper float64) {
	for i := range vec {
		if vec[i] > upper {
			vec[i] = upper
		}
	}
}

// clampVec confines every component of vec to [lower, upper], in place.
func clampVec(vec []float64, lower, upper float64) {
	maxVec(vec, lower)
	minVec(vec, upper)
}

// sanitizeVec repairs NaN and infinite components of vec by redrawing them
// uniformly from [lower, upper].
//
// The Lévy walk is the reason this exists: a heavy-tailed multiplicative step
// can overflow to ±Inf in a single iteration, and Inf - Inf in the next
// separation or cohesion sum turns the whole vector into NaN. Repairing the
// position instead of aborting keeps one unlucky draw from poisoning the run.
func sanitizeVec(vec []float64, lower, upper float64, rng *rand.Rand) {
	for i := range vec {
		if math.IsNaN(vec[i]) || math.IsInf(vec[i], 0) {
			vec[i] = unifrnd(lower, upper, rng)
		}
	}
}

// sanitizeCost maps an unusable cost to positive infinity, the worst value the
// comparisons in this package can express.
//
// NaN is mapped because every comparison against it is false, so an objective
// that returns one would leave a candidate neither better nor worse than any
// other and could never be discarded. Negative infinity is mapped for the same
// reason from the other side: it would win every comparison for the rest of the
// run and pin the incumbent to a position that carries no information. Positive
// infinity is already the worst value and is returned unchanged.
//
// This differs from Mayfly, which maps to the large finite values ±1e100.
func sanitizeCost(cost float64) float64 {
	if math.IsNaN(cost) || math.IsInf(cost, -1) {
		return math.Inf(1)
	}

	return cost
}

// copyVec returns a fresh copy of vec. A nil input yields a nil result.
func copyVec(vec []float64) []float64 {
	if vec == nil {
		return nil
	}

	clone := make([]float64, len(vec))
	copy(clone, vec)

	return clone
}

// applyBounds returns an out-of-range position to [lower, upper], adjusting the
// matching component of step according to method.
//
// The three methods differ in what leaving the box means:
//
//   - BoundaryWrap treats the search space as a torus. A component past a bound
//     reappears at the opposite bound and its step is redrawn from [0,1). This
//     is the paper's rule and part of the algorithm's exploration behaviour.
//   - BoundaryClamp pins the component to the bound it crossed and leaves the
//     step alone, which is what Mayfly does.
//   - BoundaryReflect mirrors the component back into the box and flips the sign
//     of its step, as if the bound were a wall. Repeated reflection is applied
//     until the component lands inside, so a step longer than the box width
//     still terminates.
//
// An unrecognised method falls back to wrapping, so a zero-valued Config that
// never reached validateConfig still behaves like the paper.
func applyBounds(position, step []float64, lower, upper float64, method BoundaryMethod, rng *rand.Rand) {
	for i := range position {
		switch method {
		case BoundaryClamp:
			applyClampAt(position, i, lower, upper)
		case BoundaryReflect:
			applyReflectAt(position, step, i, lower, upper)
		default:
			applyWrapAt(position, step, i, lower, upper, rng)
		}
	}
}

func applyClampAt(position []float64, i int, lower, upper float64) {
	if position[i] > upper {
		position[i] = upper
	}

	if position[i] < lower {
		position[i] = lower
	}
}

func applyWrapAt(position, step []float64, i int, lower, upper float64, rng *rand.Rand) {
	switch {
	case position[i] > upper:
		position[i] = lower
	case position[i] < lower:
		position[i] = upper
	default:
		return
	}

	if i < len(step) {
		step[i] = unifrnd(0, 1, rng)
	}
}

func applyReflectAt(position, step []float64, i int, lower, upper float64) {
	if position[i] >= lower && position[i] <= upper {
		return
	}

	width := upper - lower

	// A degenerate box has nothing to reflect off; pin the component instead of
	// looping forever on a zero-width interval.
	if width <= 0 {
		position[i] = lower

		return
	}

	for position[i] < lower || position[i] > upper {
		if position[i] < lower {
			position[i] = 2*lower - position[i]
		}

		if position[i] > upper {
			position[i] = 2*upper - position[i]
		}
	}

	if i < len(step) {
		step[i] = -step[i]
	}
}

// effectiveBoundaryMethod reports the boundary rule a run will actually use.
//
// An unset field resolves to the paper's BoundaryWrap. The value is never
// written back, so a Config keeps meaning what its author wrote.
func effectiveBoundaryMethod(config *Config) BoundaryMethod {
	switch config.BoundaryMethod {
	case BoundaryClamp, BoundaryReflect, BoundaryWrap:
		return config.BoundaryMethod
	default:
		return BoundaryWrap
	}
}

// effectiveMaxWorkers reports the worker count a parallel run will use,
// resolving a non-positive MaxWorkers to one worker per CPU.
func effectiveMaxWorkers(config *Config) int {
	if config.MaxWorkers > 0 {
		return config.MaxWorkers
	}

	return defaultMaxWorkers()
}

// validateConfig rejects a configuration the optimizer cannot run.
//
// The checks are ordered from the required fields a caller most likely forgot
// to the tuning parameters they most likely mistyped, so the first error a
// newcomer sees is about the field they actually left out.
func validateConfig(config *Config) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	if config.ObjectiveFunc == nil {
		return errors.New("ObjectiveFunc must be set")
	}

	if config.ProblemSize <= 0 {
		return fmt.Errorf("problem_size must be positive, got %d", config.ProblemSize)
	}

	if err := validateBounds(config); err != nil {
		return err
	}

	if config.NPop <= 0 {
		return fmt.Errorf("npop must be positive, got %d", config.NPop)
	}

	if config.MaxIterations <= 0 {
		return fmt.Errorf("max_iterations must be positive, got %d", config.MaxIterations)
	}

	if config.MaxWorkers < 0 {
		return fmt.Errorf("max_workers must be non-negative, got %d", config.MaxWorkers)
	}

	if err := validateWeights(config); err != nil {
		return err
	}

	if err := validateSchedules(config); err != nil {
		return err
	}

	switch config.BoundaryMethod {
	case "", BoundaryWrap, BoundaryClamp, BoundaryReflect:
	default:
		return fmt.Errorf("boundary_method must be %q, %q or %q, got %q",
			BoundaryWrap, BoundaryClamp, BoundaryReflect, config.BoundaryMethod)
	}

	if err := validateConvergenceBlock(config.Convergence, config.MaxIterations); err != nil {
		return fmt.Errorf("invalid convergence config: %w", err)
	}

	return nil
}

// validateBounds checks the search box itself. Equal bounds are rejected along
// with inverted ones: a zero-width box makes every schedule that divides by
// (ub-lb) degenerate and there is nothing to search.
func validateBounds(config *Config) error {
	if !isFinite(config.LowerBound) || !isFinite(config.UpperBound) {
		return fmt.Errorf("lower_bound and upper_bound must be finite, got %v and %v",
			config.LowerBound, config.UpperBound)
	}

	if config.LowerBound >= config.UpperBound {
		return fmt.Errorf("lower_bound (%v) must be less than upper_bound (%v)",
			config.LowerBound, config.UpperBound)
	}

	return nil
}

// namedValue pairs a configuration field with the JSON name validation errors
// quote it by, so the checks below can iterate in a fixed order -- a map would
// make the reported field depend on Go's randomized map iteration.
type namedValue struct {
	name  string
	value float64
}

// validateWeights checks the inertia bracket and the five swarming weights.
// WeightAuto is the default for the latter and is accepted as written; any
// other value has to be a finite number the schedules can use literally.
func validateWeights(config *Config) error {
	inertia := []namedValue{
		{"inertia_weight_start", config.InertiaWeightStart},
		{"inertia_weight_end", config.InertiaWeightEnd},
	}
	for _, item := range inertia {
		name, value := item.name, item.value

		if !isFinite(value) {
			return fmt.Errorf("%s must be finite, got %v", name, value)
		}
	}

	weights := []namedValue{
		{"separation_weight", config.SeparationWeight},
		{"alignment_weight", config.AlignmentWeight},
		{"cohesion_weight", config.CohesionWeight},
		{"food_weight", config.FoodWeight},
		{"enemy_weight", config.EnemyWeight},
	}
	for _, item := range weights {
		name, value := item.name, item.value

		if value != WeightAuto && !isFinite(value) {
			return fmt.Errorf("%s must be finite or WeightAuto (%v), got %v", name, WeightAuto, value)
		}
	}

	return nil
}

// validateSchedules checks the radius, step-clamp, enemy-cutoff and Lévy
// parameters.
func validateSchedules(config *Config) error {
	if !isFinite(config.RadiusInitialDivisor) || config.RadiusInitialDivisor <= 0 {
		return fmt.Errorf("radius_initial_divisor must be a positive finite number, got %v",
			config.RadiusInitialDivisor)
	}

	if !isFinite(config.RadiusGrowth) || config.RadiusGrowth < 0 {
		return fmt.Errorf("radius_growth must be a non-negative finite number, got %v",
			config.RadiusGrowth)
	}

	if !isFinite(config.MaxStepRatio) || config.MaxStepRatio <= 0 {
		return fmt.Errorf("max_step_ratio must be a positive finite number, got %v", config.MaxStepRatio)
	}

	if !isFinite(config.EnemyCutoffFraction) ||
		config.EnemyCutoffFraction < 0 || config.EnemyCutoffFraction > 1 {
		return fmt.Errorf("enemy_cutoff_fraction must be in [0,1], got %v", config.EnemyCutoffFraction)
	}

	// Mantegna's σ divides by Γ((1+β)/2)·β·2^((β-1)/2) and raises the result to
	// 1/β, so β outside (0,2) is not a heavy-tailed distribution at all -- it is
	// a division by zero or a negative exponent.
	if !isFinite(config.LevyBeta) || config.LevyBeta <= 0 || config.LevyBeta >= 2 {
		return fmt.Errorf("levy_beta must be in the open interval (0,2), got %v", config.LevyBeta)
	}

	if !isFinite(config.LevyScale) || config.LevyScale < 0 {
		return fmt.Errorf("levy_scale must be a non-negative finite number, got %v", config.LevyScale)
	}

	return nil
}

// validateConvergenceBlock checks an optional early-termination block. A nil
// block disables early stopping and is valid.
func validateConvergenceBlock(config *ConvergenceConfig, maxIterations int) error {
	if config == nil {
		return nil
	}

	if config.TargetCost != nil && !isFinite(*config.TargetCost) {
		return fmt.Errorf("target cost must be finite, got %v", *config.TargetCost)
	}

	if !isFinite(config.MinImprovement) || config.MinImprovement < 0 {
		return fmt.Errorf("minimum improvement must be finite and non-negative, got %v",
			config.MinImprovement)
	}

	if config.StagnationIterations < 0 {
		return fmt.Errorf("stagnation iterations must be non-negative, got %d",
			config.StagnationIterations)
	}

	if config.MinIterations < 0 || config.MinIterations > maxIterations {
		return fmt.Errorf("minimum iterations must be in [0, %d], got %d",
			maxIterations, config.MinIterations)
	}

	return nil
}

// isFinite reports whether value is a real number -- neither NaN nor infinite.
func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
