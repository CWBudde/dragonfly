// Core type definitions for the Dragonfly Algorithm.

package dragonfly

import "math/rand"

// ObjectiveFunction represents a function to be optimized.
// It takes a read-only position vector and returns a fitness cost.
type ObjectiveFunction func([]float64) float64

// ConstraintFunction evaluates a constraint at a position. Inequality
// constraints are satisfied when the returned value is less than or equal to
// zero. Equality constraints are satisfied when the absolute returned value is
// within the configured equality tolerance.
type ConstraintFunction func([]float64) float64

// WeightAuto makes Optimize derive a swarming weight from the paper's adaptive
// schedule instead of taking the field literally. It is the default for every
// weight field, and it is a distinct sentinel rather than the zero value
// because zero is a meaningful weight -- "switch this behaviour off" -- and a
// caller who wrote it must keep getting it.
//
// It is the same convention Mayfly uses for NCAuto and AquilaWeightAuto.
const WeightAuto = -1.0

// BoundaryMethod names the rule that returns an out-of-range position to the
// search space.
type BoundaryMethod string

const (
	// BoundaryWrap is the paper's rule: a component that leaves the box
	// reappears at the opposite bound and its step component is redrawn
	// uniformly from [0,1). Wrapping is genuinely part of the algorithm's
	// exploration behaviour, so it is the default.
	BoundaryWrap BoundaryMethod = "wrap"
	// BoundaryClamp pins an out-of-range component to the bound it crossed and
	// leaves the step untouched. It is Mayfly's maxVec/minVec idiom, and the
	// least surprising choice for constrained problems.
	BoundaryClamp BoundaryMethod = "clamp"
	// BoundaryReflect mirrors an out-of-range component back into the box and
	// inverts the sign of its step component, as if the boundary were a wall.
	BoundaryReflect BoundaryMethod = "reflect"
)

// TerminationReason describes why an optimization run ended.
type TerminationReason string

const (
	// TerminationMaxIterations means the configured iteration cap was reached.
	TerminationMaxIterations TerminationReason = "maximum_iterations"
	// TerminationTargetCost means the configured target cost was reached.
	TerminationTargetCost TerminationReason = "target_cost"
	// TerminationStagnation means the best cost did not improve sufficiently
	// within the configured stagnation window.
	TerminationStagnation TerminationReason = "stagnation"
)

// Best represents the best position and cost found.
type Best struct {
	Position            []float64
	Cost                float64
	ConstraintViolation float64
}

// Dragonfly represents a single dragonfly in the swarm.
//
// Step is the paper's ΔX, the velocity analogue: it is carried between
// iterations through the inertia weight, clamped to ±ΔX_max, and reset by the
// boundary handler and the Lévy branch.
type Dragonfly struct {
	Position            []float64
	Step                []float64
	Cost                float64
	ConstraintViolation float64
}

// ConvergenceConfig controls optional early termination. MaxIterations remains
// the hard upper bound; successful target or stagnation checks may shorten a
// run after MinIterations completed iterations.
type ConvergenceConfig struct {
	// TargetCost stops the run when the best cost is less than or equal to the
	// pointed-to value. A pointer distinguishes a disabled target from a target
	// of zero.
	TargetCost *float64 `json:"target_cost,omitempty"`

	// MinImprovement is the absolute cost, penalty score, or constraint-
	// violation reduction required to reset the stagnation counter. It must be
	// non-negative.
	MinImprovement float64 `json:"min_improvement"`

	// StagnationIterations stops the run after this many consecutive iterations
	// without a sufficient improvement. Zero disables stagnation detection.
	StagnationIterations int `json:"stagnation_iterations"`

	// MinIterations is the minimum number of iterations completed before either
	// stopping criterion can terminate the run. Zero behaves as one because
	// convergence is checked at iteration boundaries.
	MinIterations int `json:"min_iterations"`
}

// Config holds the configuration parameters for the Dragonfly Algorithm.
//
// You must set ObjectiveFunc, ProblemSize, LowerBound and UpperBound; every
// other field has a usable default from NewDefaultConfig.
//
// When EnableParallel is true, ObjectiveFunc may be called concurrently with
// distinct position vectors and must be safe for concurrent use.
type Config struct {
	ObjectiveFunc  ObjectiveFunction  `json:"-"`
	Rand           *rand.Rand         `json:"-"`
	Convergence    *ConvergenceConfig `json:"convergence,omitempty"`
	BoundaryMethod BoundaryMethod     `json:"boundary_method"`
	LowerBound     float64            `json:"lower_bound"`
	UpperBound     float64            `json:"upper_bound"`

	// InertiaWeightStart and InertiaWeightEnd bracket the linearly decreasing
	// inertia weight w = start - t*(start-end)/T.
	InertiaWeightStart float64 `json:"inertia_weight_start"`
	InertiaWeightEnd   float64 `json:"inertia_weight_end"`

	// The five swarming weights. At WeightAuto each follows the paper's
	// schedule; any other finite value is used literally for the whole run.
	SeparationWeight float64 `json:"separation_weight"`
	AlignmentWeight  float64 `json:"alignment_weight"`
	CohesionWeight   float64 `json:"cohesion_weight"`
	FoodWeight       float64 `json:"food_weight"`
	EnemyWeight      float64 `json:"enemy_weight"`

	// RadiusInitialDivisor and RadiusGrowth shape the neighbourhood radius
	// r = (ub-lb)/divisor + (ub-lb)*(t/T)*growth.
	RadiusInitialDivisor float64 `json:"radius_initial_divisor"`
	RadiusGrowth         float64 `json:"radius_growth"`

	// MaxStepRatio sets the step clamp ΔX_max = MaxStepRatio*(ub-lb).
	MaxStepRatio float64 `json:"max_step_ratio"`

	// EnemyCutoffFraction is the fraction of the run after which the enemy
	// weight is forced to zero. The paper uses three quarters.
	EnemyCutoffFraction float64 `json:"enemy_cutoff_fraction"`

	// LevyBeta and LevyScale parameterise Mantegna's algorithm.
	LevyBeta  float64 `json:"levy_beta"`
	LevyScale float64 `json:"levy_scale"`

	ProblemSize   int `json:"problem_size"`
	NPop          int `json:"npop"`
	MaxIterations int `json:"max_iterations"`
	MaxWorkers    int `json:"max_workers"`

	// UseLevyWalk selects the paper's Lévy random walk for a dragonfly with no
	// neighbours and no food in range. Disabling it keeps such a dragonfly
	// still for that iteration.
	UseLevyWalk    bool `json:"use_levy_walk"`
	EnableParallel bool `json:"enable_parallel"`
}

// Result holds the results of the optimization.
type Result struct {
	// ConvergenceCurve holds the best cost known at the end of each completed
	// iteration, so it has IterationCount entries. It is non-increasing for
	// unconstrained optimization; a constrained incumbent's raw cost may rise
	// when feasibility or lower violation takes priority. Without early
	// stopping, IterationCount equals MaxIterations. It is a history of costs,
	// not a point in the search space.
	//
	// The solution itself is GlobalBest.Position.
	ConvergenceCurve []float64

	TerminationReason TerminationReason
	GlobalBest        Best

	// Worst is the enemy: the worst position seen during the run, which the
	// enemy term of every step is computed against. It is reported for
	// inspection -- it is specific to this algorithm and has no counterpart in
	// Mayfly's Result.
	Worst Best

	FuncEvalCount  int
	IterationCount int
	Seed           int64 // Random seed used for reproducibility
}
