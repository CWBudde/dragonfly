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

// ConstraintHandlingMethod selects how constrained candidates are ranked
// against one another.
type ConstraintHandlingMethod string

const (
	// ConstraintHandlingFeasibility applies Deb's feasibility rules: a feasible
	// candidate always beats an infeasible one, two feasible candidates are
	// ranked by cost, and two infeasible ones by aggregate violation.
	ConstraintHandlingFeasibility ConstraintHandlingMethod = "feasibility"
	// ConstraintHandlingPenalty ranks candidates by their penalized cost.
	ConstraintHandlingPenalty ConstraintHandlingMethod = "penalty"
)

// PenaltyMethod selects how aggregate constraint violation is folded into the
// objective cost.
type PenaltyMethod string

const (
	// PenaltyLinear adds factor * violation to the objective cost.
	PenaltyLinear PenaltyMethod = "linear"
	// PenaltyQuadratic adds factor * violation squared to the objective cost.
	PenaltyQuadratic PenaltyMethod = "quadratic"
)

// ConstraintConfig configures optional problem constraints. The function
// fields are not serialized; JSON round-trips carry the policy only.
type ConstraintConfig struct {
	Handling          ConstraintHandlingMethod `json:"handling,omitempty"`
	PenaltyMethod     PenaltyMethod            `json:"penalty_method,omitempty"`
	Inequalities      []ConstraintFunction     `json:"-"`
	Equalities        []ConstraintFunction     `json:"-"`
	PenaltyFactor     float64                  `json:"penalty_factor,omitempty"`
	EqualityTolerance float64                  `json:"equality_tolerance,omitempty"`
}

// WeightAuto makes Optimize derive a swarming weight from the paper's adaptive
// schedule instead of taking the field literally. It is the default for every
// weight field, and it is a distinct sentinel rather than the zero value
// because zero is a meaningful weight -- "switch this behavior off" -- and a
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
	// exploration behavior, so it is the default.
	BoundaryWrap BoundaryMethod = "wrap"
	// BoundaryClamp pins an out-of-range component to the bound it crossed and
	// leaves the step untouched. It is Mayfly's maxVec/minVec idiom, and the
	// least surprising choice for constrained problems.
	BoundaryClamp BoundaryMethod = "clamp"
	// BoundaryReflect mirrors an out-of-range component back into the box and
	// inverts the sign of its step component, as if the boundary were a wall.
	BoundaryReflect BoundaryMethod = "reflect"
)

// FidelityMode selects which published description of DA is reproduced when
// the paper and the author's MATLAB implementation differ.
type FidelityMode string

const (
	// FidelityPaper follows the equations and pseudocode in the 2016 paper. It
	// is the default for an unset Config.FidelityMode.
	FidelityPaper FidelityMode = "paper"
	// FidelityMATLAB reproduces the control-flow and operator details of the
	// author's reference MATLAB implementations.
	FidelityMATLAB FidelityMode = "matlab"
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
// Step is the paper's ΔX, the velocity analog: it is carried between
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
// When EnableParallel is true, ObjectiveFunc and every configured constraint
// callback may be called concurrently with distinct position vectors and must
// be safe for concurrent use.
type Config struct {
	ObjectiveFunc ObjectiveFunction `json:"-"`
	Rand          *rand.Rand        `json:"-"`
	// Seed requests a reproducible random stream and records its value in the
	// result. When Rand is supplied directly without Seed, the generator still
	// drives the run but its original seed is unknowable.
	Seed           *int64             `json:"seed,omitempty"`
	Convergence    *ConvergenceConfig `json:"convergence,omitempty"`
	Constraints    *ConstraintConfig  `json:"constraints,omitempty"`
	BoundaryMethod BoundaryMethod     `json:"boundary_method"`
	FidelityMode   FidelityMode       `json:"fidelity_mode"`

	// TransferFunc names the transfer function the binary variant turns a step
	// component into a bit-flip probability with. An empty value means
	// DefaultTransferFunction, the paper's v3. It is ignored by the continuous
	// entry points.
	TransferFunc TransferFunction `json:"transfer_function,omitempty"`
	ChaosMap     ChaosMap         `json:"chaos_map,omitempty"`

	LowerBound float64 `json:"lower_bound"`
	UpperBound float64 `json:"upper_bound"`

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

	// RadiusInitialDivisor and RadiusGrowth shape the neighborhood radius
	// r = (ub-lb)/divisor + (ub-lb)*(t/T)*growth.
	RadiusInitialDivisor float64 `json:"radius_initial_divisor"`
	RadiusGrowth         float64 `json:"radius_growth"`

	// MaxStepRatio sets the step clamp ΔX_max = MaxStepRatio*(ub-lb).
	MaxStepRatio float64 `json:"max_step_ratio"`

	// EnemyCutoffFraction is the fraction of the run after which the enemy
	// weight is forced to zero. The paper uses three quarters, and at that
	// default the field is inert: the convergence factor mc that the enemy
	// weight otherwise follows already reaches zero at half the run, so the
	// cutoff only ever replaces a zero with a zero. Only a fraction below 0.5
	// changes any value. It is also ignored entirely when EnemyWeight is pinned
	// to anything other than WeightAuto, since a pinned weight bypasses the
	// schedule. The field stays because the paper and the author's DA.m carry
	// both rules, and a reader checking the code against them should find the
	// cutoff where they expect it.
	EnemyCutoffFraction float64 `json:"enemy_cutoff_fraction"`

	// LevyBeta and LevyScale parameterize Mantegna's algorithm.
	LevyBeta  float64 `json:"levy_beta"`
	LevyScale float64 `json:"levy_scale"`

	// PSOCognitiveWeight and PSOSocialWeight are MHDA's acceleration
	// constants. They are ignored by the other variants.
	PSOCognitiveWeight float64 `json:"pso_cognitive_weight"`
	PSOSocialWeight    float64 `json:"pso_social_weight"`

	// ChaosSeed is CDA's deterministic initial condition. The published Gauss
	// profile starts at 0.7. It is independent of Seed: Seed owns stochastic
	// initialization while ChaosSeed owns the chaotic coefficient stream.
	ChaosSeed float64 `json:"chaos_seed"`

	// GaussianMutationWeight and QuantumRotationAngle parameterize QGDA's two
	// improvement operators. The paper reports 1 and 0.005*pi respectively.
	GaussianMutationWeight float64 `json:"gaussian_mutation_weight"`
	QuantumRotationAngle   float64 `json:"quantum_rotation_angle"`

	ProblemSize   int `json:"problem_size"`
	NPop          int `json:"npop"`
	MaxIterations int `json:"max_iterations"`
	MaxWorkers    int `json:"max_workers"`

	// UseLevyWalk selects the paper's Lévy random walk for a dragonfly with no
	// neighbors and no food in range. Disabling it keeps such a dragonfly
	// still for that iteration.
	UseLevyWalk    bool `json:"use_levy_walk"`
	EnableParallel bool `json:"enable_parallel"`

	// UseBinary marks a configuration as belonging to the binary variant: 0/1
	// positions, the bit-flip position update, and the transfer function in
	// TransferFunc. It is what NewBinaryConfig sets and what the variant
	// registry dispatches on; OptimizeBinary and OptimizeBinaryContext run the
	// binary variant regardless of it, and Optimize ignores it.
	UseBinary bool `json:"use_binary"`
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
	Seed           int64 // Random seed used for reproducibility, when SeedKnown is true
	SeedKnown      bool  // Whether Seed identifies the random stream that drove the run
}
