// Configuration factories for the Dragonfly Algorithm.

package dragonfly

import (
	"math"
	"runtime"
)

// NewDefaultConfig creates a default configuration for the standard Dragonfly
// Algorithm, with every weight left on its adaptive schedule.
// You must set ObjectiveFunc, ProblemSize, LowerBound, and UpperBound.
func NewDefaultConfig() *Config {
	return &Config{
		FidelityMode: FidelityPaper,
		// Wrapping is the paper's boundary rule; see BoundaryWrap.
		BoundaryMethod: BoundaryWrap,
		// The inertia weight decays linearly from 0.9 to 0.4 across the run.
		InertiaWeightStart: 0.9,
		InertiaWeightEnd:   0.4,
		// Every swarming weight defers to the schedules in weights.go. Write a
		// finite value into any one of them to pin it for the whole run.
		SeparationWeight: WeightAuto,
		AlignmentWeight:  WeightAuto,
		CohesionWeight:   WeightAuto,
		FoodWeight:       WeightAuto,
		EnemyWeight:      WeightAuto,
		// r = (ub-lb)/4 + (ub-lb)*(t/T)*2: a quarter of the box at the start,
		// growing to cover it entirely as the swarm converges.
		RadiusInitialDivisor: 4.0,
		RadiusGrowth:         2.0,
		// ΔX_max = (ub-lb)/10.
		MaxStepRatio: 0.1,
		// The enemy term is switched off for the last quarter of the run.
		EnemyCutoffFraction: 0.75,
		// Mantegna's algorithm with the paper's β and scale.
		LevyBeta:  1.5,
		LevyScale: 0.01,
		// Defaults used only by the corresponding improved variants.
		PSOCognitiveWeight:     2,
		PSOSocialWeight:        2,
		ChaosSeed:              0.7,
		GaussianMutationWeight: 1,
		QuantumRotationAngle:   0.005 * math.Pi,
		NPop:                   40,
		MaxIterations:          1000,
		MaxWorkers:             defaultMaxWorkers(),
		UseLevyWalk:            true,
	}
}

// NewMemoryHybridConfig creates the published memory-based hybrid Dragonfly
// Algorithm (MHDA) profile. MHDA adds per-dragonfly personal bests and follows
// each DA movement with a PSO exploitation movement initialized from that
// memory. You must set ObjectiveFunc, ProblemSize, LowerBound and UpperBound.
func NewMemoryHybridConfig() *Config {
	return NewDefaultConfig()
}

// NewChaoticConfig creates the published Chaotic Dragonfly Algorithm (CDA)
// profile. CDA replaces all five movement weights and inertia with one value
// from a deterministic chaotic sequence per iteration. The paper's
// best-performing Gauss map and initial value 0.7 are the defaults. You must
// set ObjectiveFunc, ProblemSize, LowerBound and UpperBound.
func NewChaoticConfig() *Config {
	config := NewDefaultConfig()
	config.ChaosMap = ChaosGauss
	config.ChaosSeed = 0.7

	return config
}

// NewQuantumConfig creates the published quantum-behaved and Gaussian
// mutational Dragonfly Algorithm (QGDA) profile. You must set ObjectiveFunc,
// ProblemSize, LowerBound and UpperBound.
func NewQuantumConfig() *Config {
	return NewDefaultConfig()
}

func defaultMaxWorkers() int {
	return runtime.NumCPU()
}

// NewHighDimensionalConfig creates a configuration tuned for problems with many
// dimensions, where the search space is too large for the default swarm to
// cover in the default number of iterations.
// You must set ObjectiveFunc, ProblemSize, LowerBound, and UpperBound.
//
// It enlarges the swarm and lengthens the run, and it slows the radius growth
// so that neighborhoods stay local for longer -- in high dimensions a radius
// that reaches the whole box early makes every dragonfly a neighbor of every
// other and collapses the swarm onto the food source.
func NewHighDimensionalConfig() *Config {
	config := NewDefaultConfig()
	config.NPop = 100
	config.MaxIterations = 3000
	config.RadiusGrowth = 1.0

	return config
}

// NewFastConvergenceConfig creates a configuration for a short run on a cheap
// or well-behaved objective, where a good answer soon beats the best answer
// eventually.
// You must set ObjectiveFunc, ProblemSize, LowerBound, and UpperBound.
//
// It shortens the run, shrinks the swarm, and grows the neighborhood radius
// faster so the swarm becomes a single flock -- and therefore exploits the food
// source -- earlier. A larger step clamp lets it get there in fewer iterations.
func NewFastConvergenceConfig() *Config {
	config := NewDefaultConfig()
	config.NPop = 30
	config.MaxIterations = 300
	config.RadiusGrowth = 4.0
	config.MaxStepRatio = 0.2

	return config
}

// NewBinaryConfig creates a configuration for BDA, the binary variant, where a
// position is a bit string and the step is turned into a per-bit flip
// probability by a transfer function.
// You must set ObjectiveFunc and ProblemSize; the bounds are fixed by the
// variant and are already set.
//
// The search box is the unit interval, because a position component is a bit
// and every schedule that scales with (ub-lb) is written for that box.
// Config.BoundaryMethod and Config.UseLevyWalk are ignored in binary mode --
// see OptimizeBinaryContext for why neither has a meaning for a 0/1 vector.
//
// The step clamp is widened from a tenth of the box to six times it. The
// transfer functions saturate by |Δx| ≈ 6, so clamping there is what makes the
// whole range of flip probabilities reachable; the continuous default of 0.1
// would cap every flip probability at about a tenth and freeze the swarm.
// Treat the exact value as this implementation's choice rather than a quoted
// paper constant until it has been checked against BDA.m.
func NewBinaryConfig() *Config {
	config := NewDefaultConfig()
	config.LowerBound = 0
	config.UpperBound = 1
	config.TransferFunc = DefaultTransferFunction
	config.MaxStepRatio = 6.0
	config.UseBinary = true

	return config
}
