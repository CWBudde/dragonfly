// Configuration factories for the Dragonfly Algorithm.

package dragonfly

import "runtime"

// NewDefaultConfig creates a default configuration for the standard Dragonfly
// Algorithm, with every weight left on its adaptive schedule.
// You must set ObjectiveFunc, ProblemSize, LowerBound, and UpperBound.
func NewDefaultConfig() *Config {
	return &Config{
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
		LevyBeta:      1.5,
		LevyScale:     0.01,
		NPop:          40,
		MaxIterations: 1000,
		MaxWorkers:    defaultMaxWorkers(),
		UseLevyWalk:   true,
	}
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
