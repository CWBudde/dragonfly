// Adaptive weight schedules for the Dragonfly Algorithm.

package dragonfly

import "math/rand"

// weightSchedule holds the per-iteration coefficients of one DA step.
//
// It is the whole of the algorithm's time dependence: everything in the step
// update that changes between iteration t and t+1 arrives through one of these
// fields, so the update itself can be read as a plain formula.
type weightSchedule struct {
	Inertia    float64
	Separation float64
	Alignment  float64
	Cohesion   float64
	Food       float64
	Enemy      float64
	Radius     float64
	MaxStep    float64
}

// computeWeights resolves every coefficient for the given iteration.
// iteration is 0-based; maxIterations is the configured total.
//
// The schedules are the paper's:
//
//	w  = start - t*(start-end)/T
//	mc = max(0, 0.1 - t*0.1/(T/2))
//	s  = 2*rand*mc,  a = 2*rand*mc,  c = 2*rand*mc
//	f  = 2*rand
//	e  = mc, forced to 0 once t > EnemyCutoffFraction*T
//	     (inert at the default 0.75, since mc is already 0 past T/2)
//	r  = (ub-lb)/RadiusInitialDivisor + (ub-lb)*(t/T)*RadiusGrowth
//	ΔX_max = (ub-lb)*MaxStepRatio
//
// A weight field left at WeightAuto follows its schedule; any other value is
// used literally for the whole run.
//
// # RNG consumption
//
// Exactly four uniform draws are taken per call -- separation, alignment,
// cohesion, food, in that order -- regardless of how many of those weights the
// caller pinned to a fixed value. A pinned weight discards its draw rather than
// skipping it.
//
// This costs four cheap draws that a fully pinned configuration does not need,
// and buys the property that pinning a weight changes only that weight: the
// random stream every other part of the run consumes stays aligned with a
// default-config run of the same seed. The alternative -- drawing only for the
// automatic weights -- would make overriding a single weight silently shift
// every subsequent position, step and Lévy draw of the run, so a run pinned at
// its own schedule's value would not reproduce the schedule's results and no
// override could be assessed in isolation. Determinism is measured against the
// seed, not against the number of draws, so the wasted draws cost nothing that
// matters.
//
// The enemy weight and the radius take no draw at all; they are deterministic
// functions of the iteration.
func computeWeights(config *Config, iteration, maxIterations int, rng *rand.Rand) weightSchedule {
	progress := scheduleProgress(iteration, maxIterations)
	mc := convergenceFactor(iteration, maxIterations)
	span := config.UpperBound - config.LowerBound

	// Drawn unconditionally so that pinning a weight does not desynchronize the
	// random stream; see the RNG consumption note above.
	randSeparation := unifrnd(0, 1, rng)
	randAlignment := unifrnd(0, 1, rng)
	randCohesion := unifrnd(0, 1, rng)
	randFood := unifrnd(0, 1, rng)

	return weightSchedule{
		Inertia:    config.InertiaWeightStart - progress*(config.InertiaWeightStart-config.InertiaWeightEnd),
		Separation: resolveWeight(config.SeparationWeight, 2*randSeparation*mc),
		Alignment:  resolveWeight(config.AlignmentWeight, 2*randAlignment*mc),
		Cohesion:   resolveWeight(config.CohesionWeight, 2*randCohesion*mc),
		Food:       resolveWeight(config.FoodWeight, 2*randFood),
		Enemy:      resolveWeight(config.EnemyWeight, scheduledEnemyWeight(config, mc, iteration, maxIterations)),
		Radius:     neighborhoodRadius(config, iteration, maxIterations),
		MaxStep:    span * config.MaxStepRatio,
	}
}

// neighborhoodRadius returns the radius r for the given iteration.
//
// It starts at a fixed fraction of the search box and grows linearly with the
// run, so that neighborhoods are local while the swarm explores and eventually
// cover the whole box, turning the swarm into a single flock around the food
// source.
func neighborhoodRadius(config *Config, iteration, maxIterations int) float64 {
	span := config.UpperBound - config.LowerBound
	progress := scheduleProgress(iteration, maxIterations)

	divisor := config.RadiusInitialDivisor
	// A non-positive divisor is rejected by validateConfig; falling back to the
	// documented default keeps a hand-built Config from yielding an infinite
	// radius here rather than an error there.
	if divisor <= 0 {
		divisor = 4.0
	}

	return span/divisor + span*progress*config.RadiusGrowth
}

// convergenceFactor returns the shared factor mc for the given iteration.
//
// mc = max(0, 0.1 - t*0.1/(T/2)) decays from 0.1 to zero over the first half of
// the run and stays there. It scales separation, alignment, cohesion and the
// enemy term together, which is what makes the swarm switch from exploration to
// exploitation: past the halfway point only the food term (and inertia) still
// moves a dragonfly.
func convergenceFactor(iteration, maxIterations int) float64 {
	// 0.1 - t*0.1/(T/2) = 0.1*(1 - 2*t/T), written in terms of the clamped
	// progress so that t beyond T cannot drive mc negative before the max.
	mc := 0.1 * (1 - 2*scheduleProgress(iteration, maxIterations))
	if mc < 0 {
		return 0
	}

	return mc
}

// scheduledEnemyWeight returns the scheduled enemy coefficient: mc until the cutoff, and
// exactly zero after it. The paper switches the enemy term off for the last
// quarter of the run, once the swarm should be exploiting the food source
// rather than being pushed away from the worst position seen.
//
// At the default cutoff of three quarters the branch changes nothing: mc is
// already zero for every t past T/2, so only a fraction below 0.5 can make the
// cutoff bite. See Config.EnemyCutoffFraction.
func scheduledEnemyWeight(config *Config, mc float64, iteration, maxIterations int) float64 {
	if float64(iteration) > config.EnemyCutoffFraction*float64(maxIterations) {
		return 0
	}

	return mc
}

// resolveWeight returns the caller's fixed weight, or the scheduled one when
// the field was left at the WeightAuto sentinel.
func resolveWeight(configured, scheduled float64) float64 {
	if configured == WeightAuto {
		return scheduled
	}

	return configured
}

// scheduleProgress returns t/T clamped to [0,1].
//
// A run of one iteration (or a nonsensical non-positive T) divides by one, so
// the schedules degenerate to their endpoints instead of to NaN: iteration 0 of
// such a run sees the start of every schedule.
func scheduleProgress(iteration, maxIterations int) float64 {
	denominator := max(maxIterations, 1)

	progress := float64(iteration) / float64(denominator)

	switch {
	case progress < 0:
		return 0
	case progress > 1:
		return 1
	default:
		return progress
	}
}
