// The prepare phase: everything one iteration does before the objective
// function is called.
//
// This is where the Dragonfly Algorithm differs from a swarm whose individuals
// move independently. The radius, the food source, the enemy and every
// neighborhood are properties of the whole swarm, so a step cannot be computed
// in isolation. One iteration therefore runs two sequential passes on the
// calling goroutine:
//
//  1. resolve the schedules for this iteration -- w, s, a, c, f, e, r and
//     ΔX_max -- against the food source and enemy left by the previous
//     evaluation (computeWeights, in weights.go);
//  2. per dragonfly, scan the swarm for neighbors, build S, A, C, F and E from
//     them, and turn those into a step and a new position (prepareSwarm,
//     below).
//
// Both passes draw random numbers, and both stay here. Only the objective
// evaluation of the positions this file produces fans out across goroutines,
// which is why a seeded run is bit-identical whether or not it is parallel.
//
// The sequential and the parallel path share these functions rather than each
// carrying its own copy of the step rules: there is exactly one place in this
// package where the paper's update is written down.

package dragonfly

import (
	"math"
	"math/rand"
)

// prepareSwarm advances every dragonfly by one step, in index order. It is the
// second pass of the prepare phase and consumes the RNG; the swarm it leaves
// behind is scored either by the sequential loop or by the evaluation pool.
func prepareSwarm(state *runState, config *Config, weights weightSchedule, rng *rand.Rand) {
	for i := range state.swarm {
		prepareSwarmStep(state, i, config, weights, rng)
	}
}

// prepareSwarmStep advances one dragonfly by a single step, following PLAN.md
// §1.2.
//
// The three branches are not interchangeable and collapsing them into one
// unconditional five-factor step is the classic porting bug: it still converges
// on an easy function, which is what lets it survive. With the food source out
// of range a dragonfly swarms locally, with per-dimension random coefficients
// and no food or enemy term at all; with the food out of range and no
// neighbors to swarm with, it falls through to a Lévy walk instead.
//
// Boundary repair ordering: the reference DA.m repairs a dragonfly at the top
// of its per-dragonfly block, so the swarm it computes S, A and C against is
// partly repaired and partly not, and the positions left in the population when
// the loop ends are the unrepaired ones. This implementation repairs at the
// bottom instead, immediately after the position update. The deviation buys two
// properties worth more than bit-fidelity with the MATLAB source: every
// position handed to the objective function is inside [LowerBound, UpperBound],
// and so is every position in the returned Result. Because the repair still
// happens exactly once per dragonfly per iteration, the dynamics are otherwise
// the same -- only the point in the iteration at which the wrap becomes visible
// to the other dragonflies moves.
func prepareSwarmStep(state *runState, index int, config *Config, weights weightSchedule, rng *rand.Rand) {
	fly := &state.swarm[index]

	neighbors := findNeighbors(state.swarm, index, weights.Radius)
	separation := separationVector(state.swarm, index, neighbors)
	alignment := alignmentVector(state.swarm, index, neighbors)
	cohesion := cohesionVector(state.swarm, index, neighbors)
	food := foodVector(fly.Position, state.food.Position)
	enemy := enemyVector(fly.Position, state.enemy.Position)

	switch {
	case foodInRadius(fly.Position, state.food.Position, weights.Radius):
		applyFullStep(fly, weights, separation, alignment, cohesion, food, enemy)
	case len(neighbors) > 1:
		applySwarmingStep(fly, weights, separation, alignment, cohesion, rng)
	default:
		prepareLevyStep(fly, config, rng)
	}

	sanitizeVec(fly.Position, config.LowerBound, config.UpperBound, rng)
	applyBounds(fly.Position, fly.Step, config.LowerBound, config.UpperBound,
		effectiveBoundaryMethod(config), rng)
}

// applyFullStep is the food-in-range branch: the five-factor step of the paper,
//
//	ΔX = (s·S + a·A + c·C + f·F + e·E) + w·ΔX
//
// with the weights taken from the schedule, one set per iteration rather than
// per dimension.
func applyFullStep(fly *Dragonfly, weights weightSchedule, separation, alignment, cohesion, food, enemy []float64) {
	for j := range fly.Step {
		fly.Step[j] = weights.Separation*separation[j] +
			weights.Alignment*alignment[j] +
			weights.Cohesion*cohesion[j] +
			weights.Food*food[j] +
			weights.Enemy*enemy[j] +
			weights.Inertia*fly.Step[j]
	}

	commitStep(fly, weights.MaxStep)
}

// applySwarmingStep is the food-out-of-range branch for a dragonfly with more
// than one neighbor: local swarming only, with no food and no enemy term,
//
//	ΔX_j = w·ΔX_j + rand·A_j + rand·C_j + rand·S_j
//
// The three rand factors are drawn per dimension, inside this loop, exactly as
// DA.m draws them -- not once per dragonfly. Drawing them once would make the
// three primitives share a scaling factor within a step and change the search
// behavior, not just the random stream. The draw order (alignment, cohesion,
// separation) is the reference implementation's evaluation order.
func applySwarmingStep(
	fly *Dragonfly, weights weightSchedule, separation, alignment, cohesion []float64, rng *rand.Rand,
) {
	for j := range fly.Step {
		randAlignment := unifrnd(0, 1, rng)
		randCohesion := unifrnd(0, 1, rng)
		randSeparation := unifrnd(0, 1, rng)

		fly.Step[j] = weights.Inertia*fly.Step[j] +
			randAlignment*alignment[j] +
			randCohesion*cohesion[j] +
			randSeparation*separation[j]
	}

	commitStep(fly, weights.MaxStep)
}

// prepareLevyStep is the food-out-of-range branch for an isolated dragonfly:
// the multiplicative Lévy random walk X += Levy(d) ⊙ X, after which ΔX is reset
// to zero because the walk replaces the step rather than contributing to it.
//
// With Config.UseLevyWalk disabled the dragonfly stays where it is for this
// iteration and only its step is reset, which is the documented alternative.
func prepareLevyStep(fly *Dragonfly, config *Config, rng *rand.Rand) {
	if config.UseLevyWalk {
		walk := levyVector(len(fly.Position), config.LevyBeta, config.LevyScale, rng)
		for j := range fly.Position {
			fly.Position[j] += walk[j] * fly.Position[j]
		}
	}

	for j := range fly.Step {
		fly.Step[j] = 0
	}
}

// commitStep clamps ΔX componentwise to ±maxStep and adds it to the position.
//
// The step is sanitized before the clamp because clamping cannot repair a NaN:
// every comparison against NaN is false, so a NaN component would survive the
// clamp and poison the position.
func commitStep(fly *Dragonfly, maxStep float64) {
	for j := range fly.Step {
		if math.IsNaN(fly.Step[j]) {
			fly.Step[j] = 0
		}
	}

	clampVec(fly.Step, -maxStep, maxStep)

	for j := range fly.Position {
		fly.Position[j] += fly.Step[j]
	}
}
