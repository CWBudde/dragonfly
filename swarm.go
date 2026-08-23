// The five swarming primitives of the Dragonfly Algorithm: separation,
// alignment, cohesion, food attraction and enemy distraction, plus the
// neighbourhood scan they are all computed over.
//
// Two details here are worth stating twice, because both look like typos and
// neither shows up as an obvious failure in an end-to-end convergence test:
//
//   - The enemy term is a SUM, E_i = X⁻ + X_i, not a difference. That is what
//     the paper and the reference DA.m compute.
//   - The neighbourhood test is PER-DIMENSION, not Euclidean. The reference
//     code builds a component-wise distance vector and requires every component
//     to be within the radius. The per-dimension rule accepts a box, the
//     Euclidean rule accepts the inscribed ball, so a Euclidean shortcut
//     silently shrinks every neighbourhood and degrades convergence.

package dragonfly

import "math"

// withinRadius reports whether b lies inside the per-dimension radius r around
// a, excluding the degenerate all-zero distance (a dragonfly is not its own
// neighbour).
//
// The rule is the reference implementation's, component by component:
//
//	all(|a_k - b_k| <= radius)  and  any(a_k - b_k != 0)
//
// This is a box test, not a ball test. It is deliberately not Euclidean: see
// the file comment.
//
// Vectors of different lengths, and empty vectors, are never neighbours -- an
// empty distance vector is vacuously all-zero, which is exactly the degenerate
// case the second clause exists to reject. A NaN component is not a neighbour
// either, since every comparison against NaN is false.
func withinRadius(a, b []float64, radius float64) bool {
	if len(a) != len(b) || len(a) == 0 {
		return false
	}

	allZero := true

	for k := range a {
		distance := math.Abs(a[k] - b[k])

		if !(distance <= radius) { //nolint:staticcheck // NaN must fail the test, so the negation is not equivalent to >
			return false
		}

		if a[k] != b[k] {
			allZero = false
		}
	}

	return !allZero
}

// findNeighbours returns the indices of the neighbours of swarm[index], in
// ascending index order.
//
// The result is freshly allocated and may be empty; it is never the dragonfly's
// own index. A dragonfly that happens to share a position with another is not
// its neighbour either, because withinRadius rejects an all-zero distance --
// that is the reference implementation's behaviour, not an oversight.
//
// An out-of-range index yields no neighbours rather than a panic, so a caller
// scanning a swarm it did not build cannot crash the run.
func findNeighbours(swarm []Dragonfly, index int, radius float64) []int {
	if index < 0 || index >= len(swarm) {
		return nil
	}

	self := swarm[index].Position
	neighbours := make([]int, 0, len(swarm)-1)

	for j := range swarm {
		if j == index {
			continue
		}

		if withinRadius(self, swarm[j].Position, radius) {
			neighbours = append(neighbours, j)
		}
	}

	return neighbours
}

// separationVector returns S_i, the repulsion from local crowding:
//
//	S_i = -Σ_j (X_i - X_j)
//
// With no neighbours the sum is empty and S_i is the zero vector, which is what
// the reference code computes as well. The result is freshly allocated and has
// the dimension of swarm[index].Position; the inputs are not touched.
func separationVector(swarm []Dragonfly, index int, neighbours []int) []float64 {
	self := swarmPositionAt(swarm, index)
	separation := make([]float64, len(self))

	for _, j := range neighbours {
		other := swarmPositionAt(swarm, j)

		// -(X_i - X_j) accumulated directly as (X_j - X_i).
		for k := range separation {
			if k < len(other) {
				separation[k] += other[k] - self[k]
			}
		}
	}

	return separation
}

// alignmentVector returns A_i, the mean step of the neighbours:
//
//	A_i = (Σ_j V_j) / N
//
// where V_j is neighbour j's Step, the algorithm's ΔX.
//
// With N == 0 the formula is undefined. The reference implementation falls back
// to the dragonfly's own step, so A_i is a copy of swarm[index].Step and the
// alignment term contributes nothing new to the update. That fallback is
// reproduced here on purpose.
//
// The result is freshly allocated; the inputs are not touched.
func alignmentVector(swarm []Dragonfly, index int, neighbours []int) []float64 {
	if len(neighbours) == 0 {
		return copyVec(swarmStepAt(swarm, index))
	}

	alignment := make([]float64, len(swarmStepAt(swarm, index)))

	for _, j := range neighbours {
		other := swarmStepAt(swarm, j)

		for k := range alignment {
			if k < len(other) {
				alignment[k] += other[k]
			}
		}
	}

	swarmScale(alignment, 1/float64(len(neighbours)))

	return alignment
}

// cohesionVector returns C_i, the pull toward the local centroid:
//
//	C_i = (Σ_j X_j) / N - X_i
//
// With N == 0 the formula is undefined. The reference implementation falls back
// to the dragonfly's own position as the centroid, which makes C_i the zero
// vector. That fallback is reproduced here on purpose.
//
// The result is freshly allocated; the inputs are not touched.
func cohesionVector(swarm []Dragonfly, index int, neighbours []int) []float64 {
	self := swarmPositionAt(swarm, index)
	cohesion := make([]float64, len(self))

	if len(neighbours) == 0 {
		return cohesion
	}

	for _, j := range neighbours {
		other := swarmPositionAt(swarm, j)

		for k := range cohesion {
			if k < len(other) {
				cohesion[k] += other[k]
			}
		}
	}

	swarmScale(cohesion, 1/float64(len(neighbours)))

	for k := range cohesion {
		cohesion[k] -= self[k]
	}

	return cohesion
}

// foodVector returns F_i = food - position, the attraction to the best position
// found so far.
//
// The result is freshly allocated and has the dimension of position; the inputs
// are not touched. A food vector shorter than position contributes nothing past
// its own length, and a nil food source yields the zero vector.
func foodVector(position, food []float64) []float64 {
	attraction := make([]float64, len(position))

	for k := range attraction {
		if k < len(food) {
			attraction[k] = food[k] - position[k]
		}
	}

	return attraction
}

// enemyVector returns E_i = enemy + position, the distraction away from the
// worst position found so far.
//
// The sum is intentional and is not a transcription error: the paper and the
// reference DA.m both add the enemy to the current position rather than
// subtracting it, and the difference is not visible in a convergence curve.
// Read the corresponding test before "fixing" this.
//
// The result is freshly allocated and has the dimension of position; the inputs
// are not touched.
func enemyVector(position, enemy []float64) []float64 {
	distraction := make([]float64, len(position))

	for k := range distraction {
		if k < len(enemy) {
			distraction[k] = enemy[k] + position[k]
		}
	}

	return distraction
}

// swarmPositionAt returns the position of swarm[index], or nil for an out-of-range
// index. The slice is the dragonfly's own and must not be modified.
func swarmPositionAt(swarm []Dragonfly, index int) []float64 {
	if index < 0 || index >= len(swarm) {
		return nil
	}

	return swarm[index].Position
}

// swarmStepAt returns the step of swarm[index], or nil for an out-of-range index.
// The slice is the dragonfly's own and must not be modified.
func swarmStepAt(swarm []Dragonfly, index int) []float64 {
	if index < 0 || index >= len(swarm) {
		return nil
	}

	return swarm[index].Step
}

// swarmScale multiplies every component of vec by factor, in place.
func swarmScale(vec []float64, factor float64) {
	for k := range vec {
		vec[k] *= factor
	}
}
