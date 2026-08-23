// The five swarming primitives of the Dragonfly Algorithm: separation,
// alignment, cohesion, food attraction and enemy distraction, plus the
// neighborhood scan they are all computed over.
//
// Two details here are worth stating twice, because both look like typos and
// neither shows up as an obvious failure in an end-to-end convergence test:
//
//   - The enemy term is a SUM, E_i = X⁻ + X_i, not a difference. That is what
//     the paper and the reference DA.m compute.
//   - The neighborhood test is PER-DIMENSION, not Euclidean. The reference
//     code builds a component-wise distance vector and requires every component
//     to be within the radius. The per-dimension rule accepts a box, the
//     Euclidean rule accepts the inscribed ball, so a Euclidean shortcut
//     silently shrinks every neighborhood and degrades convergence.

package dragonfly

import "math"

// withinRadius reports whether b lies inside the per-dimension radius r around
// a. The reference MATLAB test requires every component distance to be
// non-zero, in addition to being within the radius.
//
// The rule is the reference implementation's, component by component:
//
//	all(|a_k - b_k| <= radius)  and  all(a_k - b_k != 0)
//
// This is a box test, not a ball test. It is deliberately not Euclidean: see
// the file comment.
//
// Vectors of different lengths, and empty vectors, are never neighbors -- an
// empty distance vector is vacuously all-zero, which is exactly the degenerate
// case the second clause exists to reject. A NaN component is not a neighbor
// either, since every comparison against NaN is false.
func withinRadius(a, b []float64, radius float64) bool {
	if len(a) != len(b) || len(a) == 0 {
		return false
	}

	for k := range a {
		distance := math.Abs(a[k] - b[k])

		// Written as a negated <= rather than a >, so that a NaN component
		// fails the test: every comparison against NaN is false.
		if !(distance <= radius) {
			return false
		}

		if distance == 0 {
			return false
		}
	}

	return true
}

// findNeighbors returns the indices of the neighbors of swarm[index], in
// ascending index order.
//
// The result is freshly allocated and may be empty; it is never the dragonfly's
// own index. A dragonfly that happens to share a position with another is not
// its neighbor either, because withinRadius rejects an all-zero distance --
// that is the reference implementation's behavior, not an oversight.
//
// An out-of-range index yields no neighbors rather than a panic, so a caller
// scanning a swarm it did not build cannot crash the run.
func findNeighbors(swarm []Dragonfly, index int, radius float64) []int {
	if index < 0 || index >= len(swarm) {
		return nil
	}

	self := swarm[index].Position
	neighbors := make([]int, 0, len(swarm)-1)

	for j := range swarm {
		if j == index {
			continue
		}

		if withinRadius(self, swarm[j].Position, radius) {
			neighbors = append(neighbors, j)
		}
	}

	return neighbors
}

// separationVector returns S_i, the repulsion from local crowding:
//
//	S_i = -Σ_j (X_i - X_j)
//
// With no neighbors the sum is empty and S_i is the zero vector, which is what
// the reference code computes as well. The result is freshly allocated and has
// the dimension of swarm[index].Position; the inputs are not touched.
func separationVector(swarm []Dragonfly, index int, neighbors []int) []float64 {
	self := swarmPositionAt(swarm, index)
	separation := make([]float64, len(self))

	for _, j := range neighbors {
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

// alignmentVector returns A_i, the mean step of the neighbors:
//
//	A_i = (Σ_j V_j) / N
//
// where V_j is neighbor j's Step, the algorithm's ΔX.
//
// With N == 0 the formula is undefined. The reference implementation falls back
// to the dragonfly's own step, so A_i is a copy of swarm[index].Step and the
// alignment term contributes nothing new to the update. That fallback is
// reproduced here on purpose.
//
// The result is freshly allocated; the inputs are not touched.
func alignmentVector(swarm []Dragonfly, index int, neighbors []int) []float64 {
	if len(neighbors) == 0 {
		return copyVec(swarmStepAt(swarm, index))
	}

	alignment := make([]float64, len(swarmStepAt(swarm, index)))

	for _, j := range neighbors {
		other := swarmStepAt(swarm, j)

		for k := range alignment {
			if k < len(other) {
				alignment[k] += other[k]
			}
		}
	}

	swarmScale(alignment, 1/float64(len(neighbors)))

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
func cohesionVector(swarm []Dragonfly, index int, neighbors []int) []float64 {
	self := swarmPositionAt(swarm, index)
	cohesion := make([]float64, len(self))

	if len(neighbors) == 0 {
		return cohesion
	}

	for _, j := range neighbors {
		other := swarmPositionAt(swarm, j)

		for k := range cohesion {
			if k < len(other) {
				cohesion[k] += other[k]
			}
		}
	}

	swarmScale(cohesion, 1/float64(len(neighbors)))

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

// NeighborhoodRadius returns the radius r the swarm uses at the given
// iteration, which is the schedule described on Config.RadiusInitialDivisor
// and Config.RadiusGrowth. iteration is zero-based; maxIterations is the
// configured total.
//
// It is exported because r is the one schedule whose effect a caller can see
// without instrumenting the run: it decides who counts as a neighbor, and
// therefore whether a dragonfly swarms locally or flies to the food source. A
// caller tuning either field, or drawing the neighborhood a PopulationSnapshot
// implies, needs the same number the optimizer used rather than its own
// reading of the formula.
//
// A nil config has no bounds and no radius, and returns zero.
func NeighborhoodRadius(config *Config, iteration, maxIterations int) float64 {
	if config == nil {
		return 0
	}

	return neighborhoodRadius(config, iteration, maxIterations)
}

// WithinRadius reports whether b is a neighbor of a at the given radius.
//
// The test is per-dimension and is a box, not a ball: every component of the
// distance must be within the radius and non-zero, so a dragonfly is never its
// own neighbor. Vectors of unequal or zero length, and
// any vector with a NaN component, are never neighbors.
//
// It is exported for the same reason NeighborhoodRadius is, and for one more:
// a Euclidean reading of this test is the most common way to get the algorithm
// subtly wrong, and a caller reconstructing a neighborhood from a
// PopulationSnapshot should be able to ask the library rather than reimplement
// the rule and drift from it.
func WithinRadius(a, b []float64, radius float64) bool {
	return withinRadius(a, b, radius)
}
