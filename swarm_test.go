package dragonfly

import (
	"math"
	"testing"
)

// swarmTolerance is tight enough that any algebraically different formula --
// a sign flip in the enemy term, a missing division by N -- fails, while still
// tolerating the last bit of a float64 division.
const swarmTolerance = 1e-12

// The hand-worked reference swarm.
//
// Three dragonflies in two dimensions, chosen so that no two components repeat
// and every expected vector below is a distinct pair of numbers -- a
// transposition or an off-by-one in the accumulation loops cannot accidentally
// produce the right answer.
//
//	i | position X_i | step V_i
//	--+--------------+-----------
//	0 | (1, 2)       | (0.5, -1)
//	1 | (2, 3)       | (-1, 2)
//	2 | (5, 9)       | (3, 4)
//
//	food  X⁺ = (4, -1)
//	enemy X⁻ = (-2, 7)
//
// Every expected value in this file is worked out by hand from the formulas in
// PLAN.md §1.1 and written as a literal. Nothing here is produced by calling the
// functions under test.
func referenceSwarm() []Dragonfly {
	return []Dragonfly{
		{Position: []float64{1, 2}, Step: []float64{0.5, -1}},
		{Position: []float64{2, 3}, Step: []float64{-1, 2}},
		{Position: []float64{5, 9}, Step: []float64{3, 4}},
	}
}

func referenceFood() []float64 { return []float64{4, -1} }

func referenceEnemy() []float64 { return []float64{-2, 7} }

// assertVec compares got against the hand-computed want.
func assertVec(t *testing.T, name string, got, want []float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s has dimension %d, want %d (got %v)", name, len(got), len(want), got)

		return
	}

	for k := range want {
		if math.Abs(got[k]-want[k]) > swarmTolerance {
			t.Errorf("%s = %v, want %v (component %d differs by %g)",
				name, got, want, k, math.Abs(got[k]-want[k]))

			return
		}
	}
}

// TestSwarmPrimitivesHandComputed is the centerpiece of this file: the five
// primitives evaluated on the reference swarm above, against values worked out
// by hand.
//
// Dragonfly 0, radius 10, so both other dragonflies are neighbors (N = 2):
//
//	S_0 = -((X_0-X_1) + (X_0-X_2)) = -((-1,-1) + (-4,-7)) = (5, 8)
//	A_0 = (V_1 + V_2)/2 = (2, 6)/2                        = (1, 3)
//	C_0 = (X_1 + X_2)/2 - X_0 = (3.5, 6) - (1, 2)         = (2.5, 4)
//	F_0 = X⁺ - X_0 = (4, -1) - (1, 2)                     = (3, -3)
//	E_0 = X⁻ + X_0 = (-2, 7) + (1, 2)                     = (-1, 9)
//
// Dragonfly 1, radius 10, N = 2:
//
//	S_1 = (X_0-X_1) + (X_2-X_1) = (-1,-1) + (3, 6)        = (2, 5)
//	A_1 = (V_0 + V_2)/2 = (3.5, 3)/2                      = (1.75, 1.5)
//	C_1 = (X_0 + X_2)/2 - X_1 = (3, 5.5) - (2, 3)         = (1, 2.5)
//	F_1 = (4, -1) - (2, 3)                                = (2, -4)
//	E_1 = (-2, 7) + (2, 3)                                = (0, 10)
//
// Dragonfly 0, radius 2, where only dragonfly 1 is in range (N = 1):
//
//	S_0 = X_1 - X_0 = (1, 1)
//	A_0 = V_1       = (-1, 2)
//	C_0 = X_1 - X_0 = (1, 1)
func TestSwarmPrimitivesHandComputed(t *testing.T) {
	tests := []struct {
		name                string
		wantNeighbors       []int
		wantS, wantA, wantC []float64
		wantF, wantE        []float64
		index               int
		radius              float64
	}{
		{
			name:          "dragonfly 0, radius 10, both neighbors",
			index:         0,
			radius:        10,
			wantNeighbors: []int{1, 2},
			wantS:         []float64{5, 8},
			wantA:         []float64{1, 3},
			wantC:         []float64{2.5, 4},
			wantF:         []float64{3, -3},
			wantE:         []float64{-1, 9},
		},
		{
			name:          "dragonfly 1, radius 10, both neighbors",
			index:         1,
			radius:        10,
			wantNeighbors: []int{0, 2},
			wantS:         []float64{2, 5},
			wantA:         []float64{1.75, 1.5},
			wantC:         []float64{1, 2.5},
			wantF:         []float64{2, -4},
			wantE:         []float64{0, 10},
		},
		{
			name:          "dragonfly 0, radius 2, one neighbor",
			index:         0,
			radius:        2,
			wantNeighbors: []int{1},
			wantS:         []float64{1, 1},
			wantA:         []float64{-1, 2},
			wantC:         []float64{1, 1},
			wantF:         []float64{3, -3},
			wantE:         []float64{-1, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			swarm := referenceSwarm()

			neighbors := findNeighbors(swarm, tt.index, tt.radius)
			assertIndices(t, "findNeighbors", neighbors, tt.wantNeighbors)

			position := swarm[tt.index].Position

			assertVec(t, "separationVector", separationVector(swarm, tt.index, neighbors), tt.wantS)
			assertVec(t, "alignmentVector", alignmentVector(swarm, tt.index, neighbors), tt.wantA)
			assertVec(t, "cohesionVector", cohesionVector(swarm, tt.index, neighbors), tt.wantC)
			assertVec(t, "foodVector", foodVector(position, referenceFood()), tt.wantF)
			assertVec(t, "enemyVector", enemyVector(position, referenceEnemy()), tt.wantE)
		})
	}
}

func assertIndices(t *testing.T, name string, got, want []int) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v", name, got, want)

		return
	}

	for k := range want {
		if got[k] != want[k] {
			t.Fatalf("%s = %v, want %v", name, got, want)

			return
		}
	}
}

// TestWithinRadiusIsPerDimensionNotEuclidean is the test that catches a
// Euclidean shortcut.
//
// Because the Euclidean norm is never smaller than the largest component, a
// point inside the ball of radius r is always inside the box of half-width r as
// well -- so the two rules cannot be told apart by a point that the Euclidean
// rule accepts. They differ only the other way round: the box accepts points
// the ball rejects, in the corners.
//
// (2.5, 2.5) at radius 3 is exactly such a corner point. Its Euclidean distance
// from the origin is sqrt(12.5) = 3.5355..., comfortably outside the ball, yet
// every component is within 3, so the per-dimension rule -- the one the
// reference implementation uses -- must accept it. An implementation that
// compares a Euclidean distance against r fails here.
func TestWithinRadiusIsPerDimensionNotEuclidean(t *testing.T) {
	const radius = 3.0

	origin := []float64{0, 0}
	corner := []float64{2.5, 2.5}

	if euclid := math.Hypot(corner[0], corner[1]); euclid <= radius {
		t.Fatalf("test premise broken: |%v| = %v, want a value greater than the radius %v",
			corner, euclid, radius)
	}

	if !withinRadius(origin, corner, radius) {
		t.Errorf("withinRadius(%v, %v, %v) = false, want true: the neighborhood test is "+
			"per-dimension (a box), not Euclidean (a ball)", origin, corner, radius)
	}

	// The mirror image of the same point: a single component past the radius
	// disqualifies the whole candidate, however small the others are.
	outside := []float64{3.5, 0.1}
	if withinRadius(origin, outside, radius) {
		t.Errorf("withinRadius(%v, %v, %v) = true, want false: component 0 exceeds the radius",
			origin, outside, radius)
	}
}

func TestWithinRadiusBoundaryAndDegenerateCases(t *testing.T) {
	tests := []struct {
		name   string
		a, b   []float64
		radius float64
		want   bool
	}{
		{name: "exactly on the radius counts as inside", a: []float64{0, 0}, b: []float64{3, -3}, radius: 3, want: true},
		{name: "one component just outside", a: []float64{0, 0}, b: []float64{3, 3.0000001}, radius: 3, want: false},
		{name: "identical positions are not neighbors", a: []float64{1, 2}, b: []float64{1, 2}, radius: 3, want: false},
		{name: "shared coordinate is excluded", a: []float64{1, 2}, b: []float64{1, 2.5}, radius: 3, want: false},
		{name: "every coordinate differs", a: []float64{1, 2}, b: []float64{1.5, 2.5}, radius: 3, want: true},
		{name: "zero radius rejects any difference", a: []float64{1, 2}, b: []float64{1, 2.5}, radius: 0, want: false},
		{name: "negative radius rejects everything", a: []float64{1, 2}, b: []float64{1, 3}, radius: -1, want: false},
		{name: "mismatched dimensions", a: []float64{1, 2}, b: []float64{1, 2, 3}, radius: 10, want: false},
		{name: "empty vectors", a: []float64{}, b: []float64{}, radius: 10, want: false},
		{name: "NaN component is never within radius", a: []float64{1, 2}, b: []float64{math.NaN(), 2}, radius: 10, want: false},
		{name: "infinite component is never within radius", a: []float64{1, 2}, b: []float64{math.Inf(1), 2}, radius: 10, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := withinRadius(tt.a, tt.b, tt.radius); got != tt.want {
				t.Errorf("withinRadius(%v, %v, %v) = %v, want %v", tt.a, tt.b, tt.radius, got, tt.want)
			}
		})
	}
}

// TestDragonflyIsNeverItsOwnNeighbor covers both ways self-neighboring could
// creep in: the dragonfly's own index, and a second dragonfly that happens to
// sit on exactly the same position -- the all-zero distance vector the
// reference implementation excludes.
func TestDragonflyIsNeverItsOwnNeighbor(t *testing.T) {
	swarm := referenceSwarm()

	// A radius wide enough to cover the whole swarm several times over.
	for index := range swarm {
		for _, j := range findNeighbors(swarm, index, 1000) {
			if j == index {
				t.Errorf("findNeighbors(swarm, %d, 1000) returned the dragonfly's own index", index)
			}
		}
	}

	duplicates := []Dragonfly{
		{Position: []float64{1, 2}, Step: []float64{0, 0}},
		{Position: []float64{1, 2}, Step: []float64{1, 1}},
	}

	if got := findNeighbors(duplicates, 0, 5); len(got) != 0 {
		t.Errorf("findNeighbors over co-located dragonflies = %v, want none: an all-zero "+
			"distance vector is excluded", got)
	}
}

func TestFindNeighborsOutOfRangeIndex(t *testing.T) {
	swarm := referenceSwarm()

	for _, index := range []int{-1, len(swarm), len(swarm) + 7} {
		if got := findNeighbors(swarm, index, 1000); len(got) != 0 {
			t.Errorf("findNeighbors(swarm, %d, 1000) = %v, want none", index, got)
		}
	}
}

// TestEnemyVectorAdds pins down the detail that reads as a typo: E_i is
// X⁻ + X_i, so enemyVector([1,2], [3,4]) is [4,6] and emphatically not the
// difference [2,2].
func TestEnemyVectorAdds(t *testing.T) {
	position := []float64{1, 2}
	enemy := []float64{3, 4}

	got := enemyVector(position, enemy)
	want := []float64{4, 6}
	notWanted := []float64{2, 2} // what a "fixed" difference would produce

	assertVec(t, "enemyVector", got, want)

	if math.Abs(got[0]-notWanted[0]) <= swarmTolerance && math.Abs(got[1]-notWanted[1]) <= swarmTolerance {
		t.Errorf("enemyVector(%v, %v) = %v: the enemy term is a sum, X⁻ + X_i, not a difference",
			position, enemy, got)
	}
}

// TestFoodVectorSubtracts is the counterpart: F_i really is X⁺ - X_i.
func TestFoodVectorSubtracts(t *testing.T) {
	assertVec(t, "foodVector", foodVector([]float64{1, 2}, []float64{3, 4}), []float64{2, 2})
}

// TestNoNeighborFallback covers the N == 0 case, where the formulas for
// alignment and cohesion divide by zero. The reference implementation falls
// back to the dragonfly's own step for alignment and its own position for
// cohesion, which makes cohesion the zero vector.
func TestNoNeighborFallback(t *testing.T) {
	swarm := referenceSwarm()

	// Radius 2 leaves dragonfly 2 alone: its distance to 0 is (4,7) and to 1 is
	// (3,6), both with components past the radius.
	neighbors := findNeighbors(swarm, 2, 2)
	if len(neighbors) != 0 {
		t.Fatalf("findNeighbors(swarm, 2, 2) = %v, want none", neighbors)
	}

	assertVec(t, "separationVector with no neighbors", separationVector(swarm, 2, neighbors), []float64{0, 0})
	// A_2 falls back to V_2 = (3, 4).
	assertVec(t, "alignmentVector with no neighbors", alignmentVector(swarm, 2, neighbors), []float64{3, 4})
	// C_2 = X_2 - X_2 = (0, 0).
	assertVec(t, "cohesionVector with no neighbors", cohesionVector(swarm, 2, neighbors), []float64{0, 0})
}

// TestAlignmentFallbackDoesNotAliasTheStep guards the fallback against the
// obvious shortcut of returning swarm[index].Step itself: a caller that scales
// the returned vector would then rewrite the dragonfly's step.
func TestAlignmentFallbackDoesNotAliasTheStep(t *testing.T) {
	swarm := referenceSwarm()

	alignment := alignmentVector(swarm, 2, nil)
	alignment[0] = 999

	if swarm[2].Step[0] != 3 {
		t.Errorf("alignmentVector aliased the dragonfly's step: Step[0] = %v, want 3", swarm[2].Step[0])
	}
}

// TestPrimitivesDoNotMutateInputs asserts the contract that every primitive
// allocates a fresh result and leaves the swarm, the food and the enemy exactly
// as it found them.
func TestPrimitivesDoNotMutateInputs(t *testing.T) {
	swarm := referenceSwarm()
	before := referenceSwarm()
	food := referenceFood()
	enemy := referenceEnemy()
	neighbors := []int{1, 2}

	results := [][]float64{
		separationVector(swarm, 0, neighbors),
		alignmentVector(swarm, 0, neighbors),
		cohesionVector(swarm, 0, neighbors),
		alignmentVector(swarm, 0, nil),
		cohesionVector(swarm, 0, nil),
		foodVector(swarm[0].Position, food),
		enemyVector(swarm[0].Position, enemy),
	}

	// Scribble over every result: an aliased return would show up below.
	for _, result := range results {
		for k := range result {
			result[k] = math.NaN()
		}
	}

	for i := range before {
		assertVec(t, "position after the primitives", swarm[i].Position, before[i].Position)
		assertVec(t, "step after the primitives", swarm[i].Step, before[i].Step)
	}

	assertVec(t, "food after the primitives", food, referenceFood())
	assertVec(t, "enemy after the primitives", enemy, referenceEnemy())

	// The neighbor list is an input too.
	assertIndices(t, "neighbors after the primitives", neighbors, []int{1, 2})
}

// TestDimensionConsistency asserts that every returned vector has the problem's
// dimension, for a swarm wider than the two dimensions used above and for both
// the populated and the empty neighborhood.
func TestDimensionConsistency(t *testing.T) {
	const dimension = 5

	swarm := []Dragonfly{
		{Position: []float64{0, 1, 2, 3, 4}, Step: []float64{1, 1, 1, 1, 1}},
		{Position: []float64{1, 1, 1, 1, 1}, Step: []float64{2, 0, 2, 0, 2}},
		{Position: []float64{9, 9, 9, 9, 9}, Step: []float64{0, 0, 0, 0, 0}},
	}
	food := []float64{5, 5, 5, 5, 5}
	enemy := []float64{-5, -5, -5, -5, -5}

	for _, radius := range []float64{0.5, 3, 100} {
		neighbors := findNeighbors(swarm, 0, radius)

		vectors := map[string][]float64{
			"separationVector": separationVector(swarm, 0, neighbors),
			"alignmentVector":  alignmentVector(swarm, 0, neighbors),
			"cohesionVector":   cohesionVector(swarm, 0, neighbors),
			"foodVector":       foodVector(swarm[0].Position, food),
			"enemyVector":      enemyVector(swarm[0].Position, enemy),
		}

		for name, vector := range vectors {
			if len(vector) != dimension {
				t.Errorf("%s at radius %v has dimension %d, want %d", name, radius, len(vector), dimension)
			}
		}
	}
}

// TestSeparationIsAntisymmetric checks a property the hand-computed table
// cannot: for a two-dragonfly swarm in which each is the other's only
// neighbor, the separation vectors must be exact negatives.
func TestSeparationIsAntisymmetric(t *testing.T) {
	swarm := []Dragonfly{
		{Position: []float64{1, 2}, Step: []float64{0, 0}},
		{Position: []float64{2, 4}, Step: []float64{0, 0}},
	}

	first := separationVector(swarm, 0, findNeighbors(swarm, 0, 5))
	second := separationVector(swarm, 1, findNeighbors(swarm, 1, 5))

	assertVec(t, "S_0", first, []float64{1, 2})
	assertVec(t, "S_1", second, []float64{-1, -2})
}

// TestNeighborhoodRadiusMatchesTheSchedule checks the exported wrapper against
// hand-computed values of the schedule, not against the unexported function it
// forwards to -- a wrapper that agreed with a wrong implementation would prove
// nothing.
func TestNeighborhoodRadiusMatchesTheSchedule(t *testing.T) {
	config := NewDefaultConfig()
	config.LowerBound = -10
	config.UpperBound = 10

	// span = 20, divisor = 4, growth = 2:
	//   r(t) = 20/4 + 20 * (t/T) * 2 = 5 + 40*t/T
	tests := []struct {
		iteration int
		want      float64
	}{
		{iteration: 0, want: 5},
		{iteration: 50, want: 25},
		{iteration: 100, want: 45},
	}

	for _, test := range tests {
		got := NeighborhoodRadius(config, test.iteration, 100)
		if math.Abs(got-test.want) > 1e-12 {
			t.Errorf("NeighborhoodRadius(t=%d) = %v, want %v", test.iteration, got, test.want)
		}
	}

	if NeighborhoodRadius(nil, 0, 100) != 0 {
		t.Error("a nil config must report no radius")
	}
}

// TestWithinRadiusIsABoxNotABall is the exported guard on the pitfall that the
// neighborhood test is per-dimension. The point below is inside the Euclidean
// ball of radius 1.5 and outside the box, so a Euclidean implementation passes
// every end-to-end test and fails exactly here.
func TestWithinRadiusIsABoxNotABall(t *testing.T) {
	a := []float64{0, 0}

	if WithinRadius(a, []float64{1.4, 0}, 1.5) {
		t.Error("a point sharing one coordinate must be excluded")
	}

	if WithinRadius(a, []float64{1.4, 1.4}, 1.5) != true {
		t.Error("a point inside the box on both axes was not a neighbor")
	}

	// Euclidean distance 1.42 < 1.5, but the second component exceeds it.
	if WithinRadius(a, []float64{0.2, 1.6}, 1.5) {
		t.Error("a point outside the box on one axis was reported as a neighbor")
	}

	if WithinRadius(a, []float64{0, 0}, 1.5) {
		t.Error("a dragonfly must not be its own neighbor")
	}

	if WithinRadius(a, []float64{1}, 1.5) {
		t.Error("vectors of unequal length must never be neighbors")
	}

	if WithinRadius(a, []float64{math.NaN(), 0}, 1.5) {
		t.Error("a NaN component must never be a neighbor")
	}
}
