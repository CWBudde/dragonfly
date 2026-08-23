// MODA, the multi-objective Dragonfly Algorithm: a Pareto archive, the
// hypercube grid that keeps it spread out, and the separate entry point
// OptimizeMultiObjective.
//
// The step update is the single-objective one, unchanged -- the same five
// primitives from swarm.go, the same two-branch rule from dragonfly.go, the
// same weight schedules from weights.go. Only two things differ:
//
//   - There is no single best and no single worst position, so the food source
//     and the enemy are drawn from the archive instead: the food from a sparsely
//     populated hypercube (to push the swarm toward the thin parts of the front)
//     and the enemy from a crowded one (to push it away from the parts that are
//     already well covered).
//   - The run's output is the archive rather than one incumbent.
//
// The archive, the domination predicate and the sorting helpers are ported from
// the sibling Mayfly library's multiobjective.go and extended here; the
// hypercube grid is new, and is what MODA adds on top.

package dragonfly

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
)

// MultiObjectiveFunction represents a multi-objective optimization function.
// It takes a position vector and returns one value per objective, all of them
// minimized. The returned slice must have the same length on every call.
type MultiObjectiveFunction func([]float64) []float64

// ParetoSolution is one member of a Pareto archive: a position, the objective
// vector it scored, and the bookkeeping the sorting and grid helpers hang off
// it.
//
// GridIndex and GridKey are MODA's addition to Mayfly's version of this type.
// They are maintained by the owning ParetoArchive and are only meaningful
// relative to that archive's current grid bounds, which move as the archive
// does: a solution's cell coordinates can change without the solution itself
// changing at all.
type ParetoSolution struct {
	Position           []float64
	ObjectiveValues    []float64
	DominatedSolutions []int
	// GridIndex is the solution's hypercube coordinate, one component per
	// objective, each in [0, NGrid-1].
	GridIndex        []int
	CrowdingDistance float64
	// ConstraintViolation is the aggregate constraint violation of Position,
	// as EvaluateConstraints reports it: zero for a feasible solution, positive
	// for one outside the feasible region. It participates in domination
	// through constrainedDominates, so an unconstrained run -- where every
	// solution carries zero -- behaves exactly as it did before constraints
	// existed.
	ConstraintViolation float64
	Rank                int
	DominationCount     int
	// GridKey flattens GridIndex into a single integer in base NGrid, so that
	// occupancy can be counted with one map lookup rather than a slice compare.
	GridKey int
}

// clone returns a deep copy of the solution's position and objective values.
// The sorting bookkeeping is not copied: it is scratch space that is recomputed
// wherever it is used.
func (solution *ParetoSolution) clone() *ParetoSolution {
	if solution == nil {
		return nil
	}

	return &ParetoSolution{
		Position:            copyVec(solution.Position),
		ObjectiveValues:     copyVec(solution.ObjectiveValues),
		GridIndex:           append([]int(nil), solution.GridIndex...),
		ConstraintViolation: solution.ConstraintViolation,
		GridKey:             solution.GridKey,
	}
}

// dominates reports whether objective vector a Pareto-dominates b.
//
// For minimization: a[i] <= b[i] for all i, and a[j] < b[j] for at least one j.
// Vectors of different lengths are incomparable and never dominate.
func dominates(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	strictlyBetter := false

	for i := range a {
		if a[i] > b[i] {
			// a is worse in this objective
			return false
		}

		if a[i] < b[i] {
			// a is strictly better in this objective
			strictlyBetter = true
		}
	}

	return strictlyBetter
}

// constrainedDominates reports whether solution a dominates solution b once
// constraint violations are taken into account. Every archive decision that
// asks "does one solution dominate another" goes through this function; plain
// dominates is the feasible-feasible case it delegates to.
//
// The rule is Deb's, lifted from the total order in constraints.go to the
// partial order a Pareto archive needs:
//
//  1. a feasible solution dominates an infeasible one;
//  2. between two infeasible solutions, the smaller aggregate violation
//     dominates, and equal violations are incomparable;
//  3. between two feasible solutions, ordinary Pareto dominance decides.
//
// BetterConstrainedCandidate is deliberately not reused. It is a total order --
// it always names a winner -- and feeding a total order to an archive that
// keeps everything mutually non-dominated would collapse the front to a single
// point.
func constrainedDominates(a, b *ParetoSolution) bool {
	if a == nil || b == nil {
		return false
	}

	feasibleA := IsFeasible(a.ConstraintViolation)

	feasibleB := IsFeasible(b.ConstraintViolation)
	if feasibleA != feasibleB {
		return feasibleA
	}

	if !feasibleA {
		return a.ConstraintViolation < b.ConstraintViolation
	}

	return dominates(a.ObjectiveValues, b.ObjectiveValues)
}

// duplicateSolutions reports whether two solutions occupy the same point of the
// archive's ordering: the same objective vector and the same constraint
// violation. Only such a pair is a genuine duplicate -- two solutions that
// score alike but differ in feasibility are ordered by rule 1 above, not
// deduplicated.
func duplicateSolutions(a, b *ParetoSolution) bool {
	return a.ConstraintViolation == b.ConstraintViolation &&
		equalObjectives(a.ObjectiveValues, b.ObjectiveValues)
}

// equalObjectives reports whether two objective vectors are identical. An
// archive rejects a duplicate rather than storing the same point twice, which
// would skew every occupancy count the grid is built on.
func equalObjectives(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// ArchivePolicy selects the archive diversity mechanism used by MODA.
type ArchivePolicy string

const (
	// ArchivePolicyPaperSegments partitions objective space into equal segments
	// and weights them by 1/N for food and N for enemy/overflow selection.
	ArchivePolicyPaperSegments ArchivePolicy = "paper_segments"
	// ArchivePolicyMATLABDensity reproduces MODA.m's per-solution density rank:
	// neighbors are points within one twentieth of every objective span.
	ArchivePolicyMATLABDensity ArchivePolicy = "matlab_density"
	// ArchivePolicyMOPSOGrid retains v0.1.0's configurable exponent-weighted
	// hypercube grid as an explicitly selected extension.
	ArchivePolicyMOPSOGrid ArchivePolicy = "mopso_grid"

	// ArchivePolicyPaper, ArchivePolicyMATLAB, and ArchivePolicyMOPSO are short
	// aliases that keep configuration code readable.
	ArchivePolicyPaper  = ArchivePolicyPaperSegments
	ArchivePolicyMATLAB = ArchivePolicyMATLABDensity
	ArchivePolicyMOPSO  = ArchivePolicyMOPSOGrid
)

// DefaultArchiveBeta, Gamma and Delta are retained as the v0.1.0 MOPSO-grid
// extension defaults. Paper selection does not consult them: the constant c in
// c/N and N/c cancels during roulette normalization, leaving fixed exponents of
// one.
const (
	// DefaultArchiveBeta is the MOPSO food-selection exponent.
	DefaultArchiveBeta = 4.0
	// DefaultArchiveGamma is the MOPSO enemy-selection exponent.
	DefaultArchiveGamma = 2.0
	// DefaultArchiveDelta is the MOPSO overflow-deletion exponent.
	DefaultArchiveDelta = 2.0
	// DefaultArchiveNGrid is the number of hypercubes per objective.
	DefaultArchiveNGrid = 10
	// DefaultArchiveSize is the archive capacity.
	DefaultArchiveSize = 100
)

// The bucketing budget occupiedCells applies before it counting-sorts the
// archive by cell key: the key space may be countingKeyBase plus
// countingKeySlack per archived solution, and never more than maxCountingKeys.
// A two- or three-objective run on the default grid stays well inside it; a
// high-dimensional objective vector falls back to the comparison sort, which is
// also what keeps the key-space product from overflowing.
const (
	countingKeyBase  = 256
	countingKeySlack = 16
	maxCountingKeys  = 1 << 16
)

// ParetoArchive holds a mutually non-dominated set of solutions, partitioned
// into a hypercube grid over objective space.
//
// The grid is what turns "keep the non-dominated set" into "keep a non-dominated
// set that is spread out": objective space is divided into NGrid equal bins per
// objective, and every selection and deletion decision is a roulette draw over
// the occupied cells, weighted by how crowded each one is.
//
// The archive is not safe for concurrent use.
type ParetoArchive struct {
	Policy      ArchivePolicy
	Solutions   []*ParetoSolution
	lowerBounds []float64
	upperBounds []float64
	cellBuf     []hypercubeCell
	memberBuf   []int
	countBuf    []int
	Beta        float64
	Gamma       float64
	Delta       float64
	MaxSize     int
	NGrid       int
}

// NewParetoArchive creates an archive of the given capacity with the default
// grid parameters. A non-positive capacity falls back to DefaultArchiveSize.
func NewParetoArchive(maxSize int) *ParetoArchive {
	return newParetoArchive(maxSize, DefaultArchiveNGrid,
		DefaultArchiveBeta, DefaultArchiveGamma, DefaultArchiveDelta, ArchivePolicyPaperSegments)
}

// NewParetoArchiveWithGrid creates an archive with explicit grid parameters.
//
// A non-positive maxSize or nGrid falls back to the corresponding default, and
// a negative exponent is raised to zero -- a negative exponent inverts the
// preference the roulette draw exists to express, which is never what a caller
// means.
func NewParetoArchiveWithGrid(maxSize, nGrid int, beta, gamma, delta float64) *ParetoArchive {
	return newParetoArchive(maxSize, nGrid, beta, gamma, delta, ArchivePolicyMOPSOGrid)
}

func newParetoArchive(
	maxSize, nGrid int,
	beta, gamma, delta float64,
	policy ArchivePolicy,
) *ParetoArchive {
	if maxSize <= 0 {
		maxSize = DefaultArchiveSize
	}

	if nGrid <= 0 {
		nGrid = DefaultArchiveNGrid
	}

	return &ParetoArchive{
		Solutions: make([]*ParetoSolution, 0, maxSize),
		Beta:      nonNegative(beta),
		Gamma:     nonNegative(gamma),
		Delta:     nonNegative(delta),
		Policy:    policy,
		MaxSize:   maxSize,
		NGrid:     nGrid,
	}
}

// nonNegative maps a negative or non-finite exponent to zero, which makes the
// corresponding roulette draw uniform rather than inverted.
func nonNegative(value float64) float64 {
	if !(value >= 0) || math.IsInf(value, 1) {
		return 0
	}

	return value
}

// Len reports the number of archived solutions.
func (pa *ParetoArchive) Len() int {
	if pa == nil {
		return 0
	}

	return len(pa.Solutions)
}

// GridBounds returns the per-objective extent of the archive's contents -- the
// lower bounds first, then the upper -- which is also the extent of the
// hypercube grid: bin b of objective m spans one NGrid-th of
// [lower[m], upper[m]], and that is the frame every solution's GridIndex is
// expressed in.
//
// Both slices are copies. The archive rewrites its bounds through the existing
// backing arrays on every mutation (see recomputeBounds), so a shared slice
// would silently change under the caller and would be a write path back into a
// running optimizer.
//
// A nil or empty archive has no extent and returns two nil slices.
func (pa *ParetoArchive) GridBounds() ([]float64, []float64) {
	if pa == nil {
		return nil, nil
	}

	return copyVec(pa.lowerBounds), copyVec(pa.upperBounds)
}

// Add offers a solution to the archive and reports whether it was accepted.
//
// The candidate is rejected when an archived solution dominates it or already
// occupies its exact objective vector; otherwise every solution the candidate
// dominates is removed and the candidate is appended. An insert that overflows
// MaxSize evicts one member of the most crowded hypercube, chosen by a roulette
// draw weighted N^Delta, so the archive never exceeds its capacity.
//
// The archive stores a deep copy, so the caller may reuse the candidate.
//
// rng is the last parameter by the package convention and is used only for the
// overflow eviction. A nil rng makes that eviction deterministic (the first
// member of the most crowded cell), which keeps the archive usable outside a
// seeded run.
func (pa *ParetoArchive) Add(solution *ParetoSolution, rng *rand.Rand) bool {
	if pa == nil || !validParetoSolution(solution, 0) {
		return false
	}

	if pa.MaxSize <= 0 {
		pa.MaxSize = DefaultArchiveSize
	}

	if pa.NGrid <= 0 {
		pa.NGrid = DefaultArchiveNGrid
	}

	expectedObjectives := len(solution.ObjectiveValues)
	if len(pa.Solutions) > 0 {
		if !validParetoSolution(pa.Solutions[0], 0) {
			return false
		}

		expectedObjectives = len(pa.Solutions[0].ObjectiveValues)
		if len(solution.ObjectiveValues) != expectedObjectives {
			return false
		}
	}

	for _, existing := range pa.Solutions {
		if !validParetoSolution(existing, expectedObjectives) {
			return false
		}

		if duplicateSolutions(existing, solution) || constrainedDominates(existing, solution) {
			return false
		}
	}

	before := append([]*ParetoSolution(nil), pa.Solutions...)

	// The survivors are compacted in place: the write index never overtakes the
	// read index, so the archive keeps its own backing array and an accepted
	// insert does not allocate a fresh slice of every member on the way past.
	// The consequence is that a slice of Solutions taken before the insert is
	// not a snapshot of it -- read the field again after any mutation.
	kept := pa.Solutions[:0]

	for _, existing := range pa.Solutions {
		if !constrainedDominates(solution, existing) {
			kept = append(kept, existing)
		}
	}

	inserted := solution.clone()

	// The clone carries whatever cell coordinate the candidate happened to hold,
	// which means nothing in this archive. Clearing it is what marks the member
	// as the one updateGrid still has to assign.
	inserted.GridIndex = nil
	inserted.GridKey = 0

	kept = append(kept, inserted)
	pa.Solutions = kept

	pa.updateGrid()
	pa.evictOverflow(rng)

	return !sameSolutionSequence(before, pa.Solutions)
}

func validParetoSolution(solution *ParetoSolution, objectives int) bool {
	if solution == nil || len(solution.ObjectiveValues) == 0 ||
		!isFinite(solution.ConstraintViolation) || solution.ConstraintViolation < 0 {
		return false
	}

	if objectives > 0 && len(solution.ObjectiveValues) != objectives {
		return false
	}

	for _, value := range solution.ObjectiveValues {
		if !isFinite(value) {
			return false
		}
	}

	return true
}

func sameSolutionSequence(first, second []*ParetoSolution) bool {
	if len(first) != len(second) {
		return false
	}

	for i := range first {
		if first[i] != second[i] {
			return false
		}
	}

	return true
}

// UpdateFromPopulation offers every member of a population to the archive and
// returns how many were accepted.
//
// Candidates are offered in slice order, so a seeded run is reproducible: the
// archive's contents depend on the order of the offers, not only on the set.
func (pa *ParetoArchive) UpdateFromPopulation(population []*ParetoSolution, rng *rand.Rand) int {
	accepted := 0

	for _, candidate := range population {
		if pa.Add(candidate, rng) {
			accepted++
		}
	}

	return accepted
}

// IsNonDominated reports whether every archived solution is mutually
// non-dominated. It is the archive's central invariant, and it is cheap enough
// (O(n²·m)) that tests assert it after every mutation rather than once at the
// end of a run.
func (pa *ParetoArchive) IsNonDominated() bool {
	if pa == nil {
		return true
	}

	expectedObjectives := 0
	if len(pa.Solutions) > 0 && pa.Solutions[0] != nil {
		expectedObjectives = len(pa.Solutions[0].ObjectiveValues)
	}

	for i, first := range pa.Solutions {
		if !validParetoSolution(first, expectedObjectives) {
			return false
		}

		for j, second := range pa.Solutions {
			if i != j && constrainedDominates(first, second) {
				return false
			}
		}
	}

	return true
}

// updateGrid recomputes the grid bounds from the current contents and reassigns
// every solution's cell.
//
// The bounds are the exact per-objective minimum and maximum of the archive,
// with no inflation: the extreme solutions land in the first and last bin. That
// keeps a cell assignment something a reader can work out by hand from the
// contents, which is what the grid tests check against.
func (pa *ParetoArchive) updateGrid() {
	if len(pa.Solutions) == 0 {
		pa.lowerBounds = nil
		pa.upperBounds = nil

		return
	}

	boundsChanged := pa.recomputeBounds()

	// A cell assignment is a pure function of the bounds and the solution's own
	// objective vector, so bit-identical bounds mean bit-identical indices for
	// every member that already carries one. Only the members this archive has
	// not indexed yet -- the freshly inserted candidate, whose index Add clears
	// -- need the assignment. That turns the common case, an insert that does
	// not widen the front, from a full sweep into a single assignment.
	for _, solution := range pa.Solutions {
		if !boundsChanged && len(solution.GridIndex) == len(solution.ObjectiveValues) {
			continue
		}

		solution.GridIndex = pa.gridIndexInto(solution.GridIndex, solution.ObjectiveValues)
		solution.GridKey = pa.gridKeyFor(solution.GridIndex)
	}
}

// recomputeBounds refreshes lowerBounds and upperBounds from the current
// contents and reports whether either moved.
//
// It writes through the existing backing arrays rather than allocating a pair
// per call, and compares the old value against the new one before overwriting
// it. The comparison is exact on purpose: the question is not whether the
// bounds are close but whether every derived index is unchanged, and that holds
// exactly when the bounds are bit-identical.
func (pa *ParetoArchive) recomputeBounds() bool {
	objectives := len(pa.Solutions[0].ObjectiveValues)

	changed := len(pa.lowerBounds) != objectives || len(pa.upperBounds) != objectives
	if changed {
		pa.lowerBounds = make([]float64, objectives)
		pa.upperBounds = make([]float64, objectives)
	}

	for m := range objectives {
		lower := math.Inf(1)
		upper := math.Inf(-1)

		for _, solution := range pa.Solutions {
			if m >= len(solution.ObjectiveValues) {
				continue
			}

			lower = math.Min(lower, solution.ObjectiveValues[m])
			upper = math.Max(upper, solution.ObjectiveValues[m])
		}

		if lower != pa.lowerBounds[m] || upper != pa.upperBounds[m] {
			changed = true
		}

		pa.lowerBounds[m] = lower
		pa.upperBounds[m] = upper
	}

	return changed
}

// gridIndexInto writes the hypercube coordinate of an objective vector under
// the current bounds into dst, reusing dst's backing array when it is large
// enough. The archive reassigns the same solutions call after call, so reusing
// each solution's existing index array keeps a grid refresh allocation-free.
//
// Bin m has width (upper[m]-lower[m])/NGrid, and the index is the floor of the
// normalized coordinate, clamped to [0, NGrid-1] so that a value sitting exactly
// on the upper bound lands in the last bin rather than one past it. A degenerate
// objective, where every archived solution scores the same, has every solution
// in bin zero.
func (pa *ParetoArchive) gridIndexInto(dst []int, values []float64) []int {
	var index []int

	if cap(dst) >= len(values) {
		index = dst[:len(values)]
		for m := range index {
			index[m] = 0
		}
	} else {
		index = make([]int, len(values))
	}

	for m, value := range values {
		if m >= len(pa.lowerBounds) || m >= len(pa.upperBounds) {
			continue
		}

		span := pa.upperBounds[m] - pa.lowerBounds[m]
		if !(span > 0) || !isFinite(value) {
			continue
		}

		normalized := (value - pa.lowerBounds[m]) / span
		if !isFinite(span) {
			// Halving before subtraction preserves the ratio when two finite
			// opposite-sign extrema have a span larger than MaxFloat64.
			normalized = (value/2 - pa.lowerBounds[m]/2) /
				(pa.upperBounds[m]/2 - pa.lowerBounds[m]/2)
		}

		bin := int(math.Floor(normalized * float64(pa.NGrid)))
		index[m] = min(max(bin, 0), pa.NGrid-1)
	}

	return index
}

// gridKeyFor flattens a hypercube coordinate into a single integer in base
// NGrid, so occupancy can be counted with one map lookup per solution.
func (pa *ParetoArchive) gridKeyFor(index []int) int {
	key := 0
	stride := 1
	maxInt := int(^uint(0) >> 1)

	for _, component := range index {
		if component < 0 || component >= pa.NGrid ||
			component > (maxInt-key)/stride ||
			stride > maxInt/pa.NGrid {
			return hashedGridKey(index)
		}

		key += component * stride
		stride *= pa.NGrid
	}

	return key
}

// hashedGridKey provides a deterministic negative key when base-NGrid
// flattening would overflow int. Negative keys deliberately bypass counting
// sort; sortingOrder disambiguates the vanishingly unlikely hash collision by
// comparing the full GridIndex.
func hashedGridKey(index []int) int {
	hasher := fnv.New64a()
	for _, component := range index {
		_, _ = io.WriteString(hasher, strconv.Itoa(component))
		_, _ = hasher.Write([]byte{0})
	}

	maxInt := uint64(^uint(0) >> 1)

	return -int(hasher.Sum64()&maxInt) - 1
}

// hypercubeCell is one occupied cell of the grid: its flattened key and the
// archive indices of the solutions inside it.
type hypercubeCell struct {
	Members []int
	Key     int
}

// occupiedCells groups the archive by hypercube, in ascending key order, with
// each cell's members in ascending archive-index order.
//
// The ordering is deliberate and load-bearing: every roulette draw walks these
// cells, so an order that depended on anything but the archive's contents --
// Go's randomized map iteration, say -- would stop a seeded run reproducing.
//
// The result borrows the archive's scratch buffers and is only valid until the
// next call or the next mutation. Every caller consumes it immediately, which
// is what lets the grouping be allocation-free on the hot path.
func (pa *ParetoArchive) occupiedCells() []hypercubeCell {
	count := len(pa.Solutions)

	pa.memberBuf = growInts(pa.memberBuf, count)
	members := pa.memberBuf[:count]

	if !pa.countingOrder(members) {
		pa.sortingOrder(members)
	}

	cells := pa.cellBuf[:0]

	for start := 0; start < count; {
		key := pa.Solutions[members[start]].GridKey

		end := start + 1
		for end < count && pa.Solutions[members[end]].GridKey == key &&
			equalGridIndex(
				pa.Solutions[members[start]].GridIndex,
				pa.Solutions[members[end]].GridIndex,
			) {
			end++
		}

		cells = append(cells, hypercubeCell{Members: members[start:end], Key: key})
		start = end
	}

	pa.cellBuf = cells

	return cells
}

// countingOrder fills members with the archive indices ordered by (cell key,
// index) using a counting sort over the key space, and reports whether it
// could.
//
// The keys are small integers -- NGrid^objectives of them -- so for the usual
// two- or three-objective problem the whole grid is a hundred or a thousand
// buckets and the grouping is linear. It declines, leaving the comparison sort
// to do the work, when the key space is too large to bucket or when a key falls
// outside it.
func (pa *ParetoArchive) countingOrder(members []int) bool {
	keyRange, ok := pa.keySpace()
	if !ok {
		return false
	}

	pa.countBuf = growInts(pa.countBuf, keyRange)
	counts := pa.countBuf[:keyRange]
	clear(counts)

	for _, solution := range pa.Solutions {
		if solution.GridKey < 0 || solution.GridKey >= keyRange {
			return false
		}

		counts[solution.GridKey]++
	}

	offset := 0
	for key, occupancy := range counts {
		counts[key] = offset
		offset += occupancy
	}

	// Walking the archive in index order places each cell's members in
	// ascending index order, which is the order the cells are expected to
	// carry.
	for i, solution := range pa.Solutions {
		members[counts[solution.GridKey]] = i
		counts[solution.GridKey]++
	}

	return true
}

// keySpace reports the number of distinct cell keys the current grid can
// produce, and whether that is few enough to bucket. It refuses a key space
// beyond maxCountingKeys, which is also what keeps the exponentiation from
// overflowing on a high-dimensional objective vector.
func (pa *ParetoArchive) keySpace() (int, bool) {
	if pa.NGrid <= 0 || len(pa.Solutions) == 0 {
		return 0, false
	}

	// A key space much larger than the archive itself costs more to clear than
	// the comparison sort costs to run, so the budget scales with the contents
	// and maxCountingKeys is only the hard ceiling.
	budget := min(countingKeyBase+countingKeySlack*len(pa.Solutions), maxCountingKeys)
	keyRange := 1

	for range len(pa.Solutions[0].ObjectiveValues) {
		if keyRange > budget/pa.NGrid {
			return 0, false
		}

		keyRange *= pa.NGrid
		if keyRange > budget {
			return 0, false
		}
	}

	return keyRange, true
}

// sortingOrder fills members with the archive indices ordered by (cell key,
// index) with a comparison sort. It is countingOrder's fallback, and the
// comparison is a total order, so the result is deterministic.
func (pa *ParetoArchive) sortingOrder(members []int) {
	for i := range members {
		members[i] = i
	}

	sort.Slice(members, func(i, j int) bool {
		left, right := members[i], members[j]
		if key := pa.Solutions[left].GridKey; key != pa.Solutions[right].GridKey {
			return key < pa.Solutions[right].GridKey
		}

		if comparison := compareGridIndex(
			pa.Solutions[left].GridIndex,
			pa.Solutions[right].GridIndex,
		); comparison != 0 {
			return comparison < 0
		}

		return left < right
	})
}

func equalGridIndex(first, second []int) bool {
	return compareGridIndex(first, second) == 0
}

func compareGridIndex(first, second []int) int {
	for i := range min(len(first), len(second)) {
		if first[i] < second[i] {
			return -1
		}

		if first[i] > second[i] {
			return 1
		}
	}

	if len(first) < len(second) {
		return -1
	}

	if len(first) > len(second) {
		return 1
	}

	return 0
}

// growInts returns a slice of at least the given length, reusing the buffer
// when it is already large enough.
func growInts(buf []int, length int) []int {
	if cap(buf) >= length {
		return buf
	}

	return make([]int, length)
}

// selectSparse draws a solution from a sparsely populated hypercube, weighting
// each occupied cell by 1/N^Beta. It is MODA's food source: the swarm is pulled
// toward the thin parts of the front, which is what spreads the archive out
// rather than piling it up where it already has coverage.
//
// Returns nil for an empty archive.
func (pa *ParetoArchive) selectSparse(rng *rand.Rand) *ParetoSolution {
	if pa != nil && pa.effectivePolicy() == ArchivePolicyMATLABDensity {
		return pa.selectByDensity(true, rng)
	}

	exponent := 1.0
	if pa != nil && pa.effectivePolicy() == ArchivePolicyMOPSOGrid {
		exponent = pa.Beta
	}

	return pa.selectFromCells(func(count float64) float64 {
		return -exponent * math.Log(count)
	}, rng)
}

// selectCrowded draws a solution from a crowded hypercube, weighting each
// occupied cell by N^Gamma. It is MODA's enemy: the swarm is pushed away from
// the parts of the front that are already well covered.
//
// Returns nil for an empty archive.
func (pa *ParetoArchive) selectCrowded(rng *rand.Rand) *ParetoSolution {
	if pa != nil && pa.effectivePolicy() == ArchivePolicyMATLABDensity {
		return pa.selectByDensity(false, rng)
	}

	exponent := 1.0
	if pa != nil && pa.effectivePolicy() == ArchivePolicyMOPSOGrid {
		exponent = pa.Gamma
	}

	return pa.selectFromCells(func(count float64) float64 {
		return exponent * math.Log(count)
	}, rng)
}

// selectFromCells performs the two-stage draw both selections share: a roulette
// over the occupied cells under the given weight function, then a uniform draw
// among the members of the chosen cell.
func (pa *ParetoArchive) selectFromCells(logWeight func(count float64) float64, rng *rand.Rand) *ParetoSolution {
	if pa == nil || len(pa.Solutions) == 0 {
		return nil
	}

	cells := pa.occupiedCells()
	weights := make([]float64, len(cells))

	for i, cell := range cells {
		weights[i] = logWeight(float64(len(cell.Members)))
	}

	chosen := cells[rouletteLogIndex(weights, rng)]
	member := chosen.Members[0]

	if rng != nil && len(chosen.Members) > 1 {
		member = chosen.Members[rng.Intn(len(chosen.Members))]
	}

	return pa.Solutions[member]
}

func (pa *ParetoArchive) effectivePolicy() ArchivePolicy {
	if pa == nil || pa.Policy == "" {
		return ArchivePolicyPaperSegments
	}

	return pa.Policy
}

// densityRanks reproduces RankingProcess.m. A point's rank is the number of
// archive points strictly closer than span/20 in every objective, including
// itself when every objective span is non-zero.
func (pa *ParetoArchive) densityRanks() []int {
	if pa == nil || len(pa.Solutions) == 0 {
		return nil
	}

	objectives := len(pa.Solutions[0].ObjectiveValues)
	lower := make([]float64, objectives)
	upper := make([]float64, objectives)

	copy(lower, pa.Solutions[0].ObjectiveValues)
	copy(upper, pa.Solutions[0].ObjectiveValues)

	for _, solution := range pa.Solutions[1:] {
		for m, value := range solution.ObjectiveValues {
			lower[m] = math.Min(lower[m], value)
			upper[m] = math.Max(upper[m], value)
		}
	}

	radius := make([]float64, objectives)
	for m := range radius {
		span := upper[m] - lower[m]

		radius[m] = span / 20
		if !isFinite(span) {
			radius[m] = upper[m]/20 - lower[m]/20
		}
	}

	ranks := make([]int, len(pa.Solutions))
	for i, first := range pa.Solutions {
		for _, second := range pa.Solutions {
			inside := true

			for m := range objectives {
				if !(math.Abs(second.ObjectiveValues[m]-first.ObjectiveValues[m]) < radius[m]) {
					inside = false

					break
				}
			}

			if inside {
				ranks[i]++
			}
		}
	}

	return ranks
}

func (pa *ParetoArchive) selectByDensity(sparse bool, rng *rand.Rand) *ParetoSolution {
	if pa == nil || len(pa.Solutions) == 0 {
		return nil
	}

	ranks := pa.densityRanks()
	logWeights := make([]float64, len(ranks))
	positiveRank := false

	for i, rank := range ranks {
		if rank == 0 {
			// MODA.m's roulette cannot select from all-zero or infinite
			// weights and falls back to the first member.
			if sparse {
				return pa.Solutions[0]
			}

			logWeights[i] = math.Inf(-1)

			continue
		}

		positiveRank = true

		logWeights[i] = math.Log(float64(rank))
		if sparse {
			logWeights[i] = -logWeights[i]
		}
	}

	if !positiveRank {
		return pa.Solutions[0]
	}

	return pa.Solutions[rouletteLogIndex(logWeights, rng)]
}

// evictOverflow trims the archive back to MaxSize, deleting from the most
// crowded hypercube -- a roulette over the occupied cells weighted N^Delta,
// then a member of the chosen cell.
//
// It loops rather than deleting once, so an archive that somehow overshot by
// more than one still lands on capacity.
func (pa *ParetoArchive) evictOverflow(rng *rand.Rand) {
	for len(pa.Solutions) > pa.MaxSize {
		if pa.effectivePolicy() == ArchivePolicyMATLABDensity {
			ranks := pa.densityRanks()
			logWeights := make([]float64, len(ranks))
			positiveRank := false

			for i, rank := range ranks {
				if rank == 0 {
					logWeights[i] = math.Inf(-1)
				} else {
					positiveRank = true
					logWeights[i] = math.Log(float64(rank))
				}
			}

			victim := 0
			if positiveRank {
				victim = rouletteLogIndex(logWeights, rng)
			}

			pa.Solutions = append(pa.Solutions[:victim], pa.Solutions[victim+1:]...)
			pa.updateGrid()

			continue
		}

		cells := pa.occupiedCells()
		logWeights := make([]float64, len(cells))

		exponent := 1.0
		if pa.effectivePolicy() == ArchivePolicyMOPSOGrid {
			exponent = pa.Delta
		}

		for i, cell := range cells {
			logWeights[i] = exponent * math.Log(float64(len(cell.Members)))
		}

		chosen := cells[rouletteLogIndex(logWeights, rng)]
		victim := chosen.Members[0]

		if rng != nil && len(chosen.Members) > 1 {
			victim = chosen.Members[rng.Intn(len(chosen.Members))]
		}

		pa.Solutions = append(pa.Solutions[:victim], pa.Solutions[victim+1:]...)
		pa.updateGrid()
	}
}

// rouletteIndex draws an index from weights with probability proportional to
// the weight, and falls back to a uniform draw when the weights carry no usable
// information -- all zero, negative, or non-finite. A nil rng always returns the
// first index, which is what makes an rng-free archive deterministic.
func rouletteIndex(weights []float64, rng *rand.Rand) int {
	if len(weights) == 0 {
		return 0
	}

	if rng == nil {
		return 0
	}

	maxWeight := 0.0

	infinite := make([]int, 0, len(weights))
	for i, weight := range weights {
		if math.IsInf(weight, 1) {
			infinite = append(infinite, i)
		} else if isFinite(weight) && weight > maxWeight {
			maxWeight = weight
		}
	}

	if len(infinite) > 0 {
		return infinite[rng.Intn(len(infinite))]
	}

	if !(maxWeight > 0) {
		return rng.Intn(len(weights))
	}

	total := 0.0

	for _, weight := range weights {
		if isFinite(weight) && weight > 0 {
			total += weight / maxWeight
		}
	}

	draw := rng.Float64() * total
	cumulative := 0.0

	for i, weight := range weights {
		if isFinite(weight) && weight > 0 {
			cumulative += weight / maxWeight
		}

		if draw < cumulative {
			return i
		}
	}

	return len(weights) - 1
}

// rouletteLogIndex performs a stable roulette draw from logarithmic weights.
// Subtracting the largest log weight before exponentiation prevents both
// overflow and underflow from turning an intentional bias into a uniform draw.
func rouletteLogIndex(logWeights []float64, rng *rand.Rand) int {
	if len(logWeights) == 0 {
		return 0
	}

	positiveInfinity := make([]int, 0, len(logWeights))

	maxLog := math.Inf(-1)
	for i, weight := range logWeights {
		if math.IsInf(weight, 1) {
			positiveInfinity = append(positiveInfinity, i)
		} else if !math.IsNaN(weight) && weight > maxLog {
			maxLog = weight
		}
	}

	if len(positiveInfinity) > 0 {
		if rng == nil {
			return positiveInfinity[0]
		}

		return positiveInfinity[rng.Intn(len(positiveInfinity))]
	}

	if math.IsInf(maxLog, -1) {
		if rng == nil {
			return 0
		}

		return rng.Intn(len(logWeights))
	}

	weights := make([]float64, len(logWeights))
	for i, weight := range logWeights {
		if !math.IsNaN(weight) && !math.IsInf(weight, -1) {
			weights[i] = math.Exp(weight - maxLog)
		}
	}

	if rng == nil {
		for i, weight := range weights {
			if weight == 1 {
				return i
			}
		}
	}

	return rouletteIndex(weights, rng)
}

// MultiObjectiveConfig configures a MODA run.
//
// Swarm carries the shared mechanics -- bounds, population, iterations, weight
// schedules, boundary rule, Lévy parameters and the RNG -- so that everything
// the single-objective algorithm already documents keeps meaning the same thing
// here. Swarm.ObjectiveFunc is ignored: a multi-objective run scores positions
// through ObjectiveFunc below.
//
// Use NewMultiObjectiveConfig to build one; you must then set ObjectiveFunc and
// Swarm's ProblemSize, LowerBound and UpperBound.
type MultiObjectiveConfig struct {
	ObjectiveFunc MultiObjectiveFunction `json:"-"`
	Swarm         *Config                `json:"swarm"`
	// ArchivePolicy selects paper segments, MATLAB density ranking, or the
	// v0.1.0 MOPSO grid extension. Empty follows Swarm.FidelityMode.
	ArchivePolicy ArchivePolicy `json:"archive_policy,omitempty"`

	// Beta, Gamma and Delta are used only by ArchivePolicyMOPSOGrid. They are
	// legacy extension parameters, not paper or MODA.m constants.
	Beta  float64 `json:"beta"`
	Gamma float64 `json:"gamma"`
	Delta float64 `json:"delta"`

	ArchiveSize int `json:"archive_size"`
	NGrid       int `json:"n_grid"`
}

// NewMultiObjectiveConfig creates a default MODA configuration.
// You must set ObjectiveFunc and Swarm's ProblemSize, LowerBound and UpperBound.
//
// The exponent fields retain the v0.1 MOPSO extension defaults; paper mode does
// not consult them. ArchiveSize 100 is verified against MODA.m.
func NewMultiObjectiveConfig() *MultiObjectiveConfig {
	return &MultiObjectiveConfig{
		Swarm:       NewDefaultConfig(),
		Beta:        DefaultArchiveBeta,
		Gamma:       DefaultArchiveGamma,
		Delta:       DefaultArchiveDelta,
		ArchiveSize: DefaultArchiveSize,
		NGrid:       DefaultArchiveNGrid,
	}
}

// MultiObjectiveResult holds the outcome of a MODA run.
//
// There is no single best position to report, so Archive is the result: the
// approximation of the Pareto front the run converged on.
type MultiObjectiveResult struct {
	// TerminationReason is TerminationMaxIterations for a run that used its
	// whole iteration budget, and TerminationStagnation for one that
	// Swarm.Convergence stopped early. A canceled run has no result at all --
	// OptimizeMultiObjective returns ctx.Err() instead -- so there is no
	// cancellation reason to report here.
	TerminationReason TerminationReason

	Archive *ParetoArchive

	// ArchiveSizeCurve records the archive's size after each completed
	// iteration, the multi-objective analog of Result.ConvergenceCurve. A
	// curve that stops growing early is the usual sign of a run that has
	// stagnated.
	ArchiveSizeCurve []int

	FuncEvalCount  int
	IterationCount int
	Seed           int64 // Random seed used for reproducibility, when SeedKnown is true
	SeedKnown      bool  // Whether Seed identifies the random stream that drove the run
}

// validateMultiObjectiveConfig rejects a configuration MODA cannot run.
//
// The swarm block is checked by the single-objective validateConfig, against a
// shallow copy carrying a stub ObjectiveFunc: the shared fields must mean
// exactly what they mean for a single-objective run, and the copy keeps that one
// check from demanding a single-objective function a MODA caller has no reason
// to supply.
func validateMultiObjectiveConfig(config *MultiObjectiveConfig) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	if config.ObjectiveFunc == nil {
		return errors.New("ObjectiveFunc must be set")
	}

	if config.Swarm == nil {
		return errors.New("swarm config must be set; start from NewMultiObjectiveConfig")
	}

	if config.Swarm.UseBinary {
		return errors.New("swarm.use_binary is not supported for a multi-objective run")
	}

	policy := effectiveArchivePolicy(config)
	if policy != ArchivePolicyPaperSegments &&
		policy != ArchivePolicyMATLABDensity &&
		policy != ArchivePolicyMOPSOGrid {
		return fmt.Errorf("unknown archive_policy %q", config.ArchivePolicy)
	}

	swarm := *config.Swarm
	swarm.ObjectiveFunc = func([]float64) float64 { return 0 }

	swarmErr := validateConfig(&swarm)
	if swarmErr != nil {
		return fmt.Errorf("invalid swarm config: %w", swarmErr)
	}

	constraintErr := validateMultiObjectiveConstraints(config.Swarm.Constraints)
	if constraintErr != nil {
		return constraintErr
	}

	convergenceErr := validateMultiObjectiveConvergence(config.Swarm.Convergence)
	if convergenceErr != nil {
		return convergenceErr
	}

	if config.ArchiveSize <= 0 {
		return fmt.Errorf("archive_size must be positive, got %d", config.ArchiveSize)
	}

	if config.NGrid <= 0 {
		return fmt.Errorf("n_grid must be positive, got %d", config.NGrid)
	}

	for _, field := range []namedValue{
		{name: "beta", value: config.Beta},
		{name: "gamma", value: config.Gamma},
		{name: "delta", value: config.Delta},
	} {
		if !isFinite(field.value) || field.value < 0 {
			return fmt.Errorf("%s must be a non-negative finite number, got %v", field.name, field.value)
		}
	}

	return nil
}

// validateMultiObjectiveConstraints rejects the one constraint policy that has
// no defensible multi-objective reading.
//
// MODA handles constraints by constrained domination (see
// constrainedDominates), which needs no penalty factor. Penalty handling folds
// a violation into a scalar cost, and a multi-objective run has no scalar cost
// to fold it into: applying the penalty to every objective component would
// invent a trade-off the caller never described. It fails loudly rather than
// being approximated.
func validateMultiObjectiveConstraints(constraints *ConstraintConfig) error {
	if constraints == nil || effectiveConstraintHandling(constraints) != ConstraintHandlingPenalty {
		return nil
	}

	return errors.New(
		"constraints.handling = penalty is not supported for a multi-objective run: " +
			"there is no single cost to penalize; use the default feasibility rules")
}

// validateMultiObjectiveConvergence rejects the one early-stopping criterion
// that has no multi-objective meaning.
//
// A target cost is a statement about a single incumbent, and a MODA run does
// not have one: every archive member is optimal in the only sense a Pareto
// front recognizes. Silently ignoring the field would let a caller believe a
// budget was capped when it was not, so it is rejected instead. Stagnation and
// MinIterations do carry over -- see moStagnationTracker.
func validateMultiObjectiveConvergence(convergence *ConvergenceConfig) error {
	if convergence == nil {
		return nil
	}

	if convergence.TargetCost != nil {
		return errors.New(
			"convergence.target_cost has no meaning for a multi-objective run: " +
				"a Pareto front has no single best cost; use stagnation_iterations instead")
	}

	if convergence.MinImprovement != 0 {
		return errors.New(
			"convergence.min_improvement has no meaning for a multi-objective run: " +
				"archive changes are discrete; leave it at zero")
	}

	return nil
}

func effectiveArchivePolicy(config *MultiObjectiveConfig) ArchivePolicy {
	if config != nil && config.ArchivePolicy != "" {
		return config.ArchivePolicy
	}

	if config != nil && config.Swarm != nil && config.Swarm.FidelityMode == FidelityMATLAB {
		return ArchivePolicyMATLABDensity
	}

	return ArchivePolicyPaperSegments
}

// moStagnationTracker is MODA's early-stopping rule: the archive is the
// incumbent, so an iteration counts as an improvement exactly when the archive
// accepted at least one candidate.
//
// It deliberately does not reuse convergenceTracker. That type ranks a scalar
// Best through a constraintEvaluator and measures improvement as a cost margin,
// and neither quantity exists here. What it does reuse is the shape of
// convergenceTracker.observe: the counter is maintained from the first
// observation, but MinIterations gates whether it may stop the run, so a run
// that stalls during the warm-up period stops as soon as the gate opens.
//
// Config.Convergence.MinImprovement is not consulted. Archive acceptance is a
// yes-or-no answer with no margin to compare against.
type moStagnationTracker struct {
	config             *ConvergenceConfig
	stagnantIterations int
}

// observe records one completed iteration and reports whether the run should
// stop. iteration is the 1-based count of completed iterations, matching what
// OptimizeContext feeds convergenceTracker.observe.
func (tracker *moStagnationTracker) observe(iteration, accepted int) (TerminationReason, bool) {
	if tracker == nil || tracker.config == nil {
		return "", false
	}

	if accepted > 0 {
		tracker.stagnantIterations = 0
	} else {
		tracker.stagnantIterations++
	}

	if iteration < max(tracker.config.MinIterations, 1) {
		return "", false
	}

	if tracker.config.StagnationIterations > 0 &&
		tracker.stagnantIterations >= tracker.config.StagnationIterations {
		return TerminationStagnation, true
	}

	return "", false
}

// moEvaluation is one dragonfly's raw score for the batch in flight: the
// objective vector exactly as the caller's function returned it, and the
// aggregate constraint violation of the position that produced it.
//
// It is the unit of work the parallel path fans out. Nothing here is
// sanitized, ranked or offered to the archive -- all of that happens on the
// calling goroutine afterwards, in swarm index order.
type moEvaluation struct {
	values    []float64
	violation float64
}

// moState is the mutable state of one MODA run.
type moState struct {
	archive *ParetoArchive
	swarm   []Dragonfly
	// scores is the scratch buffer one evaluation pass writes into, one element
	// per dragonfly. Each worker writes exactly one element and reads none, so
	// the buffer needs no synchronization beyond parallelFor's join.
	scores    []moEvaluation
	funcEvals int
	// objectiveCount is fixed by the first completed evaluation batch and
	// checked before any candidate from later batches can mutate the archive.
	objectiveCount int
	// maxWorkers is the fan-out of the objective evaluation, or zero when the
	// run scores the swarm on the calling goroutine. It is the only thing
	// EnableParallel changes about a MODA run: every random draw happens in the
	// prepare phase either way, so a seeded run is bit-identical with it set or
	// not.
	maxWorkers int
}

// OptimizeMultiObjective runs the multi-objective Dragonfly Algorithm, honoring
// context cancellation.
//
// It is a separate entry point rather than a mode of OptimizeContext: a
// multi-objective run has no single best cost, so Result would have to report an
// incumbent that does not exist. Cancellation is checked at the top of every
// iteration and again inside a parallel evaluation pass, and a canceled run
// returns a nil result and ctx.Err(), so a caller cannot mistake an aborted run
// for a completed one.
//
// The swarm mechanics are identical to the single-objective algorithm. Only the
// food source and the enemy differ: both are drawn from the archive once per
// iteration, the food from a sparse hypercube and the enemy from a crowded one.
//
// Swarm.Convergence may end the run early: an iteration in which the archive
// accepted nothing counts as stagnant, and StagnationIterations consecutive
// stagnant iterations stop the run with TerminationStagnation. A target cost is
// rejected by validation rather than ignored.
//
// Swarm.Constraints, when set, is honored by constrained domination rather than
// by a penalty: a feasible archive member dominates every infeasible one, and
// two infeasible members are ordered by aggregate violation. Penalty handling is
// rejected by validation; see validateMultiObjectiveConstraints.
//
// Swarm.EnableParallel fans the objective calls out across worker goroutines.
// It changes nothing else: a seeded run is bit-identical with it set or not,
// because no worker draws a random number and the archive is built in swarm
// index order on the calling goroutine either way.
func OptimizeMultiObjective(
	ctx context.Context,
	config *MultiObjectiveConfig,
	options ...RunOption,
) (*MultiObjectiveResult, error) {
	contextErr := requireContext(ctx)
	if contextErr != nil {
		return nil, contextErr
	}

	resolved, optionsErr := resolveRunOptions(options)
	if optionsErr != nil {
		return nil, optionsErr
	}

	validationErr := validateMultiObjectiveConfig(config)
	if validationErr != nil {
		return nil, validationErr
	}

	optionMeaningErr := validateMultiObjectiveRunOptions(resolved)
	if optionMeaningErr != nil {
		return nil, optionMeaningErr
	}

	swarmConfig := config.Swarm

	// As in OptimizeContext, the seeded population is checked only after the
	// config is known good: the check is against ProblemSize, NPop and the
	// bounds.
	populationErr := validateInitialPopulation(swarmConfig, resolved)
	if populationErr != nil {
		return nil, populationErr
	}

	rng, seed, seedKnown := resolveRandomSource(swarmConfig)

	state, initErr := initializeMultiObjectiveRun(ctx, config, resolved, rng)
	if initErr != nil {
		return nil, initErr
	}

	tracker := &moStagnationTracker{config: swarmConfig.Convergence}
	curve := make([]int, 0, swarmConfig.MaxIterations)
	reason := TerminationMaxIterations
	completed := 0

	for t := range swarmConfig.MaxIterations {
		ctxErr := ctx.Err()
		if ctxErr != nil {
			return nil, ctxErr
		}

		var (
			accepted int
			stepErr  error
		)
		if effectiveFidelityMode(swarmConfig) == FidelityMATLAB {
			accepted, stepErr = state.advanceMATLAB(ctx, config, t, rng)
		} else {
			accepted, stepErr = state.advance(ctx, config, t, rng)
		}

		if stepErr != nil {
			return nil, stepErr
		}

		completed = t + 1

		curve = append(curve, state.archive.Len())

		notifyArchive(resolved.archiveObserver, completed, state.funcEvals, state.archive)

		stopReason, stop := tracker.observe(completed, accepted)
		if stop {
			reason = stopReason

			break
		}
	}

	return &MultiObjectiveResult{
		Archive:           state.archive,
		ArchiveSizeCurve:  curve,
		TerminationReason: reason,
		FuncEvalCount:     state.funcEvals,
		IterationCount:    completed,
		Seed:              seed,
		SeedKnown:         seedKnown,
	}, nil
}

// initializeMultiObjectiveRun builds the starting swarm, scores it, and seeds
// the archive, so the first step already has a food source and an enemy to be
// computed against.
//
// Positions and steps are drawn exactly as initializeRun draws them for a
// single-objective run.
func initializeMultiObjectiveRun(
	ctx context.Context,
	config *MultiObjectiveConfig,
	options runOptions,
	rng *rand.Rand,
) (*moState, error) {
	swarmConfig := config.Swarm
	swarm := make([]Dragonfly, swarmConfig.NPop)

	for i := range swarm {
		// Both draws are taken before a seeded position replaces the first,
		// exactly as initializeRun does it: skipping the draw for a seeded
		// slot would shift the random stream for every dragonfly after it, so
		// seeding one position would change the whole run.
		position := unifrndVec(swarmConfig.LowerBound, swarmConfig.UpperBound, swarmConfig.ProblemSize, rng)
		step := unifrndVec(swarmConfig.LowerBound, swarmConfig.UpperBound, swarmConfig.ProblemSize, rng)

		if i < len(options.initialPositions) {
			position = copyVec(options.initialPositions[i])
		}

		swarm[i] = Dragonfly{
			Position: position,
			Step:     step,
			Cost:     math.Inf(1),
		}
	}

	state := &moState{
		archive: newParetoArchive(config.ArchiveSize, config.NGrid,
			config.Beta, config.Gamma, config.Delta, effectiveArchivePolicy(config)),
		swarm:  swarm,
		scores: make([]moEvaluation, len(swarm)),
	}

	if swarmConfig.EnableParallel {
		state.maxWorkers = effectiveMaxWorkers(swarmConfig)
	}

	// Paper mode starts from an evaluated archive and then evaluates every
	// moved population. MODA.m instead evaluates at the start of each loop and
	// leaves the final moved population unevaluated, so its initialization must
	// stay raw until advanceMATLAB performs generation one.
	if effectiveFidelityMode(swarmConfig) != FidelityMATLAB {
		_, err := state.evaluateSwarm(ctx, config, rng)
		if err != nil {
			return nil, err
		}
	}

	return state, nil
}

// evaluateSwarm scores every dragonfly and offers the results to the archive.
//
// Every objective component passes through sanitizeCost, so a NaN or -Inf from a
// misbehaving objective becomes +Inf rather than a point that dominates the
// whole archive and can never be displaced.
//
// Swarm.Constraints, when set, is evaluated here too, exactly once per
// objective evaluation, and the aggregate violation is recorded on both the
// dragonfly and the candidate. The archive then ranks by constrained
// domination, so a feasible point displaces an infeasible one whatever their
// objective vectors say.
//
// The count returned is how many candidates the archive accepted. It is the
// multi-objective analog of an improvement in the incumbent, and it is what the
// stagnation rule in moStagnationTracker counts iterations against.
//
// The objective calls themselves may fan out across goroutines when
// Swarm.EnableParallel is set; see scorePositions. Everything else -- the
// sanitizing, the ordering and the archive insert -- happens on the calling
// goroutine, in swarm index order, so the archive a run builds does not depend
// on how the workers interleaved.
func (state *moState) evaluateSwarm(
	ctx context.Context,
	config *MultiObjectiveConfig,
	rng *rand.Rand,
) (int, error) {
	scoreErr := state.scorePositions(ctx, config)
	if scoreErr != nil {
		return 0, scoreErr
	}

	state.funcEvals += len(state.swarm)

	expectedObjectives := state.objectiveCount
	for i := range state.swarm {
		values := state.scores[i].values
		if len(values) == 0 {
			return 0, errors.New("ObjectiveFunc must return at least one objective value")
		}

		if expectedObjectives == 0 {
			expectedObjectives = len(values)
		}

		if len(values) != expectedObjectives {
			return 0, fmt.Errorf("ObjectiveFunc must return a fixed number of objectives, got %d then %d",
				expectedObjectives, len(values))
		}

		for m, value := range values {
			if !isFinite(value) {
				return 0, fmt.Errorf("%w: objective %d returned %v", ErrNoFiniteObjective, m, value)
			}
		}

		if violation := state.scores[i].violation; !isFinite(violation) || violation < 0 {
			return 0, fmt.Errorf("constraints returned invalid aggregate violation %v", violation)
		}
	}

	state.objectiveCount = expectedObjectives
	candidates := make([]*ParetoSolution, 0, len(state.swarm))

	for i := range state.swarm {
		fly := &state.swarm[i]
		values := state.scores[i].values

		// Cost carries the first objective purely so an inspecting caller sees
		// something meaningful; nothing in a MODA run reads it.
		fly.Cost = values[0]
		fly.ConstraintViolation = state.scores[i].violation

		candidates = append(candidates, &ParetoSolution{
			Position:            copyVec(fly.Position),
			ObjectiveValues:     copyVec(values),
			ConstraintViolation: fly.ConstraintViolation,
		})
	}

	return state.archive.UpdateFromPopulation(candidates, rng), nil
}

// scorePositions fills state.scores with one raw evaluation per dragonfly,
// either on the calling goroutine or across maxWorkers of them.
//
// This is the only part of a MODA iteration that may run concurrently, and it
// is safe to fan out for the reason the whole package is built around: it draws
// no random numbers. The objective function is the caller's and is documented
// to be safe for concurrent use when EnableParallel is set; EvaluateConstraints
// reads the position and nothing else. Everything that consumes the RNG -- the
// weight schedules, the food and enemy draws, the step update, the archive
// insert -- has already happened, or happens afterwards, on the calling
// goroutine.
//
// Nothing is written back to the swarm here, so a canceled pass leaves the
// swarm carrying the previous iteration's scores rather than a mixture.
func (state *moState) scorePositions(ctx context.Context, config *MultiObjectiveConfig) error {
	if len(state.scores) < len(state.swarm) {
		state.scores = make([]moEvaluation, len(state.swarm))
	}

	score := func(i int) {
		state.scores[i] = moEvaluation{
			// Copy at the callback boundary: callers are allowed to reuse a
			// scratch result slice on their next invocation.
			values:    copyVec(config.ObjectiveFunc(state.swarm[i].Position)),
			violation: EvaluateConstraints(state.swarm[i].Position, config.Swarm.Constraints).Violation,
		}
	}

	if state.maxWorkers > 0 {
		return parallelFor(ctx, len(state.swarm), state.maxWorkers, score)
	}

	for i := range state.swarm {
		err := ctx.Err()
		if err != nil {
			return err
		}

		score(i)
	}

	return ctx.Err()
}

// advance runs one MODA iteration: draw the food source and the enemy from the
// archive, move every dragonfly, then rescore and update the archive. It
// returns how many candidates the archive accepted this iteration.
func (state *moState) advance(
	ctx context.Context,
	config *MultiObjectiveConfig,
	iteration int,
	rng *rand.Rand,
) (int, error) {
	swarmConfig := config.Swarm
	weights := computeWeights(swarmConfig, iteration+1, swarmConfig.MaxIterations, rng)

	// Drawn once per iteration, as the reference MODA does, rather than once per
	// dragonfly: the whole swarm is pulled toward the same sparse region and
	// pushed away from the same crowded one for the length of an iteration.
	food := copyVec(archivePosition(state.archive.selectSparse(rng)))
	enemy := copyVec(archivePosition(state.archive.selectCrowded(rng)))

	// The step update is the single-objective one, called through the very same
	// helper -- MODA changes what the swarm is attracted to, not how it moves.
	// This view shares the swarm's backing array, so the moves it makes are the
	// moves this run makes; only the food and enemy positions are substituted.
	// Cost is not read by the step, so the two Best values carry positions only.
	view := &runState{
		swarm:         state.swarm,
		food:          Best{Position: food},
		enemy:         Best{Position: enemy},
		movementEnemy: Best{Position: enemy},
	}

	for i := range state.swarm {
		prepareSwarmStep(view, i, swarmConfig, weights, rng)
	}

	return state.evaluateSwarm(ctx, config, rng)
}

// advanceMATLAB reproduces MODA.m's generation lifecycle. The current swarm is
// evaluated and offered to the archive first; food and enemy are then selected
// from that evaluated archive and the swarm moves once. The moved population is
// the starting population of the next generation, and the final movement is
// deliberately never evaluated.
func (state *moState) advanceMATLAB(
	ctx context.Context,
	config *MultiObjectiveConfig,
	iteration int,
	rng *rand.Rand,
) (int, error) {
	swarmConfig := config.Swarm
	weights := computeMATLABMODAWeights(swarmConfig, iteration+1, swarmConfig.MaxIterations, rng)

	accepted, evaluationErr := state.evaluateSwarm(ctx, config, rng)
	if evaluationErr != nil {
		return 0, evaluationErr
	}

	food := copyVec(archivePosition(state.archive.selectSparse(rng)))
	enemy := copyVec(archivePosition(state.archive.selectCrowded(rng)))
	view := &runState{
		swarm:         state.swarm,
		food:          Best{Position: food},
		enemy:         Best{Position: enemy},
		movementEnemy: Best{Position: enemy},
	}

	for i := range state.swarm {
		prepareSwarmStep(view, i, swarmConfig, weights, rng)
	}

	return accepted, ctx.Err()
}

// computeMATLABMODAWeights is the schedule in the author's MODA.m rather than
// DA.m's schedule used by the shared paper implementation. Automatic S/A/C/E
// weights all follow mc directly, the food term alone draws a random factor,
// and inertia falls from 0.9 to 0.2. Explicit fixed weights remain deliberate
// library extensions and continue to override their automatic counterparts.
func computeMATLABMODAWeights(
	config *Config,
	iteration, maxIterations int,
	rng *rand.Rand,
) weightSchedule {
	progress := scheduleProgress(iteration, maxIterations)
	mc := convergenceFactor(iteration, maxIterations)
	automaticSwarmWeight := mc

	if float64(iteration) >= 0.75*float64(maxIterations) && iteration > 0 {
		automaticSwarmWeight /= float64(iteration)
	}

	span := config.UpperBound - config.LowerBound

	return weightSchedule{
		Inertia:    0.9 - progress*(0.9-0.2),
		Separation: resolveWeight(config.SeparationWeight, automaticSwarmWeight),
		Alignment:  resolveWeight(config.AlignmentWeight, automaticSwarmWeight),
		Cohesion:   resolveWeight(config.CohesionWeight, automaticSwarmWeight),
		Food:       resolveWeight(config.FoodWeight, 2*unifrnd(0, 1, rng)),
		Enemy:      resolveWeight(config.EnemyWeight, automaticSwarmWeight),
		Radius:     neighborhoodRadius(config, iteration, maxIterations),
		MaxStep:    span * config.MaxStepRatio,
	}
}

// archivePosition returns a solution's position, or nil for a nil solution. A
// nil food source or enemy makes the corresponding primitive the zero vector,
// which is the documented behavior of foodVector and enemyVector.
func archivePosition(solution *ParetoSolution) []float64 {
	if solution == nil {
		return nil
	}

	return solution.Position
}

// ParetoPoint is one exported archive member: its objective vector and the
// position that produced it.
type ParetoPoint struct {
	Position   []float64 `json:"position"`
	Objectives []float64 `json:"objectives"`
	// ConstraintViolation is the aggregate violation of Position, zero for a
	// feasible solution and for every solution of an unconstrained run.
	ConstraintViolation float64 `json:"constraint_violation"`
	Index               int     `json:"index"`
}

// ParetoExport is the document ExportParetoJSON writes: the archived front plus
// the run-level summary it belongs to.
type ParetoExport struct {
	TerminationReason TerminationReason `json:"termination_reason,omitempty"`
	Front             []ParetoPoint     `json:"front"`
	ArchiveSizeCurve  []int             `json:"archive_size_curve,omitempty"`
	Seed              int64             `json:"seed"`
	SeedKnown         bool              `json:"seed_known"`
	ArchiveSize       int               `json:"archive_size"`
	FuncEvalCount     int               `json:"func_eval_count"`
	IterationCount    int               `json:"iteration_count"`
}

var (
	errNilMultiObjectiveResult = errors.New("multi-objective result cannot be nil")
	errNilParetoArchive        = errors.New("pareto archive cannot be nil")
	errNilParetoSolution       = errors.New("pareto archive contains a nil solution")
)

// ExportParetoCSV writes the archived front to path, one row per solution, with
// an index column, one column per objective and one per decision variable.
//
// The column count follows the archive's contents, so an empty archive yields a
// header-only file with just the index column rather than an error.
func (result *MultiObjectiveResult) ExportParetoCSV(path string) error {
	points, err := result.paretoPoints()
	if err != nil {
		return err
	}

	header := paretoCSVHeader(points)

	return writeExportFile(path, "Pareto CSV", func(sink io.Writer) error {
		writer := csv.NewWriter(sink)

		headerErr := writer.Write(header)
		if headerErr != nil {
			return fmt.Errorf("write Pareto CSV header: %w", headerErr)
		}

		for _, point := range points {
			rowErr := writer.Write(paretoCSVRow(point))
			if rowErr != nil {
				return fmt.Errorf("write Pareto CSV row: %w", rowErr)
			}
		}

		writer.Flush()

		flushErr := writer.Error()
		if flushErr != nil {
			return fmt.Errorf("flush Pareto CSV: %w", flushErr)
		}

		return nil
	})
}

// paretoCSVHeader builds the column names from the first exported point. The
// objective and variable counts follow the archive's contents, so an empty
// archive yields the two fixed columns and nothing else.
func paretoCSVHeader(points []ParetoPoint) []string {
	header := []string{"index"}

	if len(points) > 0 {
		for m := range points[0].Objectives {
			header = append(header, "objective_"+strconv.Itoa(m))
		}

		for j := range points[0].Position {
			header = append(header, "x_"+strconv.Itoa(j))
		}
	}

	return append(header, "constraint_violation")
}

// paretoCSVRow formats one exported point as CSV fields.
func paretoCSVRow(point ParetoPoint) []string {
	row := make([]string, 0, 2+len(point.Objectives)+len(point.Position))
	row = append(row, strconv.Itoa(point.Index))

	for _, value := range point.Objectives {
		row = append(row, strconv.FormatFloat(value, 'g', -1, 64))
	}

	for _, value := range point.Position {
		row = append(row, strconv.FormatFloat(value, 'g', -1, 64))
	}

	return append(row, strconv.FormatFloat(point.ConstraintViolation, 'g', -1, 64))
}

// ExportParetoJSON writes the archived front and the run summary to path as an
// indented ParetoExport document.
func (result *MultiObjectiveResult) ExportParetoJSON(path string) error {
	points, err := result.paretoPoints()
	if err != nil {
		return err
	}

	document := ParetoExport{
		TerminationReason: result.TerminationReason,
		Front:             points,
		ArchiveSizeCurve:  result.ArchiveSizeCurve,
		Seed:              result.Seed,
		SeedKnown:         result.SeedKnown,
		ArchiveSize:       len(points),
		FuncEvalCount:     result.FuncEvalCount,
		IterationCount:    result.IterationCount,
	}

	return writeExportFile(path, "Pareto JSON", func(sink io.Writer) error {
		encoder := json.NewEncoder(sink)
		encoder.SetIndent("", "  ")

		encodeErr := encoder.Encode(document)
		if encodeErr != nil {
			return fmt.Errorf("encode Pareto JSON: %w", encodeErr)
		}

		return nil
	})
}

// paretoPoints flattens the archive into export records, in archive order.
func (result *MultiObjectiveResult) paretoPoints() ([]ParetoPoint, error) {
	if result == nil {
		return nil, errNilMultiObjectiveResult
	}

	if result.Archive == nil {
		return nil, errNilParetoArchive
	}

	points := make([]ParetoPoint, 0, result.Archive.Len())
	expectedObjectives := 0

	for i, solution := range result.Archive.Solutions {
		if solution == nil {
			return nil, fmt.Errorf("%w at index %d", errNilParetoSolution, i)
		}

		if expectedObjectives == 0 {
			expectedObjectives = len(solution.ObjectiveValues)
		}

		if !validParetoSolution(solution, expectedObjectives) {
			return nil, fmt.Errorf("invalid Pareto solution at index %d", i)
		}

		points = append(points, ParetoPoint{
			Position:            copyVec(solution.Position),
			Objectives:          copyVec(solution.ObjectiveValues),
			ConstraintViolation: solution.ConstraintViolation,
			Index:               i,
		})
	}

	return points, nil
}

// validateMultiObjectiveRunOptions rejects the run options that have no
// multi-objective reading.
//
// It is the rule validateMultiObjectiveConvergence applies to a target cost,
// applied one layer out: a caller who registers an observer is waiting for
// something, and a run that quietly never calls it is worse than one that
// refuses to start. WithInitialPopulation and WithArchiveObserver carry over
// unchanged and are not checked here.
func validateMultiObjectiveRunOptions(options runOptions) error {
	if options.observer != nil {
		return errors.New(
			"WithProgressObserver has no meaning for a multi-objective run: " +
				"a Pareto front has no single best cost; use WithArchiveObserver")
	}

	if options.populationObserver != nil {
		return errors.New(
			"WithPopulationObserver has no meaning for a multi-objective run: " +
				"the food source and the enemy are per-iteration archive draws " +
				"rather than incumbents, so Best and Worst have nothing to report; " +
				"use WithArchiveObserver")
	}

	if options.logger != nil {
		return errors.New(
			"WithLogger is not supported for a multi-objective run: " +
				"its iteration and completion events report a single best cost " +
				"and a *Result, neither of which a Pareto run has")
	}

	return nil
}
