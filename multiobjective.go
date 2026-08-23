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
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"time"
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

// Default hypercube-grid parameters.
//
// UNVERIFIED. These are the MOPSO defaults -- Coello Coello, Pulido & Lechuga
// (2004), the lineage MODA borrows its archive from -- and they have NOT been
// read off the author's MODA.m, which is not available to this repository. Do
// not cite them as settled values from the DA paper. Treat them as the working
// defaults they are until someone checks them against the reference source; see
// PLAN.md §1.7 and CLAUDE.md "Common Pitfalls" #5.
const (
	// DefaultArchiveBeta is the exponent of the food-selection weight 1/N^beta.
	DefaultArchiveBeta = 4.0
	// DefaultArchiveGamma is the exponent of the enemy-selection weight N^gamma.
	DefaultArchiveGamma = 2.0
	// DefaultArchiveDelta is the exponent of the overflow-deletion weight N^delta.
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
	// Solutions is the archive contents, in insertion order. Every member is
	// mutually non-dominated with every other; that invariant is restored by
	// each mutation, not merely at the end of a run.
	//
	// A mutation compacts the survivors in place rather than allocating a fresh
	// slice, so a slice held across one is not a snapshot of the archive as it
	// was. Read the field again after any Add or UpdateFromPopulation, or copy
	// it if an older view is needed.
	Solutions []*ParetoSolution

	// lowerBounds and upperBounds are the per-objective extent of the current
	// contents, and therefore the extent of the grid. They move whenever the
	// archive does.
	lowerBounds []float64
	upperBounds []float64

	// cellBuf, memberBuf and orderBuf are occupiedCells' scratch space. They
	// are reused across calls, which is what keeps grouping the archive by
	// hypercube -- something every insert and every selection does -- free of
	// allocation. See occupiedCells for the lifetime this imposes on its
	// result.
	cellBuf   []hypercubeCell
	memberBuf []int
	countBuf  []int

	// Beta, Gamma and Delta are the roulette exponents; see the UNVERIFIED note
	// on DefaultArchiveBeta.
	Beta  float64
	Gamma float64
	Delta float64

	// MaxSize is the capacity. A successful insert past capacity evicts a member
	// of the most crowded hypercube, so the archive never exceeds it.
	MaxSize int
	// NGrid is the number of hypercubes per objective.
	NGrid int
}

// NewParetoArchive creates an archive of the given capacity with the default
// grid parameters. A non-positive capacity falls back to DefaultArchiveSize.
func NewParetoArchive(maxSize int) *ParetoArchive {
	return NewParetoArchiveWithGrid(maxSize, DefaultArchiveNGrid,
		DefaultArchiveBeta, DefaultArchiveGamma, DefaultArchiveDelta)
}

// NewParetoArchiveWithGrid creates an archive with explicit grid parameters.
//
// A non-positive maxSize or nGrid falls back to the corresponding default, and
// a negative exponent is raised to zero -- a negative exponent inverts the
// preference the roulette draw exists to express, which is never what a caller
// means.
func NewParetoArchiveWithGrid(maxSize, nGrid int, beta, gamma, delta float64) *ParetoArchive {
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
	if pa == nil || solution == nil || len(solution.ObjectiveValues) == 0 {
		return false
	}

	for _, existing := range pa.Solutions {
		if duplicateSolutions(existing, solution) || constrainedDominates(existing, solution) {
			return false
		}
	}

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

	for i, first := range pa.Solutions {
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

		bin := int(math.Floor((value - pa.lowerBounds[m]) / span * float64(pa.NGrid)))
		index[m] = min(max(bin, 0), pa.NGrid-1)
	}

	return index
}

// gridKeyFor flattens a hypercube coordinate into a single integer in base
// NGrid, so occupancy can be counted with one map lookup per solution.
func (pa *ParetoArchive) gridKeyFor(index []int) int {
	key := 0
	stride := 1

	for _, component := range index {
		key += component * stride
		stride *= pa.NGrid
	}

	return key
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
		for end < count && pa.Solutions[members[end]].GridKey == key {
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

		return left < right
	})
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
	return pa.selectFromCells(func(count float64) float64 {
		return 1 / math.Pow(count, pa.Beta)
	}, rng)
}

// selectCrowded draws a solution from a crowded hypercube, weighting each
// occupied cell by N^Gamma. It is MODA's enemy: the swarm is pushed away from
// the parts of the front that are already well covered.
//
// Returns nil for an empty archive.
func (pa *ParetoArchive) selectCrowded(rng *rand.Rand) *ParetoSolution {
	return pa.selectFromCells(func(count float64) float64 {
		return math.Pow(count, pa.Gamma)
	}, rng)
}

// selectFromCells performs the two-stage draw both selections share: a roulette
// over the occupied cells under the given weight function, then a uniform draw
// among the members of the chosen cell.
func (pa *ParetoArchive) selectFromCells(weight func(count float64) float64, rng *rand.Rand) *ParetoSolution {
	if pa == nil || len(pa.Solutions) == 0 {
		return nil
	}

	cells := pa.occupiedCells()
	weights := make([]float64, len(cells))

	for i, cell := range cells {
		weights[i] = weight(float64(len(cell.Members)))
	}

	chosen := cells[rouletteIndex(weights, rng)]
	member := chosen.Members[0]

	if rng != nil && len(chosen.Members) > 1 {
		member = chosen.Members[rng.Intn(len(chosen.Members))]
	}

	return pa.Solutions[member]
}

// evictOverflow trims the archive back to MaxSize, deleting from the most
// crowded hypercube -- a roulette over the occupied cells weighted N^Delta,
// then a member of the chosen cell.
//
// It loops rather than deleting once, so an archive that somehow overshot by
// more than one still lands on capacity.
func (pa *ParetoArchive) evictOverflow(rng *rand.Rand) {
	for len(pa.Solutions) > pa.MaxSize {
		cells := pa.occupiedCells()
		weights := make([]float64, len(cells))

		for i, cell := range cells {
			weights[i] = math.Pow(float64(len(cell.Members)), pa.Delta)
		}

		chosen := cells[rouletteIndex(weights, rng)]
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

	total := 0.0

	for _, weight := range weights {
		if isFinite(weight) && weight > 0 {
			total += weight
		}
	}

	if rng == nil {
		return 0
	}

	if !(total > 0) || !isFinite(total) {
		return rng.Intn(len(weights))
	}

	draw := rng.Float64() * total
	cumulative := 0.0

	for i, weight := range weights {
		if isFinite(weight) && weight > 0 {
			cumulative += weight
		}

		if draw < cumulative {
			return i
		}
	}

	return len(weights) - 1
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

	// Beta, Gamma and Delta are the hypercube roulette exponents. See the
	// UNVERIFIED note on DefaultArchiveBeta before treating the defaults as
	// paper values.
	Beta  float64 `json:"beta"`
	Gamma float64 `json:"gamma"`
	Delta float64 `json:"delta"`

	ArchiveSize int `json:"archive_size"`
	NGrid       int `json:"n_grid"`
}

// NewMultiObjectiveConfig creates a default MODA configuration.
// You must set ObjectiveFunc and Swarm's ProblemSize, LowerBound and UpperBound.
//
// The archive parameters are the defaults documented -- and flagged as
// unverified -- on DefaultArchiveBeta.
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
	Seed           int64
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
	if convergence == nil || convergence.TargetCost == nil {
		return nil
	}

	return errors.New(
		"convergence.target_cost has no meaning for a multi-objective run: " +
			"a Pareto front has no single best cost; use stagnation_iterations instead")
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
func OptimizeMultiObjective(ctx context.Context, config *MultiObjectiveConfig) (*MultiObjectiveResult, error) {
	contextErr := requireContext(ctx)
	if contextErr != nil {
		return nil, contextErr
	}

	validationErr := validateMultiObjectiveConfig(config)
	if validationErr != nil {
		return nil, validationErr
	}

	swarmConfig := config.Swarm

	// The seed is drawn whether or not it is used, so Seed is always populated;
	// this is the convention OptimizeContext follows.
	seed := time.Now().UnixNano()
	if swarmConfig.Rand == nil {
		swarmConfig.Rand = rand.New(rand.NewSource(seed))
	}

	rng := swarmConfig.Rand

	state, initErr := initializeMultiObjectiveRun(ctx, config, rng)
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

		accepted, stepErr := state.advance(ctx, config, t, rng)
		if stepErr != nil {
			return nil, stepErr
		}

		completed = t + 1

		curve = append(curve, state.archive.Len())

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
	rng *rand.Rand,
) (*moState, error) {
	swarmConfig := config.Swarm
	swarm := make([]Dragonfly, swarmConfig.NPop)

	for i := range swarm {
		swarm[i] = Dragonfly{
			Position: unifrndVec(swarmConfig.LowerBound, swarmConfig.UpperBound, swarmConfig.ProblemSize, rng),
			Step:     unifrndVec(swarmConfig.LowerBound, swarmConfig.UpperBound, swarmConfig.ProblemSize, rng),
			Cost:     math.Inf(1),
		}
	}

	state := &moState{
		archive: NewParetoArchiveWithGrid(config.ArchiveSize, config.NGrid,
			config.Beta, config.Gamma, config.Delta),
		swarm:  swarm,
		scores: make([]moEvaluation, len(swarm)),
	}

	if swarmConfig.EnableParallel {
		state.maxWorkers = effectiveMaxWorkers(swarmConfig)
	}

	_, err := state.evaluateSwarm(ctx, config, rng)
	if err != nil {
		return nil, err
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

	candidates := make([]*ParetoSolution, 0, len(state.swarm))

	for i := range state.swarm {
		fly := &state.swarm[i]
		values := state.scores[i].values

		if len(values) == 0 {
			return 0, errors.New("ObjectiveFunc must return at least one objective value")
		}

		if state.archive.Len() > 0 && len(values) != len(state.archive.Solutions[0].ObjectiveValues) {
			return 0, fmt.Errorf("ObjectiveFunc must return a fixed number of objectives, got %d then %d",
				len(state.archive.Solutions[0].ObjectiveValues), len(values))
		}

		sanitized := make([]float64, len(values))
		for m, value := range values {
			sanitized[m] = sanitizeCost(value)
		}

		// Cost carries the first objective purely so an inspecting caller sees
		// something meaningful; nothing in a MODA run reads it.
		fly.Cost = sanitized[0]
		fly.ConstraintViolation = state.scores[i].violation

		candidates = append(candidates, &ParetoSolution{
			Position:            copyVec(fly.Position),
			ObjectiveValues:     sanitized,
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
			values:    config.ObjectiveFunc(state.swarm[i].Position),
			violation: EvaluateConstraints(state.swarm[i].Position, config.Swarm.Constraints).Violation,
		}
	}

	if state.maxWorkers > 0 {
		return parallelFor(ctx, len(state.swarm), state.maxWorkers, score)
	}

	for i := range state.swarm {
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
	weights := computeWeights(swarmConfig, iteration, swarmConfig.MaxIterations, rng)

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
		swarm: state.swarm,
		food:  Best{Position: food},
		enemy: Best{Position: enemy},
	}

	for i := range state.swarm {
		prepareSwarmStep(view, i, swarmConfig, weights, rng)
	}

	return state.evaluateSwarm(ctx, config, rng)
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
	ArchiveSize       int               `json:"archive_size"`
	FuncEvalCount     int               `json:"func_eval_count"`
	IterationCount    int               `json:"iteration_count"`
}

var errNilMultiObjectiveResult = errors.New("multi-objective result cannot be nil")

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

	points := make([]ParetoPoint, 0, result.Archive.Len())

	for i, solution := range result.Archive.Solutions {
		points = append(points, ParetoPoint{
			Position:            copyVec(solution.Position),
			Objectives:          copyVec(solution.ObjectiveValues),
			ConstraintViolation: solution.ConstraintViolation,
			Index:               i,
		})
	}

	return points, nil
}
