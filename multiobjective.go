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
	Rank             int
	DominationCount  int
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
		Position:        copyVec(solution.Position),
		ObjectiveValues: copyVec(solution.ObjectiveValues),
		GridIndex:       append([]int(nil), solution.GridIndex...),
		GridKey:         solution.GridKey,
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
	Solutions []*ParetoSolution

	// lowerBounds and upperBounds are the per-objective extent of the current
	// contents, and therefore the extent of the grid. They move whenever the
	// archive does.
	lowerBounds []float64
	upperBounds []float64

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
		if equalObjectives(existing.ObjectiveValues, solution.ObjectiveValues) ||
			dominates(existing.ObjectiveValues, solution.ObjectiveValues) {
			return false
		}
	}

	kept := make([]*ParetoSolution, 0, len(pa.Solutions)+1)

	for _, existing := range pa.Solutions {
		if !dominates(solution.ObjectiveValues, existing.ObjectiveValues) {
			kept = append(kept, existing)
		}
	}

	kept = append(kept, solution.clone())
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
			if i != j && dominates(first.ObjectiveValues, second.ObjectiveValues) {
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

	objectives := len(pa.Solutions[0].ObjectiveValues)
	lower := make([]float64, objectives)
	upper := make([]float64, objectives)

	for m := range objectives {
		lower[m] = math.Inf(1)
		upper[m] = math.Inf(-1)
	}

	for _, solution := range pa.Solutions {
		for m := 0; m < objectives && m < len(solution.ObjectiveValues); m++ {
			lower[m] = math.Min(lower[m], solution.ObjectiveValues[m])
			upper[m] = math.Max(upper[m], solution.ObjectiveValues[m])
		}
	}

	pa.lowerBounds = lower
	pa.upperBounds = upper

	for _, solution := range pa.Solutions {
		solution.GridIndex = pa.gridIndexFor(solution.ObjectiveValues)
		solution.GridKey = pa.gridKeyFor(solution.GridIndex)
	}
}

// gridIndexFor returns the hypercube coordinate of an objective vector under
// the current bounds.
//
// Bin m has width (upper[m]-lower[m])/NGrid, and the index is the floor of the
// normalized coordinate, clamped to [0, NGrid-1] so that a value sitting exactly
// on the upper bound lands in the last bin rather than one past it. A degenerate
// objective, where every archived solution scores the same, has every solution
// in bin zero.
func (pa *ParetoArchive) gridIndexFor(values []float64) []int {
	index := make([]int, len(values))

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

// occupiedCells groups the archive by hypercube, in ascending key order.
//
// The ordering is deliberate and load-bearing: the cells are collected through a
// map, and Go randomizes map iteration, so returning them unsorted would make
// every roulette draw depend on the map's internal ordering and a seeded run
// would not reproduce.
func (pa *ParetoArchive) occupiedCells() []hypercubeCell {
	byKey := make(map[int][]int, len(pa.Solutions))

	for i, solution := range pa.Solutions {
		byKey[solution.GridKey] = append(byKey[solution.GridKey], i)
	}

	cells := make([]hypercubeCell, 0, len(byKey))
	for key, members := range byKey {
		cells = append(cells, hypercubeCell{Members: members, Key: key})
	}

	sort.Slice(cells, func(i, j int) bool { return cells[i].Key < cells[j].Key })

	return cells
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
	// Only the iteration cap ends a MODA run today, so this is always
	// TerminationMaxIterations; it is reported anyway so that a caller can read
	// a MODA result the same way it reads a single-objective one.
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

// moState is the mutable state of one MODA run.
type moState struct {
	archive   *ParetoArchive
	swarm     []Dragonfly
	funcEvals int
}

// OptimizeMultiObjective runs the multi-objective Dragonfly Algorithm, honoring
// context cancellation.
//
// It is a separate entry point rather than a mode of OptimizeContext: a
// multi-objective run has no single best cost, so Result would have to report an
// incumbent that does not exist. Cancellation is checked at the top of every
// iteration and a canceled run returns a nil result and ctx.Err(), so a caller
// cannot mistake an aborted run for a completed one.
//
// The swarm mechanics are identical to the single-objective algorithm. Only the
// food source and the enemy differ: both are drawn from the archive once per
// iteration, the food from a sparse hypercube and the enemy from a crowded one.
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

	state, initErr := initializeMultiObjectiveRun(config, rng)
	if initErr != nil {
		return nil, initErr
	}

	curve := make([]int, 0, swarmConfig.MaxIterations)
	completed := 0

	for t := range swarmConfig.MaxIterations {
		ctxErr := ctx.Err()
		if ctxErr != nil {
			return nil, ctxErr
		}

		stepErr := state.advance(config, t, rng)
		if stepErr != nil {
			return nil, stepErr
		}

		completed = t + 1

		curve = append(curve, state.archive.Len())
	}

	return &MultiObjectiveResult{
		Archive:          state.archive,
		ArchiveSizeCurve: curve,
		// Only the iteration cap ends a MODA run: the early-stopping criteria in
		// convergence.go are defined against a single best cost, which a
		// multi-objective run does not have.
		TerminationReason: TerminationMaxIterations,
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
func initializeMultiObjectiveRun(config *MultiObjectiveConfig, rng *rand.Rand) (*moState, error) {
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
		swarm: swarm,
	}

	err := state.evaluateSwarm(config, rng)
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
// All calls happen on the calling goroutine.
func (state *moState) evaluateSwarm(config *MultiObjectiveConfig, rng *rand.Rand) error {
	candidates := make([]*ParetoSolution, 0, len(state.swarm))

	for i := range state.swarm {
		fly := &state.swarm[i]

		values := config.ObjectiveFunc(fly.Position)
		state.funcEvals++

		if len(values) == 0 {
			return errors.New("ObjectiveFunc must return at least one objective value")
		}

		if state.archive.Len() > 0 && len(values) != len(state.archive.Solutions[0].ObjectiveValues) {
			return fmt.Errorf("ObjectiveFunc must return a fixed number of objectives, got %d then %d",
				len(state.archive.Solutions[0].ObjectiveValues), len(values))
		}

		sanitized := make([]float64, len(values))
		for m, value := range values {
			sanitized[m] = sanitizeCost(value)
		}

		// Cost carries the first objective purely so an inspecting caller sees
		// something meaningful; nothing in a MODA run reads it.
		fly.Cost = sanitized[0]

		candidates = append(candidates, &ParetoSolution{
			Position:        copyVec(fly.Position),
			ObjectiveValues: sanitized,
		})
	}

	state.archive.UpdateFromPopulation(candidates, rng)

	return nil
}

// advance runs one MODA iteration: draw the food source and the enemy from the
// archive, move every dragonfly, then rescore and update the archive.
func (state *moState) advance(config *MultiObjectiveConfig, iteration int, rng *rand.Rand) error {
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

	return state.evaluateSwarm(config, rng)
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
	Index      int       `json:"index"`
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

// paretoCSVHeader builds the column names from the first exported point.
func paretoCSVHeader(points []ParetoPoint) []string {
	header := []string{"index"}

	if len(points) == 0 {
		return header
	}

	for m := range points[0].Objectives {
		header = append(header, "objective_"+strconv.Itoa(m))
	}

	for j := range points[0].Position {
		header = append(header, "x_"+strconv.Itoa(j))
	}

	return header
}

// paretoCSVRow formats one exported point as CSV fields.
func paretoCSVRow(point ParetoPoint) []string {
	row := make([]string, 0, 1+len(point.Objectives)+len(point.Position))
	row = append(row, strconv.Itoa(point.Index))

	for _, value := range point.Objectives {
		row = append(row, strconv.FormatFloat(value, 'g', -1, 64))
	}

	for _, value := range point.Position {
		row = append(row, strconv.FormatFloat(value, 'g', -1, 64))
	}

	return row
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
			Position:   copyVec(solution.Position),
			Objectives: copyVec(solution.ObjectiveValues),
			Index:      i,
		})
	}

	return points, nil
}
