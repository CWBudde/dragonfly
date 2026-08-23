//go:build js && wasm

package main

import (
	"context"
	"math"
	"syscall/js"

	"github.com/CWBudde/dragonfly"
)

// The three ways a dragonfly can move, as the two-branch step update defines
// them. They are reported per dragonfly per frame because they are the single
// most important thing to see about this algorithm: collapsing the branches
// into one unconditional five-factor step still converges on Sphere, which is
// exactly what makes that bug survive.
const (
	branchFood  = 0 // food in range: the full s·S + a·A + c·C + f·F + e·E step
	branchSwarm = 1 // food out of range, neighbors present: A, C, S only
	branchLevy  = 2 // food out of range, alone: a Levy random walk
)

// jsRun performs one optimization and returns the whole recorded history.
//
// Record-then-replay, rather than a stepped state machine driven from JS, is a
// deliberate choice and it is what keeps this page simple. At the sizes the
// Swarm Lab uses -- two plotted dimensions, tens of dragonflies, at most a
// thousand iterations -- a complete run finishes in a handful of milliseconds,
// so there is nothing to chunk, nothing to cancel, and no partial state to
// reconcile. The page then replays the history on its own clock, which is what
// makes the scrubber possible: you cannot scrub backwards through a live
// computation.
//
// It also makes reproducibility demonstrable. The same seed produces the same
// history, byte for byte, and the page proves it by re-running.
func jsRun(opts js.Value) any {
	benchmarkName := readString(opts, "benchmark", "Rastrigin")

	spec, ok := lookupBenchmark(benchmarkName)
	if !ok {
		return errorResult("run: unknown benchmark %q", benchmarkName)
	}

	var (
		dimensions = clampInt(readInt(opts, "dimensions", 2), 2, maxDimensions)
		iterations = clampInt(readInt(opts, "iterations", 200), 1, maxIterations)
		npop       = clampInt(readInt(opts, "npop", 30), 2, maxPopulation)
		seed       = int64(readFloat(opts, "seed", 42))
		lower      = readFloat(opts, "lower", spec.lower)
		upper      = readFloat(opts, "upper", spec.upper)
		axisX      = clampInt(readInt(opts, "axisX", 0), 0, dimensions-1)
		axisY      = clampInt(readInt(opts, "axisY", 1), 0, dimensions-1)
		boundary   = readString(opts, "boundary", string(dragonfly.BoundaryWrap))
		levy       = readBool(opts, "levy", true)
	)

	if lower >= upper {
		return errorResult("run: lower bound %v must be below upper bound %v", lower, upper)
	}

	config, err := configFor(spec, dimensions, iterations, npop, seed, lower, upper, boundary, levy)
	if err != nil {
		return errorResult("run: %v", err)
	}

	history := newHistory(config, iterations, npop, axisX, axisY)

	result, err := dragonfly.OptimizeContext(
		context.Background(),
		config,
		dragonfly.WithPopulationObserver(history.observe),
	)
	if err != nil {
		return errorResult("run: %v", err)
	}

	history.truncate(result.IterationCount)

	response := map[string]any{
		"benchmark":         spec.name,
		"dimensions":        dimensions,
		"axisX":             axisX,
		"axisY":             axisY,
		"npop":              npop,
		"iterations":        result.IterationCount,
		"evaluations":       result.FuncEvalCount,
		"terminationReason": string(result.TerminationReason),
		"bestCost":          jsNumber(result.GlobalBest.Cost),
		"bestPosition":      floatsToJS(result.GlobalBest.Position),
		"worstCost":         jsNumber(result.Worst.Cost),
		"optimum":           optionalNumber(spec.optimumValue(dimensions)),
		"boundary":          string(config.BoundaryMethod),
		"levy":              config.UseLevyWalk,
		"lower":             lower,
		"upper":             upper,

		// Result.Seed holds a time.Now() value that was never used once a
		// caller supplies its own *rand.Rand, so reporting it would be a lie.
		// This is the seed the run actually used.
		"seed": float64(seed),
	}

	out := opts.Get("out")

	putFloats(response, out, "convergence", toFloat32(result.ConvergenceCurve))
	putFloats(response, out, "swarm", history.swarm)
	putFloats(response, out, "cost", history.cost)
	putFloats(response, out, "foodTrail", history.foodTrail)
	putFloats(response, out, "enemyTrail", history.enemyTrail)
	putFloats(response, out, "radius", history.radius)
	putFloats(response, out, "diversity", history.diversity)
	putFloats(response, out, "branch", history.branch)
	putFloats(response, out, "neighbors", history.neighbors)

	return response
}

// configFor builds a fresh configuration for one run.
//
// Fresh, not cached: OptimizeContext writes back into the config it is given --
// it installs Rand when the field is nil -- so reusing one across runs would
// quietly break the page's central claim that the same seed reproduces the same
// history.
func configFor(
	spec benchmark,
	dimensions, iterations, npop int,
	seed int64,
	lower, upper float64,
	boundary string,
	levy bool,
) (*dragonfly.Config, error) {
	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = spec.fn
	config.ProblemSize = dimensions
	config.MaxIterations = iterations
	config.NPop = npop
	config.LowerBound = lower
	config.UpperBound = upper
	config.BoundaryMethod = dragonfly.BoundaryMethod(boundary)
	config.UseLevyWalk = levy
	config.Rand = rngFor(seed)

	// Goroutines under js/wasm are cooperatively scheduled onto the one browser
	// thread, so a worker pool costs coordination and buys nothing.
	config.EnableParallel = false

	err := dragonfly.ValidateConfig(config)
	if err != nil {
		return nil, err
	}

	return config, nil
}

// history accumulates one run's population snapshots into the flat arrays the
// canvas wants. Everything is appended in iteration order, so the coordinate
// array is iterations x npop x 2 and the page indexes into it arithmetically
// rather than walking a nested structure across the boundary.
type history struct {
	config *dragonfly.Config

	swarm      []float32
	cost       []float32
	foodTrail  []float32
	enemyTrail []float32
	radius     []float32
	diversity  []float32
	branch     []float32
	neighbors  []float32

	iterations int
	npop       int
	axisX      int
	axisY      int
}

func newHistory(config *dragonfly.Config, iterations, npop, axisX, axisY int) *history {
	return &history{
		config:     config,
		swarm:      make([]float32, 0, iterations*npop*2),
		cost:       make([]float32, 0, iterations*npop),
		foodTrail:  make([]float32, 0, iterations*2),
		enemyTrail: make([]float32, 0, iterations*2),
		radius:     make([]float32, 0, iterations),
		diversity:  make([]float32, 0, iterations),
		branch:     make([]float32, 0, iterations*npop),
		neighbors:  make([]float32, 0, iterations*npop),
		iterations: iterations,
		npop:       npop,
		axisX:      axisX,
		axisY:      axisY,
	}
}

func (h *history) observe(snapshot dragonfly.PopulationSnapshot) {
	for i := range snapshot.Swarm {
		h.swarm = append(h.swarm,
			float32(coordinate(snapshot.Swarm[i].Position, h.axisX)),
			float32(coordinate(snapshot.Swarm[i].Position, h.axisY)),
		)
		h.cost = append(h.cost, float32(snapshot.Swarm[i].Cost))
	}

	h.foodTrail = append(h.foodTrail,
		float32(coordinate(snapshot.Best.Position, h.axisX)),
		float32(coordinate(snapshot.Best.Position, h.axisY)),
	)
	h.enemyTrail = append(h.enemyTrail,
		float32(coordinate(snapshot.Worst.Position, h.axisX)),
		float32(coordinate(snapshot.Worst.Position, h.axisY)),
	)

	h.diversity = append(h.diversity, float32(diversity(snapshot.Swarm)))

	h.recordBranches(snapshot)
}

// recordBranches classifies the step each dragonfly is about to take out of
// this frame.
//
// It is a classification of the *outgoing* move, not the one that produced this
// frame, and that is exact rather than approximate. The snapshot holds the
// positions the next iteration's neighbor scan runs over, and its Best and
// Worst are the food source and enemy that iteration will be computed against,
// because the loop re-derives both from this same swarm. The only piece the
// snapshot does not carry is the radius, and NeighborhoodRadius reports the
// number the optimizer will use.
//
// Both the radius schedule and the neighborhood test come from the library
// rather than being reimplemented here. That is the whole point of the two
// being exported: a per-dimension box test is easy to write as a Euclidean ball
// by accident, and a demo that drew the wrong neighborhood would be a
// confident-looking lie about the algorithm it exists to explain.
func (h *history) recordBranches(snapshot dragonfly.PopulationSnapshot) {
	// The snapshot's one-based iteration is the zero-based index of the step
	// about to be taken.
	radius := dragonfly.NeighborhoodRadius(h.config, snapshot.Iteration, h.iterations)
	h.radius = append(h.radius, float32(radius))

	for i := range snapshot.Swarm {
		position := snapshot.Swarm[i].Position

		count := 0

		for j := range snapshot.Swarm {
			if j != i && dragonfly.WithinRadius(position, snapshot.Swarm[j].Position, radius) {
				count++
			}
		}

		h.neighbors = append(h.neighbors, float32(count))
		h.branch = append(h.branch, float32(classifyBranch(position, snapshot.Best.Position, radius, count)))
	}
}

// classifyBranch applies the two-branch rule: the five-factor step runs only
// when every component of the distance to the food source is inside the
// radius, and a dragonfly with no neighbors outside that range walks by Levy
// flight instead of swarming.
func classifyBranch(position, food []float64, radius float64, neighbors int) int {
	if foodInRange(position, food, radius) {
		return branchFood
	}

	if neighbors > 0 {
		return branchSwarm
	}

	return branchLevy
}

// foodInRange is the `any(dist2Food > r)` test, negated: the food branch is
// taken only when no component exceeds the radius. It is written as a negated
// <= so that a NaN component falls out of range rather than into it.
func foodInRange(position, food []float64, radius float64) bool {
	if len(position) != len(food) {
		return false
	}

	for k := range position {
		if !(math.Abs(position[k]-food[k]) <= radius) {
			return false
		}
	}

	return true
}

// truncate discards any tail the run never reached. Early stopping ends the
// loop before MaxIterations, and the page's frame arithmetic assumes every
// array covers exactly the same span.
func (h *history) truncate(iterations int) {
	h.swarm = clip(h.swarm, iterations*h.npop*2)
	h.cost = clip(h.cost, iterations*h.npop)
	h.foodTrail = clip(h.foodTrail, iterations*2)
	h.enemyTrail = clip(h.enemyTrail, iterations*2)
	h.radius = clip(h.radius, iterations)
	h.diversity = clip(h.diversity, iterations)
	h.branch = clip(h.branch, iterations*h.npop)
	h.neighbors = clip(h.neighbors, iterations*h.npop)
}

func clip(values []float32, length int) []float32 {
	if length < 0 || length > len(values) {
		return values
	}

	return values[:length]
}

func coordinate(position []float64, axis int) float64 {
	if axis < 0 || axis >= len(position) {
		return 0
	}

	return position[axis]
}

// diversity is the mean Euclidean distance from each dragonfly to the swarm
// centroid, over all dimensions rather than the two plotted ones. It is the
// demo's clearest single number for the exploration-to-exploitation shift: it
// starts high, and a converging swarm drives it to zero.
func diversity(swarm []dragonfly.Dragonfly) float64 {
	if len(swarm) == 0 || len(swarm[0].Position) == 0 {
		return 0
	}

	dimensions := len(swarm[0].Position)
	centroid := make([]float64, dimensions)

	for i := range swarm {
		for d := 0; d < dimensions && d < len(swarm[i].Position); d++ {
			centroid[d] += swarm[i].Position[d]
		}
	}

	for d := range centroid {
		centroid[d] /= float64(len(swarm))
	}

	total := 0.0

	for i := range swarm {
		sum := 0.0

		for d := 0; d < dimensions && d < len(swarm[i].Position); d++ {
			delta := swarm[i].Position[d] - centroid[d]
			sum += delta * delta
		}

		total += math.Sqrt(sum)
	}

	return total / float64(len(swarm))
}
