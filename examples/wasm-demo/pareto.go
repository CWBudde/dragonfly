//go:build js && wasm

package main

import (
	"context"
	"math"
	"syscall/js"

	"github.com/CWBudde/dragonfly"
)

// jsPareto performs one multi-objective run and returns the archive as it stood
// after every iteration.
//
// This is the export the library grew WithArchiveObserver for. Before it, a
// MODA run could be watched only after the fact -- the archive at the end and
// its size per iteration -- so the one thing worth animating, a front filling
// in and spreading out, was not observable at all.
//
// Same record-then-replay bargain as jsRun, for the same reason: a run of these
// sizes finishes in milliseconds and you cannot scrub backwards through a live
// computation.
func jsPareto(opts js.Value) any {
	benchmarkName := readString(opts, "benchmark", "ZDT1")

	spec, ok := lookupMultiBenchmark(benchmarkName)
	if !ok {
		return errorResult("pareto: unknown benchmark %q", benchmarkName)
	}

	var (
		dimensions  = clampInt(readInt(opts, "dimensions", 10), 1, maxDimensions)
		iterations  = clampInt(readInt(opts, "iterations", 100), 1, maxIterations)
		npop        = clampInt(readInt(opts, "npop", 50), 2, maxPopulation)
		archiveSize = clampInt(readInt(opts, "archiveSize", 100), 2, maxArchiveSize)
		nGrid       = clampInt(readInt(opts, "nGrid", 10), 2, maxNGrid)
		seed        = int64(readFloat(opts, "seed", 42))
		lower       = readFloat(opts, "lower", spec.lower)
		upper       = readFloat(opts, "upper", spec.upper)
	)

	if lower >= upper {
		return errorResult("pareto: lower bound %v must be below upper bound %v", lower, upper)
	}

	config := multiConfigFor(spec, dimensions, iterations, npop, archiveSize, nGrid, seed, lower, upper)

	film := newParetoFilm(iterations, archiveSize)

	result, err := dragonfly.OptimizeMultiObjective(
		context.Background(),
		config,
		dragonfly.WithArchiveObserver(film.observe),
	)
	if err != nil {
		return errorResult("pareto: %v", err)
	}

	response := map[string]any{
		"benchmark":         spec.name,
		"objectives":        stringsToJS(spec.objectives),
		"dimensions":        dimensions,
		"npop":              npop,
		"nGrid":             nGrid,
		"archiveSize":       archiveSize,
		"iterations":        result.IterationCount,
		"evaluations":       result.FuncEvalCount,
		"terminationReason": string(result.TerminationReason),
		"finalArchive":      result.Archive.Len(),
		"lower":             lower,
		"upper":             upper,
		"seed":              float64(seed),

		// The extent of the whole replay, so the page can hold one frame
		// steady instead of rescaling every animation step -- a front that is
		// only being rescaled looks like a front that is moving.
		"frame": []any{
			jsNumber(film.minX), jsNumber(film.minY),
			jsNumber(film.maxX), jsNumber(film.maxY),
		},
	}

	out := opts.Get("out")

	putFloats(response, out, "points", film.points)
	putFloats(response, out, "counts", film.counts)
	putFloats(response, out, "grid", film.grid)
	putFloats(response, out, "cells", film.cells)
	putFloats(response, out, "archiveCurve", toFloat32(intsToFloats(result.ArchiveSizeCurve)))

	return response
}

func multiConfigFor(
	spec multiBenchmark,
	dimensions, iterations, npop, archiveSize, nGrid int,
	seed int64,
	lower, upper float64,
) *dragonfly.MultiObjectiveConfig {
	config := dragonfly.NewMultiObjectiveConfig()
	config.ObjectiveFunc = spec.fn
	config.ArchiveSize = archiveSize
	config.NGrid = nGrid
	config.Swarm.ProblemSize = dimensions
	config.Swarm.MaxIterations = iterations
	config.Swarm.NPop = npop
	config.Swarm.LowerBound = lower
	config.Swarm.UpperBound = upper
	config.Swarm.EnableParallel = false
	config.Swarm.Rand = rngFor(seed)

	return config
}

// paretoFilm records the archive frame by frame.
//
// The archive is variable-length, so unlike the swarm it cannot be indexed
// arithmetically. counts[t] is the number of points in frame t and points is
// their concatenation; the page walks a running offset. cells is the same idea
// for the occupied hypercubes, four numbers each: grid index per objective,
// then the member count.
type paretoFilm struct {
	points []float32
	counts []float32
	grid   []float32
	cells  []float32

	minX float64
	minY float64
	maxX float64
	maxY float64
}

func newParetoFilm(iterations, archiveSize int) *paretoFilm {
	return &paretoFilm{
		points: make([]float32, 0, iterations*archiveSize*2),
		counts: make([]float32, 0, iterations),
		grid:   make([]float32, 0, iterations*4),
		cells:  make([]float32, 0, iterations*4),
		minX:   math.Inf(1),
		minY:   math.Inf(1),
		maxX:   math.Inf(-1),
		maxY:   math.Inf(-1),
	}
}

func (f *paretoFilm) observe(snapshot dragonfly.ArchiveSnapshot) {
	f.counts = append(f.counts, float32(len(snapshot.Solutions)))

	for _, solution := range snapshot.Solutions {
		x := objectiveAt(solution.ObjectiveValues, 0)
		y := objectiveAt(solution.ObjectiveValues, 1)

		f.points = append(f.points, float32(x), float32(y))

		if !math.IsNaN(x) && !math.IsNaN(y) {
			f.minX = math.Min(f.minX, x)
			f.maxX = math.Max(f.maxX, x)
			f.minY = math.Min(f.minY, y)
			f.maxY = math.Max(f.maxY, y)
		}
	}

	f.grid = append(f.grid,
		float32(boundAt(snapshot.GridLower, 0)),
		float32(boundAt(snapshot.GridLower, 1)),
		float32(boundAt(snapshot.GridUpper, 0)),
		float32(boundAt(snapshot.GridUpper, 1)),
	)

	f.appendCells(snapshot)
}

// appendCells groups the frame's solutions by hypercube, which is the quantity
// MODA's food and enemy draws turn on: the food source is drawn from the least
// populated occupied cell and the enemy from the most populated one. The page
// shades the two so the mechanism is visible rather than merely described.
//
// The grouping is over GridIndex, which the archive assigns; nothing here
// re-derives which cell a point belongs to.
func (f *paretoFilm) appendCells(snapshot dragonfly.ArchiveSnapshot) {
	counts := make(map[[2]int]int, len(snapshot.Solutions))

	for _, solution := range snapshot.Solutions {
		if len(solution.GridIndex) < 2 {
			continue
		}

		counts[[2]int{solution.GridIndex[0], solution.GridIndex[1]}]++
	}

	// A frame's cell block is terminated by its own length, appended first, so
	// the page can walk the concatenation the same way it walks the points.
	f.cells = append(f.cells, float32(len(counts)))

	for index, count := range counts {
		f.cells = append(f.cells, float32(index[0]), float32(index[1]), float32(count))
	}
}

func objectiveAt(values []float64, index int) float64 {
	if index < 0 || index >= len(values) {
		return math.NaN()
	}

	return values[index]
}

func boundAt(values []float64, index int) float64 {
	if index < 0 || index >= len(values) {
		return 0
	}

	return values[index]
}

func intsToFloats(values []int) []float64 {
	widened := make([]float64, len(values))
	for i, value := range values {
		widened[i] = float64(value)
	}

	return widened
}
