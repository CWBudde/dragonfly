//go:build js && wasm

package main

import (
	"math"
	"sort"

	"github.com/CWBudde/dragonfly"
)

// benchmark pairs one of the library's objective functions with the metadata a
// UI needs: where to search, where the answer is, and what makes the function
// worth showing.
//
// It carries only what the library does not: the bounds and the minimizer,
// which live in functions.go doc comments and nowhere programmatic. Modality
// and landscape are deliberately absent — dragonfly.BenchmarkCharacteristics
// already classifies every one of these names, and restating that here would
// be a second opinion free to drift from the one the selector acts on.
type benchmark struct {
	fn   dragonfly.ObjectiveFunction
	name string

	// optimumAt returns the minimizing position in the requested dimension,
	// and whether one is known. It is not a scalar because two of these
	// functions do not have a uniform minimizer: Dixon-Price's coordinates
	// depend on their index, and Michalewicz has no closed form at all. Pinning
	// every hidden coordinate to a single value produced heatmap slices that
	// missed the global minimum while the UI claimed they passed through it.
	optimumAt func(dimensions int) ([]float64, bool)

	// optimumValue returns the objective value at that minimum, and whether it
	// is known for this dimension. Michalewicz's is tabulated for a few
	// dimensions and unknown for the rest.
	optimumValue func(dimensions int) (float64, bool)

	blurb string
	lower float64
	upper float64
}

// uniformOptimum describes a function minimized where every coordinate takes
// the same value, which is all of them except Dixon-Price and Michalewicz.
func uniformOptimum(coordinate float64) func(int) ([]float64, bool) {
	return func(dimensions int) ([]float64, bool) {
		position := make([]float64, dimensions)
		for i := range position {
			position[i] = coordinate
		}

		return position, true
	}
}

// zeroMinimum describes a function whose minimum is zero in every dimension,
// which is all of them here except Michalewicz. It is not parameterized on the
// value: every caller passes zero, and a parameter nothing varies reads as a
// generality this table does not have.
func zeroMinimum() func(int) (float64, bool) {
	return func(int) (float64, bool) { return 0, true }
}

// dixonPriceOptimum is minimized at x_i = 2^(-(2^i - 2) / 2^i) for one-based i,
// so only the first coordinate is 1. Evaluating the all-ones vector instead
// gives 2, 14 and 54 in 2, 5 and 10 dimensions rather than 0.
func dixonPriceOptimum(dimensions int) ([]float64, bool) {
	position := make([]float64, dimensions)

	for i := range position {
		power := math.Pow(2, float64(i+1))
		position[i] = math.Pow(2, -(power-2)/power)
	}

	return position, true
}

// himmelblauOptimum alternates 3 and 2 across coordinate pairs, the pattern the
// n-dimensional extension is minimized on, with a trailing 0 for the unpaired
// coordinate an odd dimension leaves over. A uniform vector is not a minimizer:
// the all-threes vector scores 26 per pair rather than 0.
func himmelblauOptimum(dimensions int) ([]float64, bool) {
	position := make([]float64, dimensions)

	for i := 0; i+1 < dimensions; i += 2 {
		position[i] = 3
		position[i+1] = 2
	}

	return position, true
}

// michalewiczOptima are the published minima for the steepness m = 10 that
// functions.go implements. There is no closed form, and no tabulated value for
// the dimensions in between, so those are reported as unknown rather than
// guessed at.
var michalewiczOptima = map[int]float64{2: -1.8013, 5: -4.687658, 10: -9.66015}

func michalewiczValue(dimensions int) (float64, bool) {
	value, ok := michalewiczOptima[dimensions]

	return value, ok
}

// michalewiczOptimumAt knows the minimizer only in two dimensions, where it is
// approximately (2.20, 1.57). Above that the demo has no position to pin a
// projection to and says so instead of inventing one.
func michalewiczOptimumAt(dimensions int) ([]float64, bool) {
	if dimensions != 2 {
		return nil, false
	}

	return []float64{2.20, 1.57}, true
}

var benchmarks = map[string]benchmark{
	"Sphere": {
		fn: dragonfly.Sphere, name: "Sphere", lower: -10, upper: 10,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "The smooth bowl. One minimum, no structure to get lost in — a sanity check, not a challenge.",
	},
	"Rastrigin": {
		fn: dragonfly.Rastrigin, name: "Rastrigin", lower: -5.12, upper: 5.12,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "A cosine egg carton over a bowl. Local minima everywhere; the classic test of whether a swarm escapes them.",
	},
	"Rosenbrock": {
		fn: dragonfly.Rosenbrock, name: "Rosenbrock", lower: -5, upper: 10,
		optimumAt: uniformOptimum(1), optimumValue: zeroMinimum(),
		blurb: "The banana valley. Finding the valley is easy; following its curved floor to (1,1) is not.",
	},
	"Ackley": {
		fn: dragonfly.Ackley, name: "Ackley", lower: -32.768, upper: 32.768,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "A near-flat plain with a narrow central funnel. Punishes swarms that converge before they explore.",
	},
	"Griewank": {
		fn: dragonfly.Griewank, name: "Griewank", lower: -600, upper: 600,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "Product-of-cosines ripple on a wide bowl. Gets easier, not harder, as dimensions rise.",
	},
	"Schwefel": {
		fn: dragonfly.Schwefel, name: "Schwefel", lower: -500, upper: 500,
		optimumAt: uniformOptimum(420.9687), optimumValue: zeroMinimum(),
		blurb: "Deceptive: the global minimum sits far from the second best, so the gradient actively misleads.",
	},
	"Levy": {
		fn: dragonfly.Levy, name: "Levy", lower: -10, upper: 10,
		optimumAt: uniformOptimum(1), optimumValue: zeroMinimum(),
		blurb: "Sinusoidal ridges with a single global basin at (1,1).",
	},
	"Zakharov": {
		fn: dragonfly.Zakharov, name: "Zakharov", lower: -5, upper: 10,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "No local minima, but strongly coupled dimensions — a test of coordinated movement.",
	},
	"Michalewicz": {
		fn: dragonfly.Michalewicz, name: "Michalewicz", lower: 0, upper: 3.141592653589793,
		optimumAt: michalewiczOptimumAt, optimumValue: michalewiczValue,
		blurb: "Steep valleys separated by flat plateaus. The steepness parameter makes the basins nearly invisible.",
	},
	"DixonPrice": {
		fn: dragonfly.DixonPrice, name: "DixonPrice", lower: -10, upper: 10,
		optimumAt: dixonPriceOptimum, optimumValue: zeroMinimum(),
		blurb: "A curved valley whose optimum shifts with the dimension index.",
	},
	"BentCigar": {
		fn: dragonfly.BentCigar, name: "BentCigar", lower: -100, upper: 100,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "One direction is a million times cheaper than the rest. Tests handling of ill-conditioning.",
	},
	"Discus": {
		fn: dragonfly.Discus, name: "Discus", lower: -100, upper: 100,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "BentCigar inverted: one direction dominates the cost entirely.",
	},
	"Weierstrass": {
		fn: dragonfly.Weierstrass, name: "Weierstrass", lower: -0.5, upper: 0.5,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "Continuous everywhere, differentiable nowhere. Fractal roughness at every scale.",
	},
	"HappyCat": {
		fn: dragonfly.HappyCat, name: "HappyCat", lower: -2, upper: 2,
		optimumAt: uniformOptimum(-1), optimumValue: zeroMinimum(),
		blurb: "A thin curved shell of near-optimal points around a sphere of radius sqrt(n).",
	},
	"ExpandedSchafferF6": {
		fn: dragonfly.ExpandedSchafferF6, name: "ExpandedSchafferF6", lower: -100, upper: 100,
		optimumAt: uniformOptimum(0), optimumValue: zeroMinimum(),
		blurb: "Concentric ripples around the origin — every ring is a local minimum.",
	},
	"Himmelblau": {
		fn: dragonfly.Himmelblau, name: "Himmelblau", lower: -5, upper: 5,
		optimumAt: himmelblauOptimum, optimumValue: zeroMinimum(),
		blurb: "Four equally good minima per coordinate pair. Tests whether a swarm commits to one or splits between them.",
	},
}

// benchmarkNames returns the table's keys in a stable, didactic order: the
// five classics first, in rising difficulty, then the CEC-style additions.
// Map iteration order would reshuffle the UI's dropdown on every page load.
func benchmarkNames() []string {
	ordered := []string{
		"Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank",
		"Schwefel", "Levy", "Zakharov", "Michalewicz", "DixonPrice",
		"BentCigar", "Discus", "Weierstrass", "HappyCat", "ExpandedSchafferF6",
		"Himmelblau",
	}

	seen := make(map[string]bool, len(ordered))
	names := make([]string, 0, len(benchmarks))

	for _, name := range ordered {
		if _, ok := benchmarks[name]; ok {
			names = append(names, name)
			seen[name] = true
		}
	}

	// Anything added to the table but forgotten in the list above still shows
	// up, sorted, rather than silently vanishing from the UI.
	rest := make([]string, 0)

	for name := range benchmarks {
		if !seen[name] {
			rest = append(rest, name)
		}
	}

	sort.Strings(rest)

	return append(names, rest...)
}

func lookupBenchmark(name string) (benchmark, bool) {
	found, ok := benchmarks[name]

	return found, ok
}

// successTarget derives the cost that counts as solving this function in this
// dimension, and whether such a target can be expressed at all.
//
// The library treats a non-positive target as "no target set", so a function
// whose optimum is negative has no representable success threshold. Reporting
// that honestly is the only correct option: the alternative, a fixed 1e-8,
// silently scored every Michalewicz run as a success because every negative
// cost is below it.
func successTarget(spec benchmark, dimensions int) (float64, bool) {
	optimum, known := spec.optimumValue(dimensions)
	if !known {
		return 0, false
	}

	const tolerance = 1e-8

	target := optimum + tolerance
	if target <= 0 {
		return 0, false
	}

	return target, true
}

// multiBenchmark pairs one of the library's multi-objective functions with the
// metadata the Pareto page needs. There is no optimum to report: the answer to
// a multi-objective problem is a front, not a point.
type multiBenchmark struct {
	fn dragonfly.MultiObjectiveFunction

	name       string
	blurb      string
	objectives []string

	lower float64
	upper float64
}

var multiBenchmarks = map[string]multiBenchmark{
	"ZDT1": {
		fn: dragonfly.ZDT1, name: "ZDT1", lower: 0, upper: 1,
		objectives: []string{"f1", "f2"},
		blurb:      "The reference two-objective problem. Its true front is convex and continuous, so a healthy archive traces a smooth arc.",
	},
	"ZDT2": {
		fn: dragonfly.ZDT2, name: "ZDT2", lower: 0, upper: 1,
		objectives: []string{"f1", "f2"},
		blurb:      "ZDT1 with a concave front. Methods that implicitly assume convexity collapse it toward the extremes.",
	},
	"ZDT3": {
		fn: dragonfly.ZDT3, name: "ZDT3", lower: 0, upper: 1,
		objectives: []string{"f1", "f2"},
		blurb:      "A front broken into five disconnected pieces. It is the one that shows whether the hypercube grid is preserving spread.",
	},
	"SchafferN1": {
		fn: dragonfly.SchafferN1, name: "SchafferN1", lower: -10, upper: 10,
		objectives: []string{"f1", "f2"},
		blurb:      "One decision variable, two objectives pulling opposite ways. Small enough to check the archive against by hand.",
	},
}

// multiBenchmarkNames returns the multi-objective table's keys in the order the
// ZDT family is normally presented, for the same reason benchmarkNames exists:
// map iteration order would reshuffle the dropdown on every page load.
func multiBenchmarkNames() []string {
	ordered := []string{"ZDT1", "ZDT2", "ZDT3", "SchafferN1"}

	seen := make(map[string]bool, len(ordered))
	names := make([]string, 0, len(multiBenchmarks))

	for _, name := range ordered {
		if _, ok := multiBenchmarks[name]; ok {
			names = append(names, name)
			seen[name] = true
		}
	}

	rest := make([]string, 0)

	for name := range multiBenchmarks {
		if !seen[name] {
			rest = append(rest, name)
		}
	}

	sort.Strings(rest)

	return append(names, rest...)
}

func lookupMultiBenchmark(name string) (multiBenchmark, bool) {
	found, ok := multiBenchmarks[name]

	return found, ok
}

// classify reports the library's own reading of a benchmark's shape. An
// unknown name cannot happen for a table entry, but reporting empty strings
// beats inventing a classification if one is ever added here and not there.
func classify(name string) (string, string) {
	characteristics, ok := dragonfly.BenchmarkCharacteristics(name)
	if !ok {
		return "", ""
	}

	return characteristics.Modality.String(), characteristics.Landscape.String()
}
