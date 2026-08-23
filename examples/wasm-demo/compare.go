//go:build js && wasm

package main

import (
	"context"
	"syscall/js"

	"github.com/CWBudde/dragonfly"
)

/*
The shootout compares *configurations*, not variants.

The library ships two single-objective variants, DA and BDA, and a
variant-vs-variant table of two rows would say very little -- BDA searches
{0,1}^d, so putting it beside DA on Rastrigin compares two different problems
and calls the result a ranking.

What is worth comparing is the choices DA actually leaves open: which boundary
rule, whether the lone-dragonfly branch takes a Levy walk, whether the enemy
term earns its place, and how much a bigger swarm buys. Each contender below is
DA with one field changed, wrapped in the exported AlgorithmVariant interface so
ComparisonRunner -- and with it the paired seeding, the Wilcoxon tests and the
Friedman test -- applies unchanged.
*/

// contender is one configuration under test. edit is applied to a fresh
// NewDefaultConfig; ComparisonRunner then fills in the objective, the
// dimensionality, the bounds, the iteration budget and the seed.
type contender struct {
	edit func(*dragonfly.Config)

	key         string
	label       string
	description string
}

var contenders = []contender{
	{
		key: "paper", label: "paper defaults",
		description: "NewDefaultConfig as published: wrapping bounds, Levy walk on, every weight on its schedule.",
		edit:        func(*dragonfly.Config) {},
	},
	{
		key: "clamp", label: "clamped bounds",
		description: "Pin a coordinate to the bound it crossed, the way most other swarm libraries do.",
		edit:        func(c *dragonfly.Config) { c.BoundaryMethod = dragonfly.BoundaryClamp },
	},
	{
		key: "reflect", label: "reflected bounds",
		description: "Mirror the overshoot back into the feasible interval.",
		edit:        func(c *dragonfly.Config) { c.BoundaryMethod = dragonfly.BoundaryReflect },
	},
	{
		key: "no-levy", label: "no Levy walk",
		description: "A dragonfly with no neighbors and no food in range stays put instead of walking away.",
		edit:        func(c *dragonfly.Config) { c.UseLevyWalk = false },
	},
	{
		key: "no-enemy", label: "no enemy term",
		description: "EnemyWeight pinned to zero for the whole run. Note that 0 is a pinned value, not the WeightAuto sentinel.",
		edit:        func(c *dragonfly.Config) { c.EnemyWeight = 0 },
	},
	{
		key: "big-swarm", label: "double the swarm",
		description: "Twice the population on the same iteration budget, so twice the evaluations. Not a like-for-like comparison, and listed as a control.",
		edit:        func(c *dragonfly.Config) { c.NPop = dragonfly.NewDefaultConfig().NPop * 2 },
	},
}

func contenderByKey(key string) (contender, bool) {
	for _, entry := range contenders {
		if entry.key == key {
			return entry, true
		}
	}

	return contender{}, false
}

/*
configVariant adapts a contender to dragonfly.AlgorithmVariant.

The interface is exported and every method is implementable from outside the
library, which is what lets the demo reuse ComparisonRunner's statistics rather
than reimplementing paired seeding and a Wilcoxon test in JavaScript. Run just
forwards to OptimizeContext: these are all standard DA.
*/
type configVariant struct {
	spec contender
}

func (v configVariant) Name() string     { return v.spec.label }
func (v configVariant) FullName() string { return "Dragonfly Algorithm — " + v.spec.label }

func (v configVariant) Description() string { return v.spec.description }

func (v configVariant) GetConfig() *dragonfly.Config {
	config := dragonfly.NewDefaultConfig()
	v.spec.edit(config)
	config.EnableParallel = false

	return config
}

func (v configVariant) IsMultiObjective() bool { return false }

func (v configVariant) Run(
	ctx context.Context,
	config *dragonfly.Config,
	options ...dragonfly.RunOption,
) (*dragonfly.Result, error) {
	return dragonfly.OptimizeContext(ctx, config, options...)
}

// ApplicableTo and the two below are only consulted by the selector, which the
// shootout never calls. They are honest rather than clever: every contender is
// standard DA, so they report what the library's own DA variant reports.
func (v configVariant) ApplicableTo(characteristics dragonfly.ProblemCharacteristics) float64 {
	base, err := dragonfly.NewVariant("DA")
	if err != nil {
		return 0
	}

	return base.ApplicableTo(characteristics)
}

func (v configVariant) EstimatedOverhead() float64 { return 1 }

func (v configVariant) RecommendedFor() []string { return []string{v.spec.description} }

// jsCompare runs exactly one benchmark through ComparisonRunner.
//
// One benchmark per call, deliberately. A call into Go blocks its thread's
// event loop for its whole duration, so a "stop" cannot be dispatched while one
// is in flight: the page's sweep loop is in JavaScript and the chunking is what
// makes cancellation possible at all. See bench-worker.js.
func jsCompare(opts js.Value) any {
	benchmarkName := readString(opts, "benchmark", "Sphere")

	spec, ok := lookupBenchmark(benchmarkName)
	if !ok {
		return errorResult("compare: unknown benchmark %q", benchmarkName)
	}

	keys := readStrings(opts, "contenders", nil)
	if len(keys) == 0 {
		return errorResult("compare: no contenders selected")
	}

	variants := make([]dragonfly.AlgorithmVariant, 0, len(keys))
	labels := make([]any, 0, len(keys))

	for _, key := range keys {
		entry, found := contenderByKey(key)
		if !found {
			return errorResult("compare: unknown contender %q", key)
		}

		variants = append(variants, configVariant{spec: entry})
		labels = append(labels, entry.label)
	}

	var (
		dimensions = clampInt(readInt(opts, "dimensions", 10), 2, maxDimensions)
		iterations = clampInt(readInt(opts, "iterations", 200), 1, maxIterations)
		runs       = clampInt(readInt(opts, "runs", 10), 2, maxCompareRuns)
		seed       = int64(readFloat(opts, "seed", 42))
	)

	runner := dragonfly.NewComparisonRunner().
		WithVariants(variants...).
		WithRuns(runs).
		WithIterations(iterations).
		WithSeed(seed).
		WithParallel(false).
		WithVerbose(false)

	// A success rate needs a threshold that is expressible as a positive cost;
	// see successTarget for why some functions have none.
	target, hasTarget := successTarget(spec, dimensions)
	if hasTarget {
		runner = runner.WithTarget(target)
	}

	result, err := runner.CompareContext(
		context.Background(), spec.name, spec.fn, dimensions, spec.lower, spec.upper)
	if err != nil {
		return errorResult("compare: %v", err)
	}

	return map[string]any{
		"benchmark":     spec.name,
		"contenders":    labels,
		"dimensions":    dimensions,
		"iterations":    iterations,
		"runs":          runs,
		"baseSeed":      float64(result.BaseSeed),
		"bestAlgorithm": result.BestAlgorithm,
		"hasTarget":     hasTarget,
		"target":        optionalNumber(target, hasTarget),
		"statistics":    compareStatistics(result),
		"rankings":      intsToJS(result.Rankings),
		"wilcoxon":      compareWilcoxon(result),
		"friedman":      compareFriedman(result),
	}
}

func compareStatistics(result *dragonfly.ComparisonResult) []any {
	rows := make([]any, 0, len(result.Statistics))

	for i, statistics := range result.Statistics {
		rows = append(rows, map[string]any{
			"name":        result.AlgorithmNames[i],
			"mean":        jsNumber(statistics.Mean),
			"median":      jsNumber(statistics.Median),
			"stddev":      jsNumber(statistics.StdDev),
			"best":        jsNumber(statistics.Best),
			"worst":       jsNumber(statistics.Worst),
			"successRate": jsNumber(statistics.SuccessRate),
			"avgEvals":    jsNumber(statistics.AvgFuncEvals),
			"avgSeconds":  jsNumber(statistics.AvgTime),
			"rank":        result.Rankings[i],
		})
	}

	return rows
}

func compareWilcoxon(result *dragonfly.ComparisonResult) []any {
	rows := make([]any, 0)

	// Only the upper triangle: the matrix is symmetric and the diagonal is
	// left zero by the library.
	for i := range result.WilcoxonTests {
		for j := i + 1; j < len(result.WilcoxonTests[i]); j++ {
			test := result.WilcoxonTests[i][j]
			rows = append(rows, map[string]any{
				"a":           test.Algorithm1,
				"b":           test.Algorithm2,
				"winner":      test.Winner,
				"w":           jsNumber(test.WStatistic),
				"p":           jsNumber(test.PValue),
				"significant": test.Significant,
			})
		}
	}

	return rows
}

func compareFriedman(result *dragonfly.ComparisonResult) any {
	if result.FriedmanResult == nil {
		return nil
	}

	return map[string]any{
		"chiSquare":   jsNumber(result.FriedmanResult.ChiSquare),
		"p":           jsNumber(result.FriedmanResult.PValue),
		"df":          result.FriedmanResult.DegreesOfFreedom,
		"significant": result.FriedmanResult.Significant,
	}
}

func intsToJS(values []int) []any {
	items := make([]any, len(values))
	for i, value := range values {
		items[i] = value
	}

	return items
}
