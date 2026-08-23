// Command examples is the guided tour of the Dragonfly library: it runs the
// standard continuous algorithm over a spread of benchmark functions, shows
// what the built-in problem classifier recommends for each of them, and points
// at the focused examples in the subdirectories.
//
// It is what `just run` and `just compare` execute, so it is deliberately
// short: every run below finishes in a fraction of a second, and none of them
// is tuned for a good answer. For anything more than a smoke test, start from
// one of the subdirectory examples.
//
// Run it with:
//
//	cd examples && go run main.go
package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/MeKo-Christian/dragonfly"
)

const (
	problemSize   = 10
	maxIterations = 300
	// runSeed pins the random stream so this tour prints the same table every
	// time. Config.Rand is the only injection point; the package never touches
	// the math/rand global, and a run with Config.Rand left nil reports the
	// seed it drew in Result.Seed instead.
	runSeed = 2024
)

// benchmark is one entry of the tour: a function from functions.go, the bounds
// it is conventionally evaluated on, and a one-line description of its
// landscape. Every function here has a global minimum of zero, so the best cost
// a run reports is also its error.
type benchmark struct {
	name     string
	fn       dragonfly.ObjectiveFunction
	lower    float64
	upper    float64
	landmark string
}

// benchmarks spans the landscape types the algorithm behaves differently on:
// a smooth bowl, a regular lattice of local minima, a curved valley, and a
// deceptive function whose global minimum is far from the next best one.
var benchmarks = []benchmark{
	{"Sphere", dragonfly.Sphere, -100, 100, "smooth and unimodal -- the easy case"},
	{"Rastrigin", dragonfly.Rastrigin, -5.12, 5.12, "a regular lattice of local minima"},
	{"Rosenbrock", dragonfly.Rosenbrock, -30, 30, "a narrow curved valley"},
	{"Ackley", dragonfly.Ackley, -32, 32, "flat outer region, sharp central funnel"},
	{"Griewank", dragonfly.Griewank, -600, 600, "multimodal, with a wide basin"},
	{"Schwefel", dragonfly.Schwefel, -500, 500, "deceptive: the optimum sits at the edge"},
}

func main() {
	fmt.Println(strings.Repeat("=", 78))
	fmt.Println("Dragonfly Algorithm -- guided tour")
	fmt.Println(strings.Repeat("=", 78))
	fmt.Println()

	runBenchmarks()
	fmt.Println()
	recommendations()
	fmt.Println()
	directory()
}

// runBenchmarks runs the continuous algorithm once on each benchmark and prints
// the table.
func runBenchmarks() {
	fmt.Printf("Standard run: %d dimensions, %d iterations, seed %d, default configuration.\n",
		problemSize, maxIterations, runSeed)
	fmt.Println("Every function here has a global minimum of 0, so best cost is also the error.")
	fmt.Println()
	fmt.Printf("%-12s %14s %14s %10s %9s  %s\n",
		"function", "best cost", "bounds", "evals", "time", "landscape")
	fmt.Println(strings.Repeat("-", 100))

	for _, entry := range benchmarks {
		result, elapsed, err := run(entry)
		if err != nil {
			fmt.Printf("%-12s %s\n", entry.name, err)

			continue
		}

		fmt.Printf("%-12s %14.6g %14s %10d %8.0fms  %s\n",
			entry.name,
			result.GlobalBest.Cost,
			fmt.Sprintf("[%g, %g]", entry.lower, entry.upper),
			result.FuncEvalCount,
			float64(elapsed)/float64(time.Millisecond),
			entry.landmark)
	}

	fmt.Println(strings.Repeat("-", 100))
	fmt.Println("Three hundred iterations is a smoke test, not a fair evaluation: the")
	fmt.Println("Dragonfly Algorithm is a coarse global searcher and keeps improving well")
	fmt.Println("past this budget. Raise MaxIterations before drawing conclusions.")
}

// run performs one optimization and returns the result and how long it took.
func run(entry benchmark) (*dragonfly.Result, time.Duration, error) {
	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = entry.fn
	config.ProblemSize = problemSize
	config.LowerBound = entry.lower
	config.UpperBound = entry.upper
	config.MaxIterations = maxIterations
	config.Rand = rand.New(rand.NewSource(runSeed))

	start := time.Now()

	result, err := dragonfly.Optimize(config)

	return result, time.Since(start), err
}

// recommendations prints what the built-in classifier suggests for each
// benchmark. With three variants the answer is rarely a surprise, but the
// reason attached to it is the part worth reading.
func recommendations() {
	fmt.Println("What the selector recommends for each of these problems:")
	fmt.Println()

	for _, entry := range benchmarks {
		recommendation := dragonfly.RecommendForBenchmark(entry.name)
		fmt.Printf("  %-12s -> %-5s (confidence %.2f) %s\n",
			entry.name,
			recommendation.Variant.Name(),
			recommendation.Confidence,
			recommendation.Reason)
	}
}

// directory points at the focused examples. Each is its own module with a
// replace directive, so it is run from its own directory.
func directory() {
	fmt.Println("Focused examples, each its own module -- cd into one and run `go run .`:")
	fmt.Println()

	entries := [][2]string{
		{"basic", "the smallest useful run: best cost, termination reason, evaluation count"},
		{"constrained", "Deb's feasibility rules, and why the reported cost is the raw objective"},
		{"parallel", "EnableParallel off vs on: bit-identical results, and when the pool pays"},
		{"comparison", "ComparisonRunner over DA and BDA with paired seeds and significance tests"},
		{"multiobjective", "MODA on ZDT1, with the Pareto front exported to CSV"},
		{"feature_selection", "BDA on a wrapper-style feature-selection problem"},
	}

	for _, entry := range entries {
		fmt.Printf("  examples/%-18s %s\n", entry[0], entry[1])
	}
}
