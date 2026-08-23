// Command comparison runs ComparisonRunner over the two single-objective
// variants, DA and BDA, and prints the statistics table it produces.
//
// The runner pairs seeds: run k of every variant is given the seed
// BaseSeed + k, so the variants face identical starting conditions and the
// differences that remain are the algorithms'. That pairing is what the
// Wilcoxon signed-rank test assumes, and it is why a ten-run comparison says
// anything at all.
//
// Comparing a continuous variant with a binary one only makes sense on a
// problem both can express, so the benchmark here is a pattern-matching
// problem: minimise the squared distance to a fixed 0/1 pattern over the unit
// box. Its optimum is a bit string, which BDA can hit exactly, and it is a
// smooth quadratic, which DA can approach from anywhere inside the box. A
// continuous benchmark such as Sphere over [-100, 100] would not be a fair
// comparison -- BDA keeps its own unit-interval bounds by design, because
// every schedule that scales with (ub - lb) is written for that box.
//
// Run it with:
//
//	go run .
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/MeKo-Christian/dragonfly"
)

const (
	problemSize   = 20
	runs          = 10
	maxIterations = 300
	// baseSeed makes the whole comparison reproducible: run k of every variant
	// used baseSeed + k, and the number is echoed in the report.
	baseSeed = 990
	// targetCost is the success threshold behind the SuccessRate column and the
	// ConvergenceAt figures. It is loose enough that a continuous variant can
	// reach it without landing exactly on the pattern.
	targetCost = 1e-3
)

// pattern is the bit string both variants are looking for. It alternates in
// blocks rather than regularly, so a solution that keeps every bit set or every
// bit clear scores badly.
var pattern = []float64{
	1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
	1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
}

// patternDistance is the benchmark: the squared distance to the pattern. Its
// global minimum is f(pattern) = 0, and it is defined for any x in the unit box,
// which is what lets a continuous and a binary variant be scored on the same
// function.
func patternDistance(x []float64) float64 {
	total := 0.0

	for j, value := range x {
		gap := value - pattern[j]
		total += gap * gap
	}

	return total
}

func main() {
	runner := dragonfly.NewComparisonRunner().
		WithVariantNames("da", "bda").
		WithRuns(runs).
		WithIterations(maxIterations).
		WithTarget(targetCost).
		WithSeed(baseSeed)

	result := runner.Compare("PatternDistance", patternDistance, problemSize, 0, 1)
	if result.BestAlgorithm < 0 {
		fmt.Println("the comparison produced no statistics")

		return
	}

	fmt.Printf("Benchmark: squared distance to a fixed %d-bit pattern, bounds [0, 1]\n",
		problemSize)
	fmt.Printf("%d paired runs per variant, %d iterations each, base seed %d,\n",
		runs, maxIterations, baseSeed)
	fmt.Printf("success threshold %g.\n", targetCost)
	fmt.Println()

	result.PrintComparisonResults()

	reportSeeds(result)
	reportExport(result)
	fmt.Println()
	interpret(result)
}

// interpret says what the table means, so the numbers are not left to speak
// for themselves.
func interpret(result *dragonfly.ComparisonResult) {
	winner := result.AlgorithmNames[result.BestAlgorithm]

	fmt.Printf("%s wins, and it should: the optimum of this benchmark is a bit string,\n", winner)
	fmt.Println("so BDA's position update can land on it exactly while DA has to approach it")
	fmt.Println("through the interior of the box and stops short. Read that as a statement")
	fmt.Println("about the problem, not a ranking of the variants -- run the same comparison")
	fmt.Println("on Sphere over [-100, 100] and the result reverses, because BDA keeps its")
	fmt.Println("unit-interval bounds and cannot represent the answer at all.")
	fmt.Println()
	fmt.Println("What the statistics add is confidence. Ten paired runs is a small sample,")
	fmt.Println("and the Wilcoxon p-value above is the normal approximation, which is")
	fmt.Println("unreliable below roughly ten non-tied pairs; NewComparisonRunner defaults to")
	fmt.Println("30 runs for that reason. This example lowers it only to stay quick.")
}

// reportSeeds prints the per-run costs side by side, which is the raw material
// the paired tests work on: the two columns of any row came from the same seed.
func reportSeeds(result *dragonfly.ComparisonResult) {
	fmt.Println("paired runs, one row per seed:")
	fmt.Println()
	fmt.Printf("  %-6s", "seed")

	for _, name := range result.AlgorithmNames {
		fmt.Printf("  %14s", name)
	}

	fmt.Println()

	for run := range result.RunResults[0] {
		fmt.Printf("  %-6d", result.RunResults[0][run].Seed)

		for variant := range result.AlgorithmNames {
			fmt.Printf("  %14.6g", result.RunResults[variant][run].BestCost)
		}

		fmt.Println()
	}

	fmt.Println()
}

// reportExport writes the full comparison, including every run, to CSV.
func reportExport(result *dragonfly.ComparisonResult) {
	path := filepath.Join(os.TempDir(), "dragonfly_comparison.csv")

	err := result.ExportToCSV(path)
	if err != nil {
		fmt.Println("export failed:", err)

		return
	}

	fmt.Printf("full results written to %s\n", path)
	fmt.Println("one row per run, with each variant's aggregate statistics repeated on it")
	fmt.Println("so the file is usable without a join.")
}
