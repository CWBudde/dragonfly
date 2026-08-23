// Command multiobjective runs MODA, the multi-objective Dragonfly Algorithm,
// on ZDT1 and reports the Pareto front it found.
//
// ZDT1 has two objectives and a known analytic front, f2 = 1 - sqrt(f1) for
// f1 in [0, 1], reached when every decision variable except the first is zero.
// That makes it the right problem for an example: the deviation of each
// archived solution from the analytic curve is a quality number a reader can
// check, not just a picture.
//
// A multi-objective run has no single best solution, so the result is the
// archive: a set of mutually non-dominated solutions, kept spread out over
// objective space by MODA's hypercube grid. The archive is written to a CSV
// file with ExportParetoCSV so it can be plotted elsewhere.
//
// Run it with:
//
//	go run .
package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	dragonfly "github.com/CWBudde/Dragonfly"
)

const (
	problemSize   = 5
	population    = 150
	maxIterations = 1000
	archiveSize   = 50
	// optimizerSeed pins the random stream so the front below is reproducible;
	// MODA is as stochastic as the single-objective algorithm, and the archive
	// contents depend on the order candidates were offered in.
	optimizerSeed = 20240923
)

func main() {
	config := dragonfly.NewMultiObjectiveConfig()
	config.ObjectiveFunc = dragonfly.ZDT1
	config.ArchiveSize = archiveSize
	config.Swarm.ProblemSize = problemSize
	config.Swarm.LowerBound = 0
	config.Swarm.UpperBound = 1
	config.Swarm.NPop = population
	config.Swarm.MaxIterations = maxIterations
	config.Swarm.Rand = rand.New(rand.NewSource(optimizerSeed))

	result, err := dragonfly.OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		fmt.Println("optimization failed:", err)

		return
	}

	front := sortedFront(result)

	reportRun(result)
	reportFront(front)
	reportExport(result)
}

// reportRun prints the run-level summary. ArchiveSizeCurve is the
// multi-objective analog of a convergence curve: an archive that stops growing
// early is the usual sign of a stagnated run.
func reportRun(result *dragonfly.MultiObjectiveResult) {
	curve := result.ArchiveSizeCurve

	fmt.Println("MODA -- ZDT1, 5 dimensions, bounds [0, 1]")
	fmt.Println()
	fmt.Printf("iterations       : %d\n", result.IterationCount)
	fmt.Printf("objective calls  : %d\n", result.FuncEvalCount)
	fmt.Printf("termination      : %s\n", result.TerminationReason)
	fmt.Printf("archive size     : %d of %d (capacity)\n", result.Archive.Len(), result.Archive.MaxSize)
	fmt.Printf("archive is mutually non-dominated: %v\n", result.Archive.IsNonDominated())
	fmt.Printf("archive size after iteration 1 / %d / %d: %d / %d / %d\n",
		len(curve)/2, len(curve), curve[0], curve[len(curve)/2-1], curve[len(curve)-1])
	fmt.Println()
}

// reportFront prints a sample of the archive next to the analytic front, plus
// the deviation from it across the whole archive.
func reportFront(front []*dragonfly.ParetoSolution) {
	if len(front) == 0 {
		fmt.Println("the archive is empty")

		return
	}

	fmt.Println("representative solutions, sorted by f1 (analytic front: f2 = 1 - sqrt(f1)):")
	fmt.Println()
	fmt.Println("       f1        f2   analytic f2   deviation   x[0]   max |x[1:]|")

	for _, index := range sampleIndices(len(front)) {
		solution := front[index]
		f1, f2 := solution.ObjectiveValues[0], solution.ObjectiveValues[1]
		analytic := analyticF2(f1)

		fmt.Printf("%9.5f %9.5f %13.5f %11.5f %6.3f %13.4f\n",
			f1, f2, analytic, f2-analytic, solution.Position[0], maxAbs(solution.Position[1:]))
	}

	mean, worst := deviation(front)

	fmt.Println()
	fmt.Printf("deviation from the analytic front: mean %.5f, worst %.5f\n", mean, worst)
	fmt.Printf("f1 spans [%.5f, %.5f], so the archive covers %.0f%% of the front\n",
		front[0].ObjectiveValues[0],
		front[len(front)-1].ObjectiveValues[0],
		100*(front[len(front)-1].ObjectiveValues[0]-front[0].ObjectiveValues[0]))
	fmt.Println("A positive deviation means the solution sits above the true front, which is")
	fmt.Println("where an unconverged solution has to be: the analytic curve is the lower limit.")
	fmt.Println()
}

// reportExport writes the archive to a CSV file and prints its first rows.
func reportExport(result *dragonfly.MultiObjectiveResult) {
	path := filepath.Join(os.TempDir(), "dragonfly_zdt1_front.csv")

	err := result.ExportParetoCSV(path)
	if err != nil {
		fmt.Println("export failed:", err)

		return
	}

	fmt.Printf("Pareto front written to %s\n", path)
	fmt.Println("one row per archived solution: index, one column per objective, one per variable")
	fmt.Println()

	content, err := os.ReadFile(path) //nolint:gosec // the path is this program's own output
	if err != nil {
		fmt.Println("read back failed:", err)

		return
	}

	printHead(string(content), 3)
}

// sortedFront returns the archive ordered by the first objective, which is the
// order a two-objective front is read in.
func sortedFront(result *dragonfly.MultiObjectiveResult) []*dragonfly.ParetoSolution {
	front := make([]*dragonfly.ParetoSolution, len(result.Archive.Solutions))
	copy(front, result.Archive.Solutions)

	sort.Slice(front, func(i, j int) bool {
		return front[i].ObjectiveValues[0] < front[j].ObjectiveValues[0]
	})

	return front
}

// analyticF2 is ZDT1's true front.
func analyticF2(f1 float64) float64 {
	return 1 - math.Sqrt(f1)
}

// deviation returns the mean and worst vertical distance from the analytic
// front over the whole archive.
func deviation(front []*dragonfly.ParetoSolution) (float64, float64) {
	total, worst := 0.0, 0.0

	for _, solution := range front {
		gap := math.Abs(solution.ObjectiveValues[1] - analyticF2(solution.ObjectiveValues[0]))
		total += gap
		worst = math.Max(worst, gap)
	}

	return total / float64(len(front)), worst
}

// sampleIndices picks up to seven evenly spaced positions in a slice of the
// given length, so the printed sample spans the front instead of clustering.
func sampleIndices(length int) []int {
	const wanted = 7
	if length <= wanted {
		indices := make([]int, length)
		for i := range indices {
			indices[i] = i
		}

		return indices
	}

	indices := make([]int, wanted)
	for i := range indices {
		indices[i] = i * (length - 1) / (wanted - 1)
	}

	return indices
}

// maxAbs returns the largest absolute component of a vector. For ZDT1 the
// variables after the first are zero on the true front, so this column reads as
// "how far from the front this solution's decision vector still is".
func maxAbs(x []float64) float64 {
	largest := 0.0
	for _, value := range x {
		largest = math.Max(largest, math.Abs(value))
	}

	return largest
}

// printHead prints the first n lines of a text blob.
func printHead(text string, n int) {
	start := 0
	for line := 0; line < n && start < len(text); line++ {
		end := start

		for end < len(text) && text[end] != '\n' {
			end++
		}

		fmt.Println("  " + text[start:end])
		start = end + 1
	}

	fmt.Println("  ...")
}
