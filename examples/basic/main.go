// Command basic is the smallest useful continuous Dragonfly run: it minimizes
// the 10-dimensional Sphere function and reports the three numbers that
// describe any run -- the best cost found, why the run stopped, and how many
// times the objective function was called.
//
// The global minimum of Sphere is f(0, ..., 0) = 0, so the printed best cost is
// also the run's error. Do not expect it to be tiny: the Dragonfly Algorithm is
// a coarse global searcher, and this run stops as soon as it is within 1.0 of
// the optimum rather than polishing the answer.
//
// Run it with:
//
//	go run .
package main

import (
	"fmt"
	"math"
	"math/rand"

	dragonfly "github.com/CWBudde/Dragonfly"
)

const (
	problemSize   = 10
	bound         = 10.0
	maxIterations = 1500
	// optimizerSeed pins the random stream. The Dragonfly Algorithm is
	// stochastic, so without a seeded generator every run of this program would
	// print different numbers and the output could not be quoted, diffed, or
	// compared against a later change to the library. Config.Rand is the only
	// injection point; the package never touches the math/rand global.
	optimizerSeed = 1
	// targetCost stops the run early once the swarm gets this close to the
	// optimum, which is what makes TerminationReason worth printing: with this
	// seed the run ends at "target_cost" well before the iteration cap, and a
	// less lucky seed ends at "maximum_iterations" instead.
	targetCost = 1.0
)

func main() {
	target := targetCost

	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = dragonfly.Sphere
	config.ProblemSize = problemSize
	config.LowerBound = -bound
	config.UpperBound = bound
	config.MaxIterations = maxIterations
	config.Rand = rand.New(rand.NewSource(optimizerSeed))
	config.Convergence = &dragonfly.ConvergenceConfig{TargetCost: &target}

	result, err := dragonfly.Optimize(config)
	if err != nil {
		fmt.Println("optimization failed:", err)

		return
	}

	report(result)
}

// report prints the run summary and enough of the solution to see that it sits
// near the origin.
func report(result *dragonfly.Result) {
	fmt.Println("Dragonfly Algorithm -- Sphere, 10 dimensions, bounds [-10, 10]")
	fmt.Println()
	fmt.Printf("best cost           : %.6f      (global minimum is 0)\n", result.GlobalBest.Cost)
	fmt.Printf("termination reason  : %s\n", result.TerminationReason)
	fmt.Printf("iterations completed: %d of %d\n", result.IterationCount, maxIterations)
	fmt.Printf("objective calls     : %d\n", result.FuncEvalCount)
	fmt.Println()
	fmt.Printf("largest |x_j| in the reported solution: %.4f\n", maxAbs(result.GlobalBest.Position))
	fmt.Printf("first five components: %s\n", format(result.GlobalBest.Position[:5]))
	fmt.Println()
	fmt.Printf("convergence curve: %d entries, from %.4g down to %.4g\n",
		len(result.ConvergenceCurve),
		result.ConvergenceCurve[0],
		result.ConvergenceCurve[len(result.ConvergenceCurve)-1])
	fmt.Printf("worst position seen (the enemy every step is repelled by): cost %.4g\n",
		result.Worst.Cost)
	fmt.Println()
	fmt.Printf("reproduce this exact run with config.Rand = rand.New(rand.NewSource(%d)).\n",
		optimizerSeed)
	fmt.Println("Result.Seed carries the seed only when Config.Rand was left nil; here the")
	fmt.Println("supplied generator drove the run, so the seed above is the one to reuse.")
}

// maxAbs returns the largest absolute component of a vector.
func maxAbs(x []float64) float64 {
	largest := 0.0
	for _, value := range x {
		largest = math.Max(largest, math.Abs(value))
	}

	return largest
}

// format renders a short vector for display.
func format(x []float64) string {
	out := "["
	for i, value := range x {
		if i > 0 {
			out += " "
		}

		out += fmt.Sprintf("%+.2e", value)
	}

	return out + "]"
}
