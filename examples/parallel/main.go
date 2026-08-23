// Command parallel runs the same seeded optimization twice, once with
// Config.EnableParallel off and once with it on, and checks that the two
// results are bit-identical.
//
// That equality is a hard guarantee of the library, not an accident of timing:
// every random draw happens on the calling goroutine before a batch is
// dispatched, and the worker goroutines only call the objective function. So
// switching the pool on can change how long a run takes and nothing else.
//
// Whether it is worth switching on is a separate question, and the answer is
// "only when the objective function is expensive". The program therefore times
// two problems:
//
//   - a cheap objective (Sphere), where a call costs less than handing it to a
//     worker and the pool is pure overhead;
//   - an expensive objective (the same Sphere behind a synthetic workload of
//     tens of thousands of floating-point operations, standing in for a
//     simulation or a model fit), where the pool pays for itself.
//
// Run it with:
//
//	go run .
package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/MeKo-Christian/dragonfly"
)

const (
	problemSize   = 30
	bound         = 100.0
	population    = 40
	maxIterations = 100
	// optimizerSeed is what makes the comparison meaningful: both runs of a
	// pair are handed a generator seeded identically, so any difference in the
	// results would be the pool's fault and nothing else.
	optimizerSeed = 4242
	// workloadSize is the number of floating-point operations the expensive
	// objective performs on top of Sphere. It is tuned so one call costs on the
	// order of a hundred microseconds, which is where the pool starts to win
	// clearly on this machine. Around a thousand operations the two are a wash,
	// and below that the pool loses -- the crossover is a property of your
	// objective and your hardware, not of the library.
	workloadSize = 50000
)

// timedRun is one run and how long it took.
type timedRun struct {
	result   *dragonfly.Result
	duration time.Duration
}

func main() {
	fmt.Printf("Deterministic parallel evaluation -- %d dimensions, %d dragonflies, %d iterations\n",
		problemSize, population, maxIterations)
	fmt.Printf("GOMAXPROCS = %d, Config.MaxWorkers defaults to runtime.NumCPU() = %d\n",
		runtime.GOMAXPROCS(0), runtime.NumCPU())
	fmt.Println()

	compare("cheap objective (Sphere)", dragonfly.Sphere)
	fmt.Println()
	compare(fmt.Sprintf("expensive objective (Sphere + %d flops)", workloadSize), expensiveSphere)
	fmt.Println()
	fmt.Println("Read the two blocks together. The pool never changes the answer, so the")
	fmt.Println("only question it raises is one of cost: it is a loss on an objective that")
	fmt.Println("takes less time than dispatching it, and a win on one that does not.")
	fmt.Println("Measure your own objective before switching EnableParallel on.")
}

// compare runs one objective sequentially and in parallel and reports both the
// equality check and the timings.
func compare(label string, objective dragonfly.ObjectiveFunction) {
	sequential := run(objective, false)
	parallel := run(objective, true)

	if sequential.result == nil || parallel.result == nil {
		return
	}

	fmt.Printf("== %s ==\n", label)
	fmt.Printf("sequential: %8.1f ms, best cost %.17g\n",
		milliseconds(sequential.duration), sequential.result.GlobalBest.Cost)
	fmt.Printf("parallel  : %8.1f ms, best cost %.17g\n",
		milliseconds(parallel.duration), parallel.result.GlobalBest.Cost)
	fmt.Printf("speedup   : %8.2fx  (%s)\n",
		float64(sequential.duration)/float64(parallel.duration),
		verdict(sequential.duration, parallel.duration))
	fmt.Println()
	reportIdentity(sequential.result, parallel.result)
}

// reportIdentity compares the two results bit for bit rather than within a
// tolerance. A tolerance would hide exactly the kind of drift -- one random
// draw made on a worker goroutine -- that this check exists to catch.
func reportIdentity(sequential, parallel *dragonfly.Result) {
	fmt.Printf("best cost identical to the last bit : %v\n",
		bitsEqual(sequential.GlobalBest.Cost, parallel.GlobalBest.Cost))
	fmt.Printf("best position identical (%2d values) : %v\n",
		len(sequential.GlobalBest.Position),
		vectorBitsEqual(sequential.GlobalBest.Position, parallel.GlobalBest.Position))
	fmt.Printf("convergence curve identical (%3d)   : %v\n",
		len(sequential.ConvergenceCurve),
		vectorBitsEqual(sequential.ConvergenceCurve, parallel.ConvergenceCurve))
	fmt.Printf("worst position (the enemy) identical: %v\n",
		vectorBitsEqual(sequential.Worst.Position, parallel.Worst.Position))
	fmt.Printf("objective calls: %d sequential, %d parallel\n",
		sequential.FuncEvalCount, parallel.FuncEvalCount)
}

// run performs one optimization with a freshly seeded generator, so the two
// runs of a pair see the identical random stream.
func run(objective dragonfly.ObjectiveFunction, enableParallel bool) timedRun {
	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = objective
	config.ProblemSize = problemSize
	config.LowerBound = -bound
	config.UpperBound = bound
	config.NPop = population
	config.MaxIterations = maxIterations
	config.EnableParallel = enableParallel
	config.Rand = rand.New(rand.NewSource(optimizerSeed))

	start := time.Now()

	result, err := dragonfly.Optimize(config)

	elapsed := time.Since(start)

	if err != nil {
		fmt.Println("optimization failed:", err)

		return timedRun{}
	}

	return timedRun{result: result, duration: elapsed}
}

// expensiveSphere is Sphere with a synthetic workload attached, standing in for
// an objective whose evaluation is the expensive part of the run: a simulation,
// a model fit, a finite-element solve. The loop is deliberately not
// optimizable away -- its result is folded into the return value.
func expensiveSphere(x []float64) float64 {
	cost := dragonfly.Sphere(x)
	noise := 0.0

	for i := range workloadSize {
		noise += math.Sqrt(float64(i) + cost)
	}

	// The landscape stays Sphere's: noise is consumed by a test that never
	// fires, which is enough to keep the compiler from discarding the loop.
	if math.IsNaN(noise) {
		return math.Inf(1)
	}

	return cost
}

// milliseconds renders a duration as a float for the timing table.
func milliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

// verdict names the trade the measured timings show.
func verdict(sequential, parallel time.Duration) string {
	if parallel < sequential {
		return "the pool pays for itself"
	}

	return "the pool costs more than it saves"
}

// bitsEqual compares two floats by their bit patterns.
func bitsEqual(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b)
}

// vectorBitsEqual compares two vectors by their bit patterns.
func vectorBitsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if !bitsEqual(a[i], b[i]) {
			return false
		}
	}

	return true
}
