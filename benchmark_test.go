package dragonfly

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
)

// The benchmarks in this file measure wall-clock cost, not solution quality --
// that is regression_test.go's job. They are therefore deliberately short:
// enough iterations that the per-iteration work dominates setup, few enough
// that `go test -bench=.` finishes in a couple of minutes.
const (
	benchDimensions = 10
	benchIterations = 50
	benchPopulation = 30
)

// BenchmarkProblem names one benchmark function together with the box it is
// conventionally evaluated on.
type BenchmarkProblem struct {
	Func       ObjectiveFunction
	Name       string
	LowerBound float64
	UpperBound float64
}

// benchmarkProblems is the continuous suite the per-function benchmarks and the
// scaling sweeps both draw on.
var benchmarkProblems = []BenchmarkProblem{
	{Name: "Sphere", Func: Sphere, LowerBound: -100, UpperBound: 100},
	{Name: "Rastrigin", Func: Rastrigin, LowerBound: -5.12, UpperBound: 5.12},
	{Name: "Rosenbrock", Func: Rosenbrock, LowerBound: -5, UpperBound: 10},
	{Name: "Ackley", Func: Ackley, LowerBound: -32.768, UpperBound: 32.768},
	{Name: "Griewank", Func: Griewank, LowerBound: -600, UpperBound: 600},
	{Name: "Schwefel", Func: Schwefel, LowerBound: -500, UpperBound: 500},
	{Name: "Zakharov", Func: Zakharov, LowerBound: -10, UpperBound: 10},
	{Name: "BentCigar", Func: BentCigar, LowerBound: -100, UpperBound: 100},
}

// benchConfig builds a fresh, fixed-seed continuous configuration. A new one
// per operation keeps the benchmark reproducible and stops one iteration's
// optimizer mutations from leaking into the next.
func benchConfig(problem BenchmarkProblem, dimensions, iterations, population int) *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = problem.Func
	config.ProblemSize = dimensions
	config.LowerBound = problem.LowerBound
	config.UpperBound = problem.UpperBound
	config.MaxIterations = iterations
	config.NPop = population
	config.Rand = rand.New(rand.NewSource(42))

	return config
}

// benchmarkDA is the body every BenchmarkOptimize<Function>_DA shares.
func benchmarkDA(b *testing.B, problem BenchmarkProblem) {
	b.Helper()
	b.ReportAllocs()

	for range b.N {
		_, err := Optimize(benchConfig(problem, benchDimensions, benchIterations, benchPopulation))
		if err != nil {
			b.Fatal(err)
		}
	}
}

// benchmarkBDA is the body every BenchmarkOptimize<Function>_BDA shares.
//
// The objective is the same continuous benchmark evaluated on a 0/1 vector:
// BDA keeps the func([]float64) float64 signature precisely so the benchmark
// suite is reused unchanged, and what is being timed here is the bit-flip
// machinery rather than the landscape.
func benchmarkBDA(b *testing.B, problem BenchmarkProblem) {
	b.Helper()
	b.ReportAllocs()

	for range b.N {
		config := NewBinaryConfig()
		config.ObjectiveFunc = problem.Func
		config.ProblemSize = benchDimensions
		config.MaxIterations = benchIterations
		config.NPop = benchPopulation
		config.Rand = rand.New(rand.NewSource(42))

		_, err := OptimizeBinary(config)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// benchmarkMODA is the body every BenchmarkOptimize<Problem>_MODA shares.
func benchmarkMODA(b *testing.B, objective MultiObjectiveFunction) {
	b.Helper()
	b.ReportAllocs()

	for range b.N {
		config := NewMultiObjectiveConfig()
		config.ObjectiveFunc = objective
		config.Swarm.ProblemSize = benchDimensions
		config.Swarm.LowerBound = 0
		config.Swarm.UpperBound = 1
		config.Swarm.MaxIterations = benchIterations
		config.Swarm.NPop = benchPopulation
		config.Swarm.Rand = rand.New(rand.NewSource(42))

		_, err := OptimizeMultiObjective(context.Background(), config)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// problemNamed looks a benchmark problem up by name so the per-function
// benchmarks below stay one line each.
func problemNamed(name string) BenchmarkProblem {
	for _, problem := range benchmarkProblems {
		if problem.Name == name {
			return problem
		}
	}

	panic("unknown benchmark problem " + name)
}

// --- DA, one benchmark per function -----------------------------------------

func BenchmarkOptimizeSphere_DA(b *testing.B)     { benchmarkDA(b, problemNamed("Sphere")) }
func BenchmarkOptimizeRastrigin_DA(b *testing.B)  { benchmarkDA(b, problemNamed("Rastrigin")) }
func BenchmarkOptimizeRosenbrock_DA(b *testing.B) { benchmarkDA(b, problemNamed("Rosenbrock")) }
func BenchmarkOptimizeAckley_DA(b *testing.B)     { benchmarkDA(b, problemNamed("Ackley")) }
func BenchmarkOptimizeGriewank_DA(b *testing.B)   { benchmarkDA(b, problemNamed("Griewank")) }
func BenchmarkOptimizeSchwefel_DA(b *testing.B)   { benchmarkDA(b, problemNamed("Schwefel")) }
func BenchmarkOptimizeZakharov_DA(b *testing.B)   { benchmarkDA(b, problemNamed("Zakharov")) }
func BenchmarkOptimizeBentCigar_DA(b *testing.B)  { benchmarkDA(b, problemNamed("BentCigar")) }

// --- BDA, one benchmark per function ----------------------------------------

func BenchmarkOptimizeSphere_BDA(b *testing.B)     { benchmarkBDA(b, problemNamed("Sphere")) }
func BenchmarkOptimizeRastrigin_BDA(b *testing.B)  { benchmarkBDA(b, problemNamed("Rastrigin")) }
func BenchmarkOptimizeRosenbrock_BDA(b *testing.B) { benchmarkBDA(b, problemNamed("Rosenbrock")) }
func BenchmarkOptimizeAckley_BDA(b *testing.B)     { benchmarkBDA(b, problemNamed("Ackley")) }
func BenchmarkOptimizeGriewank_BDA(b *testing.B)   { benchmarkBDA(b, problemNamed("Griewank")) }

// --- MODA, one benchmark per multi-objective problem -------------------------

func BenchmarkOptimizeZDT1_MODA(b *testing.B)       { benchmarkMODA(b, ZDT1) }
func BenchmarkOptimizeZDT2_MODA(b *testing.B)       { benchmarkMODA(b, ZDT2) }
func BenchmarkOptimizeZDT3_MODA(b *testing.B)       { benchmarkMODA(b, ZDT3) }
func BenchmarkOptimizeSchafferN1_MODA(b *testing.B) { benchmarkMODA(b, SchafferN1) }

// --- scaling sweeps ----------------------------------------------------------

// BenchmarkDimensionScaling measures how the per-run cost grows with the
// problem size. Every primitive is a vector operation over d components, so the
// expected shape is linear in d on top of the objective's own cost.
func BenchmarkDimensionScaling(b *testing.B) {
	problem := problemNamed("Sphere")

	for _, dimensions := range []int{2, 10, 30, 100} {
		b.Run(fmt.Sprintf("dimensions_%d", dimensions), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_, err := Optimize(benchConfig(problem, dimensions, benchIterations, benchPopulation))
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkPopulationSize measures how the per-run cost grows with the swarm.
// The neighbor scan is O(n²·d), so this sweep is the one that shows it: the
// cost should grow faster than linearly once the scan dominates the objective.
func BenchmarkPopulationSize(b *testing.B) {
	problem := problemNamed("Sphere")

	for _, population := range []int{10, 40, 100, 250} {
		b.Run(fmt.Sprintf("population_%d", population), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_, err := Optimize(benchConfig(problem, benchDimensions, benchIterations, population))
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkIterationScaling confirms the loop itself is linear in the iteration
// count, so a run that is slower than expected is slower per iteration rather
// than running more of them.
func BenchmarkIterationScaling(b *testing.B) {
	problem := problemNamed("Sphere")

	for _, iterations := range []int{25, 100, 400} {
		b.Run(fmt.Sprintf("iterations_%d", iterations), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_, err := Optimize(benchConfig(problem, benchDimensions, iterations, benchPopulation))
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkBoundaryMethods prices the three boundary rules against each other.
// Wrapping draws a random number per repaired component and the other two do
// not, so this is where that cost would show up.
func BenchmarkBoundaryMethods(b *testing.B) {
	problem := problemNamed("Rosenbrock")

	for _, method := range []BoundaryMethod{BoundaryWrap, BoundaryClamp, BoundaryReflect} {
		b.Run(string(method), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				config := benchConfig(problem, benchDimensions, benchIterations, benchPopulation)
				config.BoundaryMethod = method

				_, err := Optimize(config)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkTransferFunctions prices the eight BDA transfer functions. They
// differ by one transcendental call per component per iteration, which is the
// kind of difference only a benchmark settles.
func BenchmarkTransferFunctions(b *testing.B) {
	for _, name := range TransferFunctionNames() {
		b.Run(string(name), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				config := NewBinaryConfig()
				config.ObjectiveFunc = oneMaxBits
				config.ProblemSize = 30
				config.MaxIterations = benchIterations
				config.NPop = benchPopulation
				config.TransferFunc = name
				config.Rand = rand.New(rand.NewSource(42))

				_, err := OptimizeBinary(config)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkParallelEvaluation contrasts a sequential run with a parallel one on
// a deliberately expensive objective. On a cheap objective the pool is pure
// overhead, which is the result worth being able to demonstrate.
func BenchmarkParallelEvaluation(b *testing.B) {
	expensive := func(x []float64) float64 {
		total := 0.0
		for range 200 {
			total += Rastrigin(x)
		}

		return total / 200
	}

	problem := BenchmarkProblem{Name: "ExpensiveRastrigin", Func: expensive, LowerBound: -5.12, UpperBound: 5.12}

	for _, parallel := range []bool{false, true} {
		name := "sequential"
		if parallel {
			name = "parallel"
		}

		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				config := benchConfig(problem, benchDimensions, benchIterations, benchPopulation)
				config.EnableParallel = parallel

				_, err := Optimize(config)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
