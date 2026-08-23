package dragonfly

import (
	"context"
	"math/rand"
	"testing"
)

// BenchmarkOptimizeBaseline is the profiling anchor. `just profile-cpu` and
// `just profile-mem` both select it by exactly this name, so renaming it breaks
// both recipes.
//
// It is one representative end-to-end run: a 30-dimensional Sphere, the default
// swarm, a hundred iterations. A fresh fixed-seed configuration per operation
// keeps the profile reproducible and stops one operation's optimizer mutations
// from carrying into the next.
func BenchmarkOptimizeBaseline(b *testing.B) {
	b.ReportAllocs()

	evaluations := 0

	for range b.N {
		config := NewDefaultConfig()
		config.ObjectiveFunc = Sphere
		config.ProblemSize = 30
		config.LowerBound = -10
		config.UpperBound = 10
		config.MaxIterations = 100
		config.Rand = rand.New(rand.NewSource(42))

		result, err := Optimize(config)
		if err != nil {
			b.Fatal(err)
		}

		evaluations = result.FuncEvalCount
	}

	b.ReportMetric(float64(evaluations), "evals/op")
}

// BenchmarkOptimizeBinaryBaseline is the BDA counterpart of the anchor above,
// for profiling the bit-flip path rather than the continuous position update.
func BenchmarkOptimizeBinaryBaseline(b *testing.B) {
	b.ReportAllocs()

	for range b.N {
		config := NewBinaryConfig()
		config.ObjectiveFunc = oneMaxBits
		config.ProblemSize = 30
		config.MaxIterations = 100
		config.Rand = rand.New(rand.NewSource(42))

		_, err := OptimizeBinary(config)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOptimizeMultiObjectiveBaseline is the MODA counterpart. It profiles
// the hypercube archive as much as the swarm: every iteration re-grids the
// archive and draws a food source and an enemy from it.
func BenchmarkOptimizeMultiObjectiveBaseline(b *testing.B) {
	b.ReportAllocs()

	for range b.N {
		config := NewMultiObjectiveConfig()
		config.ObjectiveFunc = ZDT1
		config.Swarm.ProblemSize = 30
		config.Swarm.LowerBound = 0
		config.Swarm.UpperBound = 1
		config.Swarm.MaxIterations = 100
		config.Swarm.Rand = rand.New(rand.NewSource(42))

		_, err := OptimizeMultiObjective(context.Background(), config)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOptimizeBaselineParallel is the same workload as the anchor with the
// evaluation pool switched on. Sphere is far too cheap for the pool to pay for
// itself, which is the point: the pair documents the floor of what parallel
// evaluation costs before an expensive objective starts to earn it back.
func BenchmarkOptimizeBaselineParallel(b *testing.B) {
	b.ReportAllocs()

	for range b.N {
		config := NewDefaultConfig()
		config.ObjectiveFunc = Sphere
		config.ProblemSize = 30
		config.LowerBound = -10
		config.UpperBound = 10
		config.MaxIterations = 100
		config.EnableParallel = true
		config.Rand = rand.New(rand.NewSource(42))

		_, err := Optimize(config)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkObserverOverhead prices the lifecycle hooks. Progress copies one
// Best per iteration; a population observer deep-copies the whole swarm, which
// is why the two are separate options rather than one snapshot.
func BenchmarkObserverOverhead(b *testing.B) {
	run := func(b *testing.B, options ...RunOption) {
		b.Helper()
		b.ReportAllocs()

		for range b.N {
			config := NewDefaultConfig()
			config.ObjectiveFunc = Sphere
			config.ProblemSize = 30
			config.LowerBound = -10
			config.UpperBound = 10
			config.MaxIterations = 100
			config.Rand = rand.New(rand.NewSource(42))

			_, err := OptimizeContext(context.Background(), config, options...)
			if err != nil {
				b.Fatal(err)
			}
		}
	}

	b.Run("none", func(b *testing.B) { run(b) })

	b.Run("progress", func(b *testing.B) {
		run(b, WithProgressObserver(func(Progress) {}))
	})

	b.Run("population", func(b *testing.B) {
		run(b, WithPopulationObserver(func(PopulationSnapshot) {}))
	})
}
