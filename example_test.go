// Package dragonfly_test holds the runnable documentation for the library.
//
// This is the only black-box test file in the repository: everything else is
// white-box, in package dragonfly, so it can exercise unexported helpers. These
// examples are compiled and executed by `go test`, so they cannot drift out of
// date with the API -- but they are documentation, not a coverage vehicle.
//
// Every example seeds the generator and prints only structural facts (a
// dimension, an iteration count, a termination reason). A metaheuristic's
// costs are floating-point values that depend on the exact sequence of random
// draws, so printing one would make the // Output: comment a hostage to any
// change in the random stream.
package dragonfly_test

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"math/rand"

	"github.com/CWBudde/dragonfly"
)

// ExampleOptimize shows the four fields a run requires: the objective, the
// dimensionality and the two bounds. Everything else comes from the factory.
func ExampleOptimize() {
	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = dragonfly.Sphere
	config.ProblemSize = 5
	config.LowerBound = -10
	config.UpperBound = 10
	config.MaxIterations = 50
	config.NPop = 20

	// Seeding the generator makes the run reproducible. Leave Rand nil and
	// OptimizeContext draws a seed and reports it in Result.Seed, so a run can
	// still be reproduced after the fact.
	config.Rand = rand.New(rand.NewSource(42))

	result, err := dragonfly.Optimize(config)
	if err != nil {
		panic(err)
	}

	fmt.Println(len(result.GlobalBest.Position), result.IterationCount, result.TerminationReason)
	// Output: 5 50 maximum_iterations
}

// ExampleOptimizeContext shows the run lifecycle: a cancellable context, a
// structured logger, and an observer that is called once per completed
// iteration on the calling goroutine.
func ExampleOptimizeContext() {
	config := dragonfly.NewDefaultConfig()
	config.ObjectiveFunc = dragonfly.Rastrigin
	config.ProblemSize = 4
	config.LowerBound = -5.12
	config.UpperBound = 5.12
	config.MaxIterations = 25
	config.NPop = 20
	config.Rand = rand.New(rand.NewSource(42))

	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	updates := 0

	result, err := dragonfly.OptimizeContext(
		context.Background(),
		config,
		dragonfly.WithLogger(logger),
		dragonfly.WithProgressObserver(func(dragonfly.Progress) {
			updates++
		}),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.IterationCount, updates, len(result.ConvergenceCurve))
	// Output: 25 25 25
}

// ExampleNewBuilder shows the fluent front end. The chain records the first
// error it hits rather than returning one at every link, so it is checked once
// at the end.
func ExampleNewBuilder() {
	result, err := dragonfly.NewBuilder("da").
		ForProblem(dragonfly.Rastrigin, 4, -5.12, 5.12).
		WithIterations(25).
		WithPopulation(20).
		WithConfig(func(config *dragonfly.Config) {
			config.Rand = rand.New(rand.NewSource(42))
		}).
		Optimize()
	if err != nil {
		panic(err)
	}

	fmt.Println(len(result.GlobalBest.Position), result.IterationCount)
	// Output: 4 25
}

// ExampleOptimizeBinary shows BDA. Positions are bit strings, so the objective
// keeps the ordinary func([]float64) float64 signature and simply receives
// 0/1-valued input -- here the count of zero bits, which an all-ones string
// drives to zero.
func ExampleOptimizeBinary() {
	config := dragonfly.NewBinaryConfig()
	config.ObjectiveFunc = func(bits []float64) float64 {
		zeros := 0.0
		for _, bit := range bits {
			zeros += 1 - bit
		}

		return zeros
	}
	config.ProblemSize = 20
	config.MaxIterations = 200
	config.NPop = 30
	config.Rand = rand.New(rand.NewSource(42))

	result, err := dragonfly.OptimizeBinary(config)
	if err != nil {
		panic(err)
	}

	fmt.Println(dragonfly.BinaryPositionsValid(result.GlobalBest.Position), result.GlobalBest.Cost)
	// Output: true 0
}

// ExampleOptimizeMultiObjective shows MODA. There is no single best position to
// report, so the result is the archive: an approximation of the Pareto front,
// which is non-dominated by construction after every mutation.
func ExampleOptimizeMultiObjective() {
	config := dragonfly.NewMultiObjectiveConfig()
	config.ObjectiveFunc = dragonfly.ZDT1
	config.Swarm.ProblemSize = 10
	config.Swarm.LowerBound = 0
	config.Swarm.UpperBound = 1
	config.Swarm.MaxIterations = 100
	config.Swarm.NPop = 40
	config.Swarm.Rand = rand.New(rand.NewSource(42))

	result, err := dragonfly.OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.Archive.IsNonDominated(), result.IterationCount)
	// Output: true 100
}
