// Run-scoped options, observers and logging for the Dragonfly Algorithm.

package dragonfly

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
)

// Progress describes the best solution known after a completed iteration.
// Iteration is one-based. Best and its Position are snapshots: observers may
// retain or modify them without affecting the optimizer.
type Progress struct {
	Best            Best
	Iteration       int
	EvaluationCount int
}

// ProgressObserver receives a snapshot after each completed iteration.
// OptimizeContext invokes observers synchronously on the calling goroutine.
type ProgressObserver func(Progress)

// PopulationSnapshot is the state of the swarm after a completed iteration.
// Iteration is one-based. Every dragonfly, Best and Worst, is a deep copy:
// observers may retain or modify them without affecting the optimizer.
//
// It is deliberately separate from Progress rather than an extension of it.
// Copying NPop position and step vectors once per iteration is not free, and
// the overwhelmingly common reason to observe a run is to watch the best cost
// fall, which Progress already answers. Callers who want the swarm itself --
// to animate it, to measure diversity, to debug a variant's search behavior --
// opt in with WithPopulationObserver and pay for it there.
//
// Worst is the enemy the swarm is currently repelled from. It has no Mayfly
// counterpart and is carried here because every step of the algorithm is
// computed against it, so a snapshot without it cannot explain a move.
type PopulationSnapshot struct {
	Swarm           []Dragonfly
	Best            Best
	Worst           Best
	Iteration       int
	EvaluationCount int
}

// PopulationObserver receives the whole swarm after each completed iteration.
// OptimizeContext invokes observers synchronously on the calling goroutine.
type PopulationObserver func(PopulationSnapshot)

// Logger receives structured optimization lifecycle events. *slog.Logger
// implements Logger. OptimizeContext invokes loggers synchronously on the
// calling goroutine.
type Logger interface {
	Log(ctx context.Context, level slog.Level, message string, args ...any)
}

// RunOption customizes one optimization run. Its fields are intentionally
// private; construct options with WithInitialPopulation,
// WithProgressObserver, WithPopulationObserver and WithLogger.
type RunOption struct {
	apply func(*runOptions) error
}

type runOptions struct {
	observer           ProgressObserver
	populationObserver PopulationObserver
	logger             Logger
	initialPositions   [][]float64
}

// WithInitialPopulation seeds the start of the swarm. The argument may contain
// fewer positions than the configured population; unfilled slots are
// initialized randomly. The positions are copied when this function is called
// and again when applied to a run.
func WithInitialPopulation(positions [][]float64) RunOption {
	snapshot := clonePositions(positions)

	return RunOption{apply: func(options *runOptions) error {
		options.initialPositions = clonePositions(snapshot)

		return nil
	}}
}

// WithProgressObserver registers an observer for iteration progress. Passing a
// nil observer disables progress reporting.
func WithProgressObserver(observer ProgressObserver) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.observer = observer

		return nil
	}}
}

// WithPopulationObserver registers an observer for the swarm. It is called
// once per completed iteration, after WithProgressObserver's observer. Passing
// a nil observer disables population reporting, which is the default: no
// copying happens unless an observer is registered.
func WithPopulationObserver(observer PopulationObserver) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.populationObserver = observer

		return nil
	}}
}

// WithLogger registers a structured logger for run lifecycle events. Passing
// nil disables logging. The logger receives optimization_started,
// iteration_completed, and optimization_completed events.
func WithLogger(logger Logger) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.logger = logger

		return nil
	}}
}

func resolveRunOptions(options []RunOption) (runOptions, error) {
	var resolved runOptions

	for i, option := range options {
		if option.apply == nil {
			return runOptions{}, fmt.Errorf("run option %d is invalid", i)
		}

		err := option.apply(&resolved)
		if err != nil {
			return runOptions{}, fmt.Errorf("apply run option %d: %w", i, err)
		}
	}

	return resolved, nil
}

func validateInitialPopulation(config *Config, options runOptions) error {
	if len(options.initialPositions) > config.NPop {
		return fmt.Errorf("initial population has %d positions, exceeds NPop=%d",
			len(options.initialPositions), config.NPop)
	}

	return validateInitialPositions(options.initialPositions, config)
}

func validateInitialPositions(positions [][]float64, config *Config) error {
	for i, position := range positions {
		if len(position) != config.ProblemSize {
			return fmt.Errorf("initial position %d has dimension %d, want %d",
				i, len(position), config.ProblemSize)
		}

		err := validateInitialComponents(i, position, config)
		if err != nil {
			return err
		}
	}

	return nil
}

func validateInitialComponents(index int, position []float64, config *Config) error {
	for dimension, value := range position {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("initial position %d dimension %d must be finite, got %v",
				index, dimension, value)
		}

		if value < config.LowerBound || value > config.UpperBound {
			return fmt.Errorf(
				"initial position %d dimension %d is outside bounds [%v, %v]: %v",
				index, dimension, config.LowerBound, config.UpperBound, value,
			)
		}
	}

	return nil
}

func clonePositions(positions [][]float64) [][]float64 {
	if positions == nil {
		return nil
	}

	cloned := make([][]float64, len(positions))
	for i, position := range positions {
		cloned[i] = copyVec(position)
	}

	return cloned
}

func cloneBest(best Best) Best {
	return Best{
		Position:            copyVec(best.Position),
		Cost:                best.Cost,
		ConstraintViolation: best.ConstraintViolation,
	}
}

// cloneDragonflies copies a swarm element by element, so an observer cannot
// reach back into the running optimizer through a shared slice header or a
// retained position vector.
func cloneDragonflies(swarm []Dragonfly) []Dragonfly {
	if swarm == nil {
		return nil
	}

	cloned := make([]Dragonfly, len(swarm))
	for i := range swarm {
		cloned[i] = cloneDragonfly(swarm[i])
	}

	return cloned
}

func cloneDragonfly(fly Dragonfly) Dragonfly {
	return Dragonfly{
		Position:            copyVec(fly.Position),
		Step:                copyVec(fly.Step),
		Cost:                fly.Cost,
		ConstraintViolation: fly.ConstraintViolation,
	}
}

func notifyProgress(observer ProgressObserver, iteration, evaluationCount int, best Best) {
	if observer == nil {
		return
	}

	observer(Progress{
		Best:            cloneBest(best),
		Iteration:       iteration,
		EvaluationCount: evaluationCount,
	})
}

func notifyPopulation(
	observer PopulationObserver,
	iteration, evaluationCount int,
	best, worst Best,
	swarm []Dragonfly,
) {
	if observer == nil {
		return
	}

	observer(PopulationSnapshot{
		Swarm:           cloneDragonflies(swarm),
		Best:            cloneBest(best),
		Worst:           cloneBest(worst),
		Iteration:       iteration,
		EvaluationCount: evaluationCount,
	})
}

var errNilContext = errors.New("context cannot be nil")

// requireContext rejects a nil context before a run begins, so a caller who
// passes one gets an error instead of a panic from the first cancellation
// check.
func requireContext(ctx context.Context) error {
	if ctx == nil {
		return errNilContext
	}

	return nil
}
