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

// ArchiveSnapshot is the Pareto archive of a multi-objective run after a
// completed iteration. Iteration is one-based. Every solution is a deep copy
// and both bound vectors are copies: observers may retain or modify them
// without affecting the optimizer.
//
// It is the multi-objective counterpart of PopulationSnapshot, and it is
// separate from Progress for a stronger reason than PopulationSnapshot is: a
// MODA run has no single incumbent, so there is no best cost for Progress to
// carry. The front itself is the result, and watching it fill in -- animating
// it, measuring its spread, seeing whether it is still growing -- is the only
// reason to observe such a run at all.
//
// GridLower and GridUpper are the archive's per-objective extent, which is
// also the extent of the hypercube grid, and NGrid is the number of bins along
// each objective. Together they are the frame each solution's GridIndex is
// expressed in, and they move as the archive does. All three are zero for an
// empty archive.
type ArchiveSnapshot struct {
	Solutions       []*ParetoSolution
	GridLower       []float64
	GridUpper       []float64
	Iteration       int
	EvaluationCount int
	NGrid           int
}

// ArchiveObserver receives the Pareto archive after each completed iteration.
// OptimizeMultiObjective invokes observers synchronously on the calling
// goroutine.
type ArchiveObserver func(ArchiveSnapshot)

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
	archiveObserver    ArchiveObserver
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

// WithArchiveObserver registers an observer for the Pareto archive of a
// multi-objective run. It is called once per completed iteration with a deep
// copy of the archive. Passing nil disables archive reporting, which is the
// default: nothing is copied unless an observer is registered.
//
// It is meaningful only for OptimizeMultiObjective. A single-objective run has
// no archive and rejects this option rather than accepting it and then
// reporting nothing.
func WithArchiveObserver(observer ArchiveObserver) RunOption {
	return RunOption{apply: func(options *runOptions) error {
		options.archiveObserver = observer

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

// cloneParetoSolutions copies an archive element by element, so that an
// observer cannot reach back into the running optimizer through the archive's
// backing array or through a retained position vector. The array matters as
// much as the vectors here: ParetoArchive.Add compacts the survivors in place,
// so a shared slice would keep changing after the observer was handed it.
//
// ParetoSolution.clone drops the sorting bookkeeping, which is scratch space
// recomputed wherever it is used and means nothing outside that pass.
func cloneParetoSolutions(solutions []*ParetoSolution) []*ParetoSolution {
	if solutions == nil {
		return nil
	}

	cloned := make([]*ParetoSolution, len(solutions))
	for i, solution := range solutions {
		cloned[i] = solution.clone()
	}

	return cloned
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

// notifyArchive hands an observer a deep copy of the archive after a completed
// iteration. A nil observer costs nothing: no copy is made unless someone is
// watching.
func notifyArchive(
	observer ArchiveObserver,
	iteration, evaluationCount int,
	archive *ParetoArchive,
) {
	if observer == nil {
		return
	}

	lower, upper := archive.GridBounds()

	snapshot := ArchiveSnapshot{
		GridLower:       lower,
		GridUpper:       upper,
		Iteration:       iteration,
		EvaluationCount: evaluationCount,
	}

	if archive != nil {
		snapshot.Solutions = cloneParetoSolutions(archive.Solutions)
		snapshot.NGrid = archive.NGrid
	}

	observer(snapshot)
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

// validateSingleObjectiveRunOptions rejects the run options that have no
// single-objective reading.
//
// Only WithArchiveObserver qualifies today. Accepting it and never calling it
// would be exactly the silent no-op that the multi-objective side of this
// rule, validateMultiObjectiveRunOptions, exists to refuse: a caller who
// registers an observer is waiting for something.
func validateSingleObjectiveRunOptions(options runOptions) error {
	if options.archiveObserver != nil {
		return errors.New(
			"WithArchiveObserver has no meaning for a single-objective run: " +
				"there is no Pareto archive; use WithPopulationObserver")
	}

	return nil
}
