// Structured lifecycle logging and convergence export for the Dragonfly Algorithm.

package dragonfly

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"
)

const (
	eventOptimizationStarted   = "optimization_started"
	eventIterationCompleted    = "iteration_completed"
	eventOptimizationCompleted = "optimization_completed"
)

var errNilResult = errors.New("result cannot be nil")

// ConvergencePoint is one exported sample from a convergence curve. Iteration
// is one-based and BestCost is the best cost known after that iteration.
type ConvergencePoint struct {
	Iteration int     `json:"iteration"`
	BestCost  float64 `json:"best_cost"`
}

// ConvergenceExport is the document ExportConvergenceJSON writes: the
// convergence curve plus the run-level summary it belongs to.
//
// The summary carries the enemy alongside the food source. The enemy is a
// property of the whole run rather than of any one iteration, so it has no
// column in the CSV export, but a reader of the JSON document wants both ends
// of the range the swarm searched.
type ConvergenceExport struct {
	TerminationReason        TerminationReason  `json:"termination_reason,omitempty"`
	BestPosition             []float64          `json:"best_position,omitempty"`
	WorstPosition            []float64          `json:"worst_position,omitempty"`
	Convergence              []ConvergencePoint `json:"convergence"`
	BestCost                 float64            `json:"best_cost"`
	BestConstraintViolation  float64            `json:"best_constraint_violation"`
	WorstCost                float64            `json:"worst_cost"`
	WorstConstraintViolation float64            `json:"worst_constraint_violation"`
	Seed                     int64              `json:"seed"`
	FuncEvalCount            int                `json:"func_eval_count"`
	IterationCount           int                `json:"iteration_count"`
}

func logOptimizationStarted(ctx context.Context, logger Logger, config *Config) {
	if logger == nil {
		return
	}

	logger.Log(
		ctx,
		slog.LevelInfo,
		"optimization started",
		"event", eventOptimizationStarted,
		"problem_size", config.ProblemSize,
		"max_iterations", config.MaxIterations,
		"population", config.NPop,
		"parallel", config.EnableParallel,
	)
}

func logIterationCompleted(
	ctx context.Context,
	logger Logger,
	iteration, evaluationCount int,
	best Best,
) {
	if logger == nil {
		return
	}

	logger.Log(
		ctx,
		slog.LevelInfo,
		"optimization iteration completed",
		"event", eventIterationCompleted,
		"iteration", iteration,
		"evaluations", evaluationCount,
		"best_cost", best.Cost,
		"constraint_violation", best.ConstraintViolation,
	)
}

func logOptimizationCompleted(ctx context.Context, logger Logger, result *Result) {
	if logger == nil {
		return
	}

	logger.Log(
		ctx,
		slog.LevelInfo,
		"optimization completed",
		"event", eventOptimizationCompleted,
		"iterations", result.IterationCount,
		"evaluations", result.FuncEvalCount,
		"best_cost", result.GlobalBest.Cost,
		"constraint_violation", result.GlobalBest.ConstraintViolation,
		"worst_cost", result.Worst.Cost,
		"termination_reason", result.TerminationReason,
	)
}

func (result *Result) convergencePoints() ([]ConvergencePoint, error) {
	if result == nil {
		return nil, errNilResult
	}

	points := make([]ConvergencePoint, len(result.ConvergenceCurve))
	for i, cost := range result.ConvergenceCurve {
		points[i] = ConvergencePoint{Iteration: i + 1, BestCost: cost}
	}

	return points, nil
}

// writeExportFile creates path, hands the open file to write, and closes it,
// reporting a close failure that would otherwise hide a short write. The
// what argument names the document in every error it wraps.
func writeExportFile(path, what string, write func(io.Writer) error) (returnErr error) {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", what, err)
	}

	defer func() {
		closeErr := file.Close()
		if returnErr == nil && closeErr != nil {
			returnErr = fmt.Errorf("close %s: %w", what, closeErr)
		}
	}()

	return write(file)
}

// ExportConvergenceCSV writes the convergence curve to path as iteration and
// best_cost columns, with one row per completed iteration. An empty curve
// yields a header-only file rather than an error.
//
// The enemy, Result.Worst, is a single run-level value and has no per-iteration
// column here; ExportConvergenceJSON reports it.
func (result *Result) ExportConvergenceCSV(path string) error {
	points, err := result.convergencePoints()
	if err != nil {
		return err
	}

	return writeExportFile(path, "convergence CSV", func(sink io.Writer) error {
		writer := csv.NewWriter(sink)

		err := writer.Write([]string{"iteration", "best_cost"})
		if err != nil {
			return fmt.Errorf("write convergence CSV header: %w", err)
		}

		for _, point := range points {
			err = writer.Write([]string{
				strconv.Itoa(point.Iteration),
				strconv.FormatFloat(point.BestCost, 'g', -1, 64),
			})
			if err != nil {
				return fmt.Errorf("write convergence CSV row: %w", err)
			}
		}

		writer.Flush()

		err = writer.Error()
		if err != nil {
			return fmt.Errorf("flush convergence CSV: %w", err)
		}

		return nil
	})
}

// ExportConvergenceJSON writes the convergence curve and the run summary to
// path as an indented ConvergenceExport document.
func (result *Result) ExportConvergenceJSON(path string) error {
	points, err := result.convergencePoints()
	if err != nil {
		return err
	}

	document := ConvergenceExport{
		BestPosition:             result.GlobalBest.Position,
		WorstPosition:            result.Worst.Position,
		Convergence:              points,
		TerminationReason:        result.TerminationReason,
		BestCost:                 result.GlobalBest.Cost,
		BestConstraintViolation:  result.GlobalBest.ConstraintViolation,
		WorstCost:                result.Worst.Cost,
		WorstConstraintViolation: result.Worst.ConstraintViolation,
		Seed:                     result.Seed,
		FuncEvalCount:            result.FuncEvalCount,
		IterationCount:           result.IterationCount,
	}

	return writeExportFile(path, "convergence JSON", func(sink io.Writer) error {
		encoder := json.NewEncoder(sink)
		encoder.SetIndent("", "  ")

		err := encoder.Encode(document)
		if err != nil {
			return fmt.Errorf("encode convergence JSON: %w", err)
		}

		return nil
	})
}
