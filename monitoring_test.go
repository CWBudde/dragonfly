package dragonfly

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// decodeLogEvents reads one JSON object per line, the shape
// slog.NewJSONHandler writes.
func decodeLogEvents(t *testing.T, output *bytes.Buffer) []map[string]any {
	t.Helper()

	var events []map[string]any

	scanner := bufio.NewScanner(output)
	for scanner.Scan() {
		var event map[string]any

		err := json.Unmarshal(scanner.Bytes(), &event)
		if err != nil {
			t.Fatalf("decode log event: %v", err)
		}

		events = append(events, event)
	}

	err := scanner.Err()
	if err != nil {
		t.Fatalf("scan log output: %v", err)
	}

	return events
}

func monitoringResult() *Result {
	return &Result{
		ConvergenceCurve:  []float64{3, 2.5, 1},
		TerminationReason: TerminationTargetCost,
		GlobalBest: Best{
			Position:            []float64{0.5, -0.25},
			Cost:                1,
			ConstraintViolation: 0,
		},
		Worst: Best{
			Position:            []float64{-4, 4},
			Cost:                32,
			ConstraintViolation: 0.75,
		},
		FuncEvalCount:  120,
		IterationCount: 3,
		Seed:           42,
		SeedKnown:      true,
	}
}

func TestStructuredLoggerReceivesLifecycleEvents(t *testing.T) {
	var output bytes.Buffer

	logger := slog.New(slog.NewJSONHandler(&output, nil))
	ctx := context.Background()
	result := monitoringResult()

	config := NewDefaultConfig()
	config.ProblemSize = 2
	config.MaxIterations = 3
	config.NPop = 7

	logOptimizationStarted(ctx, logger, config)
	logIterationCompleted(ctx, logger, 1, 40, result.GlobalBest)
	logOptimizationCompleted(ctx, logger, result)

	events := decodeLogEvents(t, &output)

	if len(events) != 3 {
		t.Fatalf("received %d log events, want start, iteration, and completion", len(events))
	}

	wantEvents := []string{
		eventOptimizationStarted,
		eventIterationCompleted,
		eventOptimizationCompleted,
	}
	for i, want := range wantEvents {
		if events[i]["event"] != want {
			t.Errorf("event %d name = %v, want %q", i, events[i]["event"], want)
		}
	}

	if events[0]["problem_size"] != float64(config.ProblemSize) {
		t.Errorf("start problem_size = %v, want %d", events[0]["problem_size"], config.ProblemSize)
	}

	if events[0]["population"] != float64(config.NPop) {
		t.Errorf("start population = %v, want %d", events[0]["population"], config.NPop)
	}

	if events[1]["iteration"] != float64(1) {
		t.Errorf("iteration event iteration = %v, want 1", events[1]["iteration"])
	}

	if events[1]["best_cost"] != result.GlobalBest.Cost {
		t.Errorf("iteration best_cost = %v, want %v", events[1]["best_cost"], result.GlobalBest.Cost)
	}

	if events[2]["termination_reason"] != string(TerminationTargetCost) {
		t.Errorf("completion termination_reason = %v, want %q",
			events[2]["termination_reason"], TerminationTargetCost)
	}

	if events[2]["evaluations"] != float64(result.FuncEvalCount) {
		t.Errorf("completion evaluations = %v, want %d",
			events[2]["evaluations"], result.FuncEvalCount)
	}

	// The enemy is DA-specific and is reported alongside the food source.
	if events[2]["worst_cost"] != result.Worst.Cost {
		t.Errorf("completion worst_cost = %v, want %v", events[2]["worst_cost"], result.Worst.Cost)
	}
}

func TestNilLoggerDisablesLogging(t *testing.T) {
	ctx := context.Background()
	result := monitoringResult()

	config := NewDefaultConfig()
	config.ProblemSize = 2

	// A nil Logger is the documented way to switch logging off: every event
	// must be a no-op rather than a nil dereference.
	logOptimizationStarted(ctx, nil, config)
	logIterationCompleted(ctx, nil, 1, 40, result.GlobalBest)
	logOptimizationCompleted(ctx, nil, result)
}

func TestResultExportsConvergenceCSV(t *testing.T) {
	result := monitoringResult()
	path := filepath.Join(t.TempDir(), "convergence.csv")

	err := result.ExportConvergenceCSV(path)
	if err != nil {
		t.Fatalf("ExportConvergenceCSV: %v", err)
	}

	records := readCSV(t, path)

	wantRecords := [][]string{
		{"iteration", "best_cost"},
		{"1", "3"},
		{"2", "2.5"},
		{"3", "1"},
	}
	if !reflect.DeepEqual(records, wantRecords) {
		t.Errorf("CSV records = %v, want %v", records, wantRecords)
	}

	if len(records)-1 != len(result.ConvergenceCurve) {
		t.Errorf("CSV has %d data rows, want %d",
			len(records)-1, len(result.ConvergenceCurve))
	}
}

func TestExportConvergenceCSVWritesHeaderOnlyForEmptyCurve(t *testing.T) {
	result := &Result{}
	path := filepath.Join(t.TempDir(), "empty.csv")

	err := result.ExportConvergenceCSV(path)
	if err != nil {
		t.Fatalf("ExportConvergenceCSV on an empty curve: %v", err)
	}

	records := readCSV(t, path)

	wantRecords := [][]string{{"iteration", "best_cost"}}
	if !reflect.DeepEqual(records, wantRecords) {
		t.Errorf("CSV records = %v, want a header row only", records)
	}
}

func TestResultExportsConvergenceJSON(t *testing.T) {
	result := monitoringResult()
	path := filepath.Join(t.TempDir(), "convergence.json")

	err := result.ExportConvergenceJSON(path)
	if err != nil {
		t.Fatalf("ExportConvergenceJSON: %v", err)
	}

	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read convergence JSON: %v", err)
	}

	var document ConvergenceExport

	err = json.Unmarshal(contents, &document)
	if err != nil {
		t.Fatalf("decode convergence JSON: %v", err)
	}

	want := ConvergenceExport{
		BestPosition:  result.GlobalBest.Position,
		WorstPosition: result.Worst.Position,
		Convergence: []ConvergencePoint{
			{Iteration: 1, BestCost: 3},
			{Iteration: 2, BestCost: 2.5},
			{Iteration: 3, BestCost: 1},
		},
		TerminationReason:        TerminationTargetCost,
		BestCost:                 result.GlobalBest.Cost,
		BestConstraintViolation:  result.GlobalBest.ConstraintViolation,
		WorstCost:                result.Worst.Cost,
		WorstConstraintViolation: result.Worst.ConstraintViolation,
		Seed:                     result.Seed,
		FuncEvalCount:            result.FuncEvalCount,
		IterationCount:           result.IterationCount,
		SeedKnown:                result.SeedKnown,
	}
	if !reflect.DeepEqual(document, want) {
		t.Errorf("JSON document = %+v, want %+v", document, want)
	}
}

func TestExportConvergenceRejectsUnwritablePath(t *testing.T) {
	result := monitoringResult()
	// A path below a directory that does not exist: os.Create fails, and the
	// exporters must report that rather than panic.
	path := filepath.Join(t.TempDir(), "missing-directory", "curve")

	err := result.ExportConvergenceCSV(path + ".csv")
	if err == nil {
		t.Error("ExportConvergenceCSV to an unwritable path returned no error")
	}

	err = result.ExportConvergenceJSON(path + ".json")
	if err == nil {
		t.Error("ExportConvergenceJSON to an unwritable path returned no error")
	}
}

func TestNilResultCannotExportConvergence(t *testing.T) {
	var result *Result

	err := result.ExportConvergenceCSV(filepath.Join(t.TempDir(), "curve.csv"))
	if !errors.Is(err, errNilResult) {
		t.Errorf("ExportConvergenceCSV error = %v, want %v", err, errNilResult)
	}

	err = result.ExportConvergenceJSON(filepath.Join(t.TempDir(), "curve.json"))
	if !errors.Is(err, errNilResult) {
		t.Errorf("ExportConvergenceJSON error = %v, want %v", err, errNilResult)
	}
}

func TestWriteExportFilePreservesExistingFileOnFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.json")

	original := []byte("previous complete report\n")

	writeErr := os.WriteFile(path, original, 0o600)
	if writeErr != nil {
		t.Fatalf("write original report: %v", writeErr)
	}

	wantErr := errors.New("encoding failed")

	err := writeExportFile(path, "test report", func(writer io.Writer) error {
		_, writeErr := writer.Write([]byte("partial replacement"))
		if writeErr != nil {
			return writeErr
		}

		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("writeExportFile() error = %v, want %v", err, wantErr)
	}

	contents, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("read preserved report: %v", readErr)
	}

	if !bytes.Equal(contents, original) {
		t.Fatalf("existing report = %q, want %q", contents, original)
	}
}

func readCSV(t *testing.T, path string) [][]string {
	t.Helper()

	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read convergence CSV: %v", err)
	}

	records, err := csv.NewReader(bytes.NewReader(contents)).ReadAll()
	if err != nil {
		t.Fatalf("decode convergence CSV: %v", err)
	}

	return records
}
