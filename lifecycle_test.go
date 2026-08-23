package dragonfly

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"math"
	"reflect"
	"strings"
	"testing"
)

func lifecycleConfig() *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 2
	config.LowerBound = -1
	config.UpperBound = 1
	config.MaxIterations = 3
	config.NPop = 4

	return config
}

func TestResolveRunOptionsWithoutOptions(t *testing.T) {
	resolved, err := resolveRunOptions(nil)
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if resolved.observer != nil || resolved.populationObserver != nil {
		t.Error("resolveRunOptions registered an observer without options")
	}

	if resolved.logger != nil {
		t.Error("resolveRunOptions registered a logger without options")
	}

	if resolved.initialPositions != nil {
		t.Errorf("initialPositions = %v, want nil", resolved.initialPositions)
	}
}

func TestResolveRunOptionsRejectsZeroValue(t *testing.T) {
	_, err := resolveRunOptions([]RunOption{WithLogger(nil), {}})
	if err == nil {
		t.Fatal("resolveRunOptions accepted a zero-value RunOption")
	}

	if !strings.Contains(err.Error(), "run option 1 is invalid") {
		t.Errorf("error %q does not identify the offending option", err)
	}
}

func TestResolveRunOptionsPropagatesApplyError(t *testing.T) {
	sentinel := errors.New("option failed")
	failing := RunOption{apply: func(*runOptions) error { return sentinel }}

	_, err := resolveRunOptions([]RunOption{WithLogger(nil), failing})
	if !errors.Is(err, sentinel) {
		t.Fatalf("error = %v, want %v wrapped", err, sentinel)
	}

	if !strings.Contains(err.Error(), "apply run option 1") {
		t.Errorf("error %q does not identify the offending option", err)
	}
}

func TestWithProgressObserverApplies(t *testing.T) {
	calls := 0

	resolved, err := resolveRunOptions([]RunOption{
		WithProgressObserver(func(Progress) { calls++ }),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if resolved.observer == nil {
		t.Fatal("progress observer was not registered")
	}

	resolved.observer(Progress{})

	if calls != 1 {
		t.Errorf("observer called %d times, want 1", calls)
	}
}

func TestWithPopulationObserverApplies(t *testing.T) {
	calls := 0

	resolved, err := resolveRunOptions([]RunOption{
		WithPopulationObserver(func(PopulationSnapshot) { calls++ }),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if resolved.populationObserver == nil {
		t.Fatal("population observer was not registered")
	}

	resolved.populationObserver(PopulationSnapshot{})

	if calls != 1 {
		t.Errorf("observer called %d times, want 1", calls)
	}
}

func TestWithObserversAcceptNil(t *testing.T) {
	resolved, err := resolveRunOptions([]RunOption{
		WithProgressObserver(nil),
		WithPopulationObserver(nil),
		WithLogger(nil),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if resolved.observer != nil || resolved.populationObserver != nil || resolved.logger != nil {
		t.Error("a nil argument must leave the corresponding hook disabled")
	}
}

func TestWithLoggerAcceptsSlog(t *testing.T) {
	var buffer bytes.Buffer

	logger := slog.New(slog.NewTextHandler(&buffer, nil))

	resolved, err := resolveRunOptions([]RunOption{WithLogger(logger)})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if resolved.logger == nil {
		t.Fatal("logger was not registered")
	}

	resolved.logger.Log(context.Background(), slog.LevelInfo, "optimization_started", "npop", 4)

	if !strings.Contains(buffer.String(), "optimization_started") {
		t.Errorf("logged output %q does not contain the event", buffer.String())
	}
}

func TestLastOptionWins(t *testing.T) {
	first := 0
	second := 0

	resolved, err := resolveRunOptions([]RunOption{
		WithProgressObserver(func(Progress) { first++ }),
		WithProgressObserver(func(Progress) { second++ }),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	resolved.observer(Progress{})

	if first != 0 || second != 1 {
		t.Errorf("observer calls = (%d, %d), want the last option to win", first, second)
	}
}

func TestWithInitialPopulationSnapshotsAtConstructionAndApply(t *testing.T) {
	positions := [][]float64{{0.25, -0.25}, {0, 0}}
	option := WithInitialPopulation(positions)

	// WithInitialPopulation promises a snapshot at option construction time.
	positions[0][0] = 99

	resolved, err := resolveRunOptions([]RunOption{option})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	want := [][]float64{{0.25, -0.25}, {0, 0}}
	if !reflect.DeepEqual(resolved.initialPositions, want) {
		t.Fatalf("initialPositions = %v, want %v", resolved.initialPositions, want)
	}

	// Applying the same option twice must not hand out the same backing array.
	resolved.initialPositions[0][0] = -1

	again, err := resolveRunOptions([]RunOption{option})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if !reflect.DeepEqual(again.initialPositions, want) {
		t.Errorf("second application = %v, want %v", again.initialPositions, want)
	}
}

func TestWithInitialPopulationAcceptsNil(t *testing.T) {
	resolved, err := resolveRunOptions([]RunOption{WithInitialPopulation(nil)})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	if resolved.initialPositions != nil {
		t.Errorf("initialPositions = %v, want nil", resolved.initialPositions)
	}

	err = validateInitialPopulation(lifecycleConfig(), resolved)
	if err != nil {
		t.Errorf("validateInitialPopulation(nil): %v", err)
	}
}

func TestValidateInitialPopulationAcceptsPartialSeed(t *testing.T) {
	config := lifecycleConfig()

	resolved, err := resolveRunOptions([]RunOption{
		WithInitialPopulation([][]float64{{-1, 1}, {0, 0.5}}),
	})
	if err != nil {
		t.Fatalf("resolveRunOptions: %v", err)
	}

	err = validateInitialPopulation(config, resolved)
	if err != nil {
		t.Errorf("validateInitialPopulation: %v", err)
	}
}

func TestValidateInitialPopulationRejectsInvalidSeeds(t *testing.T) {
	tests := []struct {
		name      string
		positions [][]float64
		wantErr   string
	}{
		{
			name:      "too many positions",
			positions: make([][]float64, 5),
			wantErr:   "exceeds NPop",
		},
		{
			name:      "dimension too small",
			positions: [][]float64{{0}},
			wantErr:   "dimension 1, want 2",
		},
		{
			name:      "dimension too large",
			positions: [][]float64{{0, 0, 0}},
			wantErr:   "dimension 3, want 2",
		},
		{
			name:      "not a number",
			positions: [][]float64{{math.NaN(), 0}},
			wantErr:   "must be finite",
		},
		{
			name:      "infinity",
			positions: [][]float64{{0, math.Inf(1)}},
			wantErr:   "must be finite",
		},
		{
			name:      "below lower bound",
			positions: [][]float64{{-1.01, 0}},
			wantErr:   "outside bounds",
		},
		{
			name:      "above upper bound",
			positions: [][]float64{{0, 1.01}},
			wantErr:   "outside bounds",
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := lifecycleConfig()

			resolved, err := resolveRunOptions([]RunOption{
				WithInitialPopulation(testCase.positions),
			})
			if err != nil {
				t.Fatalf("resolveRunOptions: %v", err)
			}

			err = validateInitialPopulation(config, resolved)
			if err == nil {
				t.Fatal("validateInitialPopulation accepted an invalid seed")
			}

			if !strings.Contains(err.Error(), testCase.wantErr) {
				t.Errorf("error %q does not contain %q", err, testCase.wantErr)
			}
		})
	}
}

func TestNotifyProgressHandsOutIndependentCopies(t *testing.T) {
	var updates []Progress

	best := Best{Position: []float64{1, 2}, Cost: 3, ConstraintViolation: 4}

	notifyProgress(func(update Progress) {
		updates = append(updates, update)
		update.Best.Position[0] = 999
	}, 2, 16, best)

	if len(updates) != 1 {
		t.Fatalf("observer called %d times, want 1", len(updates))
	}

	if best.Position[0] != 1 {
		t.Errorf("observer mutation reached the source: Position[0] = %v", best.Position[0])
	}

	update := updates[0]
	if update.Iteration != 2 || update.EvaluationCount != 16 {
		t.Errorf("Progress = (%d, %d), want (2, 16)", update.Iteration, update.EvaluationCount)
	}

	if update.Best.Cost != 3 || update.Best.ConstraintViolation != 4 {
		t.Errorf("Best = %+v, want cost 3 and violation 4", update.Best)
	}
}

func TestNotifyProgressWithoutObserverIsANoOp(t *testing.T) {
	best := Best{Position: []float64{1, 2}}

	notifyProgress(nil, 1, 1, best)

	if best.Position[0] != 1 {
		t.Error("notifyProgress(nil, ...) modified the source best")
	}
}

func TestNotifyPopulationHandsOutIndependentCopies(t *testing.T) {
	var snapshots []PopulationSnapshot

	swarm := []Dragonfly{
		{Position: []float64{1, 2}, Step: []float64{0.1, 0.2}, Cost: 5},
		{Position: []float64{3, 4}, Step: []float64{0.3, 0.4}, Cost: 25},
	}
	best := Best{Position: []float64{1, 2}, Cost: 5}
	worst := Best{Position: []float64{3, 4}, Cost: 25}

	notifyPopulation(func(snapshot PopulationSnapshot) {
		snapshots = append(snapshots, snapshot)

		// Reach into every copy the observer was handed. None of it may be
		// aliased to state the optimizer is still using.
		snapshot.Best.Position[0] = 999
		snapshot.Worst.Position[0] = 999
		snapshot.Swarm[0].Position[0] = 999
		snapshot.Swarm[0].Step[0] = 999
	}, 3, 24, best, worst, swarm)

	if len(snapshots) != 1 {
		t.Fatalf("observer called %d times, want 1", len(snapshots))
	}

	if best.Position[0] != 1 || worst.Position[0] != 3 {
		t.Errorf("observer mutation reached food or enemy: %v %v", best.Position, worst.Position)
	}

	if swarm[0].Position[0] != 1 || swarm[0].Step[0] != 0.1 {
		t.Errorf("observer mutation reached the swarm: %+v", swarm[0])
	}

	snapshot := snapshots[0]
	if snapshot.Iteration != 3 || snapshot.EvaluationCount != 24 {
		t.Errorf("snapshot = (%d, %d), want (3, 24)", snapshot.Iteration, snapshot.EvaluationCount)
	}

	if len(snapshot.Swarm) != len(swarm) {
		t.Fatalf("snapshot holds %d dragonflies, want %d", len(snapshot.Swarm), len(swarm))
	}

	if snapshot.Swarm[1].Cost != 25 || snapshot.Swarm[1].Position[1] != 4 {
		t.Errorf("snapshot.Swarm[1] = %+v, want a faithful copy", snapshot.Swarm[1])
	}
}

func TestNotifyPopulationWithoutObserverIsANoOp(t *testing.T) {
	swarm := []Dragonfly{{Position: []float64{1, 2}, Step: []float64{3, 4}}}

	notifyPopulation(nil, 1, 1, Best{}, Best{}, swarm)

	if swarm[0].Position[0] != 1 {
		t.Error("notifyPopulation(nil, ...) modified the source swarm")
	}
}

func TestObserversRunSynchronouslyOnTheCallingGoroutine(t *testing.T) {
	// A synchronous observer has finished before notify* returns, so a plain
	// counter -- no mutex, no channel -- is enough to observe the call.
	progressCalls := 0
	populationCalls := 0

	notifyProgress(func(Progress) { progressCalls++ }, 1, 1, Best{})

	if progressCalls != 1 {
		t.Errorf("progress observer had not run when notifyProgress returned")
	}

	notifyPopulation(func(PopulationSnapshot) { populationCalls++ }, 1, 1, Best{}, Best{}, nil)

	if populationCalls != 1 {
		t.Errorf("population observer had not run when notifyPopulation returned")
	}
}

func TestCloneHelpersProduceIndependentValues(t *testing.T) {
	t.Run("clonePositions", func(t *testing.T) {
		if clonePositions(nil) != nil {
			t.Error("clonePositions(nil) is not nil")
		}

		source := [][]float64{{1, 2}}
		cloned := clonePositions(source)
		cloned[0][0] = 99

		if source[0][0] != 1 {
			t.Error("clonePositions shares its backing array")
		}
	})

	t.Run("cloneBest", func(t *testing.T) {
		source := Best{Position: []float64{1, 2}, Cost: 3}
		cloned := cloneBest(source)
		cloned.Position[0] = 99

		if source.Position[0] != 1 {
			t.Error("cloneBest shares its position vector")
		}

		if cloned.Cost != 3 {
			t.Errorf("cloneBest lost the cost: %v", cloned.Cost)
		}
	})

	t.Run("cloneDragonflies", func(t *testing.T) {
		if cloneDragonflies(nil) != nil {
			t.Error("cloneDragonflies(nil) is not nil")
		}

		source := []Dragonfly{{
			Position:            []float64{1, 2},
			Step:                []float64{3, 4},
			Cost:                5,
			ConstraintViolation: 6,
		}}
		cloned := cloneDragonflies(source)
		cloned[0].Position[0] = 99
		cloned[0].Step[0] = 99

		if source[0].Position[0] != 1 || source[0].Step[0] != 3 {
			t.Error("cloneDragonflies shares its vectors")
		}

		if cloned[0].Cost != 5 || cloned[0].ConstraintViolation != 6 {
			t.Errorf("cloneDragonflies lost scalar state: %+v", cloned[0])
		}
	})
}

func TestRequireContext(t *testing.T) {
	// A variable, not a literal nil, because passing a nil context literal is
	// exactly what static analysis exists to stop -- and what requireContext
	// exists to survive.
	var missing context.Context

	err := requireContext(missing)
	if !errors.Is(err, errNilContext) {
		t.Errorf("requireContext(nil) = %v, want %v", err, errNilContext)
	}

	err = requireContext(context.Background())
	if err != nil {
		t.Errorf("requireContext(context.Background()) = %v, want nil", err)
	}
}
