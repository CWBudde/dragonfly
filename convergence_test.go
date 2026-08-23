package dragonfly

import (
	"math/rand"
	"testing"
)

// observation is one iteration's incumbent, fed to the tracker in order.
type observation struct {
	iteration  int
	best       Best
	wantReason TerminationReason
	wantStop   bool
}

// runObservations feeds a scripted sequence of iterations to the tracker and
// checks each verdict. It stops at the first mismatch to keep the failure
// message pinned to the iteration that produced it.
func runObservations(t *testing.T, tracker *convergenceTracker, steps []observation) {
	t.Helper()

	for _, step := range steps {
		reason, stop := tracker.observe(step.iteration, step.best)
		if stop != step.wantStop || reason != step.wantReason {
			t.Fatalf("observe(%d, %+v) = (%q, %t), want (%q, %t)",
				step.iteration, step.best, reason, stop, step.wantReason, step.wantStop)
		}
	}
}

func targetCost(value float64) *float64 {
	return &value
}

func TestConvergenceTrackerTargetCost(t *testing.T) {
	tests := []struct {
		name   string
		config *ConvergenceConfig
		steps  []observation
	}{
		{
			name:   "reached",
			config: &ConvergenceConfig{TargetCost: targetCost(1e-3)},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 5}},
				{iteration: 2, best: Best{Cost: 1e-3}, wantReason: TerminationTargetCost, wantStop: true},
			},
		},
		{
			name:   "not reached",
			config: &ConvergenceConfig{TargetCost: targetCost(1e-3)},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 5}},
				{iteration: 2, best: Best{Cost: 2e-3}},
				{iteration: 3, best: Best{Cost: 1.5e-3}},
			},
		},
		{
			name:   "zero target is a real target, not a disabled one",
			config: &ConvergenceConfig{TargetCost: targetCost(0)},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 0}, wantReason: TerminationTargetCost, wantStop: true},
			},
		},
		{
			name:   "nil target never stops",
			config: &ConvergenceConfig{},
			steps: []observation{
				{iteration: 1, best: Best{Cost: -1e9}},
				{iteration: 2, best: Best{Cost: -1e9}},
			},
		},
		{
			name:   "infeasible incumbent does not stop",
			config: &ConvergenceConfig{TargetCost: targetCost(1)},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 0.5, ConstraintViolation: 2}},
				{iteration: 2, best: Best{Cost: 0.1, ConstraintViolation: 1}},
				// Feasibility restored: now the target may fire.
				{
					iteration:  3,
					best:       Best{Cost: 0.9},
					wantReason: TerminationTargetCost,
					wantStop:   true,
				},
			},
		},
		{
			name: "gated by minimum iterations",
			config: &ConvergenceConfig{
				TargetCost:    targetCost(1),
				MinIterations: 3,
			},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 0}},
				{iteration: 2, best: Best{Cost: 0}},
				{iteration: 3, best: Best{Cost: 0}, wantReason: TerminationTargetCost, wantStop: true},
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			tracker := newConvergenceTracker(testCase.config, Best{Cost: 100})
			runObservations(t, tracker, testCase.steps)
		})
	}
}

func TestConvergenceTrackerStagnation(t *testing.T) {
	tests := []struct {
		name   string
		config *ConvergenceConfig
		steps  []observation
	}{
		{
			name:   "counts consecutive insufficient improvements",
			config: &ConvergenceConfig{MinImprovement: 0.5, StagnationIterations: 2},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 9.8}},
				{iteration: 2, best: Best{Cost: 9.4}},
				{iteration: 3, best: Best{Cost: 9.1}},
				{
					iteration:  4,
					best:       Best{Cost: 9.0},
					wantReason: TerminationStagnation,
					wantStop:   true,
				},
			},
		},
		{
			name:   "sufficient improvement resets the counter",
			config: &ConvergenceConfig{MinImprovement: 0.5, StagnationIterations: 2},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 9.9}},
				// Beats the reference of 10 by more than 0.5, so the single
				// stagnant iteration above is forgotten.
				{iteration: 2, best: Best{Cost: 9.0}},
				{iteration: 3, best: Best{Cost: 8.9}},
				{
					iteration:  4,
					best:       Best{Cost: 8.8},
					wantReason: TerminationStagnation,
					wantStop:   true,
				},
			},
		},
		{
			name:   "zero window disables stagnation",
			config: &ConvergenceConfig{MinImprovement: 0.5, StagnationIterations: 0},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 10}},
				{iteration: 2, best: Best{Cost: 10}},
				{iteration: 3, best: Best{Cost: 10}},
				{iteration: 4, best: Best{Cost: 10}},
			},
		},
		{
			name: "gated by minimum iterations",
			config: &ConvergenceConfig{
				MinImprovement:       0.5,
				StagnationIterations: 1,
				MinIterations:        3,
			},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 10}},
				{iteration: 2, best: Best{Cost: 10}},
				{
					iteration:  3,
					best:       Best{Cost: 10},
					wantReason: TerminationStagnation,
					wantStop:   true,
				},
			},
		},
		{
			name:   "a worse incumbent is stagnation, not improvement",
			config: &ConvergenceConfig{StagnationIterations: 2},
			steps: []observation{
				{iteration: 1, best: Best{Cost: 11}},
				{
					iteration:  2,
					best:       Best{Cost: 12},
					wantReason: TerminationStagnation,
					wantStop:   true,
				},
			},
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			tracker := newConvergenceTracker(testCase.config, Best{Cost: 10})
			runObservations(t, tracker, testCase.steps)
		})
	}
}

func TestConvergenceTrackerNilConfigNeverStops(t *testing.T) {
	tracker := newConvergenceTracker(nil, Best{Cost: 10})

	for iteration := 1; iteration <= 5; iteration++ {
		reason, stop := tracker.observe(iteration, Best{Cost: 10})
		if stop {
			t.Fatalf("observe(%d) stopped a run with no convergence config, reason %q",
				iteration, reason)
		}
	}
}

// TestOptimizeWithoutConvergenceConfigRunsToMaxIterations checks the end the
// tracker's nil-config verdict implies: a run with no convergence block uses
// its whole iteration budget and reports the iteration cap as the reason.
func TestOptimizeWithoutConvergenceConfigRunsToMaxIterations(t *testing.T) {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 3
	config.LowerBound = -5
	config.UpperBound = 5
	config.NPop = 10
	config.MaxIterations = 20
	config.Convergence = nil
	config.Rand = rand.New(rand.NewSource(42))

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if result.TerminationReason != TerminationMaxIterations {
		t.Fatalf("termination reason = %q, want %q",
			result.TerminationReason, TerminationMaxIterations)
	}

	if result.IterationCount != config.MaxIterations {
		t.Fatalf("iteration count = %d, want %d", result.IterationCount, config.MaxIterations)
	}
}

func TestConvergenceTrackerHonoursConstraintPolicy(t *testing.T) {
	feasibility := &ConstraintConfig{Handling: ConstraintHandlingFeasibility}
	penalty := &ConstraintConfig{
		Handling:      ConstraintHandlingPenalty,
		PenaltyMethod: PenaltyLinear,
		PenaltyFactor: 10,
	}

	tests := []struct {
		name        string
		constraints *ConstraintConfig
		reference   Best
		candidate   Best
		want        bool
	}{
		{
			name:        "feasibility gain outweighs a worse cost",
			constraints: feasibility,
			reference:   Best{Cost: 1, ConstraintViolation: 2},
			candidate:   Best{Cost: 5},
			want:        true,
		},
		{
			name:        "violation reduction counts while infeasible",
			constraints: feasibility,
			reference:   Best{Cost: 1, ConstraintViolation: 2},
			candidate:   Best{Cost: 1, ConstraintViolation: 0.5},
			want:        true,
		},
		{
			name:        "losing feasibility is never an improvement",
			constraints: feasibility,
			reference:   Best{Cost: 5},
			candidate:   Best{Cost: 1, ConstraintViolation: 2},
			want:        false,
		},
		{
			name:        "penalty handling scores cost plus violation",
			constraints: penalty,
			reference:   Best{Cost: 1},
			candidate:   Best{Cost: 0.5, ConstraintViolation: 1},
			want:        false,
		},
		{
			name:        "penalty handling accepts a lower penalized score",
			constraints: penalty,
			reference:   Best{Cost: 20, ConstraintViolation: 1},
			candidate:   Best{Cost: 5},
			want:        true,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			tracker := newConvergenceTracker(
				&ConvergenceConfig{MinImprovement: 0.1, StagnationIterations: 1},
				testCase.reference,
				newConstraintEvaluator(Sphere, testCase.constraints),
			)

			got := tracker.significantlyImproved(evaluationFromBest(testCase.candidate))
			if got != testCase.want {
				t.Fatalf("significantlyImproved(%+v) = %t, want %t",
					testCase.candidate, got, testCase.want)
			}
		})
	}
}
