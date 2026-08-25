package dragonfly

import (
	"math"
	"strings"
	"testing"
)

// atLeast returns an inequality constraint g(x) = threshold - x[0], violated
// while the first component sits below threshold.
func atLeast(threshold float64) ConstraintFunction {
	return func(position []float64) float64 { return threshold - position[0] }
}

// equals returns an equality constraint h(x) = x[index] - target.
func equals(index int, target float64) ConstraintFunction {
	return func(position []float64) float64 { return position[index] - target }
}

func TestEvaluateConstraintsAggregatesViolations(t *testing.T) {
	config := &ConstraintConfig{
		Inequalities: []ConstraintFunction{
			func(position []float64) float64 { return position[0] - 2 },
			func(position []float64) float64 { return -position[0] - 1 },
		},
		Equalities:        []ConstraintFunction{equals(1, 3)},
		EqualityTolerance: 0.25,
	}

	tests := []struct {
		name      string
		position  []float64
		violation float64
		feasible  bool
	}{
		{
			name:     "inequality and equality both violated",
			position: []float64{3, 3.5},
			// (3-2) + max(0, -4) + (|0.5| - 0.25) = 1 + 0 + 0.25
			violation: 1.25,
		},
		{
			name:      "inside the box and within tolerance",
			position:  []float64{1, 3.1},
			violation: 0,
			feasible:  true,
		},
		{
			name:      "equality violated beyond tolerance only",
			position:  []float64{1, 4},
			violation: 0.75,
		},
		{
			name:      "equality exactly at the tolerance edge",
			position:  []float64{1, 3.25},
			violation: 0,
			feasible:  true,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			result := EvaluateConstraints(testCase.position, config)
			if math.Abs(result.Violation-testCase.violation) > 1e-12 {
				t.Errorf("Violation = %v, want %v", result.Violation, testCase.violation)
			}

			if result.Feasible != testCase.feasible {
				t.Errorf("Feasible = %t, want %t", result.Feasible, testCase.feasible)
			}
		})
	}
}

func TestEvaluateConstraintsWithoutAConfigIsUnconstrained(t *testing.T) {
	result := EvaluateConstraints([]float64{1e9, -1e9}, nil)
	if !result.Feasible || result.Violation != 0 {
		t.Errorf("EvaluateConstraints(nil config) = %+v, want feasible with zero violation", result)
	}

	empty := EvaluateConstraints([]float64{1}, &ConstraintConfig{})
	if !empty.Feasible || empty.Violation != 0 {
		t.Errorf("EvaluateConstraints(empty config) = %+v, want feasible with zero violation", empty)
	}
}

func TestEvaluateConstraintsRejectsNilAndNonFiniteConstraints(t *testing.T) {
	broken := []struct {
		name       string
		constraint ConstraintFunction
	}{
		{name: "nil"},
		{name: "NaN", constraint: func([]float64) float64 { return math.NaN() }},
		{name: "positive infinity", constraint: func([]float64) float64 { return math.Inf(1) }},
		{name: "negative infinity", constraint: func([]float64) float64 { return math.Inf(-1) }},
	}

	for _, testCase := range broken {
		for _, kind := range []string{"inequality", "equality"} {
			t.Run(kind+"/"+testCase.name, func(t *testing.T) {
				config := &ConstraintConfig{}
				if kind == "inequality" {
					config.Inequalities = []ConstraintFunction{testCase.constraint}
				} else {
					config.Equalities = []ConstraintFunction{testCase.constraint}
				}

				result := EvaluateConstraints([]float64{0}, config)
				if !math.IsInf(result.Violation, 1) || result.Feasible {
					t.Errorf("EvaluateConstraints = %+v, want infeasible with +Inf violation", result)
				}
			})
		}
	}
}

func TestEvaluateConstraintsIgnoresAnUnusableTolerance(t *testing.T) {
	// A tolerance validateConstraintBlock would reject must not turn every
	// violation into a NaN that wins every comparison.
	for _, tolerance := range []float64{math.NaN(), math.Inf(1), -1} {
		config := &ConstraintConfig{
			Equalities:        []ConstraintFunction{equals(0, 0)},
			EqualityTolerance: tolerance,
		}

		result := EvaluateConstraints([]float64{2}, config)
		if result.Violation != 2 || result.Feasible {
			t.Errorf("tolerance %v: EvaluateConstraints = %+v, want violation 2", tolerance, result)
		}
	}
}

func TestIsFeasible(t *testing.T) {
	tests := []struct {
		violation float64
		want      bool
	}{
		{violation: 0, want: true},
		{violation: 1e-12},
		{violation: math.Inf(1)},
		{violation: math.NaN()},
	}

	for _, testCase := range tests {
		if got := IsFeasible(testCase.violation); got != testCase.want {
			t.Errorf("IsFeasible(%v) = %t, want %t", testCase.violation, got, testCase.want)
		}
	}
}

func TestPenalizedCost(t *testing.T) {
	tests := []struct {
		name      string
		method    PenaltyMethod
		cost      float64
		violation float64
		factor    float64
		want      float64
	}{
		{name: "linear", method: PenaltyLinear, cost: 10, violation: 2, factor: 3, want: 16},
		{name: "quadratic", method: PenaltyQuadratic, cost: 10, violation: 2, factor: 3, want: 22},
		{name: "empty defaults to quadratic", cost: 10, violation: 2, factor: 3, want: 22},
		{name: "unknown defaults to quadratic", method: "bogus", cost: 10, violation: 2, factor: 3, want: 22},
		{name: "feasible candidate is untouched", method: PenaltyLinear, cost: 10, factor: 3, want: 10},
		{name: "zero factor disables the penalty", method: PenaltyQuadratic, cost: 10, violation: 5, want: 10},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			got := PenalizedCost(testCase.cost, testCase.violation, testCase.factor, testCase.method)
			if got != testCase.want {
				t.Errorf("PenalizedCost = %v, want %v", got, testCase.want)
			}
		})
	}
}

func TestBetterConstrainedCandidateAppliesDebsRules(t *testing.T) {
	tests := []struct {
		name      string
		candidate CandidateEvaluation
		incumbent CandidateEvaluation
		want      bool
	}{
		{
			name:      "rule 1: feasible beats infeasible despite a worse cost",
			candidate: CandidateEvaluation{Cost: 100},
			incumbent: CandidateEvaluation{Cost: 0, ConstraintViolation: 1},
			want:      true,
		},
		{
			name:      "rule 1: infeasible loses to feasible despite a better cost",
			candidate: CandidateEvaluation{Cost: 0, ConstraintViolation: 1},
			incumbent: CandidateEvaluation{Cost: 100},
		},
		{
			name:      "rule 2: smaller violation wins between infeasible candidates",
			candidate: CandidateEvaluation{Cost: 100, ConstraintViolation: 1},
			incumbent: CandidateEvaluation{Cost: 0, ConstraintViolation: 2},
			want:      true,
		},
		{
			name:      "rule 2: larger violation loses even with a better cost",
			candidate: CandidateEvaluation{Cost: 0, ConstraintViolation: 2},
			incumbent: CandidateEvaluation{Cost: 100, ConstraintViolation: 1},
		},
		{
			name:      "rule 3: smaller cost wins between feasible candidates",
			candidate: CandidateEvaluation{Cost: 1},
			incumbent: CandidateEvaluation{Cost: 2},
			want:      true,
		},
		{
			name:      "rule 3: equal cost is not an improvement",
			candidate: CandidateEvaluation{Cost: 2},
			incumbent: CandidateEvaluation{Cost: 2},
		},
		{
			name:      "equal violation falls through to cost",
			candidate: CandidateEvaluation{Cost: 1, ConstraintViolation: 3},
			incumbent: CandidateEvaluation{Cost: 2, ConstraintViolation: 3},
			want:      true,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			for _, config := range []*ConstraintConfig{
				{},
				{Handling: ConstraintHandlingFeasibility},
			} {
				got := BetterConstrainedCandidate(testCase.candidate, testCase.incumbent, config)
				if got != testCase.want {
					t.Errorf("BetterConstrainedCandidate(%+v) = %t, want %t",
						config.Handling, got, testCase.want)
				}
			}
		})
	}
}

func TestBetterConstrainedCandidateWithoutAConfigMinimizesCost(t *testing.T) {
	// A nil config must ignore the violation field entirely.
	candidate := CandidateEvaluation{Cost: 1, ConstraintViolation: 99}
	incumbent := CandidateEvaluation{Cost: 2}

	if !BetterConstrainedCandidate(candidate, incumbent, nil) {
		t.Error("nil config did not prefer the cheaper candidate")
	}

	if BetterConstrainedCandidate(incumbent, candidate, nil) {
		t.Error("nil config preferred the costlier candidate")
	}
}

func TestBetterConstrainedCandidateRanksByPenalizedScore(t *testing.T) {
	feasible := CandidateEvaluation{Cost: 100}
	infeasible := CandidateEvaluation{Cost: 0, ConstraintViolation: 1}

	tests := []struct {
		name      string
		config    *ConstraintConfig
		candidate CandidateEvaluation
		incumbent CandidateEvaluation
		want      bool
	}{
		{
			name: "linear: a small factor lets a cheap infeasible candidate win",
			config: &ConstraintConfig{
				Handling: ConstraintHandlingPenalty, PenaltyMethod: PenaltyLinear, PenaltyFactor: 0.1,
			},
			candidate: infeasible,
			incumbent: feasible,
			want:      true,
		},
		{
			name: "linear: a large factor makes the feasible candidate win",
			config: &ConstraintConfig{
				Handling: ConstraintHandlingPenalty, PenaltyMethod: PenaltyLinear, PenaltyFactor: 1000,
			},
			candidate: infeasible,
			incumbent: feasible,
		},
		{
			name: "quadratic: the violation is squared before scaling",
			config: &ConstraintConfig{
				Handling: ConstraintHandlingPenalty, PenaltyMethod: PenaltyQuadratic, PenaltyFactor: 1,
			},
			// 10 + 1*3² = 19 beats 20 + 1*0 = 20.
			candidate: CandidateEvaluation{Cost: 10, ConstraintViolation: 3},
			incumbent: CandidateEvaluation{Cost: 20},
			want:      true,
		},
		{
			name: "quadratic: an unset method is quadratic",
			config: &ConstraintConfig{
				Handling: ConstraintHandlingPenalty, PenaltyFactor: 1,
			},
			// Linear would score 10+3 = 13 and win; quadratic scores 19 and loses.
			candidate: CandidateEvaluation{Cost: 10, ConstraintViolation: 3},
			incumbent: CandidateEvaluation{Cost: 14},
		},
		{
			name: "an exact tie falls back to the feasibility rules",
			config: &ConstraintConfig{
				Handling: ConstraintHandlingPenalty, PenaltyMethod: PenaltyLinear, PenaltyFactor: 1,
			},
			// Both score 10; the feasible candidate breaks the tie.
			candidate: CandidateEvaluation{Cost: 10},
			incumbent: CandidateEvaluation{Cost: 9, ConstraintViolation: 1},
			want:      true,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			got := BetterConstrainedCandidate(testCase.candidate, testCase.incumbent, testCase.config)
			if got != testCase.want {
				t.Errorf("BetterConstrainedCandidate = %t, want %t", got, testCase.want)
			}
		})
	}
}

func TestEffectiveConstraintDefaults(t *testing.T) {
	if got := effectiveConstraintHandling(nil); got != ConstraintHandlingFeasibility {
		t.Errorf("effectiveConstraintHandling(nil) = %q, want %q", got, ConstraintHandlingFeasibility)
	}

	if got := effectiveConstraintHandling(&ConstraintConfig{}); got != ConstraintHandlingFeasibility {
		t.Errorf("unset handling = %q, want %q", got, ConstraintHandlingFeasibility)
	}

	penalty := &ConstraintConfig{Handling: ConstraintHandlingPenalty}
	if got := effectiveConstraintHandling(penalty); got != ConstraintHandlingPenalty {
		t.Errorf("penalty handling = %q, want %q", got, ConstraintHandlingPenalty)
	}

	if got := effectivePenaltyMethod(nil); got != PenaltyQuadratic {
		t.Errorf("effectivePenaltyMethod(nil) = %q, want %q", got, PenaltyQuadratic)
	}

	if got := effectivePenaltyMethod(&ConstraintConfig{PenaltyMethod: PenaltyLinear}); got != PenaltyLinear {
		t.Errorf("linear method = %q, want %q", got, PenaltyLinear)
	}

	factors := []struct {
		config *ConstraintConfig
		want   float64
	}{
		{config: nil, want: 0},
		{config: &ConstraintConfig{}, want: 0},
		{config: &ConstraintConfig{PenaltyFactor: 7}, want: 7},
		{config: &ConstraintConfig{PenaltyFactor: math.NaN()}, want: 0},
		{config: &ConstraintConfig{PenaltyFactor: math.Inf(1)}, want: 0},
		{config: &ConstraintConfig{PenaltyFactor: -3}, want: 0},
	}
	for _, testCase := range factors {
		if got := effectivePenaltyFactor(testCase.config); got != testCase.want {
			t.Errorf("effectivePenaltyFactor(%+v) = %v, want %v", testCase.config, got, testCase.want)
		}
	}

	tolerances := []struct {
		config *ConstraintConfig
		want   float64
	}{
		{config: nil, want: 0},
		{config: &ConstraintConfig{}, want: 0},
		{config: &ConstraintConfig{EqualityTolerance: 1e-4}, want: 1e-4},
		{config: &ConstraintConfig{EqualityTolerance: math.NaN()}, want: 0},
		{config: &ConstraintConfig{EqualityTolerance: -1}, want: 0},
	}
	for _, testCase := range tolerances {
		if got := effectiveEqualityTolerance(testCase.config); got != testCase.want {
			t.Errorf("effectiveEqualityTolerance(%+v) = %v, want %v", testCase.config, got, testCase.want)
		}
	}
}

func TestValidateConstraintBlock(t *testing.T) {
	tests := []struct {
		name        string
		constraints *ConstraintConfig
		wantErr     string
	}{
		{name: "nil block is unconstrained"},
		{name: "empty block", constraints: &ConstraintConfig{}},
		{
			name: "fully specified penalty block",
			constraints: &ConstraintConfig{
				Inequalities:      []ConstraintFunction{atLeast(1)},
				Equalities:        []ConstraintFunction{equals(0, 0)},
				Handling:          ConstraintHandlingPenalty,
				PenaltyMethod:     PenaltyLinear,
				PenaltyFactor:     12,
				EqualityTolerance: 1e-4,
			},
		},
		{
			name:        "negative tolerance",
			constraints: &ConstraintConfig{EqualityTolerance: -1},
			wantErr:     "equality tolerance",
		},
		{
			name:        "non-finite tolerance",
			constraints: &ConstraintConfig{EqualityTolerance: math.NaN()},
			wantErr:     "equality tolerance",
		},
		{
			name:        "nil inequality",
			constraints: &ConstraintConfig{Inequalities: []ConstraintFunction{nil}},
			wantErr:     "inequality constraint 0 is nil",
		},
		{
			name:        "nil equality",
			constraints: &ConstraintConfig{Equalities: []ConstraintFunction{atLeast(1), nil}},
			wantErr:     "equality constraint 1 is nil",
		},
		{
			name:        "non-finite penalty factor",
			constraints: &ConstraintConfig{PenaltyFactor: math.NaN()},
			wantErr:     "penalty factor",
		},
		{
			name:        "negative penalty factor",
			constraints: &ConstraintConfig{PenaltyFactor: -1},
			wantErr:     "penalty factor",
		},
		{
			name:        "unknown penalty method",
			constraints: &ConstraintConfig{PenaltyMethod: "unknown"},
			wantErr:     `unknown penalty method "unknown"`,
		},
		{
			name:        "unknown handling method",
			constraints: &ConstraintConfig{Handling: "unknown"},
			wantErr:     `unknown constraint handling method "unknown"`,
		},
		{
			name:        "penalty handling without a factor",
			constraints: &ConstraintConfig{Handling: ConstraintHandlingPenalty},
			wantErr:     "penalty factor must be finite and positive",
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			err := validateConstraintBlock(testCase.constraints)
			if testCase.wantErr == "" {
				if err != nil {
					t.Fatalf("validateConstraintBlock = %v, want nil", err)
				}

				return
			}

			if err == nil || !strings.Contains(err.Error(), testCase.wantErr) {
				t.Fatalf("validateConstraintBlock = %v, want an error containing %q",
					err, testCase.wantErr)
			}
		})
	}
}

func TestConstraintEvaluatorScoresPositions(t *testing.T) {
	evaluator := newConstraintEvaluator(
		func(position []float64) float64 { return position[0] },
		&ConstraintConfig{Inequalities: []ConstraintFunction{atLeast(1)}},
	)

	feasible := evaluator.evaluate([]float64{2}, true)
	if feasible.Cost != 2 || feasible.ConstraintViolation != 0 || !evaluator.feasible(feasible) {
		t.Errorf("feasible evaluation = %+v, want cost 2 with zero violation", feasible)
	}

	infeasible := evaluator.evaluate([]float64{-1}, true)
	if infeasible.Cost != -1 || infeasible.ConstraintViolation != 2 || evaluator.feasible(infeasible) {
		t.Errorf("infeasible evaluation = %+v, want cost -1 with violation 2", infeasible)
	}
}

func TestConstraintEvaluatorWithoutConstraintsIsPlainMinimization(t *testing.T) {
	evaluator := newConstraintEvaluator(func(position []float64) float64 { return position[0] }, nil)

	evaluation := evaluator.evaluate([]float64{-5}, true)
	if evaluation.Cost != -5 || evaluation.ConstraintViolation != 0 {
		t.Errorf("evaluate = %+v, want cost -5 with zero violation", evaluation)
	}

	if !evaluator.better(CandidateEvaluation{Cost: 1, ConstraintViolation: 9}, CandidateEvaluation{Cost: 2}) {
		t.Error("nil constraints did not fall back to cost minimization")
	}
}

func TestConstraintEvaluatorSanitizesUnusableCosts(t *testing.T) {
	tests := []struct {
		name string
		cost float64
		want float64
	}{
		{name: "NaN", cost: math.NaN(), want: math.Inf(1)},
		{name: "negative infinity", cost: math.Inf(-1), want: math.Inf(1)},
		{name: "positive infinity", cost: math.Inf(1), want: math.Inf(1)},
		{name: "finite", cost: 3.5, want: 3.5},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			evaluator := newConstraintEvaluator(func([]float64) float64 { return testCase.cost }, nil)

			sanitized := evaluator.evaluate([]float64{0}, true)
			if sanitized.Cost != testCase.want && !(math.IsNaN(sanitized.Cost) && math.IsNaN(testCase.want)) {
				t.Errorf("sanitized cost = %v, want %v", sanitized.Cost, testCase.want)
			}

			raw := evaluator.evaluate([]float64{0}, false)
			if math.IsNaN(testCase.cost) {
				if !math.IsNaN(raw.Cost) {
					t.Errorf("unsanitized cost = %v, want NaN", raw.Cost)
				}

				return
			}

			if raw.Cost != testCase.cost {
				t.Errorf("unsanitized cost = %v, want %v", raw.Cost, testCase.cost)
			}
		})
	}
}

func TestConstraintEvaluatorDragonflyHelpers(t *testing.T) {
	evaluator := newConstraintEvaluator(
		func(position []float64) float64 { return position[0] },
		&ConstraintConfig{Inequalities: []ConstraintFunction{atLeast(0)}},
	)

	feasible := &Dragonfly{Position: []float64{3}}
	infeasible := &Dragonfly{Position: []float64{-2}}

	evaluator.evaluateDragonfly(feasible)
	evaluator.evaluateDragonfly(infeasible)

	if feasible.Cost != 3 || feasible.ConstraintViolation != 0 {
		t.Errorf("feasible dragonfly = %+v, want cost 3 with zero violation", feasible)
	}

	if infeasible.Cost != -2 || infeasible.ConstraintViolation != 2 {
		t.Errorf("infeasible dragonfly = %+v, want cost -2 with violation 2", infeasible)
	}

	if !evaluator.betterDragonfly(feasible, infeasible) {
		t.Error("betterDragonfly did not prefer the feasible member")
	}

	if evaluator.betterDragonfly(infeasible, feasible) {
		t.Error("betterDragonfly preferred the infeasible member")
	}

	incumbent := bestFromDragonfly(infeasible)
	if !evaluator.betterDragonflyThanBest(feasible, incumbent) {
		t.Error("betterDragonflyThanBest did not prefer the feasible member")
	}

	if !evaluator.betterBest(bestFromDragonfly(feasible), incumbent) {
		t.Error("betterBest did not prefer the feasible best")
	}
}

func TestBestFromDragonflyCopiesThePosition(t *testing.T) {
	dragonfly := &Dragonfly{Position: []float64{1, 2}, Cost: 5, ConstraintViolation: 0.5}

	best := bestFromDragonfly(dragonfly)
	dragonfly.Position[0] = 99

	if best.Position[0] != 1 {
		t.Errorf("Best.Position aliases the swarm: %v", best.Position)
	}

	if best.Cost != 5 || best.ConstraintViolation != 0.5 {
		t.Errorf("bestFromDragonfly = %+v, want cost 5 and violation 0.5", best)
	}
}

func TestCopyDragonflyToBestOverwritesInPlace(t *testing.T) {
	destination := Best{Position: []float64{0, 0}, Cost: 100, ConstraintViolation: 9}
	source := &Dragonfly{Position: []float64{1, 2}, Cost: 5, ConstraintViolation: 0.5}
	positions := destination.Position

	copyDragonflyToBest(&destination, source)

	if &positions[0] != &destination.Position[0] {
		t.Error("copyDragonflyToBest reallocated the position slice")
	}

	if destination.Position[0] != 1 || destination.Position[1] != 2 ||
		destination.Cost != 5 || destination.ConstraintViolation != 0.5 {
		t.Errorf("destination = %+v, want the source values", destination)
	}
}
