package main

import (
	"bytes"
	"context"
	"math"
	"strings"
	"testing"

	"github.com/CWBudde/dragonfly"
)

func TestRandomInitialMeanUsesPhysicalBounds(t *testing.T) {
	mean := randomInitialMean(100, -10, 30, 42)
	for i, value := range mean {
		if value < -10 || value > 30 {
			t.Fatalf("mean[%d] = %g, want within [-10,30]", i, value)
		}
	}
}

func TestEvaluationBudgetCallsObjectiveOnlyToLimit(t *testing.T) {
	calls := 0
	budget := evaluationBudget{
		objective: func([]float64) float64 {
			calls++

			return 7
		},
		limit: 2,
	}

	if got := budget.evaluate(nil); got != 7 {
		t.Fatalf("first evaluation = %g, want 7", got)
	}

	if got := budget.evaluate(nil); got != 7 {
		t.Fatalf("second evaluation = %g, want 7", got)
	}

	if got := budget.evaluate(nil); !math.IsInf(got, 1) {
		t.Fatalf("over-budget evaluation = %g, want +Inf", got)
	}

	if calls != 2 || budget.evaluations() != 2 {
		t.Fatalf("calls/evaluations = %d/%d, want 2/2", calls, budget.evaluations())
	}
}

func TestAdaptersRespectExactBudget(t *testing.T) {
	const limit = 80

	seed := int64(42)
	base := dragonfly.NewDefaultConfig()
	base.ObjectiveFunc = dragonfly.Sphere
	base.ProblemSize = 2
	base.LowerBound = 0
	base.UpperBound = 1
	base.Seed = &seed

	for name, run := range map[string]func(context.Context, *dragonfly.Config) (*dragonfly.Result, error){
		"DA":     runDragonfly(limit),
		"MA":     runMayfly(limit),
		"CMA-ES": runCMAES(limit),
	} {
		result, err := run(context.Background(), base)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		if result.FuncEvalCount != limit {
			t.Errorf("%s evaluations = %d, want %d", name, result.FuncEvalCount, limit)
		}
	}
}

func TestMarkdownNamesMethodAndRawData(t *testing.T) {
	comparison := dragonfly.NewComparisonRunner().WithVariants(
		optimizerVariant{name: "DA", run: runDragonfly(80)},
		optimizerVariant{name: "MA", run: runMayfly(80)},
	).WithRuns(1).WithIterations(80).WithSeed(7).
		Compare("Sphere", dragonfly.Sphere, 2, 0, 1)

	result := study{
		options: options{
			csvPath: "raw.csv", markdownPath: "summary.md", dimensions: 2,
			runs: 1, budget: 80, seed: 7, dragonflyRevision: "dragon",
			mayflyRevision: "may", cmaRevision: "cma",
		},
		results: []*dragonfly.ComparisonResult{comparison},
	}

	var output bytes.Buffer

	err := result.writeMarkdown(&output)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{"Dragonfly vs Mayfly", "exactly 80", "DA vs MA", "raw.csv"} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("Markdown does not contain %q", want)
		}
	}

	if strings.Contains(output.String(), "CMA-ES") {
		t.Error("DA/MA-only Markdown unexpectedly describes CMA-ES")
	}
}
