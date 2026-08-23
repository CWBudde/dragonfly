package dragonfly

import (
	"context"
	"errors"
	"math/rand"
	"testing"
)

// variantNames is the canonical order GetAllVariants must return. It is pinned
// here rather than derived from variantOrder so that a reordering of the
// registry has to be a deliberate edit in two places: reports, comparison
// tables and recommendation listings all key off this order.
var canonicalVariantNames = []string{"DA", "BDA", "MODA"}

func TestGetAllVariantsReturnsCanonicalOrder(t *testing.T) {
	variants := GetAllVariants()
	if len(variants) != len(canonicalVariantNames) {
		t.Fatalf("GetAllVariants() returned %d variants, want %d", len(variants), len(canonicalVariantNames))
	}

	for i, variant := range variants {
		if variant.Name() != canonicalVariantNames[i] {
			t.Errorf("GetAllVariants()[%d].Name() = %q, want %q", i, variant.Name(), canonicalVariantNames[i])
		}
	}
}

// TestGetAllVariantsOrderIsStable guards against a map range creeping back in:
// Go randomizes map iteration, so a registry-ranging implementation would fail
// this within a handful of calls.
func TestGetAllVariantsOrderIsStable(t *testing.T) {
	for call := range 50 {
		variants := GetAllVariants()
		for i, variant := range variants {
			if variant.Name() != canonicalVariantNames[i] {
				t.Fatalf("call %d: GetAllVariants()[%d].Name() = %q, want %q",
					call, i, variant.Name(), canonicalVariantNames[i])
			}
		}
	}
}

func TestListVariantsMatchesGetAllVariants(t *testing.T) {
	names := ListVariants()
	if len(names) != len(canonicalVariantNames) {
		t.Fatalf("ListVariants() = %v, want %v", names, canonicalVariantNames)
	}

	for i, name := range names {
		if name != canonicalVariantNames[i] {
			t.Errorf("ListVariants()[%d] = %q, want %q", i, name, canonicalVariantNames[i])
		}
	}

	// A returned slice must not alias the package-level order.
	names[0] = "mutated"

	if ListVariants()[0] != canonicalVariantNames[0] {
		t.Error("ListVariants() returns a slice aliasing variantOrder")
	}
}

func TestNewVariantResolvesEveryAlias(t *testing.T) {
	tests := []struct {
		alias string
		want  string
	}{
		{"da", "DA"},
		{"DA", "DA"},
		{"  Da  ", "DA"},
		{"standard", "DA"},
		{"bda", "BDA"},
		{"BDA", "BDA"},
		{"binary", "BDA"},
		{"moda", "MODA"},
		{"MODA", "MODA"},
	}

	for _, test := range tests {
		variant, err := NewVariant(test.alias)
		if err != nil {
			t.Errorf("NewVariant(%q) returned error %v", test.alias, err)

			continue
		}

		if variant.Name() != test.want {
			t.Errorf("NewVariant(%q).Name() = %q, want %q", test.alias, variant.Name(), test.want)
		}
	}
}

func TestNewVariantRejectsUnknownName(t *testing.T) {
	for _, name := range []string{"", "ma", "desma", "dragonfly", "moda2"} {
		variant, err := NewVariant(name)
		if err == nil {
			t.Errorf("NewVariant(%q) = %v, want an error", name, variant)
		}

		if variant != nil {
			t.Errorf("NewVariant(%q) returned a non-nil variant alongside its error", name)
		}
	}
}

// Variants are stateless empty structs, so the runtime is free to give every
// &DAVariant{} the same address; identity is not an invariant here. What must
// not be shared is the mutable Config, which
// TestVariantGetConfigReturnsFreshInstances covers.

func TestSingleObjectiveVariantsExcludesMODA(t *testing.T) {
	variants := SingleObjectiveVariants()

	want := []string{"DA", "BDA"}
	if len(variants) != len(want) {
		t.Fatalf("SingleObjectiveVariants() returned %d variants, want %d", len(variants), len(want))
	}

	for i, variant := range variants {
		if variant.Name() != want[i] {
			t.Errorf("SingleObjectiveVariants()[%d] = %q, want %q", i, variant.Name(), want[i])
		}

		if variant.IsMultiObjective() {
			t.Errorf("%s reports IsMultiObjective", variant.Name())
		}
	}
}

// TestDAVariantRejectsBinaryConfig is the point of routing runs through the
// variant layer: OptimizeContext documents that it ignores Config.UseBinary, so
// a binary config handed to it would run the continuous algorithm on a [0,1]
// box and return real-valued garbage that looks like a solution.
func TestDAVariantRejectsBinaryConfig(t *testing.T) {
	config := NewBinaryConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 5
	config.MaxIterations = 5
	config.NPop = 6
	config.Rand = rand.New(rand.NewSource(1))

	variant := &DAVariant{}

	result, err := variant.Run(context.Background(), config)
	if !errors.Is(err, ErrBinaryConfigOnContinuousVariant) {
		t.Fatalf("DAVariant.Run(binary config) error = %v, want ErrBinaryConfigOnContinuousVariant", err)
	}

	if result != nil {
		t.Error("DAVariant.Run(binary config) returned a result alongside its error")
	}
}

func TestDAVariantRunsContinuousConfig(t *testing.T) {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 4
	config.LowerBound = -5
	config.UpperBound = 5
	config.MaxIterations = 20
	config.NPop = 10
	config.Rand = rand.New(rand.NewSource(7))

	result, err := (&DAVariant{}).Run(context.Background(), config)
	if err != nil {
		t.Fatalf("DAVariant.Run: %v", err)
	}

	if len(result.GlobalBest.Position) != 4 {
		t.Errorf("GlobalBest.Position has %d components, want 4", len(result.GlobalBest.Position))
	}
}

func TestBDAVariantRunsBinaryAlgorithm(t *testing.T) {
	config := (&BDAVariant{}).GetConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 8
	config.MaxIterations = 20
	config.NPop = 10
	config.Rand = rand.New(rand.NewSource(11))

	result, err := (&BDAVariant{}).Run(context.Background(), config)
	if err != nil {
		t.Fatalf("BDAVariant.Run: %v", err)
	}

	if !BinaryPositionsValid(result.GlobalBest.Position) {
		t.Errorf("BDAVariant.Run produced a non-binary position %v", result.GlobalBest.Position)
	}
}

func TestMODAVariantRunReportsWrongEntryPoint(t *testing.T) {
	variant := &MODAVariant{}

	result, err := variant.Run(context.Background(), variant.GetConfig())
	if !errors.Is(err, ErrMultiObjectiveVariant) {
		t.Fatalf("MODAVariant.Run error = %v, want ErrMultiObjectiveVariant", err)
	}

	if result != nil {
		t.Error("MODAVariant.Run returned a result alongside its error")
	}
}

func TestMODAVariantRunMultiObjective(t *testing.T) {
	variant := &MODAVariant{}

	config := variant.GetMultiObjectiveConfig()
	config.ObjectiveFunc = ZDT1
	config.Swarm.ProblemSize = 4
	config.Swarm.LowerBound = 0
	config.Swarm.UpperBound = 1
	config.Swarm.MaxIterations = 15
	config.Swarm.NPop = 12
	config.Swarm.Rand = rand.New(rand.NewSource(3))

	result, err := variant.RunMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("MODAVariant.RunMultiObjective: %v", err)
	}

	if result.Archive == nil || result.Archive.Len() == 0 {
		t.Fatal("RunMultiObjective returned an empty archive")
	}

	if !result.Archive.IsNonDominated() {
		t.Error("RunMultiObjective returned a dominated archive")
	}
}

func TestVariantMetadataIsPopulated(t *testing.T) {
	for _, variant := range GetAllVariants() {
		if variant.Name() == "" || variant.FullName() == "" || variant.Description() == "" {
			t.Errorf("%T has an empty name, full name or description", variant)
		}

		if len(variant.RecommendedFor()) == 0 {
			t.Errorf("%s.RecommendedFor() is empty", variant.Name())
		}

		if variant.EstimatedOverhead() < 1.0 {
			t.Errorf("%s.EstimatedOverhead() = %v, want at least the 1.0 baseline",
				variant.Name(), variant.EstimatedOverhead())
		}

		config := variant.GetConfig()
		if config == nil {
			t.Errorf("%s.GetConfig() returned nil", variant.Name())
		}
	}
}

func TestVariantGetConfigReturnsFreshInstances(t *testing.T) {
	for _, variant := range GetAllVariants() {
		first := variant.GetConfig()
		first.NPop = 12345

		if variant.GetConfig().NPop == 12345 {
			t.Errorf("%s.GetConfig() shares a Config between calls", variant.Name())
		}
	}
}

func TestApplicableToStaysInUnitRange(t *testing.T) {
	shapes := []ProblemCharacteristics{
		{Dimensionality: 5, Modality: Unimodal, Landscape: Smooth},
		{Dimensionality: 100, Modality: HighlyMultimodal, Landscape: Deceptive},
		{Dimensionality: 30, Discrete: true},
		{Dimensionality: 30, MultiObjective: true},
		{Dimensionality: 30, Discrete: true, MultiObjective: true, ExpensiveEvaluations: true},
	}

	for _, shape := range shapes {
		for _, variant := range GetAllVariants() {
			score := variant.ApplicableTo(shape)
			if score < 0 || score > 1 {
				t.Errorf("%s.ApplicableTo(%+v) = %v, want a score in [0,1]", variant.Name(), shape, score)
			}
		}
	}
}

func TestBuilderBuildsAndRuns(t *testing.T) {
	config, err := NewBuilder("da").
		ForProblem(Sphere, 3, -2, 2).
		WithIterations(15).
		WithPopulation(8).
		WithConfig(func(c *Config) { c.Rand = rand.New(rand.NewSource(5)) }).
		Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if config.ProblemSize != 3 || config.LowerBound != -2 || config.UpperBound != 2 {
		t.Errorf("ForProblem did not apply: %+v", config)
	}

	if config.MaxIterations != 15 || config.NPop != 8 {
		t.Errorf("WithIterations/WithPopulation did not apply: %+v", config)
	}

	result, err := NewBuilder("da").
		ForProblem(Sphere, 3, -2, 2).
		WithIterations(15).
		WithPopulation(8).
		WithConfig(func(c *Config) { c.Rand = rand.New(rand.NewSource(5)) }).
		Optimize()
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	if len(result.ConvergenceCurve) != 15 {
		t.Errorf("ConvergenceCurve has %d entries, want 15", len(result.ConvergenceCurve))
	}
}

// TestBuilderKeepsBinaryBounds: NewBinaryConfig fixes the box at [0,1] because
// every schedule that scales with (ub-lb) is written for that box, so
// ForProblem must not overwrite it.
func TestBuilderKeepsBinaryBounds(t *testing.T) {
	config, err := NewBuilder("binary").
		ForProblem(Sphere, 6, -100, 100).
		WithIterations(10).
		WithPopulation(8).
		Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if config.LowerBound != 0 || config.UpperBound != 1 {
		t.Errorf("binary builder bounds = [%v, %v], want [0, 1]", config.LowerBound, config.UpperBound)
	}

	if !config.UseBinary {
		t.Error("binary builder produced a config without UseBinary")
	}
}

func TestBuilderErrors(t *testing.T) {
	_, err := NewBuilder("nope").ForProblem(Sphere, 3, -1, 1).Build()
	if err == nil {
		t.Error("NewBuilder with an unknown variant did not report an error from Build")
	}

	_, err = NewBuilder("da").WithIterations(10).Build()
	if !errors.Is(err, errBuilderNoObjective) {
		t.Errorf("Build without an objective = %v, want errBuilderNoObjective", err)
	}

	_, err = NewBuilder("da").ForProblem(Sphere, 0, -1, 1).Build()
	if err == nil {
		t.Error("Build with a zero problem size did not report an error")
	}

	var nilBuilder *VariantBuilder

	_, err = nilBuilder.Build()
	if !errors.Is(err, errNilBuilder) {
		t.Errorf("(*VariantBuilder)(nil).Build() = %v, want errNilBuilder", err)
	}

	if nilBuilder.GetVariant() != nil {
		t.Error("(*VariantBuilder)(nil).GetVariant() must be nil")
	}
}

func TestBuilderFromVariant(t *testing.T) {
	builder := NewBuilderFromVariant(&BDAVariant{})
	if builder.GetVariant().Name() != "BDA" {
		t.Errorf("GetVariant().Name() = %q, want BDA", builder.GetVariant().Name())
	}

	_, err := NewBuilderFromVariant(nil).ForProblem(Sphere, 2, 0, 1).Build()
	if err == nil {
		t.Error("NewBuilderFromVariant(nil).Build() did not report an error")
	}
}

func TestVariantAliasesAreSortedAndComplete(t *testing.T) {
	aliases := VariantAliases()
	want := []string{"bda", "binary", "da", "moda", "standard"}

	if len(aliases) != len(want) {
		t.Fatalf("VariantAliases() = %v, want %v", aliases, want)
	}

	for i, alias := range aliases {
		if alias != want[i] {
			t.Errorf("VariantAliases()[%d] = %q, want %q", i, alias, want[i])
		}
	}
}
