package dragonfly

import (
	"math/rand"
	"strings"
	"testing"
)

func TestRecommendBestOverProblemShapes(t *testing.T) {
	tests := []struct {
		name            string
		characteristics ProblemCharacteristics
		wantVariant     string
		wantPreset      ConfigPreset
	}{
		{
			name:            "plain continuous problem picks DA",
			characteristics: ProblemCharacteristics{Dimensionality: 10, Modality: Unimodal, Landscape: Smooth},
			wantVariant:     "DA",
			wantPreset:      PresetDefault,
		},
		{
			name: "rugged multimodal continuous still picks DA",
			characteristics: ProblemCharacteristics{
				Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged,
			},
			wantVariant: "DA",
			wantPreset:  PresetDefault,
		},
		{
			name: "discrete search space picks BDA",
			characteristics: ProblemCharacteristics{
				Dimensionality: 40, Modality: Multimodal, Landscape: Rugged, Discrete: true,
			},
			wantVariant: "BDA",
			wantPreset:  PresetBinary,
		},
		{
			name: "multi-objective picks MODA",
			characteristics: ProblemCharacteristics{
				Dimensionality: 30, Modality: Multimodal, Landscape: Smooth, MultiObjective: true,
			},
			wantVariant: "MODA",
			wantPreset:  PresetDefault,
		},
		{
			name: "discrete beats multi-objective for the preset but not the variant",
			characteristics: ProblemCharacteristics{
				Dimensionality: 30, Discrete: true, MultiObjective: true,
			},
			wantVariant: "MODA",
			wantPreset:  PresetBinary,
		},
		{
			name: "high dimensionality points at the high-dimensional preset",
			characteristics: ProblemCharacteristics{
				Dimensionality: 200, Modality: Multimodal, Landscape: Rugged,
			},
			wantVariant: "DA",
			wantPreset:  PresetHighDimensional,
		},
		{
			name: "a stated time budget points at the fast-convergence preset",
			characteristics: ProblemCharacteristics{
				Dimensionality: 10, Modality: Unimodal, Landscape: Smooth, RequiresFastConvergence: true,
			},
			wantVariant: "DA",
			wantPreset:  PresetFastConvergence,
		},
	}

	selector := NewAlgorithmSelector()

	for _, test := range tests {
		recommendation := selector.RecommendBest(test.characteristics)

		if recommendation.Variant.Name() != test.wantVariant {
			t.Errorf("%s: RecommendBest = %s, want %s",
				test.name, recommendation.Variant.Name(), test.wantVariant)
		}

		if recommendation.Preset != test.wantPreset {
			t.Errorf("%s: preset = %q, want %q", test.name, recommendation.Preset, test.wantPreset)
		}

		if strings.TrimSpace(recommendation.Reason) == "" {
			t.Errorf("%s: recommendation carries no reason", test.name)
		}

		if recommendation.Confidence <= 0 || recommendation.Confidence > 1 {
			t.Errorf("%s: confidence = %v, want a value in (0,1]", test.name, recommendation.Confidence)
		}
	}
}

// TestEveryRecommendationCarriesAReason is the invariant the whole selector
// exists for: a caller must be able to see the heuristic, not just its verdict.
func TestEveryRecommendationCarriesAReason(t *testing.T) {
	shapes := []ProblemCharacteristics{
		{},
		{Dimensionality: 2, Modality: Unimodal, Landscape: Smooth},
		{Dimensionality: 60, Modality: HighlyMultimodal, Landscape: Deceptive},
		{Dimensionality: 30, Landscape: NarrowValley, RequiresStableConvergence: true},
		{Dimensionality: 30, Discrete: true},
		{Dimensionality: 30, MultiObjective: true, ExpensiveEvaluations: true},
		{Dimensionality: 30, RequiresFastConvergence: true, ExpensiveEvaluations: true},
	}

	selector := NewAlgorithmSelector()

	for _, shape := range shapes {
		recommendations := selector.RecommendAlgorithms(shape)
		if len(recommendations) != len(GetAllVariants()) {
			t.Fatalf("%+v: got %d recommendations, want %d",
				shape, len(recommendations), len(GetAllVariants()))
		}

		for _, recommendation := range recommendations {
			if strings.TrimSpace(recommendation.Reason) == "" {
				t.Errorf("%+v: %s carries no reason", shape, recommendation.Variant.Name())
			}
		}
	}
}

func TestRecommendAlgorithmsIsSortedByScore(t *testing.T) {
	recommendations := NewAlgorithmSelector().RecommendAlgorithms(
		ProblemCharacteristics{Dimensionality: 20, Discrete: true})

	for i := 1; i < len(recommendations); i++ {
		if recommendations[i-1].Score < recommendations[i].Score {
			t.Errorf("recommendations are not sorted by score: %v then %v",
				recommendations[i-1].Score, recommendations[i].Score)
		}
	}
}

func TestRecommendBestFallsBackWithoutVariants(t *testing.T) {
	recommendation := NewAlgorithmSelectorFor().RecommendBest(ProblemCharacteristics{Dimensionality: 5})

	if recommendation.Variant.Name() != "DA" {
		t.Errorf("empty selector fell back to %s, want DA", recommendation.Variant.Name())
	}

	if strings.TrimSpace(recommendation.Reason) == "" {
		t.Error("the fallback recommendation carries no reason")
	}
}

func TestRecommendPresetTable(t *testing.T) {
	tests := []struct {
		characteristics ProblemCharacteristics
		want            ConfigPreset
	}{
		{ProblemCharacteristics{Dimensionality: 10}, PresetDefault},
		{ProblemCharacteristics{Dimensionality: highDimensionalThreshold - 1}, PresetDefault},
		{ProblemCharacteristics{Dimensionality: highDimensionalThreshold}, PresetHighDimensional},
		{ProblemCharacteristics{Dimensionality: 500}, PresetHighDimensional},
		{ProblemCharacteristics{Dimensionality: 10, RequiresFastConvergence: true}, PresetFastConvergence},
		{ProblemCharacteristics{Dimensionality: 500, Discrete: true}, PresetBinary},
		{ProblemCharacteristics{Dimensionality: 10, Discrete: true, RequiresFastConvergence: true}, PresetBinary},
	}

	for _, test := range tests {
		got := RecommendPreset(test.characteristics)
		if got != test.want {
			t.Errorf("RecommendPreset(%+v) = %q, want %q", test.characteristics, got, test.want)
		}
	}
}

// TestRecommendPresetNamesRealFactories keeps the selector honest: every preset
// it can return must actually resolve through NewPresetConfig.
func TestRecommendPresetNamesRealFactories(t *testing.T) {
	presets := []ConfigPreset{PresetDefault, PresetHighDimensional, PresetFastConvergence, PresetBinary}
	for _, preset := range presets {
		config, err := NewPresetConfig(preset)
		if err != nil || config == nil {
			t.Errorf("NewPresetConfig(%q) = %v, %v", preset, config, err)
		}
	}
}

func TestRecommendForBenchmarkTable(t *testing.T) {
	tests := []struct {
		benchmark   string
		wantVariant string
		wantPreset  ConfigPreset
	}{
		{"Sphere", "DA", PresetDefault},
		{"Rastrigin", "DA", PresetDefault},
		{"Rosenbrock", "DA", PresetDefault},
		{"Schwefel", "DA", PresetDefault},
		{"BentCigar", "DA", PresetDefault},
		{"ZDT1", "MODA", PresetDefault},
		{"ZDT3", "MODA", PresetDefault},
		{"SchafferN1", "MODA", PresetDefault},
	}

	for _, test := range tests {
		recommendation := RecommendForBenchmark(test.benchmark)

		if recommendation.Variant.Name() != test.wantVariant {
			t.Errorf("RecommendForBenchmark(%q) = %s, want %s",
				test.benchmark, recommendation.Variant.Name(), test.wantVariant)
		}

		if recommendation.Preset != test.wantPreset {
			t.Errorf("RecommendForBenchmark(%q).Preset = %q, want %q",
				test.benchmark, recommendation.Preset, test.wantPreset)
		}

		if strings.TrimSpace(recommendation.Reason) == "" {
			t.Errorf("RecommendForBenchmark(%q) carries no reason", test.benchmark)
		}
	}
}

func TestRecommendForBenchmarkFlagsUnknownNames(t *testing.T) {
	known := RecommendForBenchmark("Sphere")
	unknown := RecommendForBenchmark("NotABenchmark")

	if !strings.Contains(unknown.Reason, "not in the table") {
		t.Errorf("an unknown benchmark's reason %q does not say it was a guess", unknown.Reason)
	}

	if unknown.Confidence >= known.Confidence {
		t.Errorf("an unknown benchmark reported confidence %v, not below the known %v",
			unknown.Confidence, known.Confidence)
	}
}

func TestBenchmarkCharacteristicsCoverEveryTableEntry(t *testing.T) {
	names := BenchmarkNames()
	if len(names) == 0 {
		t.Fatal("BenchmarkNames() is empty")
	}

	for i := 1; i < len(names); i++ {
		if names[i-1] >= names[i] {
			t.Errorf("BenchmarkNames() is not sorted at %d: %q then %q", i, names[i-1], names[i])
		}
	}

	for _, name := range names {
		characteristics, ok := BenchmarkCharacteristics(name)
		if !ok {
			t.Errorf("BenchmarkCharacteristics(%q) reported unknown for a listed name", name)
		}

		if characteristics.Dimensionality <= 0 {
			t.Errorf("%q has dimensionality %d", name, characteristics.Dimensionality)
		}
	}

	if _, ok := BenchmarkCharacteristics("NotABenchmark"); ok {
		t.Error("BenchmarkCharacteristics reported an unknown name as known")
	}
}

func TestClassifyProblemIsSeedReproducible(t *testing.T) {
	first := ClassifyProblem(Sphere, 5, -5, 5, rand.New(rand.NewSource(99)))
	second := ClassifyProblem(Sphere, 5, -5, 5, rand.New(rand.NewSource(99)))

	if first != second {
		t.Errorf("ClassifyProblem is not reproducible for a seed: %+v vs %+v", first, second)
	}

	if first.Dimensionality != 5 {
		t.Errorf("Dimensionality = %d, want 5", first.Dimensionality)
	}

	// The caller-set fields must be left alone, not guessed at.
	if first.Discrete || first.MultiObjective || first.ExpensiveEvaluations ||
		first.RequiresFastConvergence {
		t.Errorf("ClassifyProblem filled in a caller-set field: %+v", first)
	}
}

// TestClassifyProblemSeparatesSmoothFromRugged pins the property the line-scan
// probe exists for and the gradient-magnitude heuristic it replaced did not
// have: the verdict follows the function, not the width of the box. Sphere is
// the same shape on [-5,5] and on [-500,500] and must classify the same way.
func TestClassifyProblemSeparatesSmoothFromRugged(t *testing.T) {
	tests := []struct {
		name          string
		fn            ObjectiveFunction
		size          int
		lower, upper  float64
		wantModality  Modality
		wantLandscape Landscape
	}{
		{"Sphere narrow box", Sphere, 5, -5, 5, Unimodal, Smooth},
		{"Sphere wide box", Sphere, 5, -500, 500, Unimodal, Smooth},
		{"Zakharov", Zakharov, 5, -5, 10, Unimodal, Smooth},
		{"Rastrigin", Rastrigin, 10, -5.12, 5.12, HighlyMultimodal, Rugged},
		{"Schwefel", Schwefel, 10, -500, 500, HighlyMultimodal, Rugged},
		{"Ackley", Ackley, 10, -32, 32, HighlyMultimodal, Rugged},
	}

	for _, test := range tests {
		got := ClassifyProblem(test.fn, test.size, test.lower, test.upper, rand.New(rand.NewSource(4)))

		if got.Modality != test.wantModality {
			t.Errorf("%s: modality = %v, want %v", test.name, got.Modality, test.wantModality)
		}

		if got.Landscape != test.wantLandscape {
			t.Errorf("%s: landscape = %v, want %v", test.name, got.Landscape, test.wantLandscape)
		}
	}
}

// TestLineShapeHandWorked checks the two scan statistics on a scan whose shape
// is obvious by inspection.
func TestLineShapeHandWorked(t *testing.T) {
	// A single V: down 4, up 4. One direction change; total variation 8 over a
	// range of 4, so roughness 2 -- the value a line crossing one basin gives,
	// which is why smoothRoughness sits above it.
	turns, roughness := lineShape([]float64{4, 2, 0, 2, 4})
	if turns != 1 {
		t.Errorf("single-basin scan turns = %v, want 1", turns)
	}

	if roughness != 2 {
		t.Errorf("single-basin scan roughness = %v, want 2", roughness)
	}

	// A sawtooth 0,1,0,1,0,1,0: five direction changes; total variation 6 over
	// a range of 1, so roughness 6.
	turns, roughness = lineShape([]float64{0, 1, 0, 1, 0, 1, 0})
	if turns != 5 {
		t.Errorf("sawtooth turns = %v, want 5", turns)
	}

	if roughness != 6 {
		t.Errorf("sawtooth roughness = %v, want 6", roughness)
	}

	// A flat scan has no direction changes and no range to normalize by.
	turns, roughness = lineShape([]float64{3, 3, 3, 3})
	if turns != 0 || roughness != 0 {
		t.Errorf("flat scan = %v, %v, want 0, 0", turns, roughness)
	}

	// A monotone ramp turns nowhere and has roughness exactly 1.
	turns, roughness = lineShape([]float64{0, 1, 2, 3})
	if turns != 0 || roughness != 1 {
		t.Errorf("ramp = %v, %v, want 0, 1", turns, roughness)
	}
}

// TestClassifyProblemNeverGuessesDeceptiveOrNarrowValley documents the limit
// stated on ClassifyProblem: neither can be established from samples, so the
// classifier must not claim them.
func TestClassifyProblemNeverGuessesDeceptiveOrNarrowValley(t *testing.T) {
	functions := map[string]ObjectiveFunction{
		"Schwefel":   Schwefel,
		"Rosenbrock": Rosenbrock,
		"BentCigar":  BentCigar,
		"Sphere":     Sphere,
	}

	for name, fn := range functions {
		got := ClassifyProblem(fn, 6, -10, 10, rand.New(rand.NewSource(21)))
		if got.Landscape != Smooth && got.Landscape != Rugged {
			t.Errorf("%s classified as %v; ClassifyProblem may only report Smooth or Rugged",
				name, got.Landscape)
		}
	}
}

func TestClassifyProblemAcceptsNilRNG(t *testing.T) {
	characteristics := ClassifyProblem(Sphere, 3, -1, 1, nil)
	if characteristics.Dimensionality != 3 {
		t.Errorf("Dimensionality = %d, want 3", characteristics.Dimensionality)
	}
}

func TestModalityAndLandscapeStrings(t *testing.T) {
	modalities := map[Modality]string{
		Unimodal:         "unimodal",
		Multimodal:       "multimodal",
		HighlyMultimodal: "highly multimodal",
		Modality(9):      "Modality(9)",
	}
	for modality, want := range modalities {
		if modality.String() != want {
			t.Errorf("Modality(%d).String() = %q, want %q", int(modality), modality.String(), want)
		}
	}

	landscapes := map[Landscape]string{
		Smooth:       "smooth",
		Rugged:       "rugged",
		Deceptive:    "deceptive",
		NarrowValley: "narrow valley",
		Landscape(9): "Landscape(9)",
	}
	for landscape, want := range landscapes {
		if landscape.String() != want {
			t.Errorf("Landscape(%d).String() = %q, want %q", int(landscape), landscape.String(), want)
		}
	}
}

func TestMeanAndStdDev(t *testing.T) {
	// Population standard deviation of {2,4,4,4,5,5,7,9}: mean 5, variance
	// (9+1+1+1+0+0+4+16)/8 = 4, so sigma = 2. The textbook example.
	mean, stdDev := meanAndStdDev([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if mean != 5 {
		t.Errorf("mean = %v, want 5", mean)
	}

	if stdDev != 2 {
		t.Errorf("stdDev = %v, want 2", stdDev)
	}

	mean, stdDev = meanAndStdDev(nil)
	if mean != 0 || stdDev != 0 {
		t.Errorf("meanAndStdDev(nil) = %v, %v, want 0, 0", mean, stdDev)
	}
}

// TestPrintRecommendations is a smoke test for the stdout table: every ranked
// recommendation must appear with its variant name, preset and reason.
func TestPrintRecommendations(t *testing.T) {
	characteristics := ProblemCharacteristics{
		Dimensionality: 30,
		Modality:       Multimodal,
		Landscape:      Rugged,
	}

	recommendations := NewAlgorithmSelector().RecommendAlgorithms(characteristics)
	if len(recommendations) == 0 {
		t.Fatal("RecommendAlgorithms returned nothing to print")
	}

	text := captureStdout(t, func() { PrintRecommendations(recommendations) })
	if text == "" {
		t.Fatal("PrintRecommendations wrote nothing to stdout")
	}

	if !strings.Contains(text, "Variant recommendations (ranked by score):") {
		t.Errorf("PrintRecommendations output has no heading:\n%s", text)
	}

	for _, recommendation := range recommendations {
		if !strings.Contains(text, recommendation.Variant.Name()) {
			t.Errorf("PrintRecommendations output omits variant %q:\n%s",
				recommendation.Variant.Name(), text)
		}

		if !strings.Contains(text, recommendation.Reason) {
			t.Errorf("PrintRecommendations output omits the reason for %q:\n%s",
				recommendation.Variant.Name(), text)
		}
	}

	// An empty slice still prints the frame, so a caller sees "no results"
	// rather than nothing at all.
	empty := captureStdout(t, func() { PrintRecommendations(nil) })
	if !strings.Contains(empty, "Variant recommendations (ranked by score):") {
		t.Errorf("PrintRecommendations(nil) printed no heading: %q", empty)
	}
}
