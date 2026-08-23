package dragonfly

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newLoaderTestConfig builds a complete, runnable configuration for the
// persistence and validation tests.
func newLoaderTestConfig() *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 5
	config.LowerBound = -10
	config.UpperBound = 10

	return config
}

func TestSaveConfigAndLoadConfigRoundTrip(t *testing.T) {
	original := newLoaderTestConfig()
	original.NPop = 37
	original.MaxIterations = 123
	original.BoundaryMethod = BoundaryReflect
	original.InertiaWeightStart = 0.8
	original.InertiaWeightEnd = 0.3
	original.RadiusInitialDivisor = 3.5
	original.RadiusGrowth = 1.25
	original.MaxStepRatio = 0.15
	original.EnemyCutoffFraction = 0.6
	original.LevyBeta = 1.25
	original.LevyScale = 0.02
	original.UseLevyWalk = false
	original.EnableParallel = true
	original.MaxWorkers = 3

	// The two settings that must survive intact and distinct: a weight pinned
	// to zero ("switch the enemy term off") and a weight left on its adaptive
	// schedule.
	original.EnemyWeight = 0
	original.FoodWeight = WeightAuto

	path := filepath.Join(t.TempDir(), "config.json")

	saveErr := SaveConfig(original, path)
	if saveErr != nil {
		t.Fatalf("SaveConfig() error = %v", saveErr)
	}

	loaded, loadErr := LoadConfig(path)
	if loadErr != nil {
		t.Fatalf("LoadConfig() error = %v", loadErr)
	}

	if loaded.EnemyWeight != 0 {
		t.Errorf("pinned enemy_weight = %v, want 0", loaded.EnemyWeight)
	}

	if loaded.FoodWeight != WeightAuto {
		t.Errorf("food_weight = %v, want WeightAuto (%v)", loaded.FoodWeight, WeightAuto)
	}

	assertScalarFieldsRoundTripped(t, original, loaded)
}

func assertScalarFieldsRoundTripped(t *testing.T, original, loaded *Config) {
	t.Helper()

	floats := []struct {
		name      string
		want, got float64
	}{
		{"lower_bound", original.LowerBound, loaded.LowerBound},
		{"upper_bound", original.UpperBound, loaded.UpperBound},
		{"inertia_weight_start", original.InertiaWeightStart, loaded.InertiaWeightStart},
		{"inertia_weight_end", original.InertiaWeightEnd, loaded.InertiaWeightEnd},
		{"separation_weight", original.SeparationWeight, loaded.SeparationWeight},
		{"alignment_weight", original.AlignmentWeight, loaded.AlignmentWeight},
		{"cohesion_weight", original.CohesionWeight, loaded.CohesionWeight},
		{"radius_initial_divisor", original.RadiusInitialDivisor, loaded.RadiusInitialDivisor},
		{"radius_growth", original.RadiusGrowth, loaded.RadiusGrowth},
		{"max_step_ratio", original.MaxStepRatio, loaded.MaxStepRatio},
		{"enemy_cutoff_fraction", original.EnemyCutoffFraction, loaded.EnemyCutoffFraction},
		{"levy_beta", original.LevyBeta, loaded.LevyBeta},
		{"levy_scale", original.LevyScale, loaded.LevyScale},
	}
	for _, item := range floats {
		if item.got != item.want {
			t.Errorf("%s = %v, want %v", item.name, item.got, item.want)
		}
	}

	ints := []struct {
		name      string
		want, got int
	}{
		{"problem_size", original.ProblemSize, loaded.ProblemSize},
		{"npop", original.NPop, loaded.NPop},
		{"max_iterations", original.MaxIterations, loaded.MaxIterations},
		{"max_workers", original.MaxWorkers, loaded.MaxWorkers},
	}
	for _, item := range ints {
		if item.got != item.want {
			t.Errorf("%s = %d, want %d", item.name, item.got, item.want)
		}
	}

	if loaded.BoundaryMethod != original.BoundaryMethod {
		t.Errorf("boundary_method = %q, want %q", loaded.BoundaryMethod, original.BoundaryMethod)
	}

	if loaded.UseLevyWalk != original.UseLevyWalk {
		t.Errorf("use_levy_walk = %v, want %v", loaded.UseLevyWalk, original.UseLevyWalk)
	}

	if loaded.EnableParallel != original.EnableParallel {
		t.Errorf("enable_parallel = %v, want %v", loaded.EnableParallel, original.EnableParallel)
	}
}

// The unserializable fields are documented as the caller's responsibility;
// a loaded config must come back without them rather than with a surprise.
func TestLoadConfigLeavesUnserializableFieldsUnset(t *testing.T) {
	original := newLoaderTestConfig()
	path := filepath.Join(t.TempDir(), "config.json")

	saveErr := SaveConfig(original, path)
	if saveErr != nil {
		t.Fatalf("SaveConfig() error = %v", saveErr)
	}

	loaded, loadErr := LoadConfig(path)
	if loadErr != nil {
		t.Fatalf("LoadConfig() error = %v", loadErr)
	}

	if loaded.ObjectiveFunc != nil {
		t.Error("ObjectiveFunc survived a round trip, but it is json:\"-\"")
	}

	if loaded.Rand != nil {
		t.Error("Rand survived a round trip, but it is json:\"-\"")
	}

	// It is not runnable until the caller supplies an objective...
	unrunnableErr := ValidateConfig(loaded)
	if unrunnableErr == nil {
		t.Fatal("ValidateConfig() on a freshly loaded config = nil, want an ObjectiveFunc error")
	}

	// ...and valid the moment they do.
	loaded.ObjectiveFunc = Sphere

	runnableErr := ValidateConfig(loaded)
	if runnableErr != nil {
		t.Errorf("ValidateConfig() after setting ObjectiveFunc = %v, want nil", runnableErr)
	}
}

func TestSaveConfigOmitsFunctionFields(t *testing.T) {
	config := newLoaderTestConfig()
	path := filepath.Join(t.TempDir(), "config.json")

	saveErr := SaveConfig(config, path)
	if saveErr != nil {
		t.Fatalf("SaveConfig() error = %v", saveErr)
	}

	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("ReadFile() error = %v", readErr)
	}

	raw := map[string]json.RawMessage{}

	unmarshalErr := json.Unmarshal(data, &raw)
	if unmarshalErr != nil {
		t.Fatalf("saved file is not a JSON object: %v", unmarshalErr)
	}

	for _, key := range []string{"ObjectiveFunc", "objective_func", "Rand", "rand"} {
		if _, found := raw[key]; found {
			t.Errorf("saved config contains key %q, which cannot be serialized", key)
		}
	}

	// Snake-case tags are a lint contract; spot-check that the file honors it.
	for _, key := range []string{"problem_size", "npop", "max_iterations", "enemy_weight"} {
		if _, found := raw[key]; !found {
			t.Errorf("saved config is missing key %q", key)
		}
	}
}

func TestSaveConfigRejectsNil(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nil.json")

	err := SaveConfig(nil, path)
	if err == nil {
		t.Fatal("SaveConfig(nil) = nil, want an error")
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err == nil {
		t.Fatal("LoadConfig() on a missing file = nil, want an error")
	}

	if !strings.Contains(err.Error(), "read config file") {
		t.Errorf("LoadConfig() error = %v, want it to mention reading the file", err)
	}
}

func TestLoadConfigMalformedJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "malformed.json")

	writeErr := os.WriteFile(path, []byte("{ this is not json"), 0o600)
	if writeErr != nil {
		t.Fatalf("WriteFile() error = %v", writeErr)
	}

	_, err := LoadConfig(path)
	if err == nil {
		t.Fatal("LoadConfig() on malformed JSON = nil, want an error")
	}

	if !strings.Contains(err.Error(), "parse config file") {
		t.Errorf("LoadConfig() error = %v, want it to mention parsing", err)
	}
}

func TestLoadConfigRejectsUnknownFieldsAndTrailingValues(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{
			name: "unknown field",
			data: `{"problem_size": 2, "lower_bound": -1, "upper_bound": 1, "npop": 4, ` +
				`"max_iterations": 2, "enable_parallel_typo": true}`,
			want: "unknown field",
		},
		{
			name: "trailing JSON value",
			data: `{}` + "\n" + `{}`,
			want: "trailing data",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")

			writeErr := os.WriteFile(path, []byte(test.data), 0o600)
			if writeErr != nil {
				t.Fatalf("WriteFile() error = %v", writeErr)
			}

			_, err := LoadConfig(path)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("LoadConfig() error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestLoadConfigRejectsInvalidContents(t *testing.T) {
	config := newLoaderTestConfig()
	config.LowerBound = 10
	config.UpperBound = -10

	path := filepath.Join(t.TempDir(), "invalid.json")

	// Written directly: SaveConfig does not validate, and the point here is
	// that LoadConfig does.
	saveErr := SaveConfig(config, path)
	if saveErr != nil {
		t.Fatalf("SaveConfig() error = %v", saveErr)
	}

	_, err := LoadConfig(path)
	if err == nil {
		t.Fatal("LoadConfig() on a config with inverted bounds = nil, want an error")
	}

	if !strings.Contains(err.Error(), "invalid config") {
		t.Errorf("LoadConfig() error = %v, want it to mention invalid config", err)
	}
}

func TestNewPresetConfigKnownPresets(t *testing.T) {
	presets := []ConfigPreset{PresetDefault, PresetHighDimensional, PresetFastConvergence}

	for _, preset := range presets {
		config, err := NewPresetConfig(preset)
		if err != nil {
			t.Fatalf("NewPresetConfig(%q) error = %v", preset, err)
		}

		config.ObjectiveFunc = Sphere
		config.ProblemSize = 5
		config.LowerBound = -10
		config.UpperBound = 10

		validateErr := ValidateConfig(config)
		if validateErr != nil {
			t.Errorf("preset %q does not validate: %v", preset, validateErr)
		}
	}
}

// A preset lookup must hand back a fresh config each time, never a shared
// pointer a previous caller could have mutated.
func TestNewPresetConfigReturnsFreshConfig(t *testing.T) {
	first, firstErr := NewPresetConfig(PresetDefault)
	if firstErr != nil {
		t.Fatalf("NewPresetConfig() error = %v", firstErr)
	}

	first.NPop = 12345
	first.EnemyWeight = 0

	second, secondErr := NewPresetConfig(PresetDefault)
	if secondErr != nil {
		t.Fatalf("NewPresetConfig() error = %v", secondErr)
	}

	if first == second {
		t.Fatal("NewPresetConfig() returned the same pointer twice")
	}

	if second.NPop == 12345 {
		t.Error("mutating one preset config changed the next one's npop")
	}

	if second.EnemyWeight != WeightAuto {
		t.Errorf("second preset enemy_weight = %v, want WeightAuto (%v)", second.EnemyWeight, WeightAuto)
	}
}

func TestNewPresetConfigUnknownPreset(t *testing.T) {
	config, err := NewPresetConfig(ConfigPreset("no-such-preset"))
	if err == nil {
		t.Fatal("NewPresetConfig() on an unknown name = nil error, want an error")
	}

	if config != nil {
		t.Error("NewPresetConfig() returned a config alongside an error")
	}

	if !strings.Contains(err.Error(), "no-such-preset") {
		t.Errorf("error = %v, want it to quote the unknown name", err)
	}
}

func TestListPresetsCoversEveryName(t *testing.T) {
	presets := ListPresets()

	for _, preset := range []ConfigPreset{PresetDefault, PresetHighDimensional, PresetFastConvergence} {
		description, found := presets[preset]
		if !found {
			t.Errorf("ListPresets() is missing %q", preset)

			continue
		}

		if description == "" {
			t.Errorf("ListPresets()[%q] has an empty description", preset)
		}

		_, resolveErr := NewPresetConfig(preset)
		if resolveErr != nil {
			t.Errorf("listed preset %q does not resolve: %v", preset, resolveErr)
		}
	}

	names := PresetNames()
	if len(names) != len(presets) {
		t.Errorf("PresetNames() has %d entries, ListPresets() has %d", len(names), len(presets))
	}

	for i := 1; i < len(names); i++ {
		if names[i-1] >= names[i] {
			t.Errorf("PresetNames() is not sorted: %q before %q", names[i-1], names[i])
		}
	}
}

func TestValidateConfigAcceptsGoodConfig(t *testing.T) {
	err := ValidateConfig(newLoaderTestConfig())
	if err != nil {
		t.Fatalf("ValidateConfig() on a default config = %v, want nil", err)
	}
}

// A pinned zero weight is a legitimate setting -- "switch the enemy term off"
// -- and must not be mistaken for an unset field.
func TestValidateConfigAcceptsPinnedZeroWeights(t *testing.T) {
	config := newLoaderTestConfig()
	config.SeparationWeight = 0
	config.AlignmentWeight = 0
	config.CohesionWeight = 0
	config.FoodWeight = 0
	config.EnemyWeight = 0

	err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig() with every weight pinned to zero = %v, want nil", err)
	}
}

func TestValidateConfigFailureModes(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantSub string
	}{
		{
			name:    "missing objective",
			mutate:  func(c *Config) { c.ObjectiveFunc = nil },
			wantSub: "ObjectiveFunc",
		},
		{
			name:    "zero problem size",
			mutate:  func(c *Config) { c.ProblemSize = 0 },
			wantSub: "problem_size",
		},
		{
			name:    "negative problem size",
			mutate:  func(c *Config) { c.ProblemSize = -3 },
			wantSub: "problem_size",
		},
		{
			name:    "lower bound equals upper bound",
			mutate:  func(c *Config) { c.LowerBound, c.UpperBound = 1, 1 },
			wantSub: "lower_bound",
		},
		{
			name:    "lower bound above upper bound",
			mutate:  func(c *Config) { c.LowerBound, c.UpperBound = 5, -5 },
			wantSub: "lower_bound",
		},
		{
			name:    "non-finite bound",
			mutate:  func(c *Config) { c.UpperBound = math.Inf(1) },
			wantSub: "finite",
		},
		{
			name:    "zero npop",
			mutate:  func(c *Config) { c.NPop = 0 },
			wantSub: "npop",
		},
		{
			name:    "negative npop",
			mutate:  func(c *Config) { c.NPop = -1 },
			wantSub: "npop",
		},
		{
			name:    "zero max iterations",
			mutate:  func(c *Config) { c.MaxIterations = 0 },
			wantSub: "max_iterations",
		},
		{
			name:    "negative max iterations",
			mutate:  func(c *Config) { c.MaxIterations = -10 },
			wantSub: "max_iterations",
		},
		{
			name:    "negative max workers",
			mutate:  func(c *Config) { c.MaxWorkers = -1 },
			wantSub: "max_workers",
		},
		{
			name:    "unknown boundary method",
			mutate:  func(c *Config) { c.BoundaryMethod = BoundaryMethod("teleport") },
			wantSub: "boundary_method",
		},
		{
			name:    "non-finite weight",
			mutate:  func(c *Config) { c.FoodWeight = math.NaN() },
			wantSub: "food_weight",
		},
		{
			name:    "non-positive radius divisor",
			mutate:  func(c *Config) { c.RadiusInitialDivisor = 0 },
			wantSub: "radius_initial_divisor",
		},
		{
			name:    "levy beta out of range",
			mutate:  func(c *Config) { c.LevyBeta = 2 },
			wantSub: "levy_beta",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := newLoaderTestConfig()
			tt.mutate(config)

			err := ValidateConfig(config)
			if err == nil {
				t.Fatalf("ValidateConfig() = nil, want an error mentioning %q", tt.wantSub)
			}

			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("ValidateConfig() error = %v, want it to mention %q", err, tt.wantSub)
			}
		})
	}
}

func TestValidateConfigRejectsNil(t *testing.T) {
	err := ValidateConfig(nil)
	if err == nil {
		t.Fatal("ValidateConfig(nil) = nil, want an error")
	}
}

// An empty boundary method is the zero value a partially written file yields,
// and Optimize resolves it to the paper's default, so validation must accept it.
func TestValidateConfigAcceptsEmptyBoundaryMethod(t *testing.T) {
	config := newLoaderTestConfig()
	config.BoundaryMethod = ""

	err := ValidateConfig(config)
	if err != nil {
		t.Fatalf("ValidateConfig() with an unset boundary_method = %v, want nil", err)
	}
}

func TestAutoTuneConfigProducesValidConfigs(t *testing.T) {
	sizes := []int{1, 2, 5, 9, 10, 20, 30, 49, 50, 100, 500}

	for _, size := range sizes {
		config := newLoaderTestConfig()
		config.ProblemSize = size

		AutoTuneConfig(config)

		err := ValidateConfig(config)
		if err != nil {
			t.Errorf("AutoTuneConfig() at ProblemSize=%d produced an invalid config: %v", size, err)
		}

		if config.NPop <= 0 {
			t.Errorf("AutoTuneConfig() at ProblemSize=%d set npop=%d", size, config.NPop)
		}

		if config.MaxIterations <= 0 {
			t.Errorf("AutoTuneConfig() at ProblemSize=%d set max_iterations=%d", size, config.MaxIterations)
		}
	}
}

// The swarm and the run only ever grow with dimensionality.
func TestAutoTuneConfigIsMonotonicInProblemSize(t *testing.T) {
	sizes := []int{1, 5, 9, 10, 25, 49, 50, 200}

	previousPop, previousIterations := 0, 0

	for _, size := range sizes {
		config := newLoaderTestConfig()
		config.ProblemSize = size

		AutoTuneConfig(config)

		if config.NPop < previousPop {
			t.Errorf("npop dropped from %d to %d at ProblemSize=%d", previousPop, config.NPop, size)
		}

		if config.MaxIterations < previousIterations {
			t.Errorf("max_iterations dropped from %d to %d at ProblemSize=%d",
				previousIterations, config.MaxIterations, size)
		}

		previousPop, previousIterations = config.NPop, config.MaxIterations
	}
}

// High dimensions get a slower-growing radius, for the reason
// NewHighDimensionalConfig gives; low dimensions keep the paper's schedule.
func TestAutoTuneConfigSlowsRadiusGrowthInHighDimensions(t *testing.T) {
	low := newLoaderTestConfig()
	low.ProblemSize = 5

	AutoTuneConfig(low)

	if low.RadiusGrowth != NewDefaultConfig().RadiusGrowth {
		t.Errorf("low-dimensional radius_growth = %v, want the default %v",
			low.RadiusGrowth, NewDefaultConfig().RadiusGrowth)
	}

	high := newLoaderTestConfig()
	high.ProblemSize = 100

	AutoTuneConfig(high)

	if high.RadiusGrowth >= low.RadiusGrowth {
		t.Errorf("high-dimensional radius_growth = %v, want less than %v",
			high.RadiusGrowth, low.RadiusGrowth)
	}
}

// Weights are a deliberate choice; a heuristic must not overwrite them.
func TestAutoTuneConfigLeavesWeightsAlone(t *testing.T) {
	config := newLoaderTestConfig()
	config.ProblemSize = 100
	config.EnemyWeight = 0
	config.FoodWeight = 1.5

	AutoTuneConfig(config)

	if config.EnemyWeight != 0 {
		t.Errorf("enemy_weight = %v, want the pinned 0", config.EnemyWeight)
	}

	if config.FoodWeight != 1.5 {
		t.Errorf("food_weight = %v, want the pinned 1.5", config.FoodWeight)
	}

	if config.SeparationWeight != WeightAuto {
		t.Errorf("separation_weight = %v, want WeightAuto (%v)", config.SeparationWeight, WeightAuto)
	}
}

func TestAutoTuneConfigIgnoresUnusableConfigs(t *testing.T) {
	AutoTuneConfig(nil) // must not panic

	config := newLoaderTestConfig()
	config.ProblemSize = 0
	config.NPop = 7

	AutoTuneConfig(config)

	if config.NPop != 7 {
		t.Errorf("npop = %d, want 7 -- there is nothing to tune without a ProblemSize", config.NPop)
	}
}

// TestPlaceholderObjective pins the stand-in ValidateConfig substitutes for a
// missing ObjectiveFunc: it must be total and finite for any input, so that a
// deserialized configuration fails validation on a real problem rather than on
// the probe.
func TestPlaceholderObjective(t *testing.T) {
	inputs := [][]float64{
		nil,
		{},
		{0},
		{1, -1, 1e300},
		{math.Inf(1), math.Inf(-1), math.NaN()},
	}

	for _, input := range inputs {
		got := placeholderObjective(input)
		if got != 0 {
			t.Errorf("placeholderObjective(%v) = %v, want 0", input, got)
		}
	}

	// LoadConfig leaves ObjectiveFunc unset -- the placeholder is a validation
	// probe, never something a loaded config keeps.
	path := filepath.Join(t.TempDir(), "config.json")

	err := SaveConfig(newLoaderTestConfig(), path)
	if err != nil {
		t.Fatalf("SaveConfig: %v", err)
	}

	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if loaded.ObjectiveFunc != nil {
		t.Error("LoadConfig left an ObjectiveFunc set; the placeholder must not escape validation")
	}
}

// TestSaveConfigReportsWriteFailures covers the write-error path: a path that
// cannot be created must surface as an error, not a silent no-op.
func TestSaveConfigReportsWriteFailures(t *testing.T) {
	dir := t.TempDir()

	tests := []struct {
		name string
		path string
	}{
		{"path is an existing directory", dir},
		{"parent directory does not exist", filepath.Join(dir, "missing", "config.json")},
		{"empty path", ""},
	}

	for _, test := range tests {
		err := SaveConfig(newLoaderTestConfig(), test.path)
		if err == nil {
			t.Errorf("%s: SaveConfig(%q) returned no error", test.name, test.path)

			continue
		}

		if !strings.Contains(err.Error(), "failed to write config file") {
			t.Errorf("%s: SaveConfig error = %q, want it to name the write failure", test.name, err)
		}
	}
}

// TestPrintPresets is a smoke test for the stdout listing: every preset name
// and its description must reach the caller's terminal.
func TestPrintPresets(t *testing.T) {
	text := captureStdout(t, PrintPresets)
	if text == "" {
		t.Fatal("PrintPresets wrote nothing to stdout")
	}

	if !strings.Contains(text, "Available Configuration Presets:") {
		t.Errorf("PrintPresets output has no heading:\n%s", text)
	}

	descriptions := ListPresets()

	for _, name := range PresetNames() {
		if !strings.Contains(text, name) {
			t.Errorf("PrintPresets output does not list %q:\n%s", name, text)
		}

		if !strings.Contains(text, descriptions[ConfigPreset(name)]) {
			t.Errorf("PrintPresets output does not describe %q:\n%s", name, text)
		}
	}
}
