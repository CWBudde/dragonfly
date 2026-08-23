package dragonfly

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// testConfig returns a valid configuration with a trivial objective, so each
// validation test can invalidate exactly one field and attribute the error to
// it.
func testConfig() *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = func(position []float64) float64 { return position[0] }
	config.ProblemSize = 3
	config.LowerBound = -5
	config.UpperBound = 5

	return config
}

func TestUnifrndStaysWithinRange(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	for range 1000 {
		value := unifrnd(-3, 7, rng)
		if value < -3 || value >= 7 {
			t.Fatalf("unifrnd = %v, want within [-3, 7)", value)
		}
	}
}

func TestUnifrndIsDeterministicForASeed(t *testing.T) {
	first := unifrnd(0, 1, rand.New(rand.NewSource(42)))
	second := unifrnd(0, 1, rand.New(rand.NewSource(42)))

	if first != second {
		t.Fatalf("unifrnd = %v then %v for the same seed, want identical", first, second)
	}
}

func TestUnifrndRejectsANilRNG(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("unifrnd(nil rng) did not panic")
		}
	}()

	unifrnd(2, 4, nil)
}

func TestUnifrndVecFillsEveryComponent(t *testing.T) {
	rng := rand.New(rand.NewSource(7))

	vec := unifrndVec(-1, 1, 5, rng)
	if len(vec) != 5 {
		t.Fatalf("len = %d, want 5", len(vec))
	}

	for i, value := range vec {
		if value < -1 || value >= 1 {
			t.Fatalf("component %d = %v, want within [-1, 1)", i, value)
		}
	}
}

func TestRandnHasRoughlyStandardNormalMoments(t *testing.T) {
	rng := rand.New(rand.NewSource(3))

	const samples = 20000

	var sum, sumSquares float64

	for range samples {
		value := randn(rng)
		sum += value
		sumSquares += value * value
	}

	mean := sum / samples
	variance := sumSquares/samples - mean*mean

	if math.Abs(mean) > 0.05 {
		t.Errorf("mean = %v, want near 0", mean)
	}

	if math.Abs(variance-1) > 0.05 {
		t.Errorf("variance = %v, want near 1", variance)
	}
}

func TestRandnRejectsANilRNG(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("randn(nil rng) did not panic")
		}
	}()

	randn(nil)
}

func TestMaxVecMinVecAndClampVec(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []float64
		apply func([]float64)
	}{
		{
			name:  "maxVec raises values below the lower bound",
			input: []float64{-5, 0, 5},
			want:  []float64{-1, 0, 5},
			apply: func(v []float64) { maxVec(v, -1) },
		},
		{
			name:  "maxVec leaves values on the bound alone",
			input: []float64{-1, -1},
			want:  []float64{-1, -1},
			apply: func(v []float64) { maxVec(v, -1) },
		},
		{
			name:  "minVec lowers values above the upper bound",
			input: []float64{-5, 0, 5},
			want:  []float64{-5, 0, 1},
			apply: func(v []float64) { minVec(v, 1) },
		},
		{
			name:  "clampVec applies both bounds",
			input: []float64{-9, -1, 0, 1, 9},
			want:  []float64{-2, -1, 0, 1, 2},
			apply: func(v []float64) { clampVec(v, -2, 2) },
		},
		{
			name:  "clampVec on an empty vector is a no-op",
			input: []float64{},
			want:  []float64{},
			apply: func(v []float64) { clampVec(v, 0, 1) },
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := copyVec(test.input)
			test.apply(got)
			assertVecEqual(t, got, test.want)
		})
	}
}

func TestSanitizeVecReplacesOnlyInvalidComponents(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	vec := []float64{math.NaN(), 1.5, math.Inf(1), -2.5, math.Inf(-1)}
	sanitizeVec(vec, -3, 3, rng)

	if vec[1] != 1.5 || vec[3] != -2.5 {
		t.Errorf("finite components changed: %v", vec)
	}

	for _, i := range []int{0, 2, 4} {
		if math.IsNaN(vec[i]) || math.IsInf(vec[i], 0) {
			t.Errorf("component %d = %v, want a repaired finite value", i, vec[i])
		}

		if vec[i] < -3 || vec[i] >= 3 {
			t.Errorf("component %d = %v, want within [-3, 3)", i, vec[i])
		}
	}
}

func TestSanitizeVecLeavesAFiniteVectorUntouched(t *testing.T) {
	rng := rand.New(rand.NewSource(12))

	vec := []float64{-1, 0, 1}
	sanitizeVec(vec, -10, 10, rng)
	assertVecEqual(t, vec, []float64{-1, 0, 1})
}

func TestSanitizeCost(t *testing.T) {
	tests := []struct {
		name string
		cost float64
		want float64
	}{
		{name: "NaN becomes positive infinity", cost: math.NaN(), want: math.Inf(1)},
		{name: "negative infinity becomes positive infinity", cost: math.Inf(-1), want: math.Inf(1)},
		{name: "positive infinity is already the worst value", cost: math.Inf(1), want: math.Inf(1)},
		{name: "a finite cost is returned unchanged", cost: -1234.5, want: -1234.5},
		{name: "zero is returned unchanged", cost: 0, want: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := sanitizeCost(test.cost); got != test.want {
				t.Errorf("sanitizeCost(%v) = %v, want %v", test.cost, got, test.want)
			}
		})
	}
}

func TestCopyVecReturnsAnIndependentCopy(t *testing.T) {
	original := []float64{1, 2, 3}

	clone := copyVec(original)
	clone[0] = 99

	if original[0] != 1 {
		t.Errorf("mutating the copy changed the original: %v", original)
	}

	if copyVec(nil) != nil {
		t.Error("copyVec(nil) = non-nil, want nil")
	}

	if empty := copyVec([]float64{}); len(empty) != 0 {
		t.Errorf("copyVec(empty) = %v, want an empty slice", empty)
	}
}

func TestApplyBoundsClamp(t *testing.T) {
	position := []float64{-7, -2, 0, 2, 7}
	step := []float64{1, 1, 1, 1, 1}

	applyBounds(position, step, -5, 5, BoundaryClamp, nil)

	assertVecEqual(t, position, []float64{-5, -2, 0, 2, 5})
	assertVecEqual(t, step, []float64{1, 1, 1, 1, 1})
}

func TestApplyBoundsWrapSendsAComponentToTheOppositeBound(t *testing.T) {
	rng := rand.New(rand.NewSource(5))

	position := []float64{7, 0, -7}
	step := []float64{3, 3, 3}

	applyBounds(position, step, -5, 5, BoundaryWrap, rng)

	assertVecEqual(t, position, []float64{-5, 0, 5})

	if step[1] != 3 {
		t.Errorf("step of the in-range component = %v, want it untouched at 3", step[1])
	}

	for _, i := range []int{0, 2} {
		if step[i] < 0 || step[i] >= 1 {
			t.Errorf("step[%d] = %v, want a fresh draw within [0, 1)", i, step[i])
		}
	}
}

func TestApplyBoundsDefaultsToWrap(t *testing.T) {
	rng := rand.New(rand.NewSource(6))

	position := []float64{9}
	step := []float64{4}

	applyBounds(position, step, -5, 5, BoundaryMethod("nonsense"), rng)

	if position[0] != -5 {
		t.Errorf("position = %v, want the wrap result of -5", position[0])
	}
}

func TestApplyBoundsReflect(t *testing.T) {
	tests := []struct {
		name         string
		position     float64
		step         float64
		wantPosition float64
		wantStep     float64
	}{
		{name: "above the upper bound", position: 7, step: 2, wantPosition: 3, wantStep: -2},
		{name: "below the lower bound", position: -7, step: -2, wantPosition: -3, wantStep: 2},
		{name: "in range is untouched", position: 1, step: 2, wantPosition: 1, wantStep: 2},
		{name: "on the bound is untouched", position: 5, step: 2, wantPosition: 5, wantStep: 2},
		// 26 reflects to -16, then to 6, then to 4: reflection repeats until
		// the component lands inside the box.
		{name: "far outside reflects repeatedly", position: 26, step: 1, wantPosition: 4, wantStep: -1},
		{name: "even number of crossings preserves step", position: 16, step: 1, wantPosition: -4, wantStep: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			position := []float64{test.position}
			step := []float64{test.step}

			applyBounds(position, step, -5, 5, BoundaryReflect, nil)

			if position[0] != test.wantPosition {
				t.Errorf("position = %v, want %v", position[0], test.wantPosition)
			}

			if step[0] != test.wantStep {
				t.Errorf("step = %v, want %v", step[0], test.wantStep)
			}
		})
	}
}

func TestValidateBoundsRejectsOverflowingSpan(t *testing.T) {
	config := testConfig()
	config.LowerBound = -math.MaxFloat64
	config.UpperBound = math.MaxFloat64

	err := validateBounds(config)
	if err == nil || !strings.Contains(err.Error(), "upper_bound-lower_bound") {
		t.Fatalf("validateBounds() error = %v, want non-finite span error", err)
	}
}

func TestApplyBoundsReflectPinsOnADegenerateBox(t *testing.T) {
	position := []float64{12}
	step := []float64{1}

	applyBounds(position, step, 3, 3, BoundaryReflect, nil)

	if position[0] != 3 {
		t.Errorf("position = %v, want the pinned bound 3", position[0])
	}
}

func TestApplyBoundsToleratesAShorterStep(t *testing.T) {
	position := []float64{9, 9}
	step := []float64{1}

	applyBounds(position, step, -5, 5, BoundaryWrap, rand.New(rand.NewSource(8)))

	assertVecEqual(t, position, []float64{-5, -5})
}

func TestApplyBoundsLeavesAnInRangePositionAlone(t *testing.T) {
	for _, method := range []BoundaryMethod{BoundaryWrap, BoundaryClamp, BoundaryReflect} {
		t.Run(string(method), func(t *testing.T) {
			position := []float64{-4, 0, 4}
			step := []float64{1, 2, 3}

			applyBounds(position, step, -5, 5, method, rand.New(rand.NewSource(9)))

			assertVecEqual(t, position, []float64{-4, 0, 4})
			assertVecEqual(t, step, []float64{1, 2, 3})
		})
	}
}

func TestEffectiveBoundaryMethod(t *testing.T) {
	tests := []struct {
		name   string
		method BoundaryMethod
		want   BoundaryMethod
	}{
		{name: "unset resolves to the paper's wrap", method: "", want: BoundaryWrap},
		{name: "wrap is taken as written", method: BoundaryWrap, want: BoundaryWrap},
		{name: "clamp is taken as written", method: BoundaryClamp, want: BoundaryClamp},
		{name: "reflect is taken as written", method: BoundaryReflect, want: BoundaryReflect},
		{name: "an unknown value resolves to wrap", method: BoundaryMethod("torus"), want: BoundaryWrap},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := &Config{BoundaryMethod: test.method}
			if got := effectiveBoundaryMethod(config); got != test.want {
				t.Errorf("effectiveBoundaryMethod = %q, want %q", got, test.want)
			}
		})
	}
}

func TestEffectiveBoundaryMethodDoesNotWriteBack(t *testing.T) {
	config := &Config{}

	effectiveBoundaryMethod(config)

	if config.BoundaryMethod != "" {
		t.Errorf("BoundaryMethod = %q, want the field left unwritten", config.BoundaryMethod)
	}
}

func TestEffectiveMaxWorkers(t *testing.T) {
	if got := effectiveMaxWorkers(&Config{MaxWorkers: 3}); got != 3 {
		t.Errorf("effectiveMaxWorkers = %d, want the written 3", got)
	}

	if got := effectiveMaxWorkers(&Config{MaxWorkers: 0}); got != defaultMaxWorkers() {
		t.Errorf("effectiveMaxWorkers = %d, want the default %d", got, defaultMaxWorkers())
	}
}

func TestValidateConfigAcceptsEveryPreset(t *testing.T) {
	presets := map[string]*Config{
		"default":          NewDefaultConfig(),
		"high dimensional": NewHighDimensionalConfig(),
		"fast convergence": NewFastConvergenceConfig(),
	}

	for name, config := range presets {
		t.Run(name, func(t *testing.T) {
			config.ObjectiveFunc = func(position []float64) float64 { return position[0] }
			config.ProblemSize = 10
			config.LowerBound = -100
			config.UpperBound = 100

			err := validateConfig(config)
			if err != nil {
				t.Errorf("validateConfig = %v, want nil", err)
			}
		})
	}
}

func TestValidateConfigAcceptsALiteralWeight(t *testing.T) {
	config := testConfig()
	config.SeparationWeight = 0
	config.EnemyWeight = 0.25

	err := validateConfig(config)
	if err != nil {
		t.Errorf("validateConfig = %v, want a literal weight to be accepted", err)
	}
}

func TestValidateConfigRejections(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantSub string
	}{
		{
			name:    "missing objective function",
			mutate:  func(c *Config) { c.ObjectiveFunc = nil },
			wantSub: "ObjectiveFunc",
		},
		{
			name:    "non-positive problem size",
			mutate:  func(c *Config) { c.ProblemSize = 0 },
			wantSub: "problem_size",
		},
		{
			name:    "inverted bounds",
			mutate:  func(c *Config) { c.LowerBound, c.UpperBound = 5, -5 },
			wantSub: "lower_bound",
		},
		{
			name:    "equal bounds",
			mutate:  func(c *Config) { c.LowerBound, c.UpperBound = 1, 1 },
			wantSub: "lower_bound",
		},
		{
			name:    "infinite upper bound",
			mutate:  func(c *Config) { c.UpperBound = math.Inf(1) },
			wantSub: "finite",
		},
		{
			name:    "NaN lower bound",
			mutate:  func(c *Config) { c.LowerBound = math.NaN() },
			wantSub: "finite",
		},
		{
			name:    "non-positive population",
			mutate:  func(c *Config) { c.NPop = 0 },
			wantSub: "npop",
		},
		{
			name:    "non-positive iteration cap",
			mutate:  func(c *Config) { c.MaxIterations = 0 },
			wantSub: "max_iterations",
		},
		{
			name:    "negative worker count",
			mutate:  func(c *Config) { c.MaxWorkers = -1 },
			wantSub: "max_workers",
		},
		{
			name:    "non-finite inertia start",
			mutate:  func(c *Config) { c.InertiaWeightStart = math.Inf(1) },
			wantSub: "inertia_weight_start",
		},
		{
			name:    "non-finite swarming weight",
			mutate:  func(c *Config) { c.CohesionWeight = math.NaN() },
			wantSub: "cohesion_weight",
		},
		{
			name:    "zero radius divisor",
			mutate:  func(c *Config) { c.RadiusInitialDivisor = 0 },
			wantSub: "radius_initial_divisor",
		},
		{
			name:    "negative radius growth",
			mutate:  func(c *Config) { c.RadiusGrowth = -1 },
			wantSub: "radius_growth",
		},
		{
			name:    "zero step ratio",
			mutate:  func(c *Config) { c.MaxStepRatio = 0 },
			wantSub: "max_step_ratio",
		},
		{
			name:    "enemy cutoff above one",
			mutate:  func(c *Config) { c.EnemyCutoffFraction = 1.5 },
			wantSub: "enemy_cutoff_fraction",
		},
		{
			name:    "enemy cutoff below zero",
			mutate:  func(c *Config) { c.EnemyCutoffFraction = -0.1 },
			wantSub: "enemy_cutoff_fraction",
		},
		{
			name:    "levy beta at the closed lower end",
			mutate:  func(c *Config) { c.LevyBeta = 0 },
			wantSub: "levy_beta",
		},
		{
			name:    "levy beta at the closed upper end",
			mutate:  func(c *Config) { c.LevyBeta = 2 },
			wantSub: "levy_beta",
		},
		{
			name:    "negative levy scale",
			mutate:  func(c *Config) { c.LevyScale = -0.01 },
			wantSub: "levy_scale",
		},
		{
			name:    "unknown boundary method",
			mutate:  func(c *Config) { c.BoundaryMethod = BoundaryMethod("bounce") },
			wantSub: "boundary_method",
		},
		{
			name:    "non-finite target cost",
			mutate:  func(c *Config) { c.Convergence = &ConvergenceConfig{TargetCost: pointerTo(math.NaN())} },
			wantSub: "target cost",
		},
		{
			name:    "negative minimum improvement",
			mutate:  func(c *Config) { c.Convergence = &ConvergenceConfig{MinImprovement: -1} },
			wantSub: "minimum improvement",
		},
		{
			name:    "negative stagnation window",
			mutate:  func(c *Config) { c.Convergence = &ConvergenceConfig{StagnationIterations: -1} },
			wantSub: "stagnation iterations",
		},
		{
			name:    "minimum iterations beyond the cap",
			mutate:  func(c *Config) { c.Convergence = &ConvergenceConfig{MinIterations: 10_000} },
			wantSub: "minimum iterations",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := testConfig()
			test.mutate(config)

			err := validateConfig(config)
			if err == nil {
				t.Fatalf("validateConfig = nil, want an error mentioning %q", test.wantSub)
			}

			if !strings.Contains(err.Error(), test.wantSub) {
				t.Errorf("validateConfig = %q, want it to mention %q", err, test.wantSub)
			}
		})
	}
}

func TestValidateConfigRejectsANilConfig(t *testing.T) {
	err := validateConfig(nil)
	if err == nil {
		t.Fatal("validateConfig(nil) = nil, want an error")
	}
}

func TestValidateConfigAcceptsAValidConvergenceBlock(t *testing.T) {
	config := testConfig()
	config.Convergence = &ConvergenceConfig{
		TargetCost:           pointerTo(1e-8),
		MinImprovement:       1e-10,
		StagnationIterations: 50,
		MinIterations:        10,
	}

	err := validateConfig(config)
	if err != nil {
		t.Errorf("validateConfig = %v, want nil", err)
	}
}

// pointerTo returns a pointer to value, so the table above can build a
// ConvergenceConfig with a target cost inline.
func pointerTo[T any](value T) *T {
	return &value
}

// assertVecEqual compares two vectors exactly. The helpers under test only ever
// copy, clamp or replace components, so no tolerance is warranted -- a
// difference in the last bit would be a real defect.
func assertVecEqual(t *testing.T, got, want []float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v vs %v)", len(got), len(want), got, want)
	}

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("component %d = %v, want %v (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
}
