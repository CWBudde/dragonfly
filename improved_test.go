package dragonfly

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestChaosMapNamesAndRanges(t *testing.T) {
	want := []ChaosMap{
		ChaosChebyshev, ChaosCircle, ChaosGauss, ChaosIterative, ChaosLogistic,
		ChaosPiecewise, ChaosSine, ChaosSinger, ChaosSinusoidal, ChaosTent,
	}
	if !reflect.DeepEqual(ChaosMapNames(), want) {
		t.Fatalf("ChaosMapNames() = %v, want %v", ChaosMapNames(), want)
	}

	for _, name := range ChaosMapNames() {
		value := 0.7

		for iteration := 1; iteration <= 100; iteration++ {
			next, err := ChaoticMapValue(name, value, iteration)
			if err != nil {
				t.Fatalf("%s iteration %d: %v", name, iteration, err)
			}

			value = next

			lower := 0.0
			if name == ChaosChebyshev || name == ChaosIterative {
				lower = -1
			}

			if value < lower || value > 1 || !isFinite(value) {
				t.Fatalf("%s iteration %d produced %v, want [%v,1]", name, iteration, value, lower)
			}
		}
	}
}

func TestRepresentativeChaosMapValues(t *testing.T) {
	tests := []struct {
		name ChaosMap
		want float64
	}{
		{ChaosLogistic, 0.84},
		{ChaosGauss, 3.0 / 7.0},
		{ChaosSine, math.Sin(0.7 * math.Pi)},
		{ChaosTent, 1},
	}

	for _, test := range tests {
		got, err := ChaoticMapValue(test.name, 0.7, 1)
		if err != nil {
			t.Fatalf("ChaoticMapValue(%q): %v", test.name, err)
		}

		if math.Abs(got-test.want) > 1e-12 {
			t.Errorf("ChaoticMapValue(%q, 0.7, 1) = %.16g, want %.16g", test.name, got, test.want)
		}
	}
}

func TestChaoticMapRejectsInvalidInput(t *testing.T) {
	_, unknownErr := ChaoticMapValue("unknown", 0.7, 1)
	if unknownErr == nil {
		t.Error("unknown chaos map was accepted")
	}

	_, nonFiniteErr := ChaoticMapValue(ChaosGauss, math.NaN(), 1)
	if nonFiniteErr == nil {
		t.Error("non-finite chaos input was accepted")
	}
}

func TestImprovedConfigFactories(t *testing.T) {
	memory := NewMemoryHybridConfig()
	if memory.UseBinary || memory.PSOCognitiveWeight != 2 || memory.PSOSocialWeight != 2 {
		t.Errorf("NewMemoryHybridConfig() = %+v", memory)
	}

	chaotic := NewChaoticConfig()
	if chaotic.UseBinary || chaotic.ChaosMap != ChaosGauss || chaotic.ChaosSeed != 0.7 {
		t.Errorf("NewChaoticConfig() = %+v", chaotic)
	}

	quantum := NewQuantumConfig()
	if quantum.UseBinary || quantum.GaussianMutationWeight != 1 ||
		math.Abs(quantum.QuantumRotationAngle-0.005*math.Pi) > 1e-15 {
		t.Errorf("NewQuantumConfig() = %+v", quantum)
	}
}

func TestImprovedVariantEvaluationAccounting(t *testing.T) {
	const (
		population = 6
		iterations = 3
	)

	tests := []struct {
		name string
		new  func() *Config
		run  func(*Config) (*Result, error)
		want int
	}{
		{
			name: "MHDA",
			new:  NewMemoryHybridConfig,
			run:  func(config *Config) (*Result, error) { return OptimizeMemoryHybrid(config) },
			want: population * (2*iterations + 1),
		},
		{
			name: "CDA",
			new:  NewChaoticConfig,
			run:  func(config *Config) (*Result, error) { return OptimizeChaotic(config) },
			want: population * (iterations + 1),
		},
		{
			name: "QGDA",
			new:  NewQuantumConfig,
			run:  func(config *Config) (*Result, error) { return OptimizeQuantum(config) },
			want: population * (3*iterations + 1),
		},
	}

	for _, test := range tests {
		config := test.new()
		config.ObjectiveFunc = Sphere
		config.ProblemSize = 4
		config.LowerBound = -5
		config.UpperBound = 5
		config.NPop = population
		config.MaxIterations = iterations
		seed := int64(17)
		config.Seed = &seed

		result, err := test.run(config)
		if err != nil {
			t.Fatalf("%s: %v", test.name, err)
		}

		if result.FuncEvalCount != test.want {
			t.Errorf("%s evaluations = %d, want %d", test.name, result.FuncEvalCount, test.want)
		}

		if result.IterationCount != iterations || len(result.ConvergenceCurve) != iterations {
			t.Errorf("%s iterations/curve = %d/%d, want %d/%d", test.name,
				result.IterationCount, len(result.ConvergenceCurve), iterations, iterations)
		}
	}
}

func TestChaoticWeightsUseOneMapValueForEveryDAFactor(t *testing.T) {
	config := NewChaoticConfig()
	config.LowerBound = -10
	config.UpperBound = 10
	sequence := &chaosSequence{name: ChaosGauss, value: 0.7, step: 0}

	weights := computeChaoticWeights(config, 1, 100, sequence)

	want, err := ChaoticMapValue(ChaosGauss, 0.7, 1)
	if err != nil {
		t.Fatal(err)
	}

	got := []float64{
		weights.Inertia, weights.Separation, weights.Alignment,
		weights.Cohesion, weights.Food, weights.Enemy,
	}
	for i, value := range got {
		if math.Abs(value-want) > 1e-15 {
			t.Errorf("weight %d = %.16g, want chaotic value %.16g", i, value, want)
		}
	}

	if sequence.step != 1 {
		t.Errorf("chaotic recurrence advanced %d times, want once", sequence.step)
	}
}

func TestImprovedVariantsAreParallelDeterministic(t *testing.T) {
	tests := []struct {
		name string
		new  func() *Config
		run  func(*Config) (*Result, error)
	}{
		{"MHDA", NewMemoryHybridConfig, func(config *Config) (*Result, error) {
			return OptimizeMemoryHybrid(config)
		}},
		{"CDA", NewChaoticConfig, func(config *Config) (*Result, error) {
			return OptimizeChaotic(config)
		}},
		{"QGDA", NewQuantumConfig, func(config *Config) (*Result, error) {
			return OptimizeQuantum(config)
		}},
	}

	for _, test := range tests {
		makeConfig := func(parallel bool) *Config {
			config := test.new()
			config.ObjectiveFunc = Rastrigin
			config.ProblemSize = 5
			config.LowerBound = -5.12
			config.UpperBound = 5.12
			config.NPop = 10
			config.MaxIterations = 12
			config.EnableParallel = parallel
			config.MaxWorkers = 3
			seed := int64(991)
			config.Seed = &seed

			return config
		}

		sequential, err := test.run(makeConfig(false))
		if err != nil {
			t.Fatalf("%s sequential: %v", test.name, err)
		}

		parallel, err := test.run(makeConfig(true))
		if err != nil {
			t.Fatalf("%s parallel: %v", test.name, err)
		}

		if !reflect.DeepEqual(sequential, parallel) {
			t.Errorf("%s sequential and parallel results differ\nsequential: %+v\nparallel: %+v",
				test.name, sequential, parallel)
		}
	}
}

func TestPSOCandidateEquation(t *testing.T) {
	config := NewMemoryHybridConfig()
	config.LowerBound = -100
	config.UpperBound = 100
	config.Rand = rand.New(rand.NewSource(7))

	swarm := []Dragonfly{{Position: []float64{2}, Step: []float64{0.5}}}
	memory := []Dragonfly{{Position: []float64{1}}}
	global := Best{Position: []float64{-1}}

	wantRNG := rand.New(rand.NewSource(7))
	r1 := wantRNG.Float64()
	r2 := wantRNG.Float64()
	wantStep := 0.4*0.5 + 2*r1*(1-2) + 2*r2*(-1-2)

	candidates := preparePSOCandidates(config, swarm, memory, global, 0.4, config.Rand)
	if math.Abs(candidates[0].Step[0]-wantStep) > 1e-12 {
		t.Errorf("PSO step = %v, want %v", candidates[0].Step[0], wantStep)
	}

	if math.Abs(candidates[0].Position[0]-(2+wantStep)) > 1e-12 {
		t.Errorf("PSO position = %v, want %v", candidates[0].Position[0], 2+wantStep)
	}
}

func TestQuantumRotationGate(t *testing.T) {
	alpha, beta := QuantumRotate(1, 0, math.Pi/2)
	if math.Abs(alpha) > 1e-12 || math.Abs(beta-1) > 1e-12 {
		t.Errorf("QuantumRotate(1,0,pi/2) = (%v,%v), want (0,1)", alpha, beta)
	}

	alpha, beta = QuantumRotate(0.3, -0.7, 0.123)
	if math.Abs(alpha*alpha+beta*beta-(0.3*0.3+0.7*0.7)) > 1e-12 {
		t.Error("quantum rotation did not preserve the two-state norm")
	}
}

func TestQuantumRotationDirectionFollowsPublishedTable(t *testing.T) {
	evaluator := newConstraintEvaluator(Sphere, nil)
	best := Best{Cost: 1, Position: []float64{2}}

	tests := []struct {
		name          string
		currentCost   float64
		alpha, beta   float64
		wantDirection float64
	}{
		{"worse same signs", 2, 1, 2, 1},
		{"worse opposite signs", 2, -1, 2, -1},
		{"better reverses direction", 0, 1, 2, -1},
		{"equal fitness does not rotate", 1, 1, 2, 0},
		{"zero beta does not rotate", 2, 1, 0, 0},
	}

	for _, test := range tests {
		current := Dragonfly{Cost: test.currentCost}

		got := quantumRotationDirection(evaluator, &current, best, test.alpha, test.beta)
		if got != test.wantDirection {
			t.Errorf("%s: direction = %v, want %v", test.name, got, test.wantDirection)
		}
	}
}

func TestImprovedVariantsRejectUnsupportedModesAndParameters(t *testing.T) {
	memory := NewMemoryHybridConfig()
	memory.ObjectiveFunc = Sphere
	memory.ProblemSize = 2
	memory.LowerBound = -1
	memory.UpperBound = 1

	memory.FidelityMode = FidelityMATLAB

	_, err := OptimizeMemoryHybrid(memory)
	if !errors.Is(err, ErrImprovedVariantMATLAB) {
		t.Errorf("MHDA MATLAB error = %v, want ErrImprovedVariantMATLAB", err)
	}

	quantum := NewQuantumConfig()
	quantum.ObjectiveFunc = Sphere
	quantum.ProblemSize = 2
	quantum.LowerBound = -1
	quantum.UpperBound = 1

	quantum.GaussianMutationWeight = 0

	_, err = OptimizeQuantum(quantum)
	if err == nil {
		t.Error("QGDA accepted zero Gaussian mutation weight")
	}

	chaotic := NewChaoticConfig()
	chaotic.ObjectiveFunc = Sphere
	chaotic.ProblemSize = 2
	chaotic.LowerBound = -5
	chaotic.UpperBound = 5

	chaotic.ChaosMap = "unknown"

	_, err = OptimizeChaoticContext(context.Background(), chaotic)
	if err == nil {
		t.Error("CDA accepted an unknown chaos map")
	}
}
