package dragonfly

import (
	"context"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// transferReference holds one independently computed value of a transfer
// function. The expectations were evaluated outside Go, from the analytic
// forms in PLAN.md §1.6, so a transcription error in binary.go cannot cancel
// out against the same error here.
type transferReference struct {
	name TransferFunction
	dx   float64
	want float64
}

var transferReferences = []transferReference{
	{"v1", 0.0, 0},
	{"v1", 0.5, 0.46911594893005931},
	{"v1", 1.0, 0.78990859455606266},
	{"v1", -1.0, 0.78990859455606266},
	{"v1", 2.0, 0.98781111781519715},
	{"v1", -3.0, 0.99983004752482341},
	{"v1", 10.0, 1},
	{"v1", -10.0, 1},

	{"v2", 0.0, 0},
	{"v2", 0.5, 0.46211715726000974},
	{"v2", 1.0, 0.76159415595576485},
	{"v2", -1.0, 0.76159415595576485},
	{"v2", 2.0, 0.9640275800758169},
	{"v2", -3.0, 0.99505475368673046},
	{"v2", 10.0, 0.99999999587769273},
	{"v2", -10.0, 0.99999999587769273},

	{"v3", 0.0, 0},
	{"v3", 0.5, 0.44721359549995793},
	{"v3", 1.0, 0.70710678118654746},
	{"v3", -1.0, 0.70710678118654746},
	{"v3", 2.0, 0.89442719099991586},
	{"v3", -3.0, 0.94868329805051377},
	{"v3", 10.0, 0.99503719020998915},
	{"v3", -10.0, 0.99503719020998915},

	{"v4", 0.0, 0},
	{"v4", 0.5, 0.42384473319136162},
	{"v4", 1.0, 0.63909292677189167},
	{"v4", -1.0, 0.63909292677189167},
	{"v4", 2.0, 0.80381347609541276},
	{"v4", -3.0, 0.86687984924793093},
	{"v4", 10.0, 0.95952614569197137},
	{"v4", -10.0, 0.95952614569197137},

	{"s1", 0.0, 0.5},
	{"s1", 0.5, 0.7310585786300049},
	{"s1", 1.0, 0.88079707797788231},
	{"s1", -1.0, 0.11920292202211755},
	{"s1", 2.0, 0.98201379003790845},
	{"s1", -3.0, 0.0024726231566347743},
	{"s1", 10.0, 0.99999999793884631},
	{"s1", -10.0, 2.0611536181902037e-09},

	{"s2", 0.0, 0.5},
	{"s2", 0.5, 0.62245933120185459},
	{"s2", 1.0, 0.7310585786300049},
	{"s2", -1.0, 0.2689414213699951},
	{"s2", 2.0, 0.88079707797788231},
	{"s2", -3.0, 0.047425873177566781},
	{"s2", 10.0, 0.99995460213129761},
	{"s2", -10.0, 4.5397868702434395e-05},

	{"s3", 0.0, 0.5},
	{"s3", 0.5, 0.56217650088579807},
	{"s3", 1.0, 0.62245933120185459},
	{"s3", -1.0, 0.37754066879814541},
	{"s3", 2.0, 0.7310585786300049},
	{"s3", -3.0, 0.18242552380635635},
	{"s3", 10.0, 0.99330714907571527},
	{"s3", -10.0, 0.0066928509242848554},

	{"s4", 0.0, 0.5},
	{"s4", 0.5, 0.5415704832167999},
	{"s4", 1.0, 0.58257020646231472},
	{"s4", -1.0, 0.41742979353768528},
	{"s4", 2.0, 0.66075636876581723},
	{"s4", -3.0, 0.2689414213699951},
	{"s4", 10.0, 0.96555480433378893},
	{"s4", -10.0, 0.034445195666211167},
}

func TestTransferFunctionsMatchTheirAnalyticForm(t *testing.T) {
	for _, reference := range transferReferences {
		transfer, err := LookupTransferFunction(reference.name)
		if err != nil {
			t.Fatalf("LookupTransferFunction(%q): %v", reference.name, err)
		}

		got := transfer(reference.dx)
		if math.Abs(got-reference.want) > 1e-12 {
			t.Errorf("%s(%v) = %.17g, want %.17g", reference.name, reference.dx, got, reference.want)
		}
	}
}

func TestTransferFunctionsStayInUnitInterval(t *testing.T) {
	for _, name := range TransferFunctionNames() {
		transfer, err := LookupTransferFunction(name)
		if err != nil {
			t.Fatalf("LookupTransferFunction(%q): %v", name, err)
		}

		for dx := -60.0; dx <= 60.0; dx += 0.01 {
			got := transfer(dx)
			if got < 0 || got > 1 || math.IsNaN(got) {
				t.Fatalf("%s(%v) = %v, want a probability in [0,1]", name, dx, got)
			}
		}
	}
}

func TestVShapedTransferFunctionsAreSymmetric(t *testing.T) {
	for _, name := range []TransferFunction{TransferV1, TransferV2, TransferV3, TransferV4} {
		transfer, err := LookupTransferFunction(name)
		if err != nil {
			t.Fatalf("LookupTransferFunction(%q): %v", name, err)
		}

		if transfer(0) != 0 {
			t.Errorf("%s(0) = %v, want 0: a V-shaped function has its minimum at zero", name, transfer(0))
		}

		for dx := 0.0; dx <= 20.0; dx += 0.01 {
			left, right := transfer(-dx), transfer(dx)
			if math.Abs(left-right) > 1e-15 {
				t.Fatalf("%s is not symmetric: T(%v) = %v, T(%v) = %v", name, -dx, left, dx, right)
			}
		}
	}
}

func TestSShapedTransferFunctionsAreMonotoneIncreasing(t *testing.T) {
	for _, name := range []TransferFunction{TransferS1, TransferS2, TransferS3, TransferS4} {
		transfer, err := LookupTransferFunction(name)
		if err != nil {
			t.Fatalf("LookupTransferFunction(%q): %v", name, err)
		}

		if math.Abs(transfer(0)-0.5) > 1e-15 {
			t.Errorf("%s(0) = %v, want 0.5: an S-shaped function crosses one half at zero", name, transfer(0))
		}

		previous := transfer(-30)
		for dx := -30.0; dx <= 30.0; dx += 0.01 {
			current := transfer(dx)
			if current < previous {
				t.Fatalf("%s decreased at dx = %v: %v after %v", name, dx, current, previous)
			}

			previous = current
		}
	}
}

func TestLookupTransferFunctionRejectsUnknownName(t *testing.T) {
	transfer, err := LookupTransferFunction("v9")
	if err == nil {
		t.Fatalf("LookupTransferFunction(%q) returned a function, want an error", "v9")
	}

	if transfer != nil {
		t.Errorf("LookupTransferFunction returned a function alongside its error")
	}
}

func TestEffectiveTransferFunctionDefaultsToV3(t *testing.T) {
	config := NewDefaultConfig()

	transfer, err := effectiveTransferFunction(config)
	if err != nil {
		t.Fatalf("effectiveTransferFunction: %v", err)
	}

	v3, err := LookupTransferFunction(TransferV3)
	if err != nil {
		t.Fatalf("LookupTransferFunction(v3): %v", err)
	}

	for _, dx := range []float64{-3, -0.5, 0, 0.5, 3} {
		if transfer(dx) != v3(dx) {
			t.Fatalf("default transfer function differs from v3 at %v", dx)
		}
	}
}

func TestEffectiveTransferFunctionRejectsUnknownName(t *testing.T) {
	config := NewBinaryConfig()
	config.TransferFunc = "sigmoid"

	_, err := effectiveTransferFunction(config)
	if err == nil {
		t.Fatal("effectiveTransferFunction accepted an unregistered name")
	}
}

// TestFlipRateMatchesTransferProbability is the statistical statement the
// bit-flip update makes: over many draws with a fixed ΔX, a component flips
// with probability exactly T(Δx).
//
// The tolerance is four standard errors of a binomial proportion,
// 4*sqrt(p(1-p)/n), which for n = 200000 is at most 0.0045. Four sigma leaves a
// false-failure probability below 1e-4 per component, and the run is seeded, so
// the test either passes forever or is genuinely broken.
func TestFlipRateMatchesTransferProbability(t *testing.T) {
	const draws = 200_000

	transfer, err := LookupTransferFunction(TransferV3)
	if err != nil {
		t.Fatalf("LookupTransferFunction(v3): %v", err)
	}

	step := []float64{0, 0.25, 1, -1, 4}
	flips := make([]int, len(step))
	rng := rand.New(rand.NewSource(1))

	for range draws {
		fly := &Dragonfly{Position: make([]float64, len(step)), Step: step}
		flipBits(fly, transfer, rng)

		for j, value := range fly.Position {
			if value == 1 {
				flips[j]++
			}
		}
	}

	for j, dx := range step {
		want := transfer(dx)
		got := float64(flips[j]) / draws
		tolerance := 4 * math.Sqrt(want*(1-want)/draws)

		if math.Abs(got-want) > tolerance {
			t.Errorf("flip rate at dx = %v: got %.5f, want %.5f ± %.5f", dx, got, want, tolerance)
		}
	}
}

func TestFlipBitsOnlyProducesBits(t *testing.T) {
	transfer, err := LookupTransferFunction(TransferS2)
	if err != nil {
		t.Fatalf("LookupTransferFunction(s2): %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	fly := &Dragonfly{Position: []float64{0, 1, 0, 1}, Step: []float64{3, -3, 0.5, -0.5}}

	for range 500 {
		flipBits(fly, transfer, rng)

		if !BinaryPositionsValid(fly.Position) {
			t.Fatalf("position left {0,1}: %v", fly.Position)
		}
	}
}

// TestBuildBinaryStepLeavesPositionUnchanged pins the contract that makes the
// reuse of dragonfly.go's step builders safe: they commit ΔX to the position,
// and buildBinaryStep has to undo that.
func TestBuildBinaryStepLeavesPositionUnchanged(t *testing.T) {
	config := NewBinaryConfig()
	config.ProblemSize = 4
	config.NPop = 5
	config.ObjectiveFunc = sumVector

	rng := rand.New(rand.NewSource(11))
	state := initializeBinaryRun(config, runOptions{}, rng)
	weights := computeWeights(config, 0, config.MaxIterations, rng)

	for i := range state.swarm {
		before := copyVec(state.swarm[i].Position)
		buildBinaryStep(state, i, weights, rng)

		for j := range before {
			if state.swarm[i].Position[j] != before[j] {
				t.Fatalf("buildBinaryStep moved dragonfly %d: %v -> %v", i, before, state.swarm[i].Position)
			}
		}
	}
}

func sumVector(x []float64) float64 {
	total := 0.0
	for _, value := range x {
		total += value
	}

	return total
}

func TestOptimizeBinaryKeepsPositionsBinaryThroughout(t *testing.T) {
	config := NewBinaryConfig()
	config.ProblemSize = 12
	config.NPop = 15
	config.MaxIterations = 60
	config.Rand = rand.New(rand.NewSource(3))
	config.ObjectiveFunc = sumVector

	observed := 0
	observer := func(snapshot PopulationSnapshot) {
		observed++

		for i, fly := range snapshot.Swarm {
			if !BinaryPositionsValid(fly.Position) {
				t.Errorf("iteration %d, dragonfly %d left {0,1}: %v", snapshot.Iteration, i, fly.Position)
			}
		}

		if !BinaryPositionsValid(snapshot.Best.Position) {
			t.Errorf("iteration %d: food source left {0,1}: %v", snapshot.Iteration, snapshot.Best.Position)
		}
	}

	result, err := OptimizeBinaryContext(context.Background(), config, WithPopulationObserver(observer))
	if err != nil {
		t.Fatalf("OptimizeBinaryContext: %v", err)
	}

	if observed != config.MaxIterations {
		t.Errorf("observed %d iterations, want %d", observed, config.MaxIterations)
	}

	if !BinaryPositionsValid(result.GlobalBest.Position) {
		t.Errorf("result position left {0,1}: %v", result.GlobalBest.Position)
	}

	// Minimizing the number of set bits has a unique optimum: the all-zero
	// string.
	if result.GlobalBest.Cost != 0 {
		t.Errorf("best cost = %v, want 0 (the all-zero bit string)", result.GlobalBest.Cost)
	}
}

func TestOptimizeBinaryRejectsNonBinaryBounds(t *testing.T) {
	config := NewBinaryConfig()
	config.ProblemSize = 4
	config.ObjectiveFunc = sumVector
	config.LowerBound = -5
	config.UpperBound = 5

	_, err := OptimizeBinary(config)
	if err == nil {
		t.Fatal("OptimizeBinary accepted bounds other than [0,1]")
	}
}

func TestOptimizeBinaryRejectsUnknownTransferFunction(t *testing.T) {
	config := NewBinaryConfig()
	config.ProblemSize = 4
	config.ObjectiveFunc = sumVector
	config.TransferFunc = "nope"

	_, err := OptimizeBinary(config)
	if err == nil {
		t.Fatal("OptimizeBinary accepted an unregistered transfer function")
	}
}

func TestOptimizeBinaryIsDeterministicForASeed(t *testing.T) {
	run := func() *Result {
		config := NewBinaryConfig()
		config.ProblemSize = 14
		config.NPop = 20
		config.MaxIterations = 80
		config.Rand = rand.New(rand.NewSource(20240823))
		config.ObjectiveFunc = knapsackObjective

		result, err := OptimizeBinary(config)
		if err != nil {
			t.Fatalf("OptimizeBinary: %v", err)
		}

		return result
	}

	first, second := run(), run()

	if first.GlobalBest.Cost != second.GlobalBest.Cost {
		t.Fatalf("costs differ between seeded runs: %v and %v", first.GlobalBest.Cost, second.GlobalBest.Cost)
	}

	for j := range first.GlobalBest.Position {
		if first.GlobalBest.Position[j] != second.GlobalBest.Position[j] {
			t.Fatalf("positions differ between seeded runs: %v and %v",
				first.GlobalBest.Position, second.GlobalBest.Position)
		}
	}

	for i := range first.ConvergenceCurve {
		if first.ConvergenceCurve[i] != second.ConvergenceCurve[i] {
			t.Fatalf("convergence curves differ at iteration %d: %v and %v",
				i, first.ConvergenceCurve[i], second.ConvergenceCurve[i])
		}
	}
}

// The knapsack instance used by the tests below: 14 items, capacity 35.
var (
	knapsackValues   = []float64{10, 5, 15, 7, 6, 18, 3, 12, 9, 20, 4, 11, 8, 14}
	knapsackWeights  = []float64{4, 2, 7, 3, 2, 8, 1, 5, 4, 9, 2, 5, 3, 6}
	knapsackCapacity = 35.0
)

// knapsackObjective is the toy 0/1 knapsack: minimize the negated value of the
// chosen items, with an overweight selection penalized so heavily that no
// infeasible selection can ever be the incumbent.
func knapsackObjective(x []float64) float64 {
	value, weight := 0.0, 0.0

	for i, bit := range x {
		if bit == 1 {
			value += knapsackValues[i]
			weight += knapsackWeights[i]
		}
	}

	if weight > knapsackCapacity {
		return 1000 + (weight - knapsackCapacity)
	}

	return -value
}

// bruteForceKnapsack enumerates all 2^14 selections, so the test compares
// against the true optimum rather than a remembered number.
func bruteForceKnapsack() float64 {
	best := math.Inf(1)
	bits := make([]float64, len(knapsackValues))

	for mask := range 1 << len(knapsackValues) {
		for i := range bits {
			bits[i] = float64((mask >> i) & 1)
		}

		if cost := knapsackObjective(bits); cost < best {
			best = cost
		}
	}

	return best
}

func TestOptimizeBinarySolvesAKnapsack(t *testing.T) {
	optimum := bruteForceKnapsack()

	config := NewBinaryConfig()
	config.ProblemSize = len(knapsackValues)
	config.NPop = 40
	config.MaxIterations = 300
	config.Rand = rand.New(rand.NewSource(42))
	config.ObjectiveFunc = knapsackObjective

	result, err := OptimizeBinary(config)
	if err != nil {
		t.Fatalf("OptimizeBinary: %v", err)
	}

	if result.GlobalBest.Cost != optimum {
		t.Errorf("best cost = %v, want the brute-force optimum %v (position %v)",
			result.GlobalBest.Cost, optimum, result.GlobalBest.Position)
	}

	if !BinaryPositionsValid(result.GlobalBest.Position) {
		t.Errorf("result position left {0,1}: %v", result.GlobalBest.Position)
	}
}

func TestNewBinaryConfigIsABinaryPreset(t *testing.T) {
	config := NewBinaryConfig()

	if !config.UseBinary {
		t.Error("NewBinaryConfig did not set UseBinary")
	}

	if config.TransferFunc != TransferV3 {
		t.Errorf("transfer function = %q, want %q", config.TransferFunc, TransferV3)
	}

	if config.LowerBound != 0 || config.UpperBound != 1 {
		t.Errorf("bounds = [%v, %v], want [0, 1]", config.LowerBound, config.UpperBound)
	}
}

// TestValidateInitialBitsRejectsNonBinaryComponents covers every rejection
// branch: a seeded population must be exactly 0/1-valued, and the error must
// name the offending element so the caller can find it.
func TestValidateInitialBitsRejectsNonBinaryComponents(t *testing.T) {
	tests := []struct {
		name      string
		positions [][]float64
		wantErr   bool
		wantIn    string
	}{
		{"nil population", nil, false, ""},
		{"empty rows", [][]float64{{}, {}}, false, ""},
		{"all bits", [][]float64{{0, 1, 0}, {1, 1, 1}}, false, ""},
		{"fractional component", [][]float64{{0, 1}, {0.5, 1}}, true, "position 1 component 0"},
		{"rounded-looking component", [][]float64{{0, 0.9999999}}, true, "position 0 component 1"},
		{"negative component", [][]float64{{-1, 0}}, true, "position 0 component 0"},
		{"two-valued component", [][]float64{{1, 1}, {1, 2}}, true, "position 1 component 1"},
		{"NaN component", [][]float64{{0, math.NaN()}}, true, "position 0 component 1"},
		{"infinite component", [][]float64{{math.Inf(1)}}, true, "position 0 component 0"},
		// Ragged rows are not this function's business: it only checks values.
		{"ragged but binary", [][]float64{{1}, {0, 1, 0}}, false, ""},
	}

	for _, test := range tests {
		err := validateInitialBits(test.positions)

		switch {
		case test.wantErr && err == nil:
			t.Errorf("%s: validateInitialBits returned no error", test.name)
		case !test.wantErr && err != nil:
			t.Errorf("%s: validateInitialBits = %v, want nil", test.name, err)
		case test.wantErr && !strings.Contains(err.Error(), test.wantIn):
			t.Errorf("%s: validateInitialBits = %q, want it to name %q", test.name, err, test.wantIn)
		}
	}
}

// TestBinaryPositionsValidRejectsNonBits pins the exported invariant check on
// both answers, including the values a rounding bug would produce.
func TestBinaryPositionsValidRejectsNonBits(t *testing.T) {
	tests := []struct {
		name     string
		position []float64
		want     bool
	}{
		{"nil", nil, true},
		{"empty", []float64{}, true},
		{"zeros and ones", []float64{0, 1, 1, 0}, true},
		{"fractional", []float64{0, 0.5, 1}, false},
		{"just above one", []float64{1, 1.0000001}, false},
		{"negative zero is zero", []float64{math.Copysign(0, -1), 1}, true},
		{"negative one", []float64{-1}, false},
		{"NaN", []float64{math.NaN()}, false},
		{"infinite", []float64{math.Inf(-1), 0}, false},
	}

	for _, test := range tests {
		if got := BinaryPositionsValid(test.position); got != test.want {
			t.Errorf("%s: BinaryPositionsValid(%v) = %v, want %v",
				test.name, test.position, got, test.want)
		}
	}
}

func TestOptimizeBinaryContextRejectsArchiveObserver(t *testing.T) {
	config := NewBinaryConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 4
	config.MaxIterations = 2

	result, err := OptimizeBinaryContext(context.Background(), config,
		WithArchiveObserver(func(ArchiveSnapshot) {}))
	if err == nil {
		t.Fatal("a binary run accepted WithArchiveObserver")
	}

	if result != nil {
		t.Error("a rejected run must not return a partial result")
	}
}
