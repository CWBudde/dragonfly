package dragonfly

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
)

func TestCEC2017SuiteLoadsAllUsableFunctions(t *testing.T) {
	suite, err := CEC2017Suite(makeCECFixture(2017, 10), 10)
	if err != nil {
		t.Fatal(err)
	}

	if len(suite) != 29 {
		t.Fatalf("CEC2017 suite has %d functions, want 29", len(suite))
	}

	for _, problem := range suite {
		got, err := problem.Evaluate(problem.Optimum())
		if err != nil {
			t.Fatalf("%s optimum: %v", problem.Name(), err)
		}

		if math.Abs(got-problem.Minimum()) > 1e-7 {
			t.Errorf("%s optimum = %.12g, want %.12g", problem.Name(), got, problem.Minimum())
		}

		if problem.MaxEvaluations() != 100000 {
			t.Errorf("%s budget = %d, want 100000", problem.Name(), problem.MaxEvaluations())
		}
	}
}

func TestCEC2020SuiteLoadsAllFunctions(t *testing.T) {
	suite, err := CEC2020Suite(makeCECFixture(2020, 5), 5)
	if err != nil {
		t.Fatal(err)
	}

	if len(suite) != 10 {
		t.Fatalf("CEC2020 suite has %d functions, want 10", len(suite))
	}

	for _, problem := range suite {
		got, err := problem.Evaluate(problem.Optimum())
		if err != nil {
			t.Fatalf("%s optimum: %v", problem.Name(), err)
		}

		if math.Abs(got-problem.Minimum()) > 1e-7 {
			t.Errorf("%s optimum = %.12g, want %.12g", problem.Name(), got, problem.Minimum())
		}

		if problem.MaxEvaluations() != 50000 {
			t.Errorf("%s budget = %d, want 50000", problem.Name(), problem.MaxEvaluations())
		}
	}
}

func TestCEC2020EvaluationBudgets(t *testing.T) {
	tests := []struct {
		dimension int
		want      int
	}{
		{5, 50000},
		{10, 1000000},
		{15, 3000000},
		{20, 10000000},
		{30, 0},
	}

	for _, test := range tests {
		if got := cec2020EvaluationBudget(test.dimension); got != test.want {
			t.Errorf("CEC2020 D%d budget = %d, want %d", test.dimension, got, test.want)
		}
	}
}

func TestCECConstructorsRejectInvalidOrIncompleteInputs(t *testing.T) {
	requireError := func(err error, message string) {
		t.Helper()

		if err == nil {
			t.Fatal(message)
		}
	}

	_, err := NewCEC2017Problem(makeCECFixture(2017, 10), 2, 10)
	requireError(err, "CEC2017 F2 should be rejected")

	_, err = NewCEC2017Problem(makeCECFixture(2017, 10), 1, 20)
	requireError(err, "non-competition CEC2017 dimension should be rejected")

	_, err = NewCEC2020Problem(makeCECFixture(2020, 5), 11, 5)
	requireError(err, "CEC2020 F11 should be rejected")

	_, err = NewCEC2020Problem(makeCECFixture(2020, 5), 1, 30)
	requireError(err, "non-competition CEC2020 dimension should be rejected")

	_, err = NewCEC2020Problem(nil, 1, 5)
	requireError(err, "nil data filesystem should be rejected")

	truncated := fstest.MapFS{
		"M_1_D10.txt":            &fstest.MapFile{Data: []byte("1 0")},
		"shift_data_1.txt":       &fstest.MapFile{Data: []byte(strings.Repeat("0 ", 10))},
		"shuffle_data_1_D10.txt": &fstest.MapFile{Data: []byte("1 2 3 4 5 6 7 8 9 10")},
	}
	_, err = NewCEC2017Problem(truncated, 1, 10)
	requireError(err, "truncated rotation data should be rejected")

	nonFiniteRotation := makeCECFixture(2020, 5)
	nonFiniteRotation["M_1_D5.txt"] = &fstest.MapFile{Data: []byte(strings.Repeat("NaN ", 25))}
	_, err = NewCEC2020Problem(nonFiniteRotation, 1, 5)
	requireError(err, "non-finite rotation data should be rejected")

	nonFiniteShift := makeCECFixture(2020, 5)
	nonFiniteShift["shift_data_1.txt"] = &fstest.MapFile{Data: []byte("NaN 0 0 0 0")}
	_, err = NewCEC2020Problem(nonFiniteShift, 1, 5)
	requireError(err, "non-finite shift data should be rejected")

	badShuffle := makeCECFixture(2020, 5)
	badShuffle["shuffle_data_4_D5.txt"] = &fstest.MapFile{Data: []byte("1 2 3 4 6")}
	_, err = NewCEC2020Problem(badShuffle, 5, 5)
	requireError(err, "out-of-range shuffle data should be rejected")

	singularLevy := makeCECFixture(2017, 10)
	singularLevy["M_9_D10.txt"] = &fstest.MapFile{Data: []byte(strings.Repeat("0 ", 100))}
	_, err = NewCEC2017Problem(singularLevy, 9, 10)
	requireError(err, "CEC2017 F9 should reject a singular rotation matrix")
}

func TestCECReferenceEvaluatorQuirksRemainCompatible(t *testing.T) {
	x := []float64{1.25, -0.75, 0.5}
	shift := []float64{0.1, -0.2, 0.3}
	rotation := []float64{
		0, 1, 0,
		1, 0, 0,
		0, 0, 1,
	}
	// The released Schaffer F7 evaluator uses its pre-rotation scratch.
	gotSchaffer := cecEvaluateBase(cecSchafferF7, x, shift, rotation)

	wantSchaffer := cecSchafferF7Value(cecTransform(x, shift, nil, 1))
	if gotSchaffer != wantSchaffer {
		t.Fatalf("Schaffer F7 reference compatibility = %v, want %v", gotSchaffer, wantSchaffer)
	}
	// Its non-continuous Rastrigin rounding writes a discarded scratch vector.
	gotStep := cecEvaluateBase(cecNonContinuousRastrigin, x, shift, rotation)

	wantRastrigin := cecEvaluateBase(cecRastrigin, x, shift, rotation)
	if gotStep != wantRastrigin {
		t.Fatalf("non-continuous Rastrigin reference compatibility = %v, want %v", gotStep, wantRastrigin)
	}

	problem, err := NewCEC2020Problem(makeCECFixture(2020, 5), 7, 5)
	if err != nil {
		t.Fatal(err)
	}

	value, err := problem.Evaluate([]float64{0.1, 0.2, 0.3, 0.4, 0.5})
	if err != nil || !isFinite(value) {
		t.Fatalf("CEC2020 F7 D5 should repair the reference's one-dimensional elliptic partition: value=%v err=%v", value, err)
	}
}

func TestCECReferenceRegressionValues(t *testing.T) {
	tests := []struct {
		year      int
		function  int
		dimension int
		want      float64
	}{
		{2017, 1, 10, 3840100.0099999998},
		{2017, 11, 10, 1101.6478203230527},
		{2017, 21, 10, 2116.1588829169646},
		{2020, 1, 5, 540100.01000000001},
		{2020, 5, 5, 251700.35379191718},
		{2020, 8, 5, 2224.6285043138464},
	}

	for _, test := range tests {
		var (
			problem *BenchmarkCase
			err     error
		)

		if test.year == 2017 {
			problem, err = NewCEC2017Problem(
				makeCECFixture(test.year, test.dimension), test.function, test.dimension,
			)
		} else {
			problem, err = NewCEC2020Problem(
				makeCECFixture(test.year, test.dimension), test.function, test.dimension,
			)
		}

		if err != nil {
			t.Fatal(err)
		}

		position := make([]float64, test.dimension)
		for i := range position {
			position[i] = float64(i+1) / 10
		}

		got, err := problem.Evaluate(position)
		if err != nil {
			t.Fatal(err)
		}

		tolerance := 1e-12 * max(1, math.Abs(test.want))
		if math.Abs(got-test.want) > tolerance {
			t.Errorf("CEC%d F%d D%d = %.17g, want %.17g", test.year, test.function, test.dimension, got, test.want)
		}
	}
}

func TestCECDataLoaderAcceptsOfficialArchiveRoots(t *testing.T) {
	tests := []struct {
		prefix    string
		year      int
		dimension int
	}{
		{"input_data/", 2020, 5},
		{"CEC17_fast_pow/input_data/", 2017, 10},
		{"Matlab version/input_data/", 2020, 5},
	}

	for _, test := range tests {
		data := prefixCECFixture(makeCECFixture(test.year, test.dimension), test.prefix)

		var err error
		if test.year == 2017 {
			_, err = NewCEC2017Problem(data, 1, test.dimension)
		} else {
			_, err = NewCEC2020Problem(data, 1, test.dimension)
		}

		if err != nil {
			t.Errorf("load CEC%d from %q: %v", test.year, test.prefix, err)
		}
	}
}

func TestCECProblemIsSafeForOptimizerUse(t *testing.T) {
	problem, err := NewCEC2020Problem(makeCECFixture(2020, 5), 1, 5)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(problem.Objective()([]float64{0}), 1) {
		t.Fatal("wrong-dimension objective should score +Inf")
	}

	_, evaluateErr := problem.Evaluate([]float64{101, 0, 0, 0, 0})
	if evaluateErr == nil {
		t.Fatal("out-of-bounds physical position should be rejected")
	}

	base := NewDefaultConfig()
	target := 123.0
	base.NPop = 17
	base.MaxIterations = 23
	base.Convergence = &ConvergenceConfig{TargetCost: &target, MinIterations: 7}

	config, err := problem.NewConfig(base)
	if err != nil {
		t.Fatal(err)
	}

	if config.ProblemSize != 5 || config.LowerBound != 0 || config.UpperBound != 1 {
		t.Fatalf("unexpected normalized config: D=%d bounds=[%v,%v]", config.ProblemSize, config.LowerBound, config.UpperBound)
	}

	if base.ProblemSize != 0 || base.ObjectiveFunc != nil {
		t.Fatal("NewConfig mutated its base configuration")
	}

	if config.NPop != 17 || config.MaxIterations != 23 {
		t.Fatal("NewConfig did not preserve algorithm tuning from its base")
	}

	*config.Convergence.TargetCost = 456
	config.Convergence.MinIterations = 9

	if *base.Convergence.TargetCost != 123 || base.Convergence.MinIterations != 7 {
		t.Fatal("NewConfig's convergence settings alias its base")
	}

	center := []float64{0.5, 0.5, 0.5, 0.5, 0.5}

	physical, err := problem.Decode(center)
	if err != nil {
		t.Fatal(err)
	}

	want, err := problem.Evaluate(physical)
	if err != nil {
		t.Fatal(err)
	}

	if got := config.ObjectiveFunc(center); got != want {
		t.Fatalf("normalized objective = %v, want physical value %v", got, want)
	}

	_, binaryErr := problem.NewConfig(NewBinaryConfig())
	if binaryErr == nil {
		t.Fatal("continuous CEC problem should reject a binary base configuration")
	}

	lower, upper := problem.Bounds()
	lower[0], upper[0] = -1, 1

	lowerAgain, upperAgain := problem.Bounds()
	if lowerAgain[0] == -1 || upperAgain[0] == 1 {
		t.Fatal("Bounds returned mutable aliases")
	}

	optimum := problem.Optimum()
	optimum[0] = 99

	if problem.Optimum()[0] == 99 {
		t.Fatal("Optimum returned a mutable alias")
	}

	_, decodeErr := problem.Decode([]float64{math.NaN(), 0, 0, 0, 0})
	if decodeErr == nil {
		t.Fatal("Decode should reject non-finite normalized coordinates")
	}

	_, evaluateErr = problem.Evaluate([]float64{math.NaN(), 0, 0, 0, 0})
	if evaluateErr == nil {
		t.Fatal("Evaluate should reject non-finite physical coordinates")
	}
}

func TestCECObjectiveIsSafeForConcurrentEvaluation(t *testing.T) {
	problem, err := NewCEC2017Problem(makeCECFixture(2017, 10), 21, 10)
	if err != nil {
		t.Fatal(err)
	}

	position := make([]float64, problem.Dimension())
	objective := problem.Objective()
	want := objective(position)
	results := make(chan float64, 32)

	var group sync.WaitGroup
	for range 32 {
		group.Add(1)

		go func() {
			defer group.Done()

			results <- objective(position)
		}()
	}

	group.Wait()
	close(results)

	for got := range results {
		if got != want {
			t.Fatalf("concurrent objective = %v, want %v", got, want)
		}
	}
}

func makeCECFixture(year, dimension int) fstest.MapFS {
	data := fstest.MapFS{}
	internalFunctions := []int{}

	if year == 2017 {
		for function := 1; function <= 30; function++ {
			if function != 2 {
				internalFunctions = append(internalFunctions, function)
			}
		}
	} else {
		internalFunctions = append(internalFunctions, cec2020Internal[1:]...)
	}

	for _, internal := range internalFunctions {
		components := 1
		if internal > 20 {
			components = 10
		}

		data[fmt.Sprintf("M_%d_D%d.txt", internal, dimension)] = &fstest.MapFile{
			Data: []byte(identityMatrices(components, dimension)),
		}
		data[fmt.Sprintf("shift_data_%d.txt", internal)] = &fstest.MapFile{
			Data: []byte(distinctShiftRows(components, dimension)),
		}
		shuffleCount := 0

		if year == 2017 {
			if internal >= 11 && internal <= 20 {
				shuffleCount = dimension
			} else if internal == 29 || internal == 30 {
				shuffleCount = 10 * dimension
			}
		} else if internal == 4 || internal == 6 || (internal >= 11 && internal <= 20) {
			shuffleCount = dimension
		}

		if shuffleCount > 0 {
			var shuffle strings.Builder
			for i := range shuffleCount {
				fmt.Fprintf(&shuffle, "%d ", i%dimension+1)
			}

			data[fmt.Sprintf("shuffle_data_%d_D%d.txt", internal, dimension)] = &fstest.MapFile{Data: []byte(shuffle.String())}
		}
	}

	return data
}

func prefixCECFixture(data fstest.MapFS, prefix string) fstest.MapFS {
	prefixed := make(fstest.MapFS, len(data))
	for name, file := range data {
		prefixed[prefix+name] = file
	}

	return prefixed
}

func identityMatrices(count, dimension int) string {
	var result strings.Builder

	for range count {
		for row := range dimension {
			for column := range dimension {
				if row == column {
					result.WriteString("1 ")
				} else {
					result.WriteString("0 ")
				}
			}
		}
	}

	return result.String()
}

func distinctShiftRows(count, dimension int) string {
	var result strings.Builder

	for row := range count {
		for range dimension {
			fmt.Fprintf(&result, "%d ", row*10)
		}

		result.WriteByte('\n')
	}

	return result.String()
}
