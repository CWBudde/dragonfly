package dragonfly

import (
	"math"
	"testing"
)

// Tolerance for floating point comparisons.
const epsilon = 1e-10

// TestSphere tests the Sphere benchmark function.
func TestSphere(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
		{
			name:     "ones",
			x:        []float64{1.0, 1.0, 1.0},
			expected: 3.0,
		},
		{
			name:     "mixed",
			x:        []float64{1.0, -2.0, 3.0},
			expected: 14.0,
		},
		{
			name:     "single_dimension",
			x:        []float64{5.0},
			expected: 25.0,
		},
		{
			name:     "high_dimensional",
			x:        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
			expected: 10.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Sphere(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Sphere(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestSphereDimensionality tests that Sphere works with various dimensions.
func TestSphereDimensionality(t *testing.T) {
	dimensions := []int{1, 2, 5, 10, 30, 50, 100}

	for _, dim := range dimensions {
		t.Run(string(rune(dim)), func(t *testing.T) {
			x := make([]float64, dim)
			// All zeros should give minimum
			result := Sphere(x)
			if result != 0.0 {
				t.Errorf("Sphere(%dd zeros) = %v, want 0.0", dim, result)
			}

			// All ones should give dimension value
			for i := range x {
				x[i] = 1.0
			}

			result = Sphere(x)
			expected := float64(dim)

			if math.Abs(result-expected) > epsilon {
				t.Errorf("Sphere(%dd ones) = %v, want %v", dim, result, expected)
			}
		})
	}
}

// TestRastrigin tests the Rastrigin benchmark function.
func TestRastrigin(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
		{
			name:     "single_dimension_zero",
			x:        []float64{0.0},
			expected: 0.0,
		},
		{
			name:     "2d_origin",
			x:        []float64{0.0, 0.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Rastrigin(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Rastrigin(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestRastriginNonZero tests Rastrigin at non-zero points.
func TestRastriginNonZero(t *testing.T) {
	// Rastrigin is highly multimodal, so we just check properties
	tests := []struct {
		name string
		x    []float64
	}{
		{
			name: "ones",
			x:    []float64{1.0, 1.0},
		},
		{
			name: "mixed",
			x:    []float64{1.5, -1.5, 2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Rastrigin(tt.x)
			// Should be positive at non-zero points
			if result < 0 {
				t.Errorf("Rastrigin(%v) = %v, expected positive value", tt.x, result)
			}
			// Should be greater than at origin
			origin := make([]float64, len(tt.x))

			originResult := Rastrigin(origin)
			if result <= originResult {
				t.Logf("Rastrigin(%v) = %v, origin = %v (expected higher at non-zero)",
					tt.x, result, originResult)
			}
		})
	}
}

// TestRosenbrock tests the Rosenbrock benchmark function.
func TestRosenbrock(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_2d",
			x:        []float64{1.0, 1.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{1.0, 1.0, 1.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_5d",
			x:        []float64{1.0, 1.0, 1.0, 1.0, 1.0},
			expected: 0.0,
		},
		{
			name:     "zeros_2d",
			x:        []float64{0.0, 0.0},
			expected: 1.0, // (1-0)^2 = 1
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Rosenbrock(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Rosenbrock(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestRosenbrockNonOptimal tests Rosenbrock at non-optimal points.
func TestRosenbrockNonOptimal(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
	}{
		{
			name: "negative",
			x:    []float64{-1.0, -1.0},
		},
		{
			name: "far_from_optimum",
			x:    []float64{5.0, 5.0, 5.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Rosenbrock(tt.x)
			// Should be positive at non-optimal points
			if result <= 0 {
				t.Errorf("Rosenbrock(%v) = %v, expected positive value", tt.x, result)
			}
			// Should be greater than at optimum
			optimum := make([]float64, len(tt.x))
			for i := range optimum {
				optimum[i] = 1.0
			}

			optimumResult := Rosenbrock(optimum)
			if result <= optimumResult {
				t.Errorf("Rosenbrock(%v) = %v, optimum = %v (expected higher at non-optimal)",
					tt.x, result, optimumResult)
			}
		})
	}
}

func TestMichalewicz(t *testing.T) {
	if result := Michalewicz(nil); result != 0 {
		t.Errorf("Michalewicz(nil) = %v, want 0", result)
	}

	input := []float64{1, 2}

	want := 0.0
	for i, value := range input {
		want -= math.Sin(value) * math.Pow(math.Sin(float64(i+1)*value*value/math.Pi), 20)
	}

	if result := Michalewicz(input); math.Abs(result-want) > epsilon {
		t.Errorf("Michalewicz(%v) = %v, want %v", input, result, want)
	}
}

// TestAckley tests the Ackley benchmark function.
func TestAckley(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_1d",
			x:        []float64{0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_2d",
			x:        []float64{0.0, 0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Ackley(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Ackley(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestAckleyNonZero tests Ackley at non-zero points.
func TestAckleyNonZero(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
	}{
		{
			name: "ones",
			x:    []float64{1.0, 1.0},
		},
		{
			name: "far_from_origin",
			x:    []float64{5.0, 5.0, 5.0},
		},
		{
			name: "negative",
			x:    []float64{-2.0, -2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Ackley(tt.x)
			// Should be positive at non-zero points
			if result < 0 {
				t.Errorf("Ackley(%v) = %v, expected non-negative value", tt.x, result)
			}
			// Should be greater than at origin
			origin := make([]float64, len(tt.x))

			originResult := Ackley(origin)
			if result <= originResult {
				t.Logf("Ackley(%v) = %v, origin = %v (expected higher at non-zero)",
					tt.x, result, originResult)
			}
		})
	}
}

// TestGriewank tests the Griewank benchmark function.
func TestGriewank(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_1d",
			x:        []float64{0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_2d",
			x:        []float64{0.0, 0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Griewank(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Griewank(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestGriewankNonZero tests Griewank at non-zero points.
func TestGriewankNonZero(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
	}{
		{
			name: "ones",
			x:    []float64{1.0, 1.0},
		},
		{
			name: "large_values",
			x:    []float64{100.0, 100.0, 100.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Griewank(tt.x)
			// Should be non-negative (formula is sum/4000 - prod + 1)
			// At origin: 0/4000 - 1 + 1 = 0
			// At other points, typically positive
			if result < -epsilon {
				t.Errorf("Griewank(%v) = %v, expected non-negative value", tt.x, result)
			}
		})
	}
}

// TestBenchmarkFunctionsSymmetry tests that functions are symmetric around origin.
func TestBenchmarkFunctionsSymmetry(t *testing.T) {
	symmetricFunctions := []struct {
		name string
		fn   func([]float64) float64
	}{
		{"Sphere", Sphere},
		{"Rastrigin", Rastrigin},
		{"Ackley", Ackley},
		{"Griewank", Griewank},
	}

	x := []float64{2.5, -1.5, 3.0}
	xNeg := []float64{-2.5, 1.5, -3.0}

	for _, tt := range symmetricFunctions {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.fn(x)
			resultNeg := tt.fn(xNeg)

			if math.Abs(result-resultNeg) > epsilon {
				t.Errorf("%s not symmetric: f(%v)=%v, f(%v)=%v",
					tt.name, x, result, xNeg, resultNeg)
			}
		})
	}
}

// TestBenchmarkFunctionsMonotonicity tests some properties of benchmark functions.
func TestBenchmarkFunctionsMonotonicity(t *testing.T) {
	// For Sphere: moving away from origin should increase cost
	origin := []float64{0.0, 0.0}
	point1 := []float64{1.0, 0.0}
	point2 := []float64{2.0, 0.0}

	cost0 := Sphere(origin)
	cost1 := Sphere(point1)
	cost2 := Sphere(point2)

	if !(cost0 < cost1 && cost1 < cost2) {
		//nolint:dupword // repeated format verbs, not repeated words
		t.Errorf("Sphere not monotonic: f(%v)=%v, f(%v)=%v, f(%v)=%v",
			origin, cost0, point1, cost1, point2, cost2)
	}
}

// TestBenchmarkFunctionsEdgeCases tests edge cases.
func TestBenchmarkFunctionsEdgeCases(t *testing.T) {
	functions := []struct {
		name string
		fn   func([]float64) float64
	}{
		{"Sphere", Sphere},
		{"Rastrigin", Rastrigin},
		{"Rosenbrock", Rosenbrock},
		{"Ackley", Ackley},
		{"Griewank", Griewank},
	}

	t.Run("large_values", func(t *testing.T) {
		x := []float64{1000.0, 1000.0}
		for _, fn := range functions {
			result := fn.fn(x)
			// Should not produce NaN or Inf
			if math.IsNaN(result) {
				t.Errorf("%s(large values) = NaN", fn.name)
			}

			if math.IsInf(result, 0) {
				t.Errorf("%s(large values) = Inf", fn.name)
			}
		}
	})

	t.Run("very_small_values", func(t *testing.T) {
		x := []float64{1e-10, 1e-10}
		for _, fn := range functions {
			result := fn.fn(x)
			// Should not produce NaN
			if math.IsNaN(result) {
				t.Errorf("%s(small values) = NaN", fn.name)
			}
		}
	})
}

// TestRosenbrockSingleDimension tests Rosenbrock edge case.
func TestRosenbrockSingleDimension(t *testing.T) {
	// Rosenbrock requires at least 2 dimensions (uses x[i+1])
	x := []float64{1.0}
	result := Rosenbrock(x)
	// With single dimension, the loop doesn't execute, so result should be 0
	if result != 0.0 {
		t.Logf("Rosenbrock(1D) = %v (edge case, no pairs to compare)", result)
	}
}

// BenchmarkSphere benchmarks the Sphere function.
func BenchmarkSphere(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.1
	}

	b.ResetTimer()

	for range b.N {
		_ = Sphere(x)
	}
}

// BenchmarkRastrigin benchmarks the Rastrigin function.
func BenchmarkRastrigin(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.1
	}

	b.ResetTimer()

	for range b.N {
		_ = Rastrigin(x)
	}
}

// BenchmarkRosenbrock benchmarks the Rosenbrock function.
func BenchmarkRosenbrock(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.1
	}

	b.ResetTimer()

	for range b.N {
		_ = Rosenbrock(x)
	}
}

// BenchmarkAckley benchmarks the Ackley function.
func BenchmarkAckley(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.1
	}

	b.ResetTimer()

	for range b.N {
		_ = Ackley(x)
	}
}

// BenchmarkGriewank benchmarks the Griewank function.
func BenchmarkGriewank(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.1
	}

	b.ResetTimer()

	for range b.N {
		_ = Griewank(x)
	}
}

// TestSchwefel tests the Schwefel benchmark function.
func TestSchwefel(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_1d",
			x:        []float64{420.9687},
			expected: 0.0,
		},
		{
			name:     "global_minimum_2d",
			x:        []float64{420.9687, 420.9687},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Schwefel(tt.x)
			if math.Abs(result-tt.expected) > 1e-2 { // Schwefel needs larger tolerance
				t.Errorf("Schwefel(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestLevy tests the Levy benchmark function.
func TestLevy(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_2d",
			x:        []float64{1.0, 1.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{1.0, 1.0, 1.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Levy(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Levy(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestZakharov tests the Zakharov benchmark function.
func TestZakharov(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_1d",
			x:        []float64{0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_2d",
			x:        []float64{0.0, 0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Zakharov(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Zakharov(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}
}

// TestDixonPrice tests the Dixon-Price benchmark function.
func TestDixonPrice(t *testing.T) {
	// Test at global minimum (approximate)
	x := []float64{1.0, 0.707107, 0.577350, 0.5}
	result := DixonPrice(x)

	// Should be near zero at optimum
	if result < 0 || result > 1.0 {
		t.Logf("DixonPrice at near-optimum = %v (expected close to 0)", result)
	}

	// Test at origin
	origin := []float64{0.0, 0.0, 0.0}

	resultOrigin := DixonPrice(origin)
	if resultOrigin < 0 {
		t.Errorf("DixonPrice should be non-negative, got %v", resultOrigin)
	}
}

// TestBentCigar tests the Bent Cigar benchmark function.
func TestBentCigar(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_1d",
			x:        []float64{0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := BentCigar(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("BentCigar(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}

	// Test ill-conditioning property
	x := []float64{1.0, 1.0, 1.0}
	result := BentCigar(x)
	expected := 1.0 + 2.0*1e6 // First dimension normal, others scaled

	if math.Abs(result-expected) > epsilon {
		t.Errorf("BentCigar(%v) = %v, want %v (ill-conditioned test)", x, result, expected)
	}
}

// TestDiscus tests the Discus benchmark function.
func TestDiscus(t *testing.T) {
	tests := []struct {
		name     string
		x        []float64
		expected float64
	}{
		{
			name:     "global_minimum_1d",
			x:        []float64{0.0},
			expected: 0.0,
		},
		{
			name:     "global_minimum_3d",
			x:        []float64{0.0, 0.0, 0.0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Discus(tt.x)
			if math.Abs(result-tt.expected) > epsilon {
				t.Errorf("Discus(%v) = %v, want %v", tt.x, result, tt.expected)
			}
		})
	}

	// Test ill-conditioning property
	x := []float64{1.0, 1.0, 1.0}
	result := Discus(x)
	expected := 1e6 + 2.0 // First dimension scaled, others normal

	if math.Abs(result-expected) > epsilon {
		t.Errorf("Discus(%v) = %v, want %v (ill-conditioned test)", x, result, expected)
	}
}

// TestWeierstrass tests the Weierstrass benchmark function.
func TestWeierstrass(t *testing.T) {
	// Test at global minimum
	x := []float64{0.0, 0.0, 0.0}
	result := Weierstrass(x)

	// Should be at or very close to zero
	if math.Abs(result) > epsilon {
		t.Errorf("Weierstrass(%v) = %v, want 0.0", x, result)
	}
}

// TestHappyCat tests the HappyCat benchmark function.
func TestHappyCat(t *testing.T) {
	// Test at global minimum
	x := []float64{-1.0, -1.0, -1.0}
	result := HappyCat(x)

	// Should be at or very close to zero
	if math.Abs(result) > epsilon {
		t.Errorf("HappyCat(%v) = %v, want 0.0", x, result)
	}

	// Test at origin
	origin := []float64{0.0, 0.0, 0.0}

	resultOrigin := HappyCat(origin)
	if resultOrigin < 0 {
		t.Errorf("HappyCat should be non-negative, got %v", resultOrigin)
	}
}

// TestExpandedSchafferF6 tests the Expanded Schaffer F6 benchmark function.
func TestExpandedSchafferF6(t *testing.T) {
	// Test at global minimum
	x := []float64{0.0, 0.0, 0.0}
	result := ExpandedSchafferF6(x)

	// Should be at or very close to zero
	if math.Abs(result) > epsilon {
		t.Errorf("ExpandedSchafferF6(%v) = %v, want 0.0", x, result)
	}

	// Test with 2D
	x2d := []float64{0.0, 0.0}
	result2d := ExpandedSchafferF6(x2d)

	if math.Abs(result2d) > epsilon {
		t.Errorf("ExpandedSchafferF6(%v) = %v, want 0.0", x2d, result2d)
	}
}

// TestHimmelblau tests the Himmelblau benchmark function.
func TestHimmelblau(t *testing.T) {
	// The three non-trivial minimizers are published rounded to six decimals,
	// so they need a looser tolerance than the exactly representable cases.
	const roundedEpsilon = 1e-8

	tests := []struct {
		name string
		x    []float64
		want float64
		tol  float64
	}{
		{"first minimum", []float64{3.0, 2.0}, 0.0, epsilon},
		{"second minimum", []float64{-2.805118, 3.131312}, 0.0, roundedEpsilon},
		{"third minimum", []float64{-3.779310, -3.283186}, 0.0, roundedEpsilon},
		{"fourth minimum", []float64{3.584428, -1.848126}, 0.0, roundedEpsilon},
		{"two pairs", []float64{3.0, 2.0, 3.0, 2.0}, 0.0, epsilon},
		{"odd tail at zero", []float64{3.0, 2.0, 0.0}, 0.0, epsilon},
		{"odd tail scored", []float64{3.0, 2.0, 1.0}, 1.0, epsilon},
		{"origin", []float64{0.0, 0.0}, 170.0, epsilon},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Himmelblau(tt.x)
			if math.Abs(result-tt.want) > tt.tol {
				t.Errorf("Himmelblau(%v) = %v, want %v", tt.x, result, tt.want)
			}
		})
	}
}

// TestCECFunctionsNonNegative tests that CEC functions produce valid outputs.
func TestCECFunctionsNonNegative(t *testing.T) {
	cecFunctions := []struct {
		name string
		fn   func([]float64) float64
		x    []float64
	}{
		{"Schwefel", Schwefel, []float64{100.0, 100.0}},
		{"Levy", Levy, []float64{5.0, 5.0}},
		{"Zakharov", Zakharov, []float64{1.0, 1.0}},
		{"DixonPrice", DixonPrice, []float64{1.0, 1.0}},
		{"BentCigar", BentCigar, []float64{1.0, 1.0}},
		{"Discus", Discus, []float64{1.0, 1.0}},
		{"Weierstrass", Weierstrass, []float64{0.1, 0.1}},
		{"HappyCat", HappyCat, []float64{0.0, 0.0}},
		{"ExpandedSchafferF6", ExpandedSchafferF6, []float64{1.0, 1.0}},
		{"Himmelblau", Himmelblau, []float64{1.0, 1.0}},
	}

	for _, tt := range cecFunctions {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.fn(tt.x)
			// Check for NaN or Inf
			if math.IsNaN(result) {
				t.Errorf("%s(%v) = NaN", tt.name, tt.x)
			}

			if math.IsInf(result, 0) {
				t.Errorf("%s(%v) = Inf", tt.name, tt.x)
			}
		})
	}
}

// BenchmarkSchwefel benchmarks the Schwefel function.
func BenchmarkSchwefel(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 10.0
	}

	b.ResetTimer()

	for range b.N {
		_ = Schwefel(x)
	}
}

// BenchmarkLevy benchmarks the Levy function.
func BenchmarkLevy(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.1
	}

	b.ResetTimer()

	for range b.N {
		_ = Levy(x)
	}
}

// BenchmarkWeierstrass benchmarks the Weierstrass function.
func BenchmarkWeierstrass(b *testing.B) {
	x := make([]float64, 30)
	for i := range x {
		x[i] = float64(i) * 0.01
	}

	b.ResetTimer()

	for range b.N {
		_ = Weierstrass(x)
	}
}

// assertObjectives compares a multi-objective result against expected values.
func assertObjectives(t *testing.T, name string, x, got, want []float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s(%v) returned %d objectives, want %d", name, x, len(got), len(want))
	}

	for m := range want {
		if math.Abs(got[m]-want[m]) > epsilon {
			t.Errorf("%s(%v)[%d] = %v, want %v", name, x, m, got[m], want[m])
		}
	}
}

// TestZDT1 tests the ZDT1 multi-objective benchmark problem. On the Pareto front
// the tail variables are zero, so g = 1 and f2 = 1 - sqrt(f1).
func TestZDT1(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		want []float64
	}{
		{name: "front_origin", x: []float64{0, 0, 0}, want: []float64{0, 1}},
		{name: "front_quarter", x: []float64{0.25, 0, 0}, want: []float64{0.25, 0.5}},
		{name: "front_upper", x: []float64{1, 0, 0}, want: []float64{1, 0}},
		// g = 1 + 9*(0.5+0.5)/2 = 5.5; f2 = 5.5*(1 - sqrt(1/5.5)).
		{name: "off_front", x: []float64{1, 0.5, 0.5}, want: []float64{1, 5.5 * (1 - math.Sqrt(1/5.5))}},
		{name: "single_dimension", x: []float64{0.36}, want: []float64{0.36, 0.4}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertObjectives(t, "ZDT1", tt.x, ZDT1(tt.x), tt.want)
		})
	}
}

// TestZDT2 tests the ZDT2 multi-objective benchmark problem, whose front is the
// concave curve f2 = 1 - f1 squared.
func TestZDT2(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		want []float64
	}{
		{name: "front_origin", x: []float64{0, 0, 0}, want: []float64{0, 1}},
		{name: "front_half", x: []float64{0.5, 0, 0}, want: []float64{0.5, 0.75}},
		{name: "front_upper", x: []float64{1, 0, 0}, want: []float64{1, 0}},
		// g = 1 + 9*1/2 = 5.5; f2 = 5.5*(1 - (1/5.5)^2).
		{name: "off_front", x: []float64{1, 0.5, 0.5}, want: []float64{1, 5.5 * (1 - 1/(5.5*5.5))}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertObjectives(t, "ZDT2", tt.x, ZDT2(tt.x), tt.want)
		})
	}
}

// TestZDT3 tests the ZDT3 multi-objective benchmark problem, whose sine term
// breaks the front into five disconnected pieces.
func TestZDT3(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		want []float64
	}{
		// sin(0) = 0, so the sine term vanishes at both ends of the interval.
		{name: "front_origin", x: []float64{0, 0, 0}, want: []float64{0, 1}},
		{name: "front_upper", x: []float64{1, 0, 0}, want: []float64{1, 0}},
		// sin(10*pi*0.1) = sin(pi) = 0, so f2 = 1 - sqrt(0.1).
		{name: "front_sine_zero", x: []float64{0.1, 0, 0}, want: []float64{0.1, 1 - math.Sqrt(0.1)}},
		// sin(10*pi*0.05) = sin(pi/2) = 1, so f2 = 1 - sqrt(0.05) - 0.05.
		{name: "front_sine_peak", x: []float64{0.05, 0, 0}, want: []float64{0.05, 1 - math.Sqrt(0.05) - 0.05}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertObjectives(t, "ZDT3", tt.x, ZDT3(tt.x), tt.want)
		})
	}
}

// TestSchafferN1 tests the Schaffer N.1 multi-objective benchmark problem, the
// two shifted parabolas f1 = x squared and f2 = (x-2) squared.
func TestSchafferN1(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		want []float64
	}{
		{name: "front_lower", x: []float64{0}, want: []float64{0, 4}},
		{name: "front_middle", x: []float64{1}, want: []float64{1, 1}},
		{name: "front_upper", x: []float64{2}, want: []float64{4, 0}},
		{name: "outside_front", x: []float64{-1}, want: []float64{1, 9}},
		{name: "extra_components_ignored", x: []float64{1, 7, -3}, want: []float64{1, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertObjectives(t, "SchafferN1", tt.x, SchafferN1(tt.x), tt.want)
		})
	}
}

// TestMultiObjectiveFunctionsDegenerateInput checks that the multi-objective
// problems handle an empty position vector without panicking.
func TestMultiObjectiveFunctionsDegenerateInput(t *testing.T) {
	for name, fn := range map[string]MultiObjectiveFunction{
		"ZDT1": ZDT1, "ZDT2": ZDT2, "ZDT3": ZDT3, "SchafferN1": SchafferN1,
	} {
		values := fn(nil)
		if len(values) != 2 {
			t.Errorf("%s(nil) returned %d objectives, want 2", name, len(values))
		}
	}
}

// TestBenchmarkFunctionsEmptyInput checks the suite-wide convention that every
// single-objective benchmark function scores an empty position vector as 0,
// for both a nil slice and an allocated empty one, without panicking.
func TestBenchmarkFunctionsEmptyInput(t *testing.T) {
	functions := map[string]ObjectiveFunction{
		"Sphere":             Sphere,
		"Rastrigin":          Rastrigin,
		"Rosenbrock":         Rosenbrock,
		"Ackley":             Ackley,
		"Griewank":           Griewank,
		"Schwefel":           Schwefel,
		"Levy":               Levy,
		"Zakharov":           Zakharov,
		"Michalewicz":        Michalewicz,
		"DixonPrice":         DixonPrice,
		"BentCigar":          BentCigar,
		"Discus":             Discus,
		"Weierstrass":        Weierstrass,
		"HappyCat":           HappyCat,
		"ExpandedSchafferF6": ExpandedSchafferF6,
		"Himmelblau":         Himmelblau,
	}

	inputs := map[string][]float64{
		"nil":   nil,
		"empty": {},
	}

	for name, fn := range functions {
		for label, x := range inputs {
			if got := fn(x); got != 0 {
				t.Errorf("%s(%s) = %v, want 0", name, label, got)
			}
		}
	}
}
