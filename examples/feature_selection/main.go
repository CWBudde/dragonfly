// Command feature_selection solves a small wrapper-style feature-selection
// problem with BDA, the binary Dragonfly Algorithm.
//
// A synthetic data set of 12 candidate features is generated, of which only
// four actually drive the response; the remaining eight are noise, and two of
// them are near-duplicates of informative ones so that a greedy selection is
// tempted to keep them. A candidate solution is a 12-bit mask saying which
// features a linear model is allowed to use. The objective is the model's
// training error plus a small per-feature price, so that a feature has to earn
// its place:
//
//	cost(mask) = MSE(mask) + lambda * |mask| / d
//
// Run it with:
//
//	go run .
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/MeKo-Christian/dragonfly"
)

const (
	samples        = 200
	features       = 12
	featurePrice   = 0.05
	dataSeed       = 7
	optimizerSeed  = 2024
	noiseAmplitude = 0.35
)

// trueCoefficients is the model the data is generated from: features 1, 4, 7
// and 9 matter and the rest are noise. Features 2 and 8 are built as copies of
// 1 and 7 with a little jitter, so a selection that keeps them fits almost as
// well and must be rejected by the per-feature price rather than by the error.
var trueCoefficients = map[int]float64{1: 2.5, 4: -1.8, 7: 3.1, 9: 1.2}

func main() {
	design, response := makeDataset()

	config := dragonfly.NewBinaryConfig()
	config.ProblemSize = features
	config.NPop = 40
	config.MaxIterations = 400
	config.Rand = rand.New(rand.NewSource(optimizerSeed))
	config.ObjectiveFunc = func(mask []float64) float64 {
		return selectionCost(design, response, mask)
	}

	result, err := dragonfly.OptimizeBinary(config)
	if err != nil {
		fmt.Println("optimization failed:", err)

		return
	}

	report(design, response, result)
}

// report prints the selected mask next to the features that actually matter.
func report(design [][]float64, response []float64, result *dragonfly.Result) {
	mask := result.GlobalBest.Position

	fmt.Printf("evaluations: %d, iterations: %d, cost: %.5f\n",
		result.FuncEvalCount, result.IterationCount, result.GlobalBest.Cost)
	fmt.Printf("mean squared error of the selection: %.5f\n", meanSquaredError(design, response, mask))
	fmt.Println()
	fmt.Println("feature  selected  informative")

	for j := range features {
		_, informative := trueCoefficients[j]
		fmt.Printf("%7d  %8v  %11v\n", j, mask[j] == 1, informative)
	}

	fmt.Println()
	fmt.Println("selected:", indicesOf(mask))
	fmt.Println("truth:   ", sortedKeys(trueCoefficients))
}

// selectionCost is the objective: training error plus a price per selected
// feature. An empty selection is rejected outright, because the error of a
// model with nothing to fit is not comparable to the others.
func selectionCost(design [][]float64, response, mask []float64) float64 {
	chosen := indicesOf(mask)
	if len(chosen) == 0 {
		return math.Inf(1)
	}

	return meanSquaredError(design, response, mask) + featurePrice*float64(len(chosen))/features
}

// meanSquaredError fits an ordinary least-squares model on the selected columns
// and returns its training error.
func meanSquaredError(design [][]float64, response, mask []float64) float64 {
	chosen := indicesOf(mask)
	if len(chosen) == 0 {
		return math.Inf(1)
	}

	coefficients := leastSquares(design, response, chosen)
	total := 0.0

	for i, row := range design {
		prediction := 0.0
		for k, j := range chosen {
			prediction += coefficients[k] * row[j]
		}

		residual := response[i] - prediction
		total += residual * residual
	}

	return total / float64(len(design))
}

// leastSquares solves the normal equations XᵀX b = Xᵀy for the selected
// columns by Gaussian elimination with partial pivoting. A tiny ridge term
// keeps a singular system (two selected columns that are near-copies of one
// another) solvable.
func leastSquares(design [][]float64, response []float64, chosen []int) []float64 {
	size := len(chosen)
	normal := make([][]float64, size)

	for a := range size {
		normal[a] = make([]float64, size+1)

		for b := range size {
			sum := 0.0
			for _, row := range design {
				sum += row[chosen[a]] * row[chosen[b]]
			}

			normal[a][b] = sum
		}

		normal[a][a] += 1e-8

		target := 0.0
		for i, row := range design {
			target += row[chosen[a]] * response[i]
		}

		normal[a][size] = target
	}

	return solve(normal)
}

// solve runs Gaussian elimination with partial pivoting on an augmented matrix.
func solve(matrix [][]float64) []float64 {
	size := len(matrix)

	for column := range size {
		pivot := column
		for row := column + 1; row < size; row++ {
			if math.Abs(matrix[row][column]) > math.Abs(matrix[pivot][column]) {
				pivot = row
			}
		}

		matrix[column], matrix[pivot] = matrix[pivot], matrix[column]

		if matrix[column][column] == 0 {
			continue
		}

		for row := column + 1; row < size; row++ {
			factor := matrix[row][column] / matrix[column][column]
			for k := column; k <= size; k++ {
				matrix[row][k] -= factor * matrix[column][k]
			}
		}
	}

	solution := make([]float64, size)

	for row := size - 1; row >= 0; row-- {
		if matrix[row][row] == 0 {
			continue
		}

		sum := matrix[row][size]
		for k := row + 1; k < size; k++ {
			sum -= matrix[row][k] * solution[k]
		}

		solution[row] = sum / matrix[row][row]
	}

	return solution
}

// makeDataset builds the synthetic problem described in the package comment.
func makeDataset() ([][]float64, []float64) {
	rng := rand.New(rand.NewSource(dataSeed))
	design := make([][]float64, samples)
	response := make([]float64, samples)

	for i := range design {
		row := make([]float64, features)
		for j := range row {
			row[j] = rng.NormFloat64()
		}

		// Features 2 and 8 are near-copies of the informative features 1 and 7.
		row[2] = row[1] + 0.05*rng.NormFloat64()
		row[8] = row[7] + 0.05*rng.NormFloat64()

		value := 0.0
		for j, coefficient := range trueCoefficients {
			value += coefficient * row[j]
		}

		design[i] = row
		response[i] = value + noiseAmplitude*rng.NormFloat64()
	}

	return design, response
}

// indicesOf returns the positions of the set bits of a mask.
func indicesOf(mask []float64) []int {
	chosen := make([]int, 0, len(mask))

	for j, bit := range mask {
		if bit == 1 {
			chosen = append(chosen, j)
		}
	}

	return chosen
}

// sortedKeys returns the feature indices of the true model in ascending order.
func sortedKeys(model map[int]float64) []int {
	keys := make([]int, 0, len(model))

	for j := range features {
		if _, ok := model[j]; ok {
			keys = append(keys, j)
		}
	}

	return keys
}
