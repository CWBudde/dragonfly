//go:build js && wasm

package main

import (
	"context"
	"math"
	"math/rand"
	"syscall/js"

	"github.com/CWBudde/dragonfly"
)

// The binary page needs objectives defined over 0/1 vectors, and the library's
// benchmark suite is continuous. These three are the demo's own, kept small on
// purpose: every one of them can be checked against the bit matrix by eye,
// which is the only way a reader can tell that the transfer function is doing
// what the curve beside it claims.
type binaryProblem struct {
	build func(bits int) dragonfly.ObjectiveFunction

	// optimum is the best attainable cost, where it is known in closed form.
	optimum func(bits int) (float64, bool)

	name  string
	blurb string
}

var binaryProblems = map[string]binaryProblem{
	"TargetPattern": {
		name:    "TargetPattern",
		build:   targetPattern,
		optimum: func(int) (float64, bool) { return 0, true },
		blurb: "Hamming distance to a hidden bit pattern. The one problem here whose answer is " +
			"visible in the matrix itself: every column converges to the same stripe.",
	},
	"Knapsack": {
		name:    "Knapsack",
		build:   knapsack,
		optimum: func(int) (float64, bool) { return 0, false },
		blurb: "Fixed weights and values, capacity at half the total weight. Cost is value " +
			"forgone, so a full column is not automatically good.",
	},
	"MaxOnes": {
		name:    "MaxOnes",
		build:   maxOnes,
		optimum: func(int) (float64, bool) { return 0, true },
		blurb: "Set every bit. The sanity check: a run that cannot solve this has a broken " +
			"transfer function, not a hard problem.",
	},
}

func binaryProblemNames() []string {
	return []string{"TargetPattern", "Knapsack", "MaxOnes"}
}

// problemRNG seeds the generated problem data from the bit count alone, never
// from the run's seed. A knapsack whose weights moved with the seed would make
// "same seed, same answer" true and "different seed, comparable answer" false,
// and the page would be quietly comparing different problems.
func problemRNG(bits int) *rand.Rand {
	return rand.New(rand.NewSource(int64(bits) * 7919))
}

func targetPattern(bits int) dragonfly.ObjectiveFunction {
	rng := problemRNG(bits)
	target := make([]float64, bits)

	for i := range target {
		if rng.Float64() < 0.5 {
			target[i] = 1
		}
	}

	return func(x []float64) float64 {
		distance := 0.0

		for i := range target {
			if i < len(x) && x[i] != target[i] {
				distance++
			}
		}

		return distance
	}
}

func maxOnes(bits int) dragonfly.ObjectiveFunction {
	return func(x []float64) float64 {
		missing := 0.0

		for i := range x {
			if x[i] < 0.5 {
				missing++
			}
		}

		return missing + float64(bits-len(x))
	}
}

func knapsack(bits int) dragonfly.ObjectiveFunction {
	rng := problemRNG(bits)

	weights := make([]float64, bits)
	values := make([]float64, bits)
	total := 0.0

	for i := range weights {
		weights[i] = 1 + math.Floor(rng.Float64()*20)
		values[i] = 1 + math.Floor(rng.Float64()*20)
		total += weights[i]
	}

	capacity := total / 2

	best := 0.0
	for _, value := range values {
		best += value
	}

	return func(x []float64) float64 {
		weight := 0.0
		value := 0.0

		for i := range weights {
			if i < len(x) && x[i] >= 0.5 {
				weight += weights[i]
				value += values[i]
			}
		}

		// Over capacity is penalized rather than rejected: a hard rejection
		// would make every infeasible position equally bad and flatten the
		// gradient the swarm needs to climb back out.
		if weight > capacity {
			return best + (weight-capacity)*10
		}

		return best - value
	}
}

// jsBinary performs one BDA run and returns the swarm as a bit matrix per
// iteration.
//
// The only thing that differs from the continuous variant is the position
// update: ΔX is computed exactly as before, and the transfer function turns its
// magnitude into a per-bit flip probability. That is why the page plots the
// chosen function's curve beside the matrix -- its height at Δx is literally
// the probability that a bit flipped this iteration.
func jsBinary(opts js.Value) any {
	problemName := readString(opts, "problem", "TargetPattern")

	problem, ok := binaryProblems[problemName]
	if !ok {
		return errorResult("binary: unknown problem %q", problemName)
	}

	var (
		bits       = clampInt(readInt(opts, "bits", 24), 2, maxDimensions)
		iterations = clampInt(readInt(opts, "iterations", 150), 1, maxIterations)
		npop       = clampInt(readInt(opts, "npop", 30), 2, maxPopulation)
		seed       = int64(readFloat(opts, "seed", 42))
		transfer   = readString(opts, "transfer", string(dragonfly.DefaultTransferFunction))
	)

	config := dragonfly.NewBinaryConfig()
	config.ObjectiveFunc = problem.build(bits)
	config.ProblemSize = bits
	config.MaxIterations = iterations
	config.NPop = npop
	config.TransferFunc = dragonfly.TransferFunction(transfer)
	config.EnableParallel = false
	config.Rand = rngFor(seed)

	film := newBitFilm(iterations, npop, bits)

	result, err := dragonfly.OptimizeBinaryContext(
		context.Background(),
		config,
		dragonfly.WithPopulationObserver(film.observe),
	)
	if err != nil {
		return errorResult("binary: %v", err)
	}

	film.truncate(result.IterationCount)

	response := map[string]any{
		"problem":           problem.name,
		"transfer":          transfer,
		"bits":              bits,
		"npop":              npop,
		"iterations":        result.IterationCount,
		"evaluations":       result.FuncEvalCount,
		"terminationReason": string(result.TerminationReason),
		"bestCost":          jsNumber(result.GlobalBest.Cost),
		"bestBits":          floatsToJS(result.GlobalBest.Position),
		"optimum":           optionalNumber(problem.optimum(bits)),
		"seed":              float64(seed),
	}

	out := opts.Get("out")

	putFloats(response, out, "convergence", toFloat32(result.ConvergenceCurve))
	putFloats(response, out, "matrix", film.matrix)
	putFloats(response, out, "setCount", film.setCount)
	putFloats(response, out, "agreement", film.agreement)

	curve, curveErr := transferSamples(config.TransferFunc, transferCurveSpan, transferCurveSamples)
	if curveErr != nil {
		return errorResult("binary: %v", curveErr)
	}

	response["transferSpan"] = transferCurveSpan

	putFloats(response, out, "transferCurve", curve)

	return response
}

const (
	transferCurveSpan    = 5.0
	transferCurveSamples = 200
)

// transferSamples evaluates the chosen transfer function across [-span, span].
//
// It goes through the library's own LookupTransferFunction rather than being
// transcribed into JavaScript, which is the same rule the rest of the demo
// follows and matters more here than anywhere else: the curve's whole job is to
// explain a bit flip the reader just watched, so a curve that disagreed with
// the function that produced the flip would be worse than no curve at all.
func transferSamples(name dragonfly.TransferFunction, span float64, samples int) ([]float32, error) {
	fn, err := dragonfly.LookupTransferFunction(name)
	if err != nil {
		return nil, err
	}

	curve := make([]float32, samples)

	for i := range curve {
		x := -span + (float64(i)/float64(samples-1))*span*2
		curve[i] = float32(fn(x))
	}

	return curve, nil
}

// bitFilm records the swarm as a bit matrix, laid out iteration x dragonfly x
// bit so the page can slice one frame with a single subarray.
type bitFilm struct {
	matrix   []float32
	setCount []float32

	// agreement is, per iteration per bit, the fraction of the swarm that has
	// that bit set. It is what turns the matrix from a flicker into a picture:
	// a bit the swarm has settled on reads as a solid row.
	agreement []float32

	npop int
	bits int
}

func newBitFilm(iterations, npop, bits int) *bitFilm {
	return &bitFilm{
		matrix:    make([]float32, 0, iterations*npop*bits),
		setCount:  make([]float32, 0, iterations),
		agreement: make([]float32, 0, iterations*bits),
		npop:      npop,
		bits:      bits,
	}
}

func (f *bitFilm) observe(snapshot dragonfly.PopulationSnapshot) {
	agreement := make([]float32, f.bits)

	for i := range snapshot.Swarm {
		position := snapshot.Swarm[i].Position

		for b := range f.bits {
			bit := float32(0)

			if b < len(position) && position[b] >= 0.5 {
				bit = 1
				agreement[b]++
			}

			f.matrix = append(f.matrix, bit)
		}
	}

	for b := range agreement {
		agreement[b] /= float32(len(snapshot.Swarm))
	}

	f.agreement = append(f.agreement, agreement...)

	set := 0

	for b := 0; b < f.bits && b < len(snapshot.Best.Position); b++ {
		if snapshot.Best.Position[b] >= 0.5 {
			set++
		}
	}

	f.setCount = append(f.setCount, float32(set))
}

func (f *bitFilm) truncate(iterations int) {
	f.matrix = clip(f.matrix, iterations*f.npop*f.bits)
	f.setCount = clip(f.setCount, iterations)
	f.agreement = clip(f.agreement, iterations*f.bits)
}
