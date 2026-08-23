// BDA, the binary variant of the Dragonfly Algorithm.

package dragonfly

import (
	"context"
	"fmt"
	"math"
	"math/rand"
)

// TransferFunction names one of the standard transfer functions that turn a
// step component into a bit-flip probability.
//
// The V-shaped family (v1..v4) is symmetric about zero: it reads the magnitude
// of a step as "how unsettled is this bit", and flips regardless of the step's
// sign. The S-shaped family (s1..s4) is monotone increasing: a positive step
// pushes the bit towards one and a negative step towards zero, in the sense
// that the flip probability crosses one half at zero.
type TransferFunction string

const (
	// TransferV1 is |erf(√π/2 · Δx)|.
	TransferV1 TransferFunction = "v1"
	// TransferV2 is |tanh(Δx)|.
	TransferV2 TransferFunction = "v2"
	// TransferV3 is |Δx / √(Δx²+1)|, the paper's default.
	TransferV3 TransferFunction = "v3"
	// TransferV4 is |(2/π)·arctan((π/2)·Δx)|.
	TransferV4 TransferFunction = "v4"
	// TransferS1 is 1/(1+e^(-2Δx)).
	TransferS1 TransferFunction = "s1"
	// TransferS2 is 1/(1+e^(-Δx)), the logistic sigmoid.
	TransferS2 TransferFunction = "s2"
	// TransferS3 is 1/(1+e^(-Δx/2)).
	TransferS3 TransferFunction = "s3"
	// TransferS4 is 1/(1+e^(-Δx/3)).
	TransferS4 TransferFunction = "s4"
)

// DefaultTransferFunction is the one BDA uses in the paper: the V-shaped v3.
// A Config that leaves TransferFunc empty runs with it.
const DefaultTransferFunction = TransferV3

// transferFunc maps a step component Δx to a bit-flip probability in [0,1].
type transferFunc func(float64) float64

// transferFunctions is the registry behind LookupTransferFunction. The forms
// are written out in the constant documentation above; they are transcribed
// here exactly once each.
var transferFunctions = map[TransferFunction]transferFunc{
	TransferV1: func(dx float64) float64 { return math.Abs(math.Erf(math.Sqrt(math.Pi) / 2 * dx)) },
	TransferV2: func(dx float64) float64 { return math.Abs(math.Tanh(dx)) },
	TransferV3: func(dx float64) float64 { return math.Abs(dx / math.Sqrt(dx*dx+1)) },
	TransferV4: func(dx float64) float64 { return math.Abs(2 / math.Pi * math.Atan(math.Pi/2*dx)) },
	TransferS1: func(dx float64) float64 { return logistic(2 * dx) },
	TransferS2: logistic,
	TransferS3: func(dx float64) float64 { return logistic(dx / 2) },
	TransferS4: func(dx float64) float64 { return logistic(dx / 3) },
}

// transferFunctionOrder is the canonical listing order: the V-shaped family
// first, then the S-shaped one, each in index order. Map iteration is
// randomized, so the order lives here rather than being recovered from the
// registry.
var transferFunctionOrder = []TransferFunction{
	TransferV1, TransferV2, TransferV3, TransferV4,
	TransferS1, TransferS2, TransferS3, TransferS4,
}

// logistic is 1/(1+e^-x), the shared shape of the S-family.
//
// A large negative x makes the exponential overflow to +Inf and the result
// underflows to zero rather than becoming NaN, which is the behavior the
// bit-flip test wants at saturation.
func logistic(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// LookupTransferFunction returns the named transfer function.
//
// An unknown name is an error rather than a silent fallback to the default: a
// misspelled name in a JSON configuration would otherwise run a different
// algorithm than the one the caller wrote down.
func LookupTransferFunction(name TransferFunction) (func(float64) float64, error) {
	transfer, ok := transferFunctions[name]
	if !ok {
		return nil, fmt.Errorf("unknown transfer function %q, expected one of %v",
			name, TransferFunctionNames())
	}

	return transfer, nil
}

// TransferFunctionNames returns every registered transfer function in a stable
// order: v1..v4 then s1..s4.
func TransferFunctionNames() []TransferFunction {
	names := make([]TransferFunction, len(transferFunctionOrder))
	copy(names, transferFunctionOrder)

	return names
}

// effectiveTransferFunction resolves the transfer function a run should use.
// An empty Config.TransferFunc means DefaultTransferFunction; anything else is
// looked up and must exist.
func effectiveTransferFunction(config *Config) (transferFunc, error) {
	name := config.TransferFunc
	if name == "" {
		name = DefaultTransferFunction
	}

	transfer, ok := transferFunctions[name]
	if !ok {
		return nil, fmt.Errorf("unknown transfer function %q, expected one of %v",
			name, TransferFunctionNames())
	}

	return transfer, nil
}

// binaryRun is everything OptimizeBinaryContext prepares before the first
// iteration: the swarm, the early-termination tracker, the resolved transfer
// function, the run options and the generator every draw comes from.
type binaryRun struct {
	state        *runState
	tracker      *convergenceTracker
	rng          *rand.Rand
	transfer     transferFunc
	transferName TransferFunction
	options      runOptions
	seed         int64
	seedKnown    bool
}

// OptimizeBinary runs the binary Dragonfly Algorithm with a background context.
//
// Start from NewBinaryConfig and set ObjectiveFunc and ProblemSize. The
// objective keeps the ordinary ObjectiveFunction signature and is handed a
// vector whose components are exactly 0 or 1, so the benchmark, constraint and
// monitoring machinery works unchanged.
func OptimizeBinary(config *Config) (*Result, error) {
	return OptimizeBinaryContext(context.Background(), config)
}

// OptimizeBinaryContext runs the binary Dragonfly Algorithm, honoring context
// cancellation and the supplied run options.
//
// BDA treats every other dragonfly as a neighbor and applies the five-factor
// step unconditionally; it has neither DA's radius branches nor its Lévy walk.
// The continuous position update is discarded in favor of
//
//	x_j <- ¬x_j  if rand < T(Δx_j)  else  x_j
//
// where T is Config.TransferFunc, defaulting to the paper's v3.
//
// # Boundary handling
//
// Config.BoundaryMethod is ignored in binary mode. A 0/1 vector cannot leave
// [0,1], so there is nothing for a wrap, clamp or reflect rule to repair, and
// applying one anyway would be worse than useless: the wrap rule's Δx reset
// would silently overwrite the step the very next bit-flip decision is made
// from. The field is left alone rather than validated away, so that one Config
// can be handed to both entry points.
//
// The Lévy walk has no binary counterpart. Config.UseLevyWalk and the radius
// schedule are ignored.
func OptimizeBinaryContext(ctx context.Context, config *Config, options ...RunOption) (*Result, error) {
	run, setupErr := setupBinaryRun(ctx, config, options)
	if setupErr != nil {
		return nil, setupErr
	}

	return runBinaryLoop(ctx, config, run)
}

// setupBinaryRun validates the configuration, seeds the generator and builds
// the starting swarm. It is separate from the loop so that neither function has
// to be read to understand the other.
func setupBinaryRun(ctx context.Context, config *Config, options []RunOption) (*binaryRun, error) {
	contextErr := requireContext(ctx)
	if contextErr != nil {
		return nil, contextErr
	}

	resolved, optionsErr := resolveRunOptions(options)
	if optionsErr != nil {
		return nil, optionsErr
	}

	transfer, configErr := validateBinaryRun(config, resolved)
	if configErr != nil {
		return nil, configErr
	}

	// A configured seed is reproducible. A directly supplied generator has no
	// introspectable seed and is marked unknown in the result.
	rng, seed, seedKnown := resolveRandomSource(config)

	state := initializeBinaryRun(config, resolved, rng)
	if !hasFiniteObjective(state.swarm) {
		return nil, ErrNoFiniteObjective
	}

	contextErr = ctx.Err()
	if contextErr != nil {
		return nil, contextErr
	}

	// The pool is built after the starting swarm has been scored, so that
	// initialization stays a plain sequential pass. Both paths produce the same
	// costs, the same food, the same enemy and the same evaluation count, so
	// scoring the first population sequentially costs a parallel run one batch
	// and keeps initializeBinaryRun free of a context it has nothing else to do
	// with.
	if config.EnableParallel {
		state.pool = newEvaluationPool(state.evaluator, effectiveMaxWorkers(config), config.NPop)
	}

	return &binaryRun{
		state:        state,
		tracker:      newConvergenceTracker(config.Convergence, state.food, state.evaluator),
		transfer:     transfer,
		transferName: effectiveTransferFunctionName(config),
		options:      resolved,
		rng:          rng,
		seed:         seed,
		seedKnown:    seedKnown,
	}, nil
}

// validateBinaryRun runs the shared configuration checks, the binary-only ones,
// and resolves the transfer function.
func validateBinaryRun(config *Config, options runOptions) (transferFunc, error) {
	optionMeaningErr := validateSingleObjectiveRunOptions(options)
	if optionMeaningErr != nil {
		return nil, optionMeaningErr
	}

	validationErr := validateConfig(config)
	if validationErr != nil {
		return nil, validationErr
	}

	binaryErr := validateBinaryConfig(config)
	if binaryErr != nil {
		return nil, binaryErr
	}

	populationErr := validateInitialPopulation(config, options)
	if populationErr != nil {
		return nil, populationErr
	}

	bitsErr := validateInitialBits(options.initialPositions)
	if bitsErr != nil {
		return nil, bitsErr
	}

	return effectiveTransferFunction(config)
}

// validateBinaryConfig checks what only the binary variant cares about: the
// search box has to be the unit interval, because a position component is a bit
// and every schedule that scales with (ub-lb) is written for that box, and the
// transfer function has to name a registered entry.
func validateBinaryConfig(config *Config) error {
	if config.LowerBound != 0 || config.UpperBound != 1 {
		return fmt.Errorf(
			"binary optimization requires lower_bound 0 and upper_bound 1, got %v and %v",
			config.LowerBound, config.UpperBound)
	}

	_, transferErr := effectiveTransferFunction(config)

	return transferErr
}

// validateInitialBits rejects a seeded population that is not 0/1-valued.
// Rounding it silently would hand the caller a different starting swarm than
// the one they wrote.
func validateInitialBits(positions [][]float64) error {
	for i, position := range positions {
		for j, value := range position {
			if value != 0 && value != 1 {
				return fmt.Errorf("initial position %d component %d must be 0 or 1, got %v", i, j, value)
			}
		}
	}

	return nil
}

// runBinaryLoop is the iteration loop: schedules, one binary move per
// dragonfly, evaluation, observers, early termination.
func runBinaryLoop(ctx context.Context, config *Config, run *binaryRun) (*Result, error) {
	logOptimizationStarted(ctx, run.options.logger, config)

	state := run.state
	curve := make([]float64, 0, config.MaxIterations)
	reason := TerminationMaxIterations
	completed := 0

	for t := range config.MaxIterations {
		ctxErr := ctx.Err()
		if ctxErr != nil {
			return nil, ctxErr
		}

		weights := computeWeights(config, t+1, config.MaxIterations, run.rng)

		for i := range state.swarm {
			moveBinaryDragonfly(state, i, weights, run.transferName, run.transfer, run.rng)
		}

		evaluationErr := state.evaluateBinary(ctx)
		if evaluationErr != nil {
			return nil, evaluationErr
		}

		curve = append(curve, state.food.Cost)
		completed = t + 1

		notifyProgress(run.options.observer, completed, state.funcEvals, state.food)
		notifyPopulation(run.options.populationObserver, completed, state.funcEvals,
			state.food, state.enemy, state.swarm)
		logIterationCompleted(ctx, run.options.logger, completed, state.funcEvals, state.food)

		stopReason, stop := run.tracker.observe(completed, state.food)
		if stop {
			reason = stopReason

			break
		}
	}

	result := &Result{
		ConvergenceCurve:  curve,
		TerminationReason: reason,
		GlobalBest:        state.food,
		Worst:             state.enemy,
		FuncEvalCount:     state.funcEvals,
		IterationCount:    completed,
		Seed:              run.seed,
		SeedKnown:         run.seedKnown,
	}

	logOptimizationCompleted(ctx, run.options.logger, result)

	return result, nil
}

// evaluateBinary scores the binary swarm and updates the food source and the
// enemy, through the worker pool when the run is parallel and inline when it is
// not. It is the binary counterpart of runState.evaluate.
//
// The dispatch lives here rather than reusing runState.evaluate so that the
// binary loop names the variant's own parallel entry point, and so that the two
// loops stay independently readable. Both paths compute the same thing; the
// parallel one can be canceled between objective calls and reports that as an
// error rather than committing a half-scored batch.
//
// Every random draw the iteration makes -- weights, step, bit flips -- has
// already happened on the calling goroutine by the time this is called, which
// is what keeps a seeded binary run bit-identical with EnableParallel on or
// off.
func (state *runState) evaluateBinary(ctx context.Context) error {
	if state.pool == nil {
		contextErr := ctx.Err()
		if contextErr != nil {
			return contextErr
		}

		state.evaluateSwarm()

		return ctx.Err()
	}

	count, err := evaluateParallelBinary(ctx, state, state.pool)
	if err != nil {
		return err
	}

	state.funcEvals += count

	return nil
}

// initializeBinaryRun builds and evaluates the starting swarm.
//
// Positions are fair coin flips rather than uniform draws from [0,1]: the
// binary variant's position space is the corners of the unit cube, and a
// dragonfly has to start on one. The step is drawn from [LowerBound,
// UpperBound] exactly as the continuous variant draws it, so the first
// iteration's flip probabilities are T of a value in [0,1) instead of T(0).
func initializeBinaryRun(config *Config, options runOptions, rng *rand.Rand) *runState {
	swarm := make([]Dragonfly, config.NPop)
	for i := range swarm {
		position := randomBits(config.ProblemSize, rng)

		step := unifrndVec(config.LowerBound, config.UpperBound, config.ProblemSize, rng)
		if effectiveFidelityMode(config) == FidelityMATLAB {
			step = randomBits(config.ProblemSize, rng)
		}

		if i < len(options.initialPositions) {
			position = copyVec(options.initialPositions[i])
		}

		swarm[i] = Dragonfly{
			Position: position,
			Step:     step,
			Cost:     math.Inf(1),
		}
	}

	state := &runState{
		evaluator: newConstraintEvaluator(config.ObjectiveFunc, config.Constraints),
		swarm:     swarm,
		food: Best{
			Position:            make([]float64, config.ProblemSize),
			Cost:                math.Inf(1),
			ConstraintViolation: math.Inf(1),
		},
		enemy: Best{
			Position: make([]float64, config.ProblemSize),
			Cost:     math.Inf(-1),
		},
	}

	state.evaluateSwarm()

	return state
}

// randomBits returns a fresh vector of size fair coin flips.
func randomBits(size int, rng *rand.Rand) []float64 {
	bits := make([]float64, size)
	for i := range bits {
		if unifrnd(0, 1, rng) < 0.5 {
			continue
		}

		bits[i] = 1
	}

	return bits
}

// moveBinaryDragonfly advances one dragonfly: the continuous step, then the
// bit-flip position update.
func moveBinaryDragonfly(
	state *runState,
	index int,
	weights weightSchedule,
	name TransferFunction,
	transfer transferFunc,
	rng *rand.Rand,
) {
	buildBinaryStep(state, index, weights)
	updateBinaryPosition(&state.swarm[index], name, transfer, rng)
}

// buildBinaryStep computes ΔX exactly as the continuous variant does, by
// calling dragonfly.go's own step builders, and leaves the position untouched.
//
// Both builders commit the finished step to the position, which is meaningless
// for a 0/1 vector, so the position is saved beforehand and restored
// afterwards. Only ΔX survives the call, which is precisely the part BDA
// shares with DA.
func buildBinaryStep(state *runState, index int, weights weightSchedule, _ ...*rand.Rand) {
	fly := &state.swarm[index]

	neighbors := make([]int, 0, len(state.swarm)-1)
	for i := range state.swarm {
		if i != index {
			neighbors = append(neighbors, i)
		}
	}

	separation := separationVector(state.swarm, index, neighbors)
	alignment := alignmentVector(state.swarm, index, neighbors)
	cohesion := cohesionVector(state.swarm, index, neighbors)
	food := foodVector(fly.Position, state.food.Position)
	enemy := enemyVector(fly.Position, state.enemy.Position)

	saved := copyVec(fly.Position)
	applyFullStep(fly, weights, separation, alignment, cohesion, food, enemy)

	copy(fly.Position, saved)
}

func effectiveTransferFunctionName(config *Config) TransferFunction {
	if config.TransferFunc == "" {
		return DefaultTransferFunction
	}

	return config.TransferFunc
}

func isSShaped(name TransferFunction) bool {
	switch name {
	case TransferS1, TransferS2, TransferS3, TransferS4:
		return true
	default:
		return false
	}
}

// updateBinaryPosition applies the distinct binary semantics of the transfer
// families: V-shaped functions complement the current bit, while S-shaped
// functions assign a sampled Bernoulli value.
func updateBinaryPosition(
	fly *Dragonfly, name TransferFunction, transfer transferFunc, rng *rand.Rand,
) {
	for j := range fly.Position {
		if j >= len(fly.Step) {
			return
		}

		draw := unifrnd(0, 1, rng)

		probability := transfer(fly.Step[j])
		if isSShaped(name) {
			if draw < probability {
				fly.Position[j] = 1
			} else {
				fly.Position[j] = 0
			}

			continue
		}

		if draw < probability {
			fly.Position[j] = flipBit(fly.Position[j])
		}
	}
}

// flipBits is the binary position update: each component is flipped with
// probability T(Δx_j).
//
// The uniform draw is taken for every component whether or not the flip
// happens, so that the random stream does not depend on the outcome of earlier
// flips. A NaN or missing step component cannot flip a bit, because every
// comparison against NaN is false.
func flipBits(fly *Dragonfly, transfer transferFunc, rng *rand.Rand) {
	updateBinaryPosition(fly, TransferV3, transfer, rng)
}

// flipBit returns the complement of a bit. Anything that is not exactly zero
// counts as a one, so a position that somehow left {0,1} is repaired towards it
// rather than being negated into a third value.
func flipBit(bit float64) float64 {
	if bit == 0 {
		return 1
	}

	return 0
}

// BinaryPositionsValid reports whether every component of position is exactly
// zero or one. It is the invariant a binary run maintains from initialization
// onwards, exposed so that a caller inspecting a Result or a population
// snapshot can assert it too.
func BinaryPositionsValid(position []float64) bool {
	for _, value := range position {
		if value != 0 && value != 1 {
			return false
		}
	}

	return true
}
