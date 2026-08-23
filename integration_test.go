package dragonfly

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/cucumber/godog"
)

// featureContext is the state one Gherkin scenario builds up across its steps.
// A fresh one is installed before every scenario, so no scenario can observe
// another's leftovers.
type featureContext struct {
	objective       ObjectiveFunction
	multiObjective  MultiObjectiveFunction
	config          *Config
	result          *Result
	secondResult    *Result
	moResult        *MultiObjectiveResult
	finalSwarm      []Dragonfly
	transferFunc    TransferFunction
	targetCost      *float64
	candidate       CandidateEvaluation
	incumbent       CandidateEvaluation
	constraints     *ConstraintConfig
	position        []float64
	step            []float64
	lowerBound      float64
	upperBound      float64
	penalizedCost   float64
	problemSize     int
	stagnationLimit int
	preferred       bool
	ranked          bool
}

// errNoResult is what a "then" step reports when the scenario's "when" step
// never produced one; it turns a mis-wired feature file into a readable
// failure instead of a nil dereference.
var errNoResult = errors.New("no optimization result available")

// benchmarkByName resolves the benchmark name a feature file spells out.
func benchmarkByName(name string) (ObjectiveFunction, error) {
	switch name {
	case "Sphere":
		return Sphere, nil
	case "Rastrigin":
		return Rastrigin, nil
	case "Rosenbrock":
		return Rosenbrock, nil
	case "Ackley":
		return Ackley, nil
	case "Griewank":
		return Griewank, nil
	case "Schwefel":
		return Schwefel, nil
	default:
		return nil, fmt.Errorf("unknown benchmark function %q", name)
	}
}

// --- shared "given" steps ---------------------------------------------------

func (fc *featureContext) aFunctionWithDimension(name string, dimension int) error {
	objective, err := benchmarkByName(name)
	if err != nil {
		return err
	}

	fc.objective = objective
	fc.problemSize = dimension

	return nil
}

func (fc *featureContext) aConstantObjectiveWithDimension(dimension int) error {
	fc.objective = func([]float64) float64 { return 42 }
	fc.problemSize = dimension

	return nil
}

func (fc *featureContext) boundsFromTo(lower, upper float64) error {
	fc.lowerBound = lower
	fc.upperBound = upper

	return nil
}

func (fc *featureContext) aTargetCostOf(target float64) error {
	fc.targetCost = &target

	return nil
}

func (fc *featureContext) aStagnationWindowOf(window int) error {
	fc.stagnationLimit = window

	return nil
}

// runConfig assembles the configuration the "when I run DA" steps share.
func (fc *featureContext) runConfig(iterations int, seed int64) *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = fc.objective
	config.ProblemSize = fc.problemSize
	config.LowerBound = fc.lowerBound
	config.UpperBound = fc.upperBound
	config.MaxIterations = iterations
	config.NPop = 40
	config.Rand = rand.New(rand.NewSource(seed))

	if fc.targetCost != nil || fc.stagnationLimit > 0 {
		config.Convergence = &ConvergenceConfig{
			TargetCost:           fc.targetCost,
			StagnationIterations: fc.stagnationLimit,
			MinIterations:        1,
		}
	}

	return config
}

func (fc *featureContext) iRunDAForIterations(iterations int) error {
	config := fc.runConfig(iterations, 20240823)

	result, err := Optimize(config)
	if err != nil {
		return err
	}

	fc.config = config
	fc.result = result

	return nil
}

func (fc *featureContext) iRunDATwiceWithSeed(iterations int, seed int64) error {
	first, firstErr := Optimize(fc.runConfig(iterations, seed))
	if firstErr != nil {
		return firstErr
	}

	second, secondErr := Optimize(fc.runConfig(iterations, seed))
	if secondErr != nil {
		return secondErr
	}

	fc.result = first
	fc.secondResult = second

	return nil
}

// --- shared "then" steps ----------------------------------------------------

func (fc *featureContext) theBestCostShouldBeLessThan(threshold float64) error {
	if fc.result == nil {
		return errNoResult
	}

	if fc.result.GlobalBest.Cost >= threshold {
		return fmt.Errorf("best cost %g is not less than %g", fc.result.GlobalBest.Cost, threshold)
	}

	return nil
}

func (fc *featureContext) theBestCostShouldBeLessThanPercentOfTheInitialBest(percent float64) error {
	if fc.result == nil {
		return errNoResult
	}

	if len(fc.result.ConvergenceCurve) == 0 {
		return errors.New("the convergence curve is empty")
	}

	start := fc.result.ConvergenceCurve[0]

	limit := start * percent / 100

	if fc.result.GlobalBest.Cost > limit {
		return fmt.Errorf("best cost %g is not within %g%% of the initial best %g",
			fc.result.GlobalBest.Cost, percent, start)
	}

	return nil
}

func (fc *featureContext) theBestPositionShouldHaveComponents(count int) error {
	if fc.result == nil {
		return errNoResult
	}

	if len(fc.result.GlobalBest.Position) != count {
		return fmt.Errorf("best position has %d components, want %d",
			len(fc.result.GlobalBest.Position), count)
	}

	return nil
}

func (fc *featureContext) theBestPositionShouldBeWithinBounds() error {
	if fc.result == nil {
		return errNoResult
	}

	return fc.checkWithinBounds("best position", fc.result.GlobalBest.Position)
}

func (fc *featureContext) checkWithinBounds(what string, position []float64) error {
	for i, value := range position {
		if value < fc.lowerBound || value > fc.upperBound {
			return fmt.Errorf("%s[%d] = %g is outside [%g, %g]",
				what, i, value, fc.lowerBound, fc.upperBound)
		}
	}

	return nil
}

func (fc *featureContext) theEvaluationCountShouldBePositive() error {
	if fc.result == nil {
		return errNoResult
	}

	if fc.result.FuncEvalCount <= 0 {
		return fmt.Errorf("evaluation count is %d, want a positive count", fc.result.FuncEvalCount)
	}

	return nil
}

func (fc *featureContext) theReportedSeedShouldBeNonZero() error {
	if fc.result == nil {
		return errNoResult
	}

	if fc.result.Seed == 0 {
		return errors.New("the reported seed is zero")
	}

	return nil
}

func (fc *featureContext) theTerminationReasonShouldBe(reason string) error {
	if fc.result == nil {
		return errNoResult
	}

	if string(fc.result.TerminationReason) != reason {
		return fmt.Errorf("termination reason is %q, want %q", fc.result.TerminationReason, reason)
	}

	return nil
}

func (fc *featureContext) theIterationCountShouldBe(count int) error {
	if fc.result == nil {
		return errNoResult
	}

	if fc.result.IterationCount != count {
		return fmt.Errorf("iteration count is %d, want %d", fc.result.IterationCount, count)
	}

	return nil
}

func (fc *featureContext) theIterationCountShouldBeLessThan(limit int) error {
	if fc.result == nil {
		return errNoResult
	}

	if fc.result.IterationCount >= limit {
		return fmt.Errorf("iteration count is %d, want fewer than %d", fc.result.IterationCount, limit)
	}

	return nil
}

func (fc *featureContext) theConvergenceCurveShouldHaveEntries(count int) error {
	if fc.result == nil {
		return errNoResult
	}

	if len(fc.result.ConvergenceCurve) != count {
		return fmt.Errorf("convergence curve has %d entries, want %d",
			len(fc.result.ConvergenceCurve), count)
	}

	return nil
}

func (fc *featureContext) theConvergenceCurveShouldBeNonIncreasing() error {
	if fc.result == nil {
		return errNoResult
	}

	curve := fc.result.ConvergenceCurve
	for i := 1; i < len(curve); i++ {
		if curve[i] > curve[i-1] {
			return fmt.Errorf("convergence curve rose at index %d: %g > %g", i, curve[i], curve[i-1])
		}
	}

	return nil
}

func (fc *featureContext) bothRunsShouldReturnIdenticalResults() error {
	if fc.result == nil || fc.secondResult == nil {
		return errNoResult
	}

	if fc.result.GlobalBest.Cost != fc.secondResult.GlobalBest.Cost {
		return fmt.Errorf("costs differ: %g vs %g",
			fc.result.GlobalBest.Cost, fc.secondResult.GlobalBest.Cost)
	}

	if fc.result.FuncEvalCount != fc.secondResult.FuncEvalCount {
		return fmt.Errorf("evaluation counts differ: %d vs %d",
			fc.result.FuncEvalCount, fc.secondResult.FuncEvalCount)
	}

	first := fc.result.GlobalBest.Position

	second := fc.secondResult.GlobalBest.Position
	for i := range first {
		if first[i] != second[i] {
			return fmt.Errorf("position[%d] differs: %g vs %g", i, first[i], second[i])
		}
	}

	return nil
}

// --- boundary handling ------------------------------------------------------

func (fc *featureContext) aNewDefaultConfiguration() error {
	fc.config = NewDefaultConfig()

	return nil
}

func (fc *featureContext) theEffectiveBoundaryMethodShouldBe(want string) error {
	if fc.config == nil {
		return errors.New("no configuration available")
	}

	got := effectiveBoundaryMethod(fc.config)
	if string(got) != want {
		return fmt.Errorf("effective boundary method is %q, want %q", got, want)
	}

	return nil
}

func (fc *featureContext) aBoxFromTo(lower, upper float64) error {
	fc.lowerBound = lower
	fc.upperBound = upper

	return nil
}

func (fc *featureContext) aComponentAtWithStep(position, step float64) error {
	fc.position = []float64{position}
	fc.step = []float64{step}

	return nil
}

func (fc *featureContext) theBoundaryRuleIsApplied(method string) error {
	if fc.position == nil {
		return errors.New("no component was set up")
	}

	rng := rand.New(rand.NewSource(20240823))
	applyBounds(fc.position, fc.step, fc.lowerBound, fc.upperBound, BoundaryMethod(method), rng)

	return nil
}

func (fc *featureContext) theComponentShouldBe(want float64) error {
	if fc.position == nil {
		return errors.New("no component was set up")
	}

	if math.Abs(fc.position[0]-want) > 1e-12 {
		return fmt.Errorf("component is %g, want %g", fc.position[0], want)
	}

	return nil
}

func (fc *featureContext) theStepComponentShouldBe(want float64) error {
	if fc.step == nil {
		return errors.New("no step was set up")
	}

	if math.Abs(fc.step[0]-want) > 1e-12 {
		return fmt.Errorf("step component is %g, want %g", fc.step[0], want)
	}

	return nil
}

// theStepComponentShouldHaveBeenRedrawn is the half of the wrap rule that is
// easiest to drop: the teleport alone is not the rule. The original step was
// 7.25, which is outside [0, 1), so a value inside that interval can only have
// come from the fresh uniform draw.
func (fc *featureContext) theStepComponentShouldHaveBeenRedrawn() error {
	if fc.step == nil {
		return errors.New("no step was set up")
	}

	if fc.step[0] < 0 || fc.step[0] >= 1 {
		return fmt.Errorf("step component is %g, want a fresh draw inside [0, 1)", fc.step[0])
	}

	return nil
}

func (fc *featureContext) iRunDAWithBoundaryMethod(iterations int, method string) error {
	config := fc.runConfig(iterations, 20240823)
	config.BoundaryMethod = BoundaryMethod(method)

	var final []Dragonfly

	result, err := OptimizeContext(context.Background(), config,
		WithPopulationObserver(func(snapshot PopulationSnapshot) {
			final = snapshot.Swarm
		}))
	if err != nil {
		return err
	}

	fc.config = config
	fc.result = result
	fc.finalSwarm = final

	return nil
}

func (fc *featureContext) everyPositionInTheFinalSwarmShouldBeWithinBounds() error {
	if len(fc.finalSwarm) == 0 {
		return errors.New("no swarm snapshot was captured")
	}

	for i := range fc.finalSwarm {
		err := fc.checkWithinBounds(fmt.Sprintf("dragonfly %d position", i), fc.finalSwarm[i].Position)
		if err != nil {
			return err
		}
	}

	return nil
}

// --- configuration validation -----------------------------------------------

func (fc *featureContext) aValidConfiguration() error {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 5
	config.LowerBound = -10
	config.UpperBound = 10
	config.MaxIterations = 20
	config.NPop = 10
	fc.config = config

	return nil
}

func (fc *featureContext) aBinaryConfiguration() error {
	config := NewBinaryConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 5
	config.MaxIterations = 20
	config.NPop = 10
	fc.config = config

	return nil
}

func (fc *featureContext) iClearTheObjectiveFunction() error {
	if fc.config == nil {
		return errors.New("no configuration available")
	}

	fc.config.ObjectiveFunc = nil

	return nil
}

// iSetField applies one mutation from a Scenario Outline's Examples column.
// The switch is exhaustive on purpose: an unrecognized phrase in a feature file
// fails the scenario rather than silently passing.
func (fc *featureContext) iSetField(mutation string) error {
	if fc.config == nil {
		return errors.New("no configuration available")
	}

	switch mutation {
	case "problem_size to 0":
		fc.config.ProblemSize = 0
	case "problem_size to -3":
		fc.config.ProblemSize = -3
	case "lower_bound above upper_bound":
		fc.config.LowerBound = 10
		fc.config.UpperBound = -10
	case "both bounds to 1":
		fc.config.LowerBound = 1
		fc.config.UpperBound = 1
	case "upper_bound to 5":
		fc.config.UpperBound = 5
	case "npop to 0":
		fc.config.NPop = 0
	case "max_iterations to 0":
		fc.config.MaxIterations = 0
	case "max_workers to -1":
		fc.config.MaxWorkers = -1
	case `boundary_method to "bounce"`:
		fc.config.BoundaryMethod = BoundaryMethod("bounce")
	case "food_weight to NaN":
		fc.config.FoodWeight = math.NaN()
	case "enemy_weight to 0":
		fc.config.EnemyWeight = 0
	case "radius_initial_divisor to 0":
		fc.config.RadiusInitialDivisor = 0
	case "stagnation_iterations to -1":
		fc.config.Convergence = &ConvergenceConfig{StagnationIterations: -1}
	case "min_improvement to -1":
		fc.config.Convergence = &ConvergenceConfig{MinImprovement: -1}
	default:
		return fmt.Errorf("unknown configuration mutation %q", mutation)
	}

	return nil
}

func (fc *featureContext) validationShouldSucceed() error {
	err := ValidateConfig(fc.config)
	if err != nil {
		return fmt.Errorf("ValidateConfig rejected a valid configuration: %w", err)
	}

	return nil
}

func (fc *featureContext) validationShouldFailWithAnErrorContaining(fragment string) error {
	err := ValidateConfig(fc.config)
	if err == nil {
		return fmt.Errorf("ValidateConfig accepted a configuration that should fail on %q", fragment)
	}

	if !strings.Contains(err.Error(), fragment) {
		return fmt.Errorf("error %q does not mention %q", err.Error(), fragment)
	}

	return nil
}

// optimizeShouldRefuseToRunIt closes the loop on ValidateConfig: the exported
// checker and the entry point have to agree, or fail-fast validation would be
// advisory rather than binding.
func (fc *featureContext) optimizeShouldRefuseToRunIt() error {
	result, err := Optimize(fc.config)
	if err == nil {
		return errors.New("Optimize accepted a configuration ValidateConfig rejected")
	}

	if result != nil {
		return errors.New("Optimize returned a result alongside an error")
	}

	return nil
}

func (fc *featureContext) theRunShouldComplete() error {
	result, err := Optimize(fc.config)
	if err != nil {
		return fmt.Errorf("Optimize failed: %w", err)
	}

	if result.IterationCount != fc.config.MaxIterations {
		return fmt.Errorf("run stopped after %d of %d iterations",
			result.IterationCount, fc.config.MaxIterations)
	}

	return nil
}

// --- constraint handling ----------------------------------------------------

func (fc *featureContext) aCandidateWithCostAndViolation(cost, violation float64) error {
	fc.candidate = CandidateEvaluation{Cost: cost, ConstraintViolation: violation}

	return nil
}

func (fc *featureContext) anIncumbentWithCostAndViolation(cost, violation float64) error {
	fc.incumbent = CandidateEvaluation{Cost: cost, ConstraintViolation: violation}

	return nil
}

func (fc *featureContext) iRankThemUnderFeasibilityRules() error {
	config := &ConstraintConfig{Handling: ConstraintHandlingFeasibility}
	fc.preferred = BetterConstrainedCandidate(fc.candidate, fc.incumbent, config)
	fc.ranked = true

	return nil
}

func (fc *featureContext) iRankThemUnderThePenaltyMethodWithFactor(factor float64) error {
	config := &ConstraintConfig{
		Handling:      ConstraintHandlingPenalty,
		PenaltyMethod: PenaltyLinear,
		PenaltyFactor: factor,
	}
	fc.preferred = BetterConstrainedCandidate(fc.candidate, fc.incumbent, config)
	fc.ranked = true

	return nil
}

func (fc *featureContext) theCandidateShouldBePreferred() error {
	if !fc.ranked {
		return errors.New("no ranking was performed")
	}

	if !fc.preferred {
		return errors.New("the incumbent was preferred, want the candidate")
	}

	return nil
}

func (fc *featureContext) theIncumbentShouldBePreferred() error {
	if !fc.ranked {
		return errors.New("no ranking was performed")
	}

	if fc.preferred {
		return errors.New("the candidate was preferred, want the incumbent")
	}

	return nil
}

func (fc *featureContext) iComputeItsPenalizedCostWithFactor(method string, factor float64) error {
	fc.penalizedCost = PenalizedCost(
		fc.candidate.Cost,
		fc.candidate.ConstraintViolation,
		factor,
		PenaltyMethod(method),
	)

	return nil
}

func (fc *featureContext) thePenalizedCostShouldBe(want float64) error {
	if math.Abs(fc.penalizedCost-want) > 1e-12 {
		return fmt.Errorf("penalized cost is %g, want %g", fc.penalizedCost, want)
	}

	return nil
}

// aConstrainedOneDimensionalProblem sets up min x² on [-5, 5] subject to
// x >= 1, whose feasible optimum is x = 1 at cost 1. The unconstrained optimum
// sits at x = 0, so a run that ignored the constraint would land there and the
// scenario would fail on feasibility rather than on cost.
func (fc *featureContext) aConstrainedOneDimensionalProblem() error {
	fc.objective = Sphere
	fc.problemSize = 1
	fc.lowerBound = -5
	fc.upperBound = 5
	fc.constraints = &ConstraintConfig{
		Handling: ConstraintHandlingFeasibility,
		Inequalities: []ConstraintFunction{
			func(position []float64) float64 { return 1 - position[0] },
		},
	}

	return nil
}

func (fc *featureContext) iOptimizeItUsingFeasibilityRules() error {
	config := fc.runConfig(400, 20240823)
	config.Constraints = fc.constraints

	result, err := Optimize(config)
	if err != nil {
		return err
	}

	fc.config = config
	fc.result = result

	return nil
}

func (fc *featureContext) theReturnedSolutionShouldSatisfyEveryConstraint() error {
	if fc.result == nil {
		return errNoResult
	}

	if fc.result.GlobalBest.ConstraintViolation != 0 {
		return fmt.Errorf("constraint violation is %g, want zero",
			fc.result.GlobalBest.ConstraintViolation)
	}

	evaluation := EvaluateConstraints(fc.result.GlobalBest.Position, fc.constraints)
	if !evaluation.Feasible {
		return fmt.Errorf("re-evaluating the reported position gives violation %g", evaluation.Violation)
	}

	return nil
}

func (fc *featureContext) theReportedCostShouldBeTheRawObjectiveCost() error {
	if fc.result == nil {
		return errNoResult
	}

	want := fc.objective(fc.result.GlobalBest.Position)
	if fc.result.GlobalBest.Cost != want {
		return fmt.Errorf("reported cost is %g, want the raw objective cost %g",
			fc.result.GlobalBest.Cost, want)
	}

	return nil
}

// --- variant execution ------------------------------------------------------

// oneMax counts the zero bits, so a perfect all-ones bit string costs zero.
func oneMax(x []float64) float64 {
	zeros := 0.0
	for _, bit := range x {
		zeros += 1 - bit
	}

	return zeros
}

func (fc *featureContext) aOneMaxProblemOverBits(bits int) error {
	fc.objective = oneMax
	fc.problemSize = bits
	fc.lowerBound = 0
	fc.upperBound = 1

	return nil
}

func (fc *featureContext) theTransferFunction(name string) error {
	fc.transferFunc = TransferFunction(name)

	return nil
}

func (fc *featureContext) iRunBDAForIterations(iterations int) error {
	config := NewBinaryConfig()
	config.ObjectiveFunc = fc.objective
	config.ProblemSize = fc.problemSize
	config.MaxIterations = iterations
	config.NPop = 30
	config.Rand = rand.New(rand.NewSource(20240823))

	if fc.transferFunc != "" {
		config.TransferFunc = fc.transferFunc
	}

	result, err := OptimizeBinary(config)
	if err != nil {
		return err
	}

	fc.config = config
	fc.result = result

	return nil
}

func (fc *featureContext) everyComponentOfTheBestPositionShouldBeABit() error {
	if fc.result == nil {
		return errNoResult
	}

	if !BinaryPositionsValid(fc.result.GlobalBest.Position) {
		return fmt.Errorf("best position %v is not a bit string", fc.result.GlobalBest.Position)
	}

	return nil
}

func (fc *featureContext) aZDT1ProblemWithDimension(dimension int) error {
	fc.multiObjective = ZDT1
	fc.problemSize = dimension
	fc.lowerBound = 0
	fc.upperBound = 1

	return nil
}

func (fc *featureContext) iRunMODAForIterations(iterations int) error {
	config := NewMultiObjectiveConfig()
	config.ObjectiveFunc = fc.multiObjective
	config.Swarm.ProblemSize = fc.problemSize
	config.Swarm.LowerBound = fc.lowerBound
	config.Swarm.UpperBound = fc.upperBound
	config.Swarm.MaxIterations = iterations
	config.Swarm.NPop = 40
	config.Swarm.Rand = rand.New(rand.NewSource(20240823))

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		return err
	}

	fc.moResult = result

	return nil
}

func (fc *featureContext) theArchiveShouldBeNonDominated() error {
	if fc.moResult == nil {
		return errors.New("no multi-objective result available")
	}

	if !fc.moResult.Archive.IsNonDominated() {
		return errors.New("the archive contains a dominated solution")
	}

	return nil
}

func (fc *featureContext) theArchiveShouldHoldAtLeastSolutions(count int) error {
	if fc.moResult == nil {
		return errors.New("no multi-objective result available")
	}

	if fc.moResult.Archive.Len() < count {
		return fmt.Errorf("the archive holds %d solutions, want at least %d",
			fc.moResult.Archive.Len(), count)
	}

	return nil
}

func (fc *featureContext) everyArchivedPositionShouldBeWithinBounds() error {
	if fc.moResult == nil {
		return errors.New("no multi-objective result available")
	}

	for i, solution := range fc.moResult.Archive.Solutions {
		err := fc.checkWithinBounds(fmt.Sprintf("archived position %d", i), solution.Position)
		if err != nil {
			return err
		}
	}

	return nil
}

// --- godog wiring -----------------------------------------------------------

// registerSharedSteps wires the steps several feature files share.
func registerSharedSteps(sc *godog.ScenarioContext, fc *featureContext) {
	sc.Step(`^an? (\w+) function with dimension (\d+)$`, fc.aFunctionWithDimension)
	sc.Step(`^a constant objective with dimension (\d+)$`, fc.aConstantObjectiveWithDimension)
	sc.Step(`^bounds from (-?[\d.]+) to (-?[\d.]+)$`, fc.boundsFromTo)
	sc.Step(`^a target cost of ([\d.]+)$`, fc.aTargetCostOf)
	sc.Step(`^a stagnation window of (\d+) iterations$`, fc.aStagnationWindowOf)
	sc.Step(`^I run DA for (\d+) iterations$`, fc.iRunDAForIterations)
	sc.Step(`^I run DA twice for (\d+) iterations with seed (\d+)$`, fc.iRunDATwiceWithSeed)

	sc.Step(`^the best cost should be less than ([\d.]+)$`, fc.theBestCostShouldBeLessThan)
	sc.Step(`^the best cost should be less than ([\d.]+) percent of the initial best$`,
		fc.theBestCostShouldBeLessThanPercentOfTheInitialBest)
	sc.Step(`^the best position should have (\d+) components$`, fc.theBestPositionShouldHaveComponents)
	sc.Step(`^the best position should be within bounds$`, fc.theBestPositionShouldBeWithinBounds)
	sc.Step(`^the evaluation count should be positive$`, fc.theEvaluationCountShouldBePositive)
	sc.Step(`^the reported seed should be non-zero$`, fc.theReportedSeedShouldBeNonZero)
	sc.Step(`^the termination reason should be "([^"]*)"$`, fc.theTerminationReasonShouldBe)
	sc.Step(`^the iteration count should be (\d+)$`, fc.theIterationCountShouldBe)
	sc.Step(`^the iteration count should be less than (\d+)$`, fc.theIterationCountShouldBeLessThan)
	sc.Step(`^the convergence curve should have (\d+) entries$`, fc.theConvergenceCurveShouldHaveEntries)
	sc.Step(`^the convergence curve should be non-increasing$`, fc.theConvergenceCurveShouldBeNonIncreasing)
	sc.Step(`^both runs should return identical results$`, fc.bothRunsShouldReturnIdenticalResults)
}

// registerBoundarySteps wires features/boundary_handling.feature.
func registerBoundarySteps(sc *godog.ScenarioContext, fc *featureContext) {
	sc.Step(`^a new default configuration$`, fc.aNewDefaultConfiguration)
	sc.Step(`^the effective boundary method should be "([^"]*)"$`, fc.theEffectiveBoundaryMethodShouldBe)
	sc.Step(`^a box from (-?[\d.]+) to (-?[\d.]+)$`, fc.aBoxFromTo)
	sc.Step(`^a component at (-?[\d.]+) with step (-?[\d.]+)$`, fc.aComponentAtWithStep)
	sc.Step(`^the "([^"]*)" boundary rule is applied$`, fc.theBoundaryRuleIsApplied)
	sc.Step(`^the component should be (-?[\d.]+)$`, fc.theComponentShouldBe)
	sc.Step(`^the step component should be (-?[\d.]+)$`, fc.theStepComponentShouldBe)
	sc.Step(`^the step component should have been redrawn inside \[0, 1\)$`,
		fc.theStepComponentShouldHaveBeenRedrawn)
	sc.Step(`^I run DA for (\d+) iterations with the "([^"]*)" boundary method$`,
		fc.iRunDAWithBoundaryMethod)
	sc.Step(`^every position in the final swarm should be within bounds$`,
		fc.everyPositionInTheFinalSwarmShouldBeWithinBounds)
}

// registerValidationSteps wires features/configuration_validation.feature.
func registerValidationSteps(sc *godog.ScenarioContext, fc *featureContext) {
	sc.Step(`^a valid configuration$`, fc.aValidConfiguration)
	sc.Step(`^a binary configuration$`, fc.aBinaryConfiguration)
	sc.Step(`^I clear the objective function$`, fc.iClearTheObjectiveFunction)
	sc.Step(`^I set (.+)$`, fc.iSetField)
	sc.Step(`^validation should succeed$`, fc.validationShouldSucceed)
	sc.Step(`^validation should fail with an error containing "([^"]*)"$`,
		fc.validationShouldFailWithAnErrorContaining)
	sc.Step(`^Optimize should refuse to run it$`, fc.optimizeShouldRefuseToRunIt)
	sc.Step(`^the run should complete$`, fc.theRunShouldComplete)
}

// registerConstraintSteps wires features/constraint_handling.feature.
func registerConstraintSteps(sc *godog.ScenarioContext, fc *featureContext) {
	sc.Step(`^a candidate with cost (-?[\d.]+) and violation (-?[\d.]+)$`,
		fc.aCandidateWithCostAndViolation)
	sc.Step(`^an incumbent with cost (-?[\d.]+) and violation (-?[\d.]+)$`,
		fc.anIncumbentWithCostAndViolation)
	sc.Step(`^I rank them under feasibility rules$`, fc.iRankThemUnderFeasibilityRules)
	sc.Step(`^I rank them under the penalty method with factor (-?[\d.]+)$`,
		fc.iRankThemUnderThePenaltyMethodWithFactor)
	sc.Step(`^the candidate should be preferred$`, fc.theCandidateShouldBePreferred)
	sc.Step(`^the incumbent should be preferred$`, fc.theIncumbentShouldBePreferred)
	sc.Step(`^I compute its "([^"]*)" penalized cost with factor (-?[\d.]+)$`,
		fc.iComputeItsPenalizedCostWithFactor)
	sc.Step(`^the penalized cost should be (-?[\d.]+)$`, fc.thePenalizedCostShouldBe)
	sc.Step(`^a one-dimensional problem minimizing x squared on \[-5, 5\] subject to x at least 1$`,
		fc.aConstrainedOneDimensionalProblem)
	sc.Step(`^I optimize it using feasibility rules$`, fc.iOptimizeItUsingFeasibilityRules)
	sc.Step(`^the returned solution should satisfy every configured constraint$`,
		fc.theReturnedSolutionShouldSatisfyEveryConstraint)
	sc.Step(`^the reported cost should be the raw objective cost$`,
		fc.theReportedCostShouldBeTheRawObjectiveCost)
}

// registerVariantSteps wires features/variant_execution.feature.
func registerVariantSteps(sc *godog.ScenarioContext, fc *featureContext) {
	sc.Step(`^a OneMax problem over (\d+) bits$`, fc.aOneMaxProblemOverBits)
	sc.Step(`^the "([^"]*)" transfer function$`, fc.theTransferFunction)
	sc.Step(`^I run BDA for (\d+) iterations$`, fc.iRunBDAForIterations)
	sc.Step(`^every component of the best position should be 0 or 1$`,
		fc.everyComponentOfTheBestPositionShouldBeABit)
	sc.Step(`^a ZDT1 problem with dimension (\d+)$`, fc.aZDT1ProblemWithDimension)
	sc.Step(`^I run MODA for (\d+) iterations$`, fc.iRunMODAForIterations)
	sc.Step(`^the archive should be non-dominated$`, fc.theArchiveShouldBeNonDominated)
	sc.Step(`^the archive should hold at least (\d+) solutions$`, fc.theArchiveShouldHoldAtLeastSolutions)
	sc.Step(`^every archived position should be within bounds$`,
		fc.everyArchivedPositionShouldBeWithinBounds)
}

// InitializeScenario registers every step definition against a scenario-scoped
// context, so that state never leaks from one scenario into the next.
func InitializeScenario(sc *godog.ScenarioContext) {
	fc := &featureContext{}

	sc.Before(func(ctx context.Context, _ *godog.Scenario) (context.Context, error) {
		*fc = featureContext{}

		return ctx, nil
	})

	registerSharedSteps(sc, fc)
	registerBoundarySteps(sc, fc)
	registerValidationSteps(sc, fc)
	registerConstraintSteps(sc, fc)
	registerVariantSteps(sc, fc)
}

// TestFeatures runs the Gherkin suite in features/ through godog.
//
// The scenarios are seeded end-to-end runs, so they are deterministic; they are
// not gated behind testing.Short() because the whole suite is a few seconds of
// short runs, not a statistical study.
func TestFeatures(t *testing.T) {
	suite := godog.TestSuite{
		ScenarioInitializer: InitializeScenario,
		Options: &godog.Options{
			Format:   "pretty",
			Paths:    []string{"features"},
			TestingT: t,
		},
	}

	if suite.Run() != 0 {
		t.Fatal("godog reported a non-zero status: feature tests failed")
	}
}
