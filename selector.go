// Problem classification and variant recommendation: given what is known about
// a problem, which Dragonfly variant should run it, and with which preset.

package dragonfly

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"
)

// ErrUnsupportedProblemClass reports a problem that needs a variant the
// library does not implement. In particular, MODA is continuous and BDA is
// single-objective; neither can solve a discrete multi-objective problem.
var ErrUnsupportedProblemClass = errors.New("discrete multi-objective problems are not supported")

// Modality describes how many optima the landscape has.
type Modality int

const (
	// Unimodal is a single optimum.
	Unimodal Modality = iota
	// Multimodal is a handful of optima.
	Multimodal
	// HighlyMultimodal is a lattice or a forest of them.
	HighlyMultimodal
)

// String names the modality for a report.
func (m Modality) String() string {
	switch m {
	case Unimodal:
		return "unimodal"
	case Multimodal:
		return "multimodal"
	case HighlyMultimodal:
		return "highly multimodal"
	default:
		return fmt.Sprintf("Modality(%d)", int(m))
	}
}

// Landscape describes the terrain between the optima.
type Landscape int

const (
	// Smooth has few local features.
	Smooth Landscape = iota
	// Rugged has many local features.
	Rugged
	// Deceptive has gradients that lead away from the global optimum.
	Deceptive
	// NarrowValley is ill-conditioned: a long thin basin.
	NarrowValley
)

// String names the landscape for a report.
func (l Landscape) String() string {
	switch l {
	case Smooth:
		return "smooth"
	case Rugged:
		return "rugged"
	case Deceptive:
		return "deceptive"
	case NarrowValley:
		return "narrow valley"
	default:
		return fmt.Sprintf("Landscape(%d)", int(l))
	}
}

// ProblemCharacteristics describes an optimization problem well enough to pick
// a variant for it.
//
// ClassifyProblem fills in Dimensionality, Modality, Landscape and
// RequiresStableConvergence by sampling the objective. Discrete,
// MultiObjective, ExpensiveEvaluations and RequiresFastConvergence are
// statements about the caller's problem and budget that no amount of sampling
// can recover, so the caller sets them.
type ProblemCharacteristics struct {
	// Dimensionality is the number of decision variables.
	Dimensionality int

	// Modality describes how many optima the landscape has.
	Modality Modality

	// Landscape describes the terrain between them.
	Landscape Landscape

	// Discrete marks a search space whose variables are binary or otherwise
	// discrete. It is what routes a problem to BDA.
	Discrete bool

	// ExpensiveEvaluations marks an objective whose evaluation dominates the
	// run time, so that a variant's overhead matters.
	ExpensiveEvaluations bool

	// RequiresFastConvergence marks a run with a wall-clock or budget limit,
	// where a good answer soon beats the best answer eventually.
	RequiresFastConvergence bool

	// RequiresStableConvergence marks a run whose variance across seeds
	// matters as much as its mean.
	RequiresStableConvergence bool

	// MultiObjective marks a problem with several objectives to trade off. It
	// is what routes a problem to MODA.
	MultiObjective bool
}

// AlgorithmRecommendation is one variant scored against a problem, with the
// reason it was scored that way.
//
// Reason is never empty. A recommendation a caller cannot interrogate is worth
// no more than a coin flip, and this layer is heuristic enough that the caller
// deserves to see the heuristic.
type AlgorithmRecommendation struct {
	// Variant is the recommended variant.
	Variant AlgorithmVariant

	// Error is non-nil when no implemented variant can solve the requested
	// problem class. Variant is nil in that case.
	Error error

	// Reason explains, in one line, why this variant scored as it did.
	Reason string

	// Preset names the configuration factory to start from. It reflects the
	// problem's shape, not the variant, so every recommendation for a given
	// problem carries the same preset -- except that a discrete problem always
	// gets PresetBinary.
	Preset ConfigPreset

	// Score is the fit in [0,1], from AlgorithmVariant.ApplicableTo.
	Score float64

	// Confidence is how much weight to put on Score, in [0,1]. It is lower
	// where the heuristics are guessing: a variant scored outside the problem
	// class it was written for, or a high-overhead variant on an expensive
	// objective.
	Confidence float64
}

// AlgorithmSelector ranks the available variants against a problem.
type AlgorithmSelector struct {
	variants []AlgorithmVariant
}

// NewAlgorithmSelector creates a selector over every variant, in the canonical
// order.
func NewAlgorithmSelector() *AlgorithmSelector {
	return &AlgorithmSelector{variants: GetAllVariants()}
}

// NewAlgorithmSelectorFor creates a selector over a given set of variants.
func NewAlgorithmSelectorFor(variants ...AlgorithmVariant) *AlgorithmSelector {
	return &AlgorithmSelector{variants: variants}
}

// RecommendAlgorithms returns every variant scored against the problem, best
// first. Ties keep the canonical variant order, so the ranking is stable.
func (s *AlgorithmSelector) RecommendAlgorithms(
	characteristics ProblemCharacteristics,
) []AlgorithmRecommendation {
	if characteristics.Discrete && characteristics.MultiObjective {
		return nil
	}

	preset := RecommendPreset(characteristics)
	recommendations := make([]AlgorithmRecommendation, 0, len(s.variants))

	for _, variant := range s.variants {
		if variant == nil {
			continue
		}

		score := variant.ApplicableTo(characteristics)
		recommendations = append(recommendations, AlgorithmRecommendation{
			Variant:    variant,
			Reason:     explainRecommendation(characteristics, variant, score),
			Preset:     preset,
			Score:      score,
			Confidence: recommendationConfidence(characteristics, variant),
		})
	}

	sort.SliceStable(recommendations, func(i, j int) bool {
		return recommendations[i].Score > recommendations[j].Score
	})

	return recommendations
}

// RecommendBest returns the single best-scoring variant for the problem.
func (s *AlgorithmSelector) RecommendBest(
	characteristics ProblemCharacteristics,
) AlgorithmRecommendation {
	if characteristics.Discrete && characteristics.MultiObjective {
		return AlgorithmRecommendation{
			Reason:     ErrUnsupportedProblemClass.Error(),
			Preset:     PresetDefault,
			Confidence: 1,
			Error:      ErrUnsupportedProblemClass,
		}
	}

	recommendations := s.RecommendAlgorithms(characteristics)
	if len(recommendations) == 0 {
		return AlgorithmRecommendation{
			Variant:    &DAVariant{},
			Reason:     "no variants were available to score; falling back to standard DA",
			Preset:     RecommendPreset(characteristics),
			Score:      0.5,
			Confidence: 0.3,
		}
	}

	return recommendations[0]
}

// RecommendPreset names the configuration factory a problem of this shape
// should start from.
//
// A discrete problem needs NewBinaryConfig regardless of anything else: it is
// the only preset whose bounds, step clamp and transfer function match a bit
// string. Otherwise dimensionality decides first, because a swarm too small to
// cover the space cannot be rescued by tuning, and a stated time budget
// decides second.
func RecommendPreset(characteristics ProblemCharacteristics) ConfigPreset {
	switch {
	case characteristics.Discrete && characteristics.MultiObjective:
		return PresetDefault
	case characteristics.Discrete:
		return PresetBinary
	case characteristics.Dimensionality >= highDimensionalThreshold:
		return PresetHighDimensional
	case characteristics.RequiresFastConvergence:
		return PresetFastConvergence
	default:
		return PresetDefault
	}
}

// highDimensionalThreshold is the dimensionality at which the default swarm of
// 40 over 1000 iterations stops covering the search space. It matches
// autoTuneLargeDims in config_loader.go on purpose: the two heuristics should
// not disagree about what "high-dimensional" means.
const highDimensionalThreshold = autoTuneLargeDims

// recommendationConfidence estimates how much to trust a variant's score.
func recommendationConfidence(
	characteristics ProblemCharacteristics,
	variant AlgorithmVariant,
) float64 {
	confidence := 0.7

	// Problem class is the one thing the heuristics are sure about: a variant
	// either handles the class or it does not.
	switch {
	case characteristics.MultiObjective:
		confidence = 0.3
		if variant.IsMultiObjective() {
			confidence = 0.95
		}
	case characteristics.Discrete:
		confidence = 0.3
		if variant.Name() == nameBDA {
			confidence = 0.9
		}
	case variant.Name() == nameDA:
		confidence = 0.85
	}

	// An expensive objective makes overhead a real cost rather than a rounding
	// error, and the overhead figures are themselves estimates.
	if characteristics.ExpensiveEvaluations && variant.EstimatedOverhead() > 1.05 {
		confidence *= 0.7
	}

	return min(confidence, 1.0)
}

// explainRecommendation builds the one-line reason. It never returns an empty
// string: the fallback quotes the score and the variant's own description.
func explainRecommendation(
	characteristics ProblemCharacteristics,
	variant AlgorithmVariant,
	score float64,
) string {
	reasons := classReasons(characteristics, variant)
	reasons = append(reasons, shapeReasons(characteristics, variant)...)

	if len(reasons) == 0 {
		return fmt.Sprintf("no decisive signal (score %.2f): %s", score, variant.Description())
	}

	return strings.Join(reasons, "; ")
}

// classReasons explains the fit between the problem class -- continuous,
// discrete or multi-objective -- and the variant. Exactly one of the three
// branches applies, so this always contributes a reason.
func classReasons(characteristics ProblemCharacteristics, variant AlgorithmVariant) []string {
	switch {
	case characteristics.MultiObjective:
		if variant.IsMultiObjective() {
			return []string{"several objectives to trade off: MODA maintains a Pareto archive"}
		}

		return []string{"several objectives, but this variant reports a single incumbent"}

	case characteristics.Discrete:
		if variant.Name() == nameBDA {
			return []string{"discrete search space: BDA flips bits through a transfer function"}
		}

		return []string{"discrete search space, but this variant moves continuous positions"}

	default:
		if variant.Name() == nameDA {
			return []string{"continuous single-objective problem: DA is the paper's baseline"}
		}

		if variant.Name() == nameMHDA || variant.Name() == nameCDA || variant.Name() == nameQGDA {
			return []string{"continuous single-objective problem: this improved DA variant matches the problem class"}
		}

		return []string{"continuous single-objective problem is outside this variant's class"}
	}
}

// shapeReasons explains the fit between the landscape, the dimensionality and
// the budget on one side and the variant on the other.
func shapeReasons(characteristics ProblemCharacteristics, variant AlgorithmVariant) []string {
	reasons := make([]string, 0, 4)

	if characteristics.Modality == HighlyMultimodal {
		switch variant.Name() {
		case nameCDA:
			reasons = append(reasons, "highly multimodal: chaotic coefficients preserve movement diversity")
		case nameQGDA:
			reasons = append(reasons, "highly multimodal: mutation and rotation add two diversity stages")
		case nameMHDA:
			reasons = append(reasons, "highly multimodal: personal memory retains promising regions")
		default:
			reasons = append(reasons,
				"highly multimodal: the growing neighborhood radius keeps the swarm exploring longer")
		}
	}

	if characteristics.Landscape == Deceptive && characteristics.Modality != Unimodal {
		reasons = append(reasons, "deceptive landscape: expect the Lévy walk to do the escaping")
	}

	if characteristics.Landscape == NarrowValley {
		reasons = append(reasons,
			"narrow valley: lower MaxStepRatio so the step clamp does not overshoot the basin")
	}

	if characteristics.Dimensionality >= highDimensionalThreshold {
		reasons = append(reasons, fmt.Sprintf(
			"%d dimensions: start from NewHighDimensionalConfig for a larger swarm and slower radius growth",
			characteristics.Dimensionality))
	}

	if characteristics.ExpensiveEvaluations {
		reasons = append(reasons, fmt.Sprintf(
			"expensive evaluations: this variant costs about %.2fx the DA baseline",
			variant.EstimatedOverhead()))
	}

	if characteristics.RequiresFastConvergence {
		reasons = append(reasons, "fast convergence required: NewFastConvergenceConfig shortens the run")
	}

	if characteristics.RequiresStableConvergence {
		reasons = append(reasons, "stable convergence required: fix Config.Rand and report the seed")
	}

	return reasons
}

// Sampling budget for ClassifyProblem. Deliberately small: classification runs
// before the optimization, and a classifier that costs a meaningful fraction of
// the run it is choosing for is not worth having.
const (
	// classifyLines is the number of random line scans across the box.
	classifyLines = 6
	// classifyLineSteps is the number of samples along each line. It has to be
	// dense enough not to alias a landscape with per-unit structure, such as
	// Rastrigin's lattice, across a box tens of units wide.
	classifyLineSteps  = 65
	classifyIterations = 20
	classifyRuns       = 3
	classifyPopulation = 10
)

// Classification thresholds, all applied to scale-free quantities so that they
// mean the same thing on Sphere over [-5,5] and on Schwefel over [-500,500].
const (
	// unimodalTurningPoints is the average number of direction changes per line
	// scan below which the landscape reads as single-basin. A quadratic bowl
	// crossed by a straight line turns exactly once.
	unimodalTurningPoints = 1.5
	// multimodalTurningPoints separates a handful of optima from a lattice.
	// Across forty seeds at d=10, Schwefel turns 5.17 to 8.83 times per
	// line. The former threshold of 6 made its verdict seed-dependent, so
	// this sits below the observed range while leaving single-basin shapes
	// (about one turn) well clear.
	multimodalTurningPoints = 5.0
	// smoothRoughness is the total variation along a line scan, in units of
	// that line's own value range, at or above which the landscape reads as
	// rugged. A line crossing a single basin cannot exceed twice its range;
	// forty seeded d=10 sweeps put Sphere and Rosenbrock below 2 while
	// Schwefel starts at 2.50. The former threshold of 3 made Schwefel's
	// landscape verdict depend on the seed.
	smoothRoughness = 2.2
)

// ClassifyProblem samples an objective function to estimate its landscape.
//
// It fills in Dimensionality, Modality, Landscape and
// RequiresStableConvergence. Discrete, MultiObjective, ExpensiveEvaluations and
// RequiresFastConvergence are left at false, because they are facts about the
// caller's problem and budget rather than about the function's values; set them
// on the returned value before passing it to a selector.
//
// # What it can and cannot see
//
// Modality and Landscape come from a handful of straight-line scans across the
// box: how often the function changes direction along a line, and how much total
// variation it accumulates relative to that line's own range. Both quantities
// are scale-free, so the same thresholds apply whatever the bounds and whatever
// the units of the cost.
//
// Landscape is therefore only ever reported as Smooth or Rugged. Deceptive
// (gradients that lead away from the global optimum) and NarrowValley (an
// ill-conditioned basin) are statements about where the optimum is relative to
// the terrain, and a few dozen samples cannot establish either. A caller who
// knows their problem is Schwefel-like or Rosenbrock-like should say so by
// setting Landscape on the returned value.
//
// The estimates are coarse heuristics. Treat a classification as a starting
// point a caller who knows their problem should override.
//
// rng is the last parameter, as every stochastic helper in this package takes
// it. Pass a seeded generator to make a classification reproducible; nil draws
// a fresh one.
func ClassifyProblem(
	fn ObjectiveFunction,
	size int,
	lower, upper float64,
	rng *rand.Rand,
) ProblemCharacteristics {
	characteristics, _ := ClassifyProblemChecked(fn, size, lower, upper, rng)

	return characteristics
}

// ClassifyProblemChecked is ClassifyProblem with input validation. Prefer it
// when the objective or bounds are supplied dynamically; the legacy wrapper
// returns only the partially known dimensionality when validation fails.
func ClassifyProblemChecked(
	fn ObjectiveFunction,
	size int,
	lower, upper float64,
	rng *rand.Rand,
) (ProblemCharacteristics, error) {
	if fn == nil {
		return ProblemCharacteristics{Dimensionality: size}, errors.New("classification objective function is required")
	}

	if size <= 0 {
		return ProblemCharacteristics{Dimensionality: size},
			fmt.Errorf("classification problem size must be positive, got %d", size)
	}

	if !isFinite(lower) || !isFinite(upper) {
		return ProblemCharacteristics{Dimensionality: size}, errors.New("classification bounds must be finite")
	}

	if lower >= upper {
		return ProblemCharacteristics{Dimensionality: size},
			fmt.Errorf("classification lower bound %v must be less than upper bound %v", lower, upper)
	}

	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	turningPoints, roughness := lineScanStatistics(fn, size, lower, upper, rng)
	stability := estimateStability(fn, size, lower, upper, rng)

	return ProblemCharacteristics{
		Dimensionality:            size,
		Modality:                  modalityFromTurningPoints(turningPoints),
		Landscape:                 landscapeFromRoughness(roughness),
		RequiresStableConvergence: stability < 0.5,
	}, nil
}

// modalityFromTurningPoints maps the average number of direction changes per
// line scan onto a modality.
func modalityFromTurningPoints(turningPoints float64) Modality {
	switch {
	case turningPoints >= multimodalTurningPoints:
		return HighlyMultimodal
	case turningPoints >= unimodalTurningPoints:
		return Multimodal
	default:
		return Unimodal
	}
}

// landscapeFromRoughness maps the average normalized total variation per line
// scan onto a landscape. It never returns Deceptive or NarrowValley; see
// ClassifyProblem.
func landscapeFromRoughness(roughness float64) Landscape {
	if roughness >= smoothRoughness {
		return Rugged
	}

	return Smooth
}

// lineScanStatistics walks several random straight lines across the search box
// and returns the average number of direction changes per line and the average
// total variation per line in units of that line's own value range.
//
// Both are scale-free by construction: the turning-point count does not look at
// magnitudes at all, and the roughness divides by the range it measured. That
// is the whole point -- the gradient-magnitude heuristic this replaced called
// Sphere over [-5,5] rugged and Sphere over [-1,1] smooth, which says more
// about the bounds than about the function.
func lineScanStatistics(
	fn ObjectiveFunction,
	size int,
	lower, upper float64,
	rng *rand.Rand,
) (float64, float64) {
	totalTurns, totalRoughness := 0.0, 0.0

	for range classifyLines {
		values := scanLine(fn, size, lower, upper, rng)
		turns, roughness := lineShape(values)
		totalTurns += turns
		totalRoughness += roughness
	}

	return totalTurns / classifyLines, totalRoughness / classifyLines
}

// scanLine samples fn at evenly spaced points along the segment between two
// uniformly drawn points of the box.
func scanLine(fn ObjectiveFunction, size int, lower, upper float64, rng *rand.Rand) []float64 {
	from := unifrndVec(lower, upper, size, rng)
	to := unifrndVec(lower, upper, size, rng)

	point := make([]float64, size)
	values := make([]float64, classifyLineSteps)

	for step := range values {
		fraction := float64(step) / float64(classifyLineSteps-1)
		for j := range point {
			point[j] = from[j] + fraction*(to[j]-from[j])
		}

		values[step] = fn(point)
	}

	return values
}

// lineShape returns the number of direction changes along a scan and its total
// variation divided by its value range. A flat or non-finite scan contributes
// nothing to either.
func lineShape(values []float64) (float64, float64) {
	turns := 0.0
	totalVariation := 0.0
	lowest, highest := math.Inf(1), math.Inf(-1)
	previousSign := 0

	for i := 1; i < len(values); i++ {
		delta := values[i] - values[i-1]
		if !isFinite(delta) {
			return 0, 0
		}

		totalVariation += math.Abs(delta)

		sign := 0
		if delta > 0 {
			sign = 1
		} else if delta < 0 {
			sign = -1
		}

		if sign != 0 {
			if previousSign != 0 && sign != previousSign {
				turns++
			}

			previousSign = sign
		}
	}

	for _, value := range values {
		if !isFinite(value) {
			return 0, 0
		}

		lowest = math.Min(lowest, value)
		highest = math.Max(highest, value)
	}

	valueRange := highest - lowest
	if valueRange <= 0 {
		return turns, 0
	}

	return turns, totalVariation / valueRange
}

// estimateStability runs a few very short optimizations and reports
// 1/(1+cv) of their final costs, in [0,1]. A low value means the outcome
// depends heavily on the seed.
func estimateStability(
	fn ObjectiveFunction,
	size int,
	lower, upper float64,
	rng *rand.Rand,
) float64 {
	costs := make([]float64, 0, classifyRuns)

	for range classifyRuns {
		config := NewDefaultConfig()
		config.ObjectiveFunc = fn
		config.ProblemSize = size
		config.LowerBound = lower
		config.UpperBound = upper
		config.MaxIterations = classifyIterations
		config.NPop = classifyPopulation
		config.Rand = rand.New(rand.NewSource(rng.Int63()))

		result, err := Optimize(config)
		if err != nil {
			continue
		}

		costs = append(costs, result.GlobalBest.Cost)
	}

	if len(costs) == 0 {
		return 0
	}

	mean, stdDev := meanAndStdDev(costs)
	cv := stdDev / (math.Abs(mean) + 1e-10)

	return 1.0 / (1.0 + cv)
}

// meanAndStdDev returns the mean and the population standard deviation of
// values. An empty slice yields two zeros.
func meanAndStdDev(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}

	mean := 0.0
	for _, value := range values {
		mean += value
	}

	mean /= float64(len(values))

	variance := 0.0

	for _, value := range values {
		diff := value - mean
		variance += diff * diff
	}

	variance /= float64(len(values))

	return mean, math.Sqrt(variance)
}

// benchmarkCharacteristics is the hand-classified shape of each benchmark
// function in functions.go. It is a table rather than a call to ClassifyProblem
// because these landscapes are known from the literature, and a sampled
// estimate of a known answer is strictly worse than the known answer.
//
// The multi-objective entries are the ZDT family and SchafferN1, whose
// MultiObjectiveFunction signature the single-objective classifier cannot even
// call.
var benchmarkCharacteristics = map[string]ProblemCharacteristics{
	"Sphere":     {Dimensionality: 30, Modality: Unimodal, Landscape: Smooth},
	"Zakharov":   {Dimensionality: 30, Modality: Unimodal, Landscape: Smooth},
	"Rosenbrock": {Dimensionality: 30, Modality: Unimodal, Landscape: NarrowValley, RequiresStableConvergence: true},
	"BentCigar":  {Dimensionality: 30, Modality: Unimodal, Landscape: NarrowValley, RequiresStableConvergence: true},
	"Discus":     {Dimensionality: 30, Modality: Unimodal, Landscape: NarrowValley, RequiresStableConvergence: true},
	"DixonPrice": {Dimensionality: 30, Modality: Unimodal, Landscape: NarrowValley, RequiresStableConvergence: true},
	"Ackley":     {Dimensionality: 30, Modality: Multimodal, Landscape: Rugged},
	"HappyCat":   {Dimensionality: 30, Modality: Multimodal, Landscape: Rugged},
	"Rastrigin":  {Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged},
	"Griewank":   {Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged},
	"Weierstrass": {
		Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged,
	},
	"ExpandedSchafferF6": {
		Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged,
	},
	"Himmelblau": {
		Dimensionality: 30, Modality: Multimodal, Landscape: Rugged,
	},
	"Schwefel":    {Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Deceptive},
	"Michalewicz": {Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Deceptive},
	"Levy":        {Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged},
	"ZDT1":        {Dimensionality: 30, Modality: Unimodal, Landscape: Smooth, MultiObjective: true},
	"ZDT2":        {Dimensionality: 30, Modality: Unimodal, Landscape: Smooth, MultiObjective: true},
	"ZDT3":        {Dimensionality: 30, Modality: HighlyMultimodal, Landscape: Rugged, MultiObjective: true},
	"SchafferN1":  {Dimensionality: 1, Modality: Unimodal, Landscape: Smooth, MultiObjective: true},
}

// BenchmarkCharacteristics returns the hand-classified characteristics of a
// benchmark function from functions.go, and whether the name is known.
func BenchmarkCharacteristics(benchmarkName string) (ProblemCharacteristics, bool) {
	characteristics, ok := benchmarkCharacteristics[benchmarkName]

	return characteristics, ok
}

// BenchmarkNames returns every benchmark BenchmarkCharacteristics knows, in a
// stable alphabetical order.
func BenchmarkNames() []string {
	names := make([]string, 0, len(benchmarkCharacteristics))
	for name := range benchmarkCharacteristics {
		names = append(names, name)
	}

	sort.Strings(names)

	return names
}

// RecommendForBenchmark recommends a variant for a named benchmark function
// from functions.go.
//
// An unrecognized name falls back to a generic 30-dimensional multimodal
// continuous problem -- the shape most benchmark suites are dominated by -- and
// says so in the Reason, so a typo does not look like a considered answer.
func RecommendForBenchmark(benchmarkName string) AlgorithmRecommendation {
	characteristics, known := BenchmarkCharacteristics(benchmarkName)
	if !known {
		characteristics = ProblemCharacteristics{
			Dimensionality: 30,
			Modality:       Multimodal,
			Landscape:      Rugged,
		}
	}

	recommendation := NewAlgorithmSelector().RecommendBest(characteristics)
	if !known {
		recommendation.Reason = fmt.Sprintf(
			"benchmark %q is not in the table; assumed a generic 30-D multimodal continuous problem (%s)",
			benchmarkName, recommendation.Reason)
		recommendation.Confidence *= 0.5
	}

	return recommendation
}

// PrintRecommendations writes a ranked recommendation table to standard output.
func PrintRecommendations(recommendations []AlgorithmRecommendation) {
	fmt.Println("Variant recommendations (ranked by score):")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Printf("%-6s | %-7s | %-10s | %-16s | %s\n", "Variant", "Score", "Confidence", "Preset", "Reason")
	fmt.Println(strings.Repeat("-", 100))

	for _, recommendation := range recommendations {
		fmt.Printf("%-6s | %6.1f%% | %9.1f%% | %-16s | %s\n",
			recommendation.Variant.Name(),
			recommendation.Score*100,
			recommendation.Confidence*100,
			string(recommendation.Preset),
			recommendation.Reason)
	}

	fmt.Println(strings.Repeat("=", 100))
}
