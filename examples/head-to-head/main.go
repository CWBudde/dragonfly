// Command head-to-head compares standard Dragonfly, Mayfly and CMA-ES under
// identical objective-evaluation budgets.
package main

import (
	"context"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"

	cmaes "github.com/CWBudde/go-cma-es"
	"github.com/cwbudde/mayfly"

	"github.com/CWBudde/dragonfly"
)

const (
	dragonflyPopulation           = 40
	mayflyMalePopulation          = 20
	mayflyFemalePopulation        = 20
	mayflyEvaluationsPerIteration = 62
)

type options struct {
	csvPath           string
	markdownPath      string
	dragonflyRevision string
	mayflyRevision    string
	cmaRevision       string
	dimensions        int
	runs              int
	budget            int
	seed              int64
	includeCMA        bool
}

type benchmark struct {
	objective  dragonfly.ObjectiveFunction
	name       string
	lowerBound float64
	upperBound float64
}

type study struct {
	results []*dragonfly.ComparisonResult
	options options
}

type optimizerVariant struct {
	run  func(context.Context, *dragonfly.Config) (*dragonfly.Result, error)
	name string
}

type evaluationBudget struct {
	objective dragonfly.ObjectiveFunction
	limit     int64
	used      atomic.Int64
}

func main() {
	opts := parseFlags()

	result, err := runStudy(context.Background(), opts)
	if err != nil {
		log.Fatal(err)
	}

	err = result.write()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Wrote %s and %s\n", opts.csvPath, opts.markdownPath)
}

func parseFlags() options {
	opts := options{}

	flag.StringVar(&opts.csvPath, "out", "head-to-head.csv", "raw CSV output path")
	flag.StringVar(&opts.markdownPath, "markdown", "", "Markdown output path (default: beside CSV)")
	flag.IntVar(&opts.dimensions, "dimensions", 30, "benchmark dimensions")
	flag.IntVar(&opts.runs, "runs", 30, "paired runs per optimizer and benchmark")
	flag.IntVar(&opts.budget, "budget", 20_000, "real objective evaluations per run")
	flag.Int64Var(&opts.seed, "seed", 20260827, "base seed")
	flag.BoolVar(&opts.includeCMA, "cma", true, "include CMA-ES as a calibration baseline")
	flag.StringVar(&opts.dragonflyRevision, "dragonfly-revision", "working-tree", "Dragonfly revision recorded in the report")
	flag.StringVar(&opts.mayflyRevision, "mayfly-revision", "working-tree", "Mayfly revision recorded in the report")
	flag.StringVar(&opts.cmaRevision, "cma-revision", "working-tree", "CMA-ES revision recorded in the report")
	flag.Parse()

	if opts.markdownPath == "" {
		extension := filepath.Ext(opts.csvPath)
		opts.markdownPath = strings.TrimSuffix(opts.csvPath, extension) + ".md"
	}

	return opts
}

func runStudy(ctx context.Context, opts options) (*study, error) {
	err := validateOptions(opts)
	if err != nil {
		return nil, err
	}

	variants := []dragonfly.AlgorithmVariant{
		optimizerVariant{name: "DA", run: runDragonfly(opts.budget)},
		optimizerVariant{name: "MA", run: runMayfly(opts.budget)},
	}
	if opts.includeCMA {
		variants = append(variants, optimizerVariant{name: "CMA-ES", run: runCMAES(opts.budget)})
	}

	result := &study{options: opts, results: make([]*dragonfly.ComparisonResult, 0, len(benchmarks()))}
	for _, problem := range benchmarks() {
		runner := dragonfly.NewComparisonRunner().
			WithVariants(variants...).
			WithRuns(opts.runs).
			WithIterations(opts.budget).
			WithSeed(opts.seed)

		comparison, err := runner.CompareContext(
			ctx, problem.name, problem.objective, opts.dimensions,
			problem.lowerBound, problem.upperBound,
		)
		if err != nil {
			return nil, fmt.Errorf("compare %s: %w", problem.name, err)
		}

		for algorithm, runs := range comparison.RunResults {
			for runIndex, run := range runs {
				if run.FuncEvals != opts.budget {
					return nil, fmt.Errorf("%s/%s run %d used %d evaluations, want %d",
						problem.name, comparison.AlgorithmNames[algorithm], runIndex+1,
						run.FuncEvals, opts.budget)
				}
			}
		}

		result.results = append(result.results, comparison)
	}

	return result, nil
}

func validateOptions(opts options) error {
	if opts.runs <= 0 {
		return fmt.Errorf("runs must be positive, got %d", opts.runs)
	}

	if opts.dimensions <= 0 {
		return fmt.Errorf("dimensions must be positive, got %d", opts.dimensions)
	}

	minimumBudget := max(mayflyMalePopulation+mayflyFemalePopulation, dragonflyPopulation)

	if opts.budget < minimumBudget {
		return fmt.Errorf("budget must be at least %d, got %d", minimumBudget, opts.budget)
	}

	if opts.csvPath == "" || opts.markdownPath == "" {
		return errors.New("CSV and Markdown output paths are required")
	}

	return nil
}

func benchmarks() []benchmark {
	return []benchmark{
		{name: "Sphere", objective: dragonfly.Sphere, lowerBound: -100, upperBound: 100},
		{name: "Rosenbrock", objective: dragonfly.Rosenbrock, lowerBound: -5, upperBound: 10},
		{name: "Rastrigin", objective: dragonfly.Rastrigin, lowerBound: -5.12, upperBound: 5.12},
		{name: "Ackley", objective: dragonfly.Ackley, lowerBound: -32.768, upperBound: 32.768},
		{name: "Griewank", objective: dragonfly.Griewank, lowerBound: -600, upperBound: 600},
	}
}

func (budget *evaluationBudget) evaluate(position []float64) float64 {
	call := budget.used.Add(1)
	if call > budget.limit {
		return math.Inf(1)
	}

	return budget.objective(position)
}

func (budget *evaluationBudget) evaluations() int {
	return int(min(budget.used.Load(), budget.limit))
}

func runDragonfly(limit int) func(context.Context, *dragonfly.Config) (*dragonfly.Result, error) {
	return func(ctx context.Context, base *dragonfly.Config) (*dragonfly.Result, error) {
		budget := &evaluationBudget{objective: base.ObjectiveFunc, limit: int64(limit)}
		config := dragonfly.NewDefaultConfig()
		config.ObjectiveFunc = budget.evaluate
		config.ProblemSize = base.ProblemSize
		config.LowerBound = base.LowerBound
		config.UpperBound = base.UpperBound
		config.NPop = dragonflyPopulation
		config.MaxIterations = max(1, ceilDiv(limit, config.NPop)-1)
		config.Seed = base.Seed

		result, err := dragonfly.OptimizeContext(ctx, config)
		if err != nil {
			return nil, err
		}

		result.FuncEvalCount = budget.evaluations()

		return result, nil
	}
}

func runMayfly(limit int) func(context.Context, *dragonfly.Config) (*dragonfly.Result, error) {
	return func(ctx context.Context, base *dragonfly.Config) (*dragonfly.Result, error) {
		budget := &evaluationBudget{objective: base.ObjectiveFunc, limit: int64(limit)}
		config := mayfly.NewDefaultConfig()
		config.ObjectiveFunc = mayfly.ObjectiveFunction(budget.evaluate)
		config.ProblemSize = base.ProblemSize
		config.LowerBound = base.LowerBound
		config.UpperBound = base.UpperBound
		config.NPop = mayflyMalePopulation
		config.NPopF = mayflyFemalePopulation
		remaining := max(0, limit-config.NPop-config.NPopF)
		config.MaxIterations = max(1, ceilDiv(remaining, mayflyEvaluationsPerIteration))
		config.Seed = base.Seed

		result, err := mayfly.OptimizeContext(ctx, config)
		if err != nil {
			return nil, err
		}

		return &dragonfly.Result{
			ConvergenceCurve: result.ConvergenceCurve,
			GlobalBest: dragonfly.Best{
				Position:            result.GlobalBest.Position,
				Cost:                result.GlobalBest.Cost,
				ConstraintViolation: result.GlobalBest.ConstraintViolation,
			},
			FuncEvalCount:  budget.evaluations(),
			IterationCount: result.IterationCount,
			Seed:           *base.Seed,
			SeedKnown:      true,
		}, nil
	}
}

func runCMAES(limit int) func(context.Context, *dragonfly.Config) (*dragonfly.Result, error) {
	return func(ctx context.Context, base *dragonfly.Config) (*dragonfly.Result, error) {
		seed := *base.Seed
		config := cmaes.NewDefaultConfig(base.ProblemSize)
		config.ObjectiveFunc = cmaes.ObjectiveFunction(base.ObjectiveFunc)
		config.LowerBound = base.LowerBound
		config.UpperBound = base.UpperBound
		config.InitialMean = randomInitialMean(
			base.ProblemSize, base.LowerBound, base.UpperBound, seed,
		)
		config.InitialSigma = 0.3 * (base.UpperBound - base.LowerBound)
		config.MaxEvaluations = limit
		config.MaxIterations = limit
		config.Convergence = nil
		config.Seed = &seed

		result, err := cmaes.OptimizeContext(ctx, config)
		if err != nil {
			return nil, err
		}

		return &dragonfly.Result{
			ConvergenceCurve: result.ConvergenceCurve,
			GlobalBest: dragonfly.Best{
				Position:            result.GlobalBest.Position,
				Cost:                result.GlobalBest.Cost,
				ConstraintViolation: result.GlobalBest.ConstraintViolation,
			},
			FuncEvalCount:  result.FuncEvalCount,
			IterationCount: result.IterationCount,
			Seed:           result.Seed,
			SeedKnown:      result.SeedKnown,
		}, nil
	}
}

func randomInitialMean(dimensions int, lower, upper float64, seed int64) []float64 {
	// Keep initialization independent from CMA-ES's internal draw stream. The
	// split constant is fixed, so a reported seed still reproduces both.
	rng := rand.New(rand.NewSource(seed ^ -7046029254386353131))
	mean := make([]float64, dimensions)

	span := upper - lower
	for i := range mean {
		mean[i] = lower + rng.Float64()*span
	}

	return mean
}

func ceilDiv(numerator, denominator int) int {
	if numerator <= 0 {
		return 0
	}

	return (numerator + denominator - 1) / denominator
}

func (variant optimizerVariant) Name() string { return variant.name }

func (variant optimizerVariant) FullName() string { return variant.name }

func (variant optimizerVariant) Description() string {
	return "head-to-head adapter with an exact objective-evaluation budget"
}

func (variant optimizerVariant) GetConfig() *dragonfly.Config {
	return dragonfly.NewDefaultConfig()
}

func (variant optimizerVariant) IsMultiObjective() bool { return false }

func (variant optimizerVariant) Run(
	ctx context.Context,
	config *dragonfly.Config,
	_ ...dragonfly.RunOption,
) (*dragonfly.Result, error) {
	return variant.run(ctx, config)
}

func (variant optimizerVariant) ApplicableTo(dragonfly.ProblemCharacteristics) float64 {
	return 1
}

func (variant optimizerVariant) EstimatedOverhead() float64 { return 1 }

func (variant optimizerVariant) RecommendedFor() []string { return nil }

func (result *study) write() error {
	err := writeAtomic(result.options.csvPath, result.writeCSV)
	if err != nil {
		return err
	}

	return writeAtomic(result.options.markdownPath, result.writeMarkdown)
}

func writeAtomic(path string, write func(io.Writer) error) error {
	directory := filepath.Dir(path)

	err := os.MkdirAll(directory, 0o755)
	if err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}

	temporary, err := os.CreateTemp(directory, ".head-to-head-*")
	if err != nil {
		return fmt.Errorf("create temporary output: %w", err)
	}

	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()

	err = write(temporary)
	if err != nil {
		_ = temporary.Close()

		return err
	}

	err = temporary.Close()
	if err != nil {
		return fmt.Errorf("close temporary output: %w", err)
	}

	err = os.Rename(temporaryPath, path)
	if err != nil {
		return fmt.Errorf("replace %s: %w", path, err)
	}

	return nil
}

func (result *study) writeCSV(sink io.Writer) error {
	writer := csv.NewWriter(sink)

	header := []string{
		"go_version", "dragonfly_revision", "mayfly_revision", "cma_es_revision",
		"benchmark", "dimensions", "budget", "base_seed", "optimizer", "rank", "run",
		"seed", "best_cost", "function_evaluations", "iterations", "execution_seconds",
	}

	err := writer.Write(header)
	if err != nil {
		return fmt.Errorf("write CSV header: %w", err)
	}

	for _, comparison := range result.results {
		for algorithm, name := range comparison.AlgorithmNames {
			for runIndex, run := range comparison.RunResults[algorithm] {
				row := []string{
					runtime.Version(), result.options.dragonflyRevision,
					result.options.mayflyRevision, result.options.cmaRevision,
					comparison.BenchmarkName, strconv.Itoa(result.options.dimensions),
					strconv.Itoa(result.options.budget), strconv.FormatInt(result.options.seed, 10),
					name, strconv.Itoa(comparison.Rankings[algorithm]), strconv.Itoa(runIndex + 1),
					strconv.FormatInt(run.Seed, 10), strconv.FormatFloat(run.BestCost, 'g', -1, 64),
					strconv.Itoa(run.FuncEvals), strconv.Itoa(run.Iterations),
					strconv.FormatFloat(run.ExecutionTime, 'g', -1, 64),
				}

				err = writer.Write(row)
				if err != nil {
					return fmt.Errorf("write CSV row: %w", err)
				}
			}
		}
	}

	writer.Flush()

	err = writer.Error()
	if err != nil {
		return fmt.Errorf("flush CSV: %w", err)
	}

	return nil
}

func (result *study) writeMarkdown(sink io.Writer) error {
	writef := func(format string, args ...any) error {
		_, err := fmt.Fprintf(sink, format, args...)

		return err
	}

	err := writef("# Dragonfly vs Mayfly head-to-head\n\n")
	if err != nil {
		return err
	}

	sourceRevisions := fmt.Sprintf("Dragonfly `%s` and Mayfly `%s`",
		result.options.dragonflyRevision, result.options.mayflyRevision)
	if result.options.includeCMA {
		sourceRevisions = fmt.Sprintf("Dragonfly `%s`, Mayfly `%s`, and CMA-ES `%s`",
			result.options.dragonflyRevision, result.options.mayflyRevision,
			result.options.cmaRevision)
	}

	err = writef(
		"Generated with Go %s from %s. "+
			"Each cell summarizes %d runs in %d dimensions with exactly %d real objective "+
			"evaluations per run; the base seed is `%d`.\n\n",
		runtime.Version(), sourceRevisions, result.options.runs, result.options.dimensions,
		result.options.budget, result.options.seed,
	)
	if err != nil {
		return err
	}

	method := "All algorithms search the benchmark's same conventional physical box. " +
		"DA uses one 40-member swarm, and MA uses its default 20 male plus 20 female populations"
	if result.options.includeCMA {
		method += ", while CMA-ES uses its dimension-derived population with a seeded uniform " +
			"initial mean and `sigma=0.3*(upper-lower)`"
	}

	method += ". Early stopping and inner parallelism are disabled. Lower cost is better."
	if result.options.includeCMA {
		method += " CMA-ES is a calibration baseline; the planned comparison is DA versus MA."
	}

	err = writef("%s\n\n", method)
	if err != nil {
		return err
	}

	err = writef("| Benchmark | DA median [best, worst] | MA median [best, worst] |")
	if err != nil {
		return err
	}

	if result.options.includeCMA {
		err = writef(" CMA-ES median [best, worst] |")
		if err != nil {
			return err
		}
	}

	err = writef(" Median winner | DA vs MA (Holm p) |\n")
	if err != nil {
		return err
	}

	err = writef("| --- | ---: | ---: |")
	if err != nil {
		return err
	}

	if result.options.includeCMA {
		err = writef(" ---: |")
		if err != nil {
			return err
		}
	}

	err = writef(" --- | --- |\n")
	if err != nil {
		return err
	}

	for _, comparison := range result.results {
		err = writef("| %s", comparison.BenchmarkName)
		if err != nil {
			return err
		}

		for _, stats := range comparison.Statistics {
			err = writef(" | %s", formatStatistics(stats))
			if err != nil {
				return err
			}
		}

		winner := medianWinner(comparison)
		pair := comparison.WilcoxonTests[0][1]

		pairWinner := pair.Winner
		if pairWinner == "Tie" {
			pairWinner = "tie"
		}

		err = writef(" | %s | %s, `%s` |\n", winner, pairWinner,
			formatProbability(pair.AdjustedPValue))
		if err != nil {
			return err
		}
	}

	wins := countMedianWins(result.results)

	err = writef("\nMedian wins: ")
	if err != nil {
		return err
	}

	names := make([]string, 0, len(wins))
	for name := range wins {
		names = append(names, name)
	}

	sort.Strings(names)

	for i, name := range names {
		if i > 0 {
			err = writef(", ")
			if err != nil {
				return err
			}
		}

		err = writef("%s %d", name, wins[name])
		if err != nil {
			return err
		}
	}

	err = writef(
		". The Wilcoxon signed-rank tests use matched run seeds and the comparison "+
			"framework's Holm correction within each benchmark at `alpha=0.05`. Timing is "+
			"retained in the CSV for diagnostics, but is not ranked because the algorithms "+
			"perform different internal work per objective call.\n\n"+
			"Raw observations: [`%s`](%s). Regenerate from the repository root with "+
			"`just head-to-head`.\n",
		filepath.Base(result.options.csvPath), filepath.Base(result.options.csvPath),
	)
	if err != nil {
		return err
	}

	return nil
}

func formatStatistics(stats dragonfly.AlgorithmStatistics) string {
	return fmt.Sprintf("%s [%s, %s]", formatCost(stats.Median), formatCost(stats.Best), formatCost(stats.Worst))
}

func formatCost(value float64) string {
	if value == 0 {
		return "0"
	}

	return strconv.FormatFloat(value, 'g', 4, 64)
}

func formatProbability(value float64) string {
	if value < 0.0001 {
		return "<0.0001"
	}

	return strconv.FormatFloat(value, 'f', 4, 64)
}

func medianWinner(result *dragonfly.ComparisonResult) string {
	best := 0
	for i := 1; i < len(result.Statistics); i++ {
		if result.Statistics[i].Median < result.Statistics[best].Median {
			best = i
		}
	}

	return result.AlgorithmNames[best]
}

func countMedianWins(results []*dragonfly.ComparisonResult) map[string]int {
	wins := make(map[string]int)
	for _, result := range results {
		wins[medianWinner(result)]++
	}

	return wins
}
