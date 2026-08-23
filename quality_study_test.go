package dragonfly

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const (
	qualityStudyEnabledEnv = "DRAGONFLY_RUN_QUALITY_STUDY"
	qualityStudyOutputEnv  = "DRAGONFLY_QUALITY_STUDY_OUTPUT"
	bdaStudyEnabledEnv     = "DRAGONFLY_RUN_BDA_STUDY"
	bdaStudyOutputEnv      = "DRAGONFLY_BDA_STUDY_OUTPUT"
	qualityRecoveryLimit   = 0.05
	qualityHVRecoveryLimit = 0.95
)

type zdtInterval struct {
	lo float64
	hi float64
}

var zdt3TrueFrontIntervals = []zdtInterval{
	{lo: 0, hi: 0.0830015349},
	{lo: 0.1822287280, hi: 0.2577623634},
	{lo: 0.4093136748, hi: 0.4538821041},
	{lo: 0.6183967944, hi: 0.6525117038},
	{lo: 0.8233317983, hi: 0.8518328654},
}

type qualityProblem struct {
	objective MultiObjectiveFunction
	front     func(float64) float64
	name      string
	intervals []zdtInterval
}

func qualityProblems() []qualityProblem {
	return []qualityProblem{
		{
			name:      "ZDT1",
			objective: ZDT1,
			front:     func(f1 float64) float64 { return 1 - math.Sqrt(f1) },
			intervals: []zdtInterval{{lo: 0, hi: 1}},
		},
		{
			name:      "ZDT2",
			objective: ZDT2,
			front:     func(f1 float64) float64 { return 1 - f1*f1 },
			intervals: []zdtInterval{{lo: 0, hi: 1}},
		},
		{
			name:      "ZDT3",
			objective: ZDT3,
			front: func(f1 float64) float64 {
				return 1 - math.Sqrt(f1) - f1*math.Sin(10*math.Pi*f1)
			},
			intervals: zdt3TrueFrontIntervals,
		},
	}
}

func qualityProblemNamed(name string) (qualityProblem, bool) {
	for _, problem := range qualityProblems() {
		if problem.name == name {
			return problem, true
		}
	}

	return qualityProblem{}, false
}

type objectivePoint struct {
	x float64
	y float64
}

type frontQuality struct {
	gd               float64
	igd              float64
	hypervolumeRatio float64
	bestG            float64
	medianG          float64
	minF1            float64
	maxF1            float64
	segmentsCovered  int
}

func evaluateFrontQuality(archive *ParetoArchive, problem qualityProblem) (frontQuality, error) {
	if archive == nil || archive.Len() == 0 {
		return frontQuality{}, errors.New("quality metrics require a non-empty archive")
	}

	reference := sampleQualityFront(problem)
	ideal, span := objectiveNormalization(reference)
	normalizedReference := normalizeObjectivePoints(reference, ideal, span)
	archivePoints := make([]objectivePoint, archive.Len())
	archiveDistances := make([]float64, archive.Len())
	gValues := make([]float64, archive.Len())

	for i, solution := range archive.Solutions {
		if solution == nil || len(solution.ObjectiveValues) != 2 || len(solution.Position) < 2 {
			return frontQuality{}, fmt.Errorf("archive solution %d is malformed", i)
		}

		point := objectivePoint{x: solution.ObjectiveValues[0], y: solution.ObjectiveValues[1]}
		if !isFinite(point.x) || !isFinite(point.y) {
			return frontQuality{}, fmt.Errorf("archive solution %d has non-finite objectives", i)
		}

		archivePoints[i] = normalizeObjectivePoint(point, ideal, span)
		archiveDistances[i] = math.Sqrt(nearestSquaredDistance(archivePoints[i], normalizedReference))
		_, gValues[i] = zdtFirstObjectiveAndG(solution.Position)
	}

	gd := rootMeanSquare(archiveDistances)
	igd := invertedGenerationalDistance(normalizedReference, archivePoints)
	trueHV := hypervolume2D(normalizedReference, objectivePoint{x: 1.1, y: 1.1})
	archiveHV := hypervolume2D(archivePoints, objectivePoint{x: 1.1, y: 1.1})
	minF1, maxF1 := firstObjectiveExtent(archive)

	return frontQuality{
		gd:               gd,
		igd:              igd,
		hypervolumeRatio: archiveHV / trueHV,
		bestG:            minFloat(gValues),
		medianG:          medianFloat(gValues),
		minF1:            minF1,
		maxF1:            maxF1,
		segmentsCovered:  countCoveredSegments(archive, archiveDistances, problem.intervals),
	}, nil
}

func sampleQualityFront(problem qualityProblem) []objectivePoint {
	const totalSamples = 10001

	counts := proportionalSampleCounts(problem.intervals, totalSamples)
	points := make([]objectivePoint, 0, totalSamples)

	for intervalIndex, interval := range problem.intervals {
		steps := counts[intervalIndex] - 1
		for i := range counts[intervalIndex] {
			f1 := interval.lo + (interval.hi-interval.lo)*float64(i)/float64(steps)
			points = append(points, objectivePoint{x: f1, y: problem.front(f1)})
		}
	}

	return points
}

func proportionalSampleCounts(intervals []zdtInterval, total int) []int {
	counts := make([]int, len(intervals))
	fractions := make([]float64, len(intervals))

	totalWidth := 0.0
	for _, interval := range intervals {
		totalWidth += interval.hi - interval.lo
	}

	assigned := 0

	for i, interval := range intervals {
		exact := float64(total) * (interval.hi - interval.lo) / totalWidth
		counts[i] = int(math.Floor(exact))
		fractions[i] = exact - float64(counts[i])
		assigned += counts[i]
	}

	for assigned < total {
		best := 0
		for i := 1; i < len(fractions); i++ {
			if fractions[i] > fractions[best] {
				best = i
			}
		}

		counts[best]++
		fractions[best] = -1
		assigned++
	}

	return counts
}

func objectiveNormalization(points []objectivePoint) (objectivePoint, objectivePoint) {
	ideal := objectivePoint{x: math.Inf(1), y: math.Inf(1)}
	nadir := objectivePoint{x: math.Inf(-1), y: math.Inf(-1)}

	for _, point := range points {
		ideal.x = math.Min(ideal.x, point.x)
		ideal.y = math.Min(ideal.y, point.y)
		nadir.x = math.Max(nadir.x, point.x)
		nadir.y = math.Max(nadir.y, point.y)
	}

	return ideal, objectivePoint{x: nadir.x - ideal.x, y: nadir.y - ideal.y}
}

func normalizeObjectivePoints(points []objectivePoint, ideal, span objectivePoint) []objectivePoint {
	normalized := make([]objectivePoint, len(points))
	for i, point := range points {
		normalized[i] = normalizeObjectivePoint(point, ideal, span)
	}

	return normalized
}

func normalizeObjectivePoint(point, ideal, span objectivePoint) objectivePoint {
	return objectivePoint{x: (point.x - ideal.x) / span.x, y: (point.y - ideal.y) / span.y}
}

func nearestSquaredDistance(point objectivePoint, candidates []objectivePoint) float64 {
	best := math.Inf(1)

	for _, candidate := range candidates {
		dx := point.x - candidate.x
		dy := point.y - candidate.y
		best = math.Min(best, dx*dx+dy*dy)
	}

	return best
}

func rootMeanSquare(values []float64) float64 {
	sumSquares := 0.0
	for _, value := range values {
		sumSquares += value * value
	}

	return math.Sqrt(sumSquares / float64(len(values)))
}

func invertedGenerationalDistance(reference, archive []objectivePoint) float64 {
	distances := make([]float64, len(reference))
	for i, point := range reference {
		distances[i] = math.Sqrt(nearestSquaredDistance(point, archive))
	}

	return rootMeanSquare(distances)
}

func hypervolume2D(points []objectivePoint, reference objectivePoint) float64 {
	sorted := append([]objectivePoint(nil), points...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].x == sorted[j].x {
			return sorted[i].y < sorted[j].y
		}

		return sorted[i].x < sorted[j].x
	})

	hypervolume := 0.0
	bestY := reference.y

	for _, point := range sorted {
		if point.x >= reference.x || point.y >= bestY {
			continue
		}

		hypervolume += (reference.x - point.x) * (bestY - point.y)
		bestY = point.y
	}

	return hypervolume
}

func firstObjectiveExtent(archive *ParetoArchive) (float64, float64) {
	minF1 := math.Inf(1)
	maxF1 := math.Inf(-1)

	for _, solution := range archive.Solutions {
		minF1 = math.Min(minF1, solution.ObjectiveValues[0])
		maxF1 = math.Max(maxF1, solution.ObjectiveValues[0])
	}

	return minF1, maxF1
}

func minFloat(values []float64) float64 {
	minimum := math.Inf(1)
	for _, value := range values {
		minimum = math.Min(minimum, value)
	}

	return minimum
}

func medianFloat(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	return sorted[len(sorted)/2]
}

func countCoveredSegments(archive *ParetoArchive, distances []float64, intervals []zdtInterval) int {
	covered := make([]bool, len(intervals))

	for i, solution := range archive.Solutions {
		if distances[i] > qualityRecoveryLimit {
			continue
		}

		f1 := solution.ObjectiveValues[0]
		for segment, interval := range intervals {
			if f1 >= interval.lo && f1 <= interval.hi {
				covered[segment] = true
			}
		}
	}

	count := 0

	for _, present := range covered {
		if present {
			count++
		}
	}

	return count
}

type qualityProfile struct {
	name     string
	fidelity FidelityMode
	policy   ArchivePolicy
}

var qualityStudyProfiles = []qualityProfile{
	{name: "paper", fidelity: FidelityPaper, policy: ArchivePolicyPaperSegments},
	{name: "matlab", fidelity: FidelityMATLAB, policy: ArchivePolicyMATLABDensity},
	{name: "legacy", fidelity: FidelityMATLAB, policy: ArchivePolicyMOPSOGrid},
}

type qualityStudyRecord struct {
	configJSON       string
	sourceHash       string
	goVersion        string
	profile          string
	problem          string
	fidelity         string
	archivePolicy    string
	termination      string
	dimensions       int
	iterations       int
	population       int
	archiveCapacity  int
	seed             int64
	archiveSize      int
	evaluations      int
	iterationsDone   int
	zdt3Segments     int
	gd               float64
	igd              float64
	hypervolumeRatio float64
	bestG            float64
	medianG          float64
	minF1            float64
	maxF1            float64
	runRecovered     bool
	profileRecovered bool
}

// TestMODAQualityStudy runs the release evidence study only when explicitly
// requested. Ordinary test, race and coverage gates compile it but skip the
// 45 deliberately expensive optimizations.
func TestMODAQualityStudy(t *testing.T) {
	if os.Getenv(qualityStudyEnabledEnv) != "1" {
		t.Skip("set " + qualityStudyEnabledEnv + "=1 to run the release quality study")
	}

	output := os.Getenv(qualityStudyOutputEnv)
	if output == "" {
		t.Fatal(qualityStudyOutputEnv + " must name the study CSV output")
	}

	records := collectQualityStudyRecords(t)
	classifyQualityStudyProfiles(records)
	writeQualityStudyOutputs(t, output, records)
}

func collectQualityStudyRecords(t *testing.T) []qualityStudyRecord {
	t.Helper()

	const (
		dimensions      = 30
		iterations      = 1000
		population      = 100
		archiveCapacity = 100
	)

	sourceHash := qualityStudySourceHash(t)
	records := make([]qualityStudyRecord, 0, len(qualityStudyProfiles)*len(qualityProblems())*5)

	for _, profile := range qualityStudyProfiles {
		for _, problem := range qualityProblems() {
			for seed := int64(3000); seed <= 3004; seed++ {
				config := qualityStudyConfig(problem, profile, seed, dimensions, iterations, population, archiveCapacity)

				result, err := OptimizeMultiObjective(context.Background(), config)
				if err != nil {
					t.Fatalf("%s/%s seed %d: %v", profile.name, problem.name, seed, err)
				}

				records = append(records, makeQualityStudyRecord(t, sourceHash, config, result, problem, profile))
			}
		}
	}

	return records
}

func qualityStudyConfig(
	problem qualityProblem,
	profile qualityProfile,
	seed int64,
	dimensions, iterations, population, archiveCapacity int,
) *MultiObjectiveConfig {
	config := NewMultiObjectiveConfig()
	config.ObjectiveFunc = problem.objective
	config.ArchivePolicy = profile.policy
	config.ArchiveSize = archiveCapacity
	config.Swarm.FidelityMode = profile.fidelity
	config.Swarm.ProblemSize = dimensions
	config.Swarm.LowerBound = 0
	config.Swarm.UpperBound = 1
	config.Swarm.MaxIterations = iterations
	config.Swarm.NPop = population
	config.Swarm.Seed = &seed

	return config
}

func makeQualityStudyRecord(
	t *testing.T,
	sourceHash string,
	config *MultiObjectiveConfig,
	result *MultiObjectiveResult,
	problem qualityProblem,
	profile qualityProfile,
) qualityStudyRecord {
	t.Helper()

	if !result.SeedKnown || result.Seed != *config.Swarm.Seed {
		t.Fatalf("%s/%s did not preserve seed provenance", profile.name, problem.name)
	}

	if !result.Archive.IsNonDominated() || result.Archive.Len() > config.ArchiveSize {
		t.Fatalf("%s/%s seed %d returned an invalid archive", profile.name, problem.name, result.Seed)
	}

	integrityErr := validateQualityStudyResult(config, result)
	if integrityErr != nil {
		t.Fatalf("%s/%s seed %d integrity: %v", profile.name, problem.name, result.Seed, integrityErr)
	}

	metrics, err := evaluateFrontQuality(result.Archive, problem)
	if err != nil {
		t.Fatalf("%s/%s seed %d metrics: %v", profile.name, problem.name, result.Seed, err)
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("marshal %s/%s config: %v", profile.name, problem.name, err)
	}

	segmentsRequired := 0
	if problem.name == "ZDT3" {
		segmentsRequired = len(zdt3TrueFrontIntervals)
	}

	return qualityStudyRecord{
		configJSON:       string(configJSON),
		sourceHash:       sourceHash,
		goVersion:        runtime.Version(),
		profile:          profile.name,
		problem:          problem.name,
		fidelity:         string(profile.fidelity),
		archivePolicy:    string(profile.policy),
		termination:      string(result.TerminationReason),
		dimensions:       config.Swarm.ProblemSize,
		iterations:       config.Swarm.MaxIterations,
		population:       config.Swarm.NPop,
		archiveCapacity:  config.ArchiveSize,
		seed:             result.Seed,
		archiveSize:      result.Archive.Len(),
		evaluations:      result.FuncEvalCount,
		iterationsDone:   result.IterationCount,
		zdt3Segments:     metrics.segmentsCovered,
		gd:               metrics.gd,
		igd:              metrics.igd,
		hypervolumeRatio: metrics.hypervolumeRatio,
		bestG:            metrics.bestG,
		medianG:          metrics.medianG,
		minF1:            metrics.minF1,
		maxF1:            metrics.maxF1,
		runRecovered: metrics.gd <= qualityRecoveryLimit &&
			metrics.igd <= qualityRecoveryLimit &&
			metrics.hypervolumeRatio >= qualityHVRecoveryLimit &&
			metrics.segmentsCovered >= segmentsRequired,
	}
}

func validateQualityStudyResult(config *MultiObjectiveConfig, result *MultiObjectiveResult) error {
	if result.TerminationReason != TerminationMaxIterations ||
		result.IterationCount != config.Swarm.MaxIterations {
		return fmt.Errorf("termination=%q iterations=%d, want max_iterations/%d",
			result.TerminationReason, result.IterationCount, config.Swarm.MaxIterations)
	}

	expectedEvaluations := config.Swarm.NPop * config.Swarm.MaxIterations
	if config.Swarm.FidelityMode == FidelityPaper {
		expectedEvaluations += config.Swarm.NPop
	}

	if result.FuncEvalCount != expectedEvaluations {
		return fmt.Errorf("evaluations=%d, want %d", result.FuncEvalCount, expectedEvaluations)
	}

	if result.Archive == nil || result.Archive.Len() == 0 || result.Archive.Len() > config.ArchiveSize {
		return fmt.Errorf("archive size=%d, want 1..%d", result.Archive.Len(), config.ArchiveSize)
	}

	for i, solution := range result.Archive.Solutions {
		if solution == nil || !isFinite(solution.ConstraintViolation) {
			return fmt.Errorf("archive solution %d has invalid constraint state", i)
		}

		for _, values := range [][]float64{solution.Position, solution.ObjectiveValues} {
			for _, value := range values {
				if !isFinite(value) {
					return fmt.Errorf("archive solution %d contains a non-finite value", i)
				}
			}
		}
	}

	return nil
}

func classifyQualityStudyProfiles(records []qualityStudyRecord) {
	passes := make(map[string]int)

	for _, record := range records {
		if record.runRecovered {
			passes[record.profile+"/"+record.problem]++
		}
	}

	for i := range records {
		records[i].profileRecovered = passes[records[i].profile+"/"+records[i].problem] >= 4
	}
}

func qualityStudySourceHash(t *testing.T) string {
	t.Helper()

	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("list production Go sources: %v", err)
	}

	sort.Strings(paths)

	digest := sha256.New()

	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}

		contents, readErr := os.ReadFile(path)
		if readErr != nil {
			t.Fatalf("hash production source %s: %v", path, readErr)
		}

		_, _ = digest.Write([]byte(path))
		_, _ = digest.Write([]byte{0})
		_, _ = digest.Write(contents)
		_, _ = digest.Write([]byte{0})
	}

	return hex.EncodeToString(digest.Sum(nil))
}

func writeQualityStudyOutputs(t *testing.T, csvPath string, records []qualityStudyRecord) {
	t.Helper()

	csvData := encodeQualityStudyCSV(t, records)
	writeAtomicTestFile(t, csvPath, csvData)

	reportPath := strings.TrimSuffix(csvPath, filepath.Ext(csvPath)) + ".md"
	writeAtomicTestFile(t, reportPath, encodeQualityStudyReport(records))
}

func encodeQualityStudyCSV(t *testing.T, records []qualityStudyRecord) []byte {
	t.Helper()

	var output bytes.Buffer

	writer := csv.NewWriter(&output)

	header := []string{
		"source_hash", "go_version", "profile", "problem", "dimensions", "iterations", "population",
		"archive_capacity", "seed", "fidelity_mode", "archive_policy", "config_json", "archive_size",
		"gd", "igd", "hypervolume_ratio", "best_g", "median_g", "min_f1", "max_f1",
		"zdt3_segments", "evaluations", "iterations_completed", "termination", "run_recovered", "profile_recovered",
	}

	writeErr := writer.Write(header)
	if writeErr != nil {
		t.Fatalf("write study CSV header: %v", writeErr)
	}

	for _, record := range records {
		writeErr = writer.Write(record.csvFields())
		if writeErr != nil {
			t.Fatalf("write study CSV row: %v", writeErr)
		}
	}

	writer.Flush()

	flushErr := writer.Error()
	if flushErr != nil {
		t.Fatalf("flush study CSV: %v", flushErr)
	}

	return output.Bytes()
}

func (record qualityStudyRecord) csvFields() []string {
	return []string{
		record.sourceHash,
		record.goVersion,
		record.profile,
		record.problem,
		strconv.Itoa(record.dimensions),
		strconv.Itoa(record.iterations),
		strconv.Itoa(record.population),
		strconv.Itoa(record.archiveCapacity),
		strconv.FormatInt(record.seed, 10),
		record.fidelity,
		record.archivePolicy,
		record.configJSON,
		strconv.Itoa(record.archiveSize),
		formatQualityFloat(record.gd),
		formatQualityFloat(record.igd),
		formatQualityFloat(record.hypervolumeRatio),
		formatQualityFloat(record.bestG),
		formatQualityFloat(record.medianG),
		formatQualityFloat(record.minF1),
		formatQualityFloat(record.maxF1),
		strconv.Itoa(record.zdt3Segments),
		strconv.Itoa(record.evaluations),
		strconv.Itoa(record.iterationsDone),
		record.termination,
		strconv.FormatBool(record.runRecovered),
		strconv.FormatBool(record.profileRecovered),
	}
}

func formatQualityFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', 17, 64)
}

func encodeQualityStudyReport(records []qualityStudyRecord) []byte {
	var report strings.Builder
	report.WriteString("# MODA v0.2.0 quality study\n\n")
	report.WriteString("Five deterministic seeds (3000..3004) were run at 30 dimensions with a population of 100, ")
	report.WriteString("1000 iterations, and an archive capacity of 100. GD, IGD, and two-objective hypervolume ")
	report.WriteString("are normalized to each analytic front. ZDT3 uses only its five true nondominated intervals.\n\n")
	report.WriteString("A profile/problem pair is labeled recovered only when at least four seeds have GD and IGD ")
	report.WriteString("at most 0.05 and hypervolume ratio at least 0.95; ZDT3 must also cover all five segments.\n\n")
	report.WriteString("| Profile | Problem | Passing seeds | Recovered | Median GD | Median IGD | Median HV | Median best g | Median archive |\n")
	report.WriteString("| --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |\n")

	for _, profile := range qualityStudyProfiles {
		for _, problem := range qualityProblems() {
			passes, recovered := qualityStudySummary(records, profile.name, problem.name)
			gd, igd, hypervolume, bestG, archiveSize := qualityStudyMedians(records, profile.name, problem.name)
			fmt.Fprintf(&report, "| %s | %s | %d/5 | %t | %.4g | %.4g | %.4g | %.4g | %.0f |\n",
				profile.name, problem.name, passes, recovered, gd, igd, hypervolume, bestG, archiveSize)
		}
	}

	return []byte(report.String())
}

func qualityStudyMedians(records []qualityStudyRecord, profile, problem string) (float64, float64, float64, float64, float64) {
	gd := make([]float64, 0, 5)
	igd := make([]float64, 0, 5)
	hypervolume := make([]float64, 0, 5)
	bestG := make([]float64, 0, 5)
	archiveSize := make([]float64, 0, 5)

	for _, record := range records {
		if record.profile != profile || record.problem != problem {
			continue
		}

		gd = append(gd, record.gd)
		igd = append(igd, record.igd)
		hypervolume = append(hypervolume, record.hypervolumeRatio)
		bestG = append(bestG, record.bestG)
		archiveSize = append(archiveSize, float64(record.archiveSize))
	}

	if len(gd) == 0 {
		return math.NaN(), math.NaN(), math.NaN(), math.NaN(), math.NaN()
	}

	return medianFloat(gd), medianFloat(igd), medianFloat(hypervolume), medianFloat(bestG), medianFloat(archiveSize)
}

func qualityStudySummary(records []qualityStudyRecord, profile, problem string) (int, bool) {
	passes := 0
	recovered := false

	for _, record := range records {
		if record.profile != profile || record.problem != problem {
			continue
		}

		if record.runRecovered {
			passes++
		}

		recovered = record.profileRecovered
	}

	return passes, recovered
}

type bdaStudyRecord struct {
	sourceHash     string
	goVersion      string
	fidelity       string
	transfer       string
	termination    string
	seed           int64
	dimensions     int
	iterations     int
	population     int
	evaluations    int
	iterationsDone int
	cost           float64
	optimumHit     bool
}

// TestBDAQualityStudy measures all eight transfer functions under the paper
// and MATLAB lifecycles. It is opt-in because the 240 deterministic runs are
// release evidence rather than an ordinary correctness test.
func TestBDAQualityStudy(t *testing.T) {
	if os.Getenv(bdaStudyEnabledEnv) != "1" {
		t.Skip("set " + bdaStudyEnabledEnv + "=1 to run the BDA transfer-family study")
	}

	output := os.Getenv(bdaStudyOutputEnv)
	if output == "" {
		t.Fatal(bdaStudyOutputEnv + " must name the study CSV output")
	}

	records := collectBDAStudyRecords(t)
	writeBDAStudyOutputs(t, output, records)
}

func collectBDAStudyRecords(t *testing.T) []bdaStudyRecord {
	t.Helper()

	const (
		dimensions = 30
		iterations = 300
		population = 30
	)

	modes := []FidelityMode{FidelityPaper, FidelityMATLAB}
	sourceHash := qualityStudySourceHash(t)
	records := make([]bdaStudyRecord, 0, len(modes)*len(TransferFunctionNames())*15)

	for _, mode := range modes {
		for _, transfer := range TransferFunctionNames() {
			for seed := int64(2000); seed <= 2014; seed++ {
				config := bdaStudyConfig(mode, transfer, seed, dimensions, iterations, population)

				result, err := OptimizeBinary(config)
				if err != nil {
					t.Fatalf("%s/%s seed %d: %v", mode, transfer, seed, err)
				}

				record, recordErr := makeBDAStudyRecord(config, result)
				if recordErr != nil {
					t.Fatalf("%s/%s seed %d integrity: %v", mode, transfer, seed, recordErr)
				}

				record.sourceHash = sourceHash
				record.goVersion = runtime.Version()
				records = append(records, record)
			}
		}
	}

	return records
}

func bdaStudyConfig(
	mode FidelityMode,
	transfer TransferFunction,
	seed int64,
	dimensions, iterations, population int,
) *Config {
	config := NewBinaryConfig()
	config.ObjectiveFunc = oneMaxBits
	config.FidelityMode = mode
	config.TransferFunc = transfer
	config.ProblemSize = dimensions
	config.MaxIterations = iterations
	config.NPop = population
	config.Seed = &seed

	return config
}

func makeBDAStudyRecord(config *Config, result *Result) (bdaStudyRecord, error) {
	if result == nil || !result.SeedKnown || config.Seed == nil || result.Seed != *config.Seed {
		return bdaStudyRecord{}, errors.New("result did not preserve seed provenance")
	}

	if result.TerminationReason != TerminationMaxIterations || result.IterationCount != config.MaxIterations {
		return bdaStudyRecord{}, fmt.Errorf("termination=%q iterations=%d, want max_iterations/%d",
			result.TerminationReason, result.IterationCount, config.MaxIterations)
	}

	expectedEvaluations := config.NPop * config.MaxIterations
	if config.FidelityMode == FidelityPaper {
		expectedEvaluations += config.NPop
	}

	if result.FuncEvalCount != expectedEvaluations {
		return bdaStudyRecord{}, fmt.Errorf("evaluations=%d, want %d", result.FuncEvalCount, expectedEvaluations)
	}

	if !isFinite(result.GlobalBest.Cost) || result.GlobalBest.Cost < 0 ||
		result.GlobalBest.Cost > float64(config.ProblemSize) {
		return bdaStudyRecord{}, fmt.Errorf("final cost %v is outside [0,%d]",
			result.GlobalBest.Cost, config.ProblemSize)
	}

	return bdaStudyRecord{
		fidelity:       string(config.FidelityMode),
		transfer:       string(config.TransferFunc),
		termination:    string(result.TerminationReason),
		seed:           result.Seed,
		dimensions:     config.ProblemSize,
		iterations:     config.MaxIterations,
		population:     config.NPop,
		evaluations:    result.FuncEvalCount,
		iterationsDone: result.IterationCount,
		cost:           result.GlobalBest.Cost,
		optimumHit:     result.GlobalBest.Cost == 0,
	}, nil
}

func writeBDAStudyOutputs(t *testing.T, csvPath string, records []bdaStudyRecord) {
	t.Helper()

	writeAtomicTestFile(t, csvPath, encodeBDAStudyCSV(t, records))
	reportPath := strings.TrimSuffix(csvPath, filepath.Ext(csvPath)) + ".md"
	writeAtomicTestFile(t, reportPath, encodeBDAStudyReport(records))
}

func encodeBDAStudyCSV(t *testing.T, records []bdaStudyRecord) []byte {
	t.Helper()

	var output bytes.Buffer

	writer := csv.NewWriter(&output)
	header := []string{
		"source_hash", "go_version", "fidelity_mode", "transfer_function", "seed", "dimensions",
		"iterations", "population", "cost", "evaluations", "iterations_completed", "termination", "optimum_hit",
	}

	writeErr := writer.Write(header)
	if writeErr != nil {
		t.Fatalf("write BDA study CSV header: %v", writeErr)
	}

	for _, record := range records {
		writeErr = writer.Write(record.csvFields())
		if writeErr != nil {
			t.Fatalf("write BDA study CSV row: %v", writeErr)
		}
	}

	writer.Flush()

	flushErr := writer.Error()
	if flushErr != nil {
		t.Fatalf("flush BDA study CSV: %v", flushErr)
	}

	return output.Bytes()
}

func (record bdaStudyRecord) csvFields() []string {
	return []string{
		record.sourceHash,
		record.goVersion,
		record.fidelity,
		record.transfer,
		strconv.FormatInt(record.seed, 10),
		strconv.Itoa(record.dimensions),
		strconv.Itoa(record.iterations),
		strconv.Itoa(record.population),
		formatQualityFloat(record.cost),
		strconv.Itoa(record.evaluations),
		strconv.Itoa(record.iterationsDone),
		record.termination,
		strconv.FormatBool(record.optimumHit),
	}
}

func encodeBDAStudyReport(records []bdaStudyRecord) []byte {
	var report strings.Builder
	report.WriteString("# BDA v0.2.0 transfer-family study\n\n")
	report.WriteString("Each transfer function was measured over deterministic seeds 2000..2014 on 30-bit OneMax ")
	report.WriteString("with a population of 30 and 300 iterations under both fidelity lifecycles. Lower cost is better.\n\n")
	report.WriteString("| Fidelity | Transfer | Mean cost | Median cost | Optimum hits |\n")
	report.WriteString("| --- | --- | ---: | ---: | ---: |\n")

	for _, mode := range []FidelityMode{FidelityPaper, FidelityMATLAB} {
		for _, transfer := range TransferFunctionNames() {
			mean, median, hits := summarizeBDAStudy(records, string(mode), string(transfer))
			fmt.Fprintf(&report, "| %s | %s | %s | %s | %d/15 |\n",
				mode, transfer, formatQualityFloat(mean), formatQualityFloat(median), hits)
		}
	}

	return []byte(report.String())
}

func summarizeBDAStudy(records []bdaStudyRecord, fidelity, transfer string) (float64, float64, int) {
	costs := make([]float64, 0, 15)
	hits := 0

	for _, record := range records {
		if record.fidelity != fidelity || record.transfer != transfer {
			continue
		}

		costs = append(costs, record.cost)
		if record.optimumHit {
			hits++
		}
	}

	if len(costs) == 0 {
		return math.NaN(), math.NaN(), 0
	}

	sum := 0.0
	for _, cost := range costs {
		sum += cost
	}

	return sum / float64(len(costs)), medianFloat(costs), hits
}

func TestBDAStudyEncodingIsDeterministic(t *testing.T) {
	records := []bdaStudyRecord{
		{
			sourceHash: "abc", goVersion: "go-test", fidelity: "paper", transfer: "v1",
			termination: string(TerminationMaxIterations), seed: 2000, dimensions: 30,
			iterations: 300, population: 30, evaluations: 9030, iterationsDone: 300,
			cost: 1, optimumHit: false,
		},
		{
			sourceHash: "abc", goVersion: "go-test", fidelity: "matlab", transfer: "s4",
			termination: string(TerminationMaxIterations), seed: 2014, dimensions: 30,
			iterations: 300, population: 30, evaluations: 9000, iterationsDone: 300,
			cost: 0, optimumHit: true,
		},
	}

	first := encodeBDAStudyCSV(t, records)

	second := encodeBDAStudyCSV(t, records)
	if !bytes.Equal(first, second) {
		t.Fatal("BDA study CSV encoding is not deterministic")
	}

	parsed, err := csv.NewReader(bytes.NewReader(first)).ReadAll()
	if err != nil {
		t.Fatalf("parse encoded BDA study CSV: %v", err)
	}

	if len(parsed) != len(records)+1 || parsed[1][3] != "v1" || parsed[2][12] != "true" {
		t.Fatalf("unexpected BDA study CSV records: %v", parsed)
	}
}

func TestBDAStudyIntegrityForFidelityLifecycles(t *testing.T) {
	tests := []struct {
		mode      FidelityMode
		wantEvals int
	}{
		{mode: FidelityPaper, wantEvals: 12},
		{mode: FidelityMATLAB, wantEvals: 8},
	}

	for _, testCase := range tests {
		t.Run(string(testCase.mode), func(t *testing.T) {
			config := bdaStudyConfig(testCase.mode, TransferV3, 42, 4, 2, 4)

			result, err := OptimizeBinary(config)
			if err != nil {
				t.Fatalf("run failed: %v", err)
			}

			record, recordErr := makeBDAStudyRecord(config, result)
			if recordErr != nil {
				t.Fatalf("study integrity rejected a valid run: %v", recordErr)
			}

			if record.evaluations != testCase.wantEvals {
				t.Errorf("evaluations = %d, want %d", record.evaluations, testCase.wantEvals)
			}
		})
	}
}

func writeAtomicTestFile(t *testing.T, path string, data []byte) {
	t.Helper()

	directory := filepath.Dir(path)

	mkdirErr := os.MkdirAll(directory, 0o755)
	if mkdirErr != nil {
		t.Fatalf("create study output directory: %v", mkdirErr)
	}

	temporary, err := os.CreateTemp(directory, ".dragonfly-quality-*")
	if err != nil {
		t.Fatalf("create temporary study output: %v", err)
	}

	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)

	_, err = temporary.Write(data)
	if err != nil {
		temporary.Close()
		t.Fatalf("write temporary study output: %v", err)
	}

	err = temporary.Sync()
	if err != nil {
		temporary.Close()
		t.Fatalf("sync temporary study output: %v", err)
	}

	err = temporary.Close()
	if err != nil {
		t.Fatalf("close temporary study output: %v", err)
	}

	err = os.Rename(temporaryPath, path)
	if err != nil {
		t.Fatalf("publish study output: %v", err)
	}
}
