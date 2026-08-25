// The framework layer: the algorithm variants behind one interface, a
// registry that resolves them by name, and a fluent builder.

package dragonfly

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Canonical variant names, as returned by AlgorithmVariant.Name().
const (
	nameDA   = "DA"
	nameBDA  = "BDA"
	nameMODA = "MODA"
	nameMHDA = "MHDA"
	nameCDA  = "CDA"
	nameQGDA = "QGDA"

	recommendedContinuous = "Continuous single-objective problems"
)

// ErrMultiObjectiveVariant is returned by AlgorithmVariant.Run for a variant
// whose IsMultiObjective reports true. MODA has no single incumbent to report,
// so it cannot produce a *Result; run it through MODAVariant.RunMultiObjective
// or OptimizeMultiObjective instead.
var ErrMultiObjectiveVariant = errors.New(
	"multi-objective variant has no single-objective result; use RunMultiObjective")

// ErrMultiObjectiveBuilder is returned when the single-objective fluent
// builder is asked to configure MODA. MODA needs MultiObjectiveConfig and its
// own objective signature, so producing a Config here would be unusable.
var ErrMultiObjectiveBuilder = errors.New(
	"the fluent builder only supports single-objective variants; use MODAVariant.GetMultiObjectiveConfig")

// ErrBinaryConfigOnContinuousVariant is returned by DAVariant.Run when it is
// handed a configuration with UseBinary set.
//
// Nothing in dragonfly.go dispatches on Config.UseBinary -- OptimizeContext
// documents that it ignores the field -- so a binary configuration handed to
// the continuous entry point runs the continuous algorithm on a swarm confined
// to [0,1] and returns a real-valued "solution" that is not a bit string. That
// is a silently wrong answer rather than a failure, so the variant layer, which
// is the one place that knows which algorithm the caller asked for, refuses it.
var ErrBinaryConfigOnContinuousVariant = errors.New(
	"config has UseBinary set: run it through the BDA variant, not DA")

// AlgorithmVariant represents one variant of the Dragonfly Algorithm, so that
// the selector, the comparison runner and the builder can work with all of them
// through one type.
//
// # Why Run returns *Result even though MODA cannot produce one
//
// MODA takes a MultiObjectiveConfig and returns a MultiObjectiveResult: an
// archive approximating a Pareto front, with no single incumbent. Three shapes
// were available for that. Widening Run's return type to an interface or `any`
// would push a type assertion onto every caller of every variant -- including
// ComparisonRunner, which computes means, medians and rank statistics over a
// scalar cost and is single-objective by construction. Splitting the interface
// in two would stop GetAllVariants from returning one slice, which the stable
// canonical order exists to provide. So Run keeps the single-objective
// signature, IsMultiObjective advertises which contract a variant honors, and
// MODAVariant.Run returns ErrMultiObjectiveVariant rather than a result it
// cannot honestly fill in. The multi-objective path is a separate method on the
// concrete type, mirroring OptimizeMultiObjective being a separate entry point.
type AlgorithmVariant interface {
	// Name returns the short canonical name of the variant, such as "DA",
	// "BDA", "MODA", "MHDA", "CDA" or "QGDA".
	Name() string

	// FullName returns the full descriptive name of the variant.
	FullName() string

	// Description returns a one-line summary of what the variant changes.
	Description() string

	// GetConfig returns a freshly allocated default configuration for this
	// variant. You must still set ObjectiveFunc and ProblemSize, and for the
	// continuous variants LowerBound and UpperBound.
	GetConfig() *Config

	// IsMultiObjective reports whether this variant optimizes several
	// objectives at once, and therefore whether Run can honor its contract.
	IsMultiObjective() bool

	// Run executes the variant's single-objective entry point. A variant whose
	// IsMultiObjective reports true returns ErrMultiObjectiveVariant.
	Run(ctx context.Context, config *Config, options ...RunOption) (*Result, error)

	// ApplicableTo scores how well this variant suits the given problem, in
	// [0,1]. Higher is a better fit.
	ApplicableTo(characteristics ProblemCharacteristics) float64

	// EstimatedOverhead returns the approximate per-iteration cost relative to
	// standard DA, as a multiplier. 1.0 is the baseline.
	EstimatedOverhead() float64

	// RecommendedFor returns the problem classes this variant excels at.
	RecommendedFor() []string
}

// variantOrder is the canonical order of the original-paper variants followed
// by the improved variants in publication order.
//
// GetAllVariants and ListVariants both walk this slice rather than ranging
// over variantRegistry, because a map range in Go is deliberately randomized
// and would make every comparison table, report and recommendation listing
// come out in a different order on every run.
var variantOrder = []string{nameDA, nameBDA, nameMODA, nameMHDA, nameCDA, nameQGDA}

// variantRegistry resolves a lowercase name or alias to a variant.
var variantRegistry = map[string]func() AlgorithmVariant{
	"da":       func() AlgorithmVariant { return &DAVariant{} },
	"standard": func() AlgorithmVariant { return &DAVariant{} },
	"bda":      func() AlgorithmVariant { return &BDAVariant{} },
	"binary":   func() AlgorithmVariant { return &BDAVariant{} },
	"moda":     func() AlgorithmVariant { return &MODAVariant{} },
	"mhda":     func() AlgorithmVariant { return &MHDAVariant{} },
	"memory":   func() AlgorithmVariant { return &MHDAVariant{} },
	"cda":      func() AlgorithmVariant { return &CDAVariant{} },
	"chaotic":  func() AlgorithmVariant { return &CDAVariant{} },
	"qgda":     func() AlgorithmVariant { return &QGDAVariant{} },
	"quantum":  func() AlgorithmVariant { return &QGDAVariant{} },
}

// NewVariant creates an algorithm variant by name, case-insensitively and
// ignoring surrounding space.
//
// Recognized names:
//   - "da" or "standard" -- the continuous Dragonfly Algorithm
//   - "bda" or "binary"  -- the binary Dragonfly Algorithm
//   - "moda"             -- the multi-objective Dragonfly Algorithm
//   - "mhda" or "memory" -- the memory-based DA/PSO hybrid
//   - "cda" or "chaotic" -- the continuous Chaotic Dragonfly Algorithm
//   - "qgda" or "quantum" -- quantum/Gaussian mutational DA
//
// An unknown name is an error rather than a nil variant: a caller that gets a
// name from a flag or a configuration file should hear about the typo here,
// not through a nil dereference several calls later.
func NewVariant(name string) (AlgorithmVariant, error) {
	key := strings.ToLower(strings.TrimSpace(name))

	factory, ok := variantRegistry[key]
	if !ok {
		return nil, fmt.Errorf("unknown variant %q (known variants: %s)",
			name, strings.Join(VariantAliases(), ", "))
	}

	return factory(), nil
}

// ListVariants returns the canonical variant names in the canonical order.
func ListVariants() []string {
	names := make([]string, len(variantOrder))
	copy(names, variantOrder)

	return names
}

// VariantAliases returns every accepted name, including aliases, in a stable
// alphabetical order. It is what an error message or a --help text lists.
func VariantAliases() []string {
	aliases := make([]string, 0, len(variantRegistry))
	for alias := range variantRegistry {
		aliases = append(aliases, alias)
	}

	sort.Strings(aliases)

	return aliases
}

// GetAllVariants returns one fresh instance of every variant, in the canonical
// order given by ListVariants. The order is stable across calls and across
// processes; a test pins it.
func GetAllVariants() []AlgorithmVariant {
	variants := make([]AlgorithmVariant, 0, len(variantOrder))

	for _, name := range variantOrder {
		variant, err := NewVariant(name)
		if err != nil {
			// Unreachable: variantOrder holds registry keys. Panicking here
			// would be worse than skipping, because GetAllVariants has no
			// error return and is called from report-formatting code.
			continue
		}

		variants = append(variants, variant)
	}

	return variants
}

// SingleObjectiveVariants returns the variants whose Run can honor its
// contract, in the canonical order. It is what ComparisonRunner defaults to,
// because the comparison statistics are all defined over a scalar cost.
func SingleObjectiveVariants() []AlgorithmVariant {
	all := GetAllVariants()
	variants := make([]AlgorithmVariant, 0, len(all))

	for _, variant := range all {
		if !variant.IsMultiObjective() {
			variants = append(variants, variant)
		}
	}

	return variants
}

// =============================================================================
// DA -- the continuous variant
// =============================================================================

// DAVariant is the standard continuous Dragonfly Algorithm, the paper's
// single-objective variant and the baseline every other variant is measured
// against.
type DAVariant struct{}

// Name returns the canonical short name.
func (v *DAVariant) Name() string { return nameDA }

// FullName returns the descriptive name.
func (v *DAVariant) FullName() string { return "Dragonfly Algorithm" }

// Description returns a one-line summary.
func (v *DAVariant) Description() string {
	return "The paper's continuous algorithm: five swarming primitives, a two-branch step " +
		"update and a Lévy walk for isolated dragonflies."
}

// GetConfig returns NewDefaultConfig.
func (v *DAVariant) GetConfig() *Config { return NewDefaultConfig() }

// IsMultiObjective reports false.
func (v *DAVariant) IsMultiObjective() bool { return false }

// Run executes the continuous algorithm through OptimizeContext.
//
// It refuses a configuration with UseBinary set. See
// ErrBinaryConfigOnContinuousVariant for why that is a rejection rather than a
// silent continuous run.
func (v *DAVariant) Run(ctx context.Context, config *Config, options ...RunOption) (*Result, error) {
	if config != nil && config.UseBinary {
		return nil, ErrBinaryConfigOnContinuousVariant
	}

	return OptimizeContext(ctx, config, options...)
}

// ApplicableTo scores the continuous variant against a problem.
func (v *DAVariant) ApplicableTo(characteristics ProblemCharacteristics) float64 {
	if characteristics.MultiObjective {
		return 0.1
	}

	if characteristics.Discrete {
		return 0.1
	}

	score := 0.6

	if characteristics.Modality == Unimodal {
		score += 0.2
	}

	if characteristics.Landscape == Smooth {
		score += 0.1
	}

	if characteristics.Dimensionality <= 50 {
		score += 0.1
	}

	return min(score, 1.0)
}

// EstimatedOverhead returns the baseline, 1.0.
func (v *DAVariant) EstimatedOverhead() float64 { return 1.0 }

// RecommendedFor lists the problem classes DA suits.
func (v *DAVariant) RecommendedFor() []string {
	return []string{
		recommendedContinuous,
		"Unimodal and mildly multimodal landscapes",
		"Baseline comparison",
	}
}

// =============================================================================
// BDA -- the binary variant
// =============================================================================

// BDAVariant is the binary Dragonfly Algorithm: the same step vector as DA,
// turned into a per-bit flip probability by a transfer function.
type BDAVariant struct{}

// Name returns the canonical short name.
func (v *BDAVariant) Name() string { return nameBDA }

// FullName returns the descriptive name.
func (v *BDAVariant) FullName() string { return "Binary Dragonfly Algorithm" }

// Description returns a one-line summary.
func (v *BDAVariant) Description() string {
	return "DA's step vector read through a V- or S-shaped transfer function as a bit-flip " +
		"probability, for 0/1-valued search spaces."
}

// GetConfig returns NewBinaryConfig, which already carries the unit bounds,
// the v3 transfer function and UseBinary.
func (v *BDAVariant) GetConfig() *Config { return NewBinaryConfig() }

// IsMultiObjective reports false.
func (v *BDAVariant) IsMultiObjective() bool { return false }

// Run executes the binary algorithm through OptimizeBinaryContext, which runs
// BDA whether or not Config.UseBinary is set.
func (v *BDAVariant) Run(ctx context.Context, config *Config, options ...RunOption) (*Result, error) {
	return OptimizeBinaryContext(ctx, config, options...)
}

// ApplicableTo scores the binary variant against a problem.
func (v *BDAVariant) ApplicableTo(characteristics ProblemCharacteristics) float64 {
	if characteristics.MultiObjective {
		return 0.1
	}

	if !characteristics.Discrete {
		// A continuous problem can be handed to BDA only through an encoding
		// the library does not supply, so this is close to unusable rather
		// than merely a poor fit.
		return 0.05
	}

	score := 0.8

	if characteristics.Dimensionality <= 200 {
		score += 0.1
	}

	if characteristics.Landscape == Rugged {
		score += 0.1
	}

	return min(score, 1.0)
}

// EstimatedOverhead returns 1.0: the bit-flip update replaces the continuous
// position update rather than adding to it.
func (v *BDAVariant) EstimatedOverhead() float64 { return 1.0 }

// RecommendedFor lists the problem classes BDA suits.
func (v *BDAVariant) RecommendedFor() []string {
	return []string{
		"Binary and discrete search spaces",
		"Feature selection",
		"Knapsack and subset-selection problems",
	}
}

// =============================================================================
// Improved single-objective variants
// =============================================================================

// MHDAVariant is the memory-based DA/PSO hybrid of Ranjini and Murugan.
type MHDAVariant struct{}

func (v *MHDAVariant) Name() string { return nameMHDA }

func (v *MHDAVariant) FullName() string { return "Memory-based Hybrid Dragonfly Algorithm" }

func (v *MHDAVariant) Description() string {
	return "Continuous DA with per-dragonfly personal memory and a PSO exploitation stage."
}

func (v *MHDAVariant) GetConfig() *Config { return NewMemoryHybridConfig() }

func (v *MHDAVariant) IsMultiObjective() bool { return false }

func (v *MHDAVariant) Run(
	ctx context.Context,
	config *Config,
	options ...RunOption,
) (*Result, error) {
	return OptimizeMemoryHybridContext(ctx, config, options...)
}

func (v *MHDAVariant) ApplicableTo(characteristics ProblemCharacteristics) float64 {
	if characteristics.MultiObjective || characteristics.Discrete {
		return 0.05
	}

	score := 0.7
	if characteristics.Modality == Multimodal || characteristics.Modality == HighlyMultimodal {
		score += 0.1
	}

	if characteristics.RequiresFastConvergence || characteristics.RequiresStableConvergence {
		score += 0.1
	}

	if characteristics.ExpensiveEvaluations {
		score -= 0.2
	}

	return min(score, 1.0)
}

func (v *MHDAVariant) EstimatedOverhead() float64 { return 2.0 }

func (v *MHDAVariant) RecommendedFor() []string {
	return []string{
		recommendedContinuous,
		"Premature-convergence-prone landscapes",
		"Problems benefiting from PSO exploitation",
	}
}

// CDAVariant is Sayed, Tharwat and Hassanien's chaotic DA.
type CDAVariant struct{}

func (v *CDAVariant) Name() string { return nameCDA }

func (v *CDAVariant) FullName() string { return "Chaotic Dragonfly Algorithm" }

func (v *CDAVariant) Description() string {
	return "Continuous DA whose movement coefficients come from a selectable chaotic map."
}

func (v *CDAVariant) GetConfig() *Config { return NewChaoticConfig() }

func (v *CDAVariant) IsMultiObjective() bool { return false }

func (v *CDAVariant) Run(
	ctx context.Context,
	config *Config,
	options ...RunOption,
) (*Result, error) {
	return OptimizeChaoticContext(ctx, config, options...)
}

func (v *CDAVariant) ApplicableTo(characteristics ProblemCharacteristics) float64 {
	if characteristics.MultiObjective {
		return 0.05
	}

	if characteristics.Discrete {
		return 0.05
	}

	score := 0.7
	if characteristics.Landscape == Rugged || characteristics.Modality == HighlyMultimodal {
		score += 0.1
	}

	return min(score, 1.0)
}

func (v *CDAVariant) EstimatedOverhead() float64 { return 1.0 }

func (v *CDAVariant) RecommendedFor() []string {
	return []string{
		recommendedContinuous,
		"Rugged and multimodal landscapes",
		"Deterministic chaotic coefficient studies",
	}
}

// QGDAVariant is Yu et al.'s quantum-behaved Gaussian mutational DA.
type QGDAVariant struct{}

func (v *QGDAVariant) Name() string { return nameQGDA }

func (v *QGDAVariant) FullName() string {
	return "Quantum-behaved and Gaussian Mutational Dragonfly Algorithm"
}

func (v *QGDAVariant) Description() string {
	return "Continuous DA followed by greedy Gaussian mutation and quantum rotation."
}

func (v *QGDAVariant) GetConfig() *Config { return NewQuantumConfig() }

func (v *QGDAVariant) IsMultiObjective() bool { return false }

func (v *QGDAVariant) Run(
	ctx context.Context,
	config *Config,
	options ...RunOption,
) (*Result, error) {
	return OptimizeQuantumContext(ctx, config, options...)
}

func (v *QGDAVariant) ApplicableTo(characteristics ProblemCharacteristics) float64 {
	if characteristics.MultiObjective || characteristics.Discrete {
		return 0.05
	}

	score := 0.7
	if characteristics.Modality == HighlyMultimodal || characteristics.Landscape == Rugged {
		score += 0.15
	}

	if characteristics.ExpensiveEvaluations {
		score -= 0.3
	}

	return min(score, 1.0)
}

func (v *QGDAVariant) EstimatedOverhead() float64 { return 3.0 }

func (v *QGDAVariant) RecommendedFor() []string {
	return []string{
		"Continuous multimodal optimization",
		"Diversity-sensitive searches",
		"Cheap objectives where extra candidate evaluation is affordable",
	}
}

// =============================================================================
// MODA -- the multi-objective variant
// =============================================================================

// MODAVariant is the multi-objective Dragonfly Algorithm: DA's swarm mechanics
// with the food source and the enemy drawn from a hypercube-gridded Pareto
// archive.
//
// It is the one variant whose Run cannot honor the AlgorithmVariant contract;
// see the note on the interface. Use GetMultiObjectiveConfig and
// RunMultiObjective.
type MODAVariant struct{}

// Name returns the canonical short name.
func (v *MODAVariant) Name() string { return nameMODA }

// FullName returns the descriptive name.
func (v *MODAVariant) FullName() string { return "Multi-Objective Dragonfly Algorithm" }

// Description returns a one-line summary.
func (v *MODAVariant) Description() string {
	return "DA over a Pareto archive: the food source is drawn from a sparse hypercube and the " +
		"enemy from a crowded one."
}

// GetConfig returns the swarm block of NewMultiObjectiveConfig, so that a
// caller inspecting the variant's mechanics through the common interface sees
// the same schedules a MODA run uses. It is not on its own runnable as MODA;
// use GetMultiObjectiveConfig.
func (v *MODAVariant) GetConfig() *Config { return NewMultiObjectiveConfig().Swarm }

// GetMultiObjectiveConfig returns a freshly allocated default MODA
// configuration. You must still set ObjectiveFunc and Swarm's ProblemSize,
// LowerBound and UpperBound.
func (v *MODAVariant) GetMultiObjectiveConfig() *MultiObjectiveConfig {
	return NewMultiObjectiveConfig()
}

// IsMultiObjective reports true.
func (v *MODAVariant) IsMultiObjective() bool { return true }

// Run always returns ErrMultiObjectiveVariant. A MODA run has no single
// incumbent, so there is no honest *Result to return; call RunMultiObjective.
func (v *MODAVariant) Run(_ context.Context, _ *Config, _ ...RunOption) (*Result, error) {
	return nil, ErrMultiObjectiveVariant
}

// RunMultiObjective executes MODA through OptimizeMultiObjective.
func (v *MODAVariant) RunMultiObjective(
	ctx context.Context,
	config *MultiObjectiveConfig,
	options ...RunOption,
) (*MultiObjectiveResult, error) {
	return OptimizeMultiObjective(ctx, config, options...)
}

// ApplicableTo scores the multi-objective variant against a problem.
func (v *MODAVariant) ApplicableTo(characteristics ProblemCharacteristics) float64 {
	if !characteristics.MultiObjective {
		// MODA on a single objective degenerates to an archive of one point.
		return 0.1
	}

	score := 0.8

	if !characteristics.Discrete {
		score += 0.1
	}

	if characteristics.Modality != Unimodal {
		score += 0.1
	}

	return min(score, 1.0)
}

// EstimatedOverhead returns 1.2: the archive update, the grid rebuild and the
// two roulette draws are per-iteration work DA does not do.
func (v *MODAVariant) EstimatedOverhead() float64 { return 1.2 }

// RecommendedFor lists the problem classes MODA suits.
func (v *MODAVariant) RecommendedFor() []string {
	return []string{
		"Multi-objective optimization",
		"Pareto front approximation",
		"Engineering design tradeoffs",
	}
}

// =============================================================================
// Fluent builder
// =============================================================================

var (
	errNilBuilder         = errors.New("builder is nil")
	errBuilderNoObjective = errors.New("objective function not set; call ForProblem")
)

// VariantBuilder is a fluent front end for configuring and running a variant.
//
// It carries only the single-objective Config: the multi-objective path takes a
// different configuration type and a different entry point, so building one
// through the same chain would mean half the methods silently doing nothing.
// Use MODAVariant.GetMultiObjectiveConfig for MODA.
type VariantBuilder struct {
	variant AlgorithmVariant
	config  *Config
	err     error
}

// NewBuilder creates a builder for the named variant. An unknown name is
// recorded on the builder and surfaces from Build, so a chain can be written
// without an error check at every link.
//
// Example:
//
//	config, err := NewBuilder("bda").ForProblem(fn, 20, 0, 1).WithIterations(500).Build()
func NewBuilder(variantName string) *VariantBuilder {
	variant, err := NewVariant(variantName)
	if err != nil {
		return &VariantBuilder{err: err}
	}

	return NewBuilderFromVariant(variant)
}

// NewBuilderFromVariant creates a builder around an existing variant instance.
func NewBuilderFromVariant(variant AlgorithmVariant) *VariantBuilder {
	if variant == nil {
		return &VariantBuilder{err: errors.New("variant cannot be nil")}
	}

	if variant.IsMultiObjective() {
		return &VariantBuilder{variant: variant, err: ErrMultiObjectiveBuilder}
	}

	return &VariantBuilder{variant: variant, config: variant.GetConfig()}
}

// ForProblem sets the objective function, the dimensionality and the bounds.
//
// The bounds of a binary variant are fixed at the unit interval and are left
// alone; pass any values.
func (b *VariantBuilder) ForProblem(fn ObjectiveFunction, size int, lower, upper float64) *VariantBuilder {
	if b == nil || b.err != nil {
		return b
	}

	b.config.ObjectiveFunc = fn
	b.config.ProblemSize = size

	if !b.config.UseBinary {
		b.config.LowerBound = lower
		b.config.UpperBound = upper
	}

	return b
}

// WithIterations sets the maximum number of iterations.
func (b *VariantBuilder) WithIterations(iterations int) *VariantBuilder {
	if b == nil || b.err != nil {
		return b
	}

	b.config.MaxIterations = iterations

	return b
}

// WithPopulation sets the swarm size.
func (b *VariantBuilder) WithPopulation(size int) *VariantBuilder {
	if b == nil || b.err != nil {
		return b
	}

	b.config.NPop = size

	return b
}

// WithConfig applies an arbitrary edit to the configuration under construction.
func (b *VariantBuilder) WithConfig(edit func(*Config)) *VariantBuilder {
	if b == nil || b.err != nil {
		return b
	}

	if edit != nil {
		edit(b.config)
	}

	return b
}

// Build returns the configured Config, or the first error the chain recorded.
func (b *VariantBuilder) Build() (*Config, error) {
	if b == nil {
		return nil, errNilBuilder
	}

	if b.err != nil {
		return nil, b.err
	}

	if b.config.ObjectiveFunc == nil {
		return nil, errBuilderNoObjective
	}

	err := validateConfig(b.config)
	if err != nil {
		return nil, err
	}

	return b.config, nil
}

// Optimize builds the configuration and runs the variant with a background
// context.
func (b *VariantBuilder) Optimize() (*Result, error) {
	return b.OptimizeContext(context.Background())
}

// OptimizeContext builds the configuration and runs the variant, honoring
// cancellation.
func (b *VariantBuilder) OptimizeContext(ctx context.Context, options ...RunOption) (*Result, error) {
	config, err := b.Build()
	if err != nil {
		return nil, err
	}

	return b.variant.Run(ctx, config, options...)
}

// GetVariant returns the variant the builder was created for, or nil if the
// name was not recognized.
func (b *VariantBuilder) GetVariant() AlgorithmVariant {
	if b == nil {
		return nil
	}

	return b.variant
}
