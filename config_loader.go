// JSON configuration persistence, exported validation, named presets and
// heuristic auto-tuning for the Dragonfly Algorithm.

package dragonfly

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ConfigPreset names one of the configuration factories in config.go, so that
// a preset can be chosen from a string -- a command-line flag, an environment
// variable or a field in a larger configuration file.
type ConfigPreset string

const (
	// PresetDefault is NewDefaultConfig: the paper's standard continuous DA.
	PresetDefault ConfigPreset = "default"
	// PresetHighDimensional is NewHighDimensionalConfig: a larger swarm, a
	// longer run and a slower-growing neighborhood radius.
	PresetHighDimensional ConfigPreset = "high-dimensional"
	// PresetFastConvergence is NewFastConvergenceConfig: a short run on a
	// cheap objective, converging early at the cost of exploration.
	PresetFastConvergence ConfigPreset = "fast-convergence"
	// PresetBinary is NewBinaryConfig: BDA on the unit interval with the
	// paper's default v3 transfer function.
	PresetBinary ConfigPreset = "binary"
)

// LoadConfig reads a Config from a JSON file written by SaveConfig.
//
// The loaded configuration is NOT runnable as it stands: ObjectiveFunc, Rand
// and the constraint function slices carry `json:"-"` because functions and
// random sources cannot be serialized. The caller must set ObjectiveFunc
// before calling Optimize, and may set Rand to reproduce a recorded seed.
// Everything else -- bounds, swarm size, weights, schedules, the convergence
// block and the constraint policy -- round-trips exactly.
//
// The file is validated on load with everything except the ObjectiveFunc
// requirement, so a malformed or contradictory file fails here rather than at
// the start of a run.
//
// Absent JSON fields decode as Go zero values, and zero is a legitimate pinned
// weight rather than a request for the adaptive schedule (see WeightAuto).
// Write configuration files with SaveConfig, which always emits every field,
// rather than hand-authoring a partial one.
func LoadConfig(path string) (*Config, error) {
	// filepath.Clean keeps the path in one canonical form before it reaches the
	// filesystem, and keeps a caller-supplied path from tripping gosec's file
	// inclusion check.
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	config := &Config{}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	unmarshalErr := decoder.Decode(config)
	if unmarshalErr != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", unmarshalErr)
	}

	trailingErr := decoder.Decode(&struct{}{})
	if !errors.Is(trailingErr, io.EOF) {
		if trailingErr == nil {
			trailingErr = errors.New("multiple JSON values")
		}

		return nil, fmt.Errorf("failed to parse config file: trailing data: %w", trailingErr)
	}

	validationErr := validateWithoutObjective(config)
	if validationErr != nil {
		return nil, fmt.Errorf("invalid config: %w", validationErr)
	}

	return config, nil
}

// SaveConfig writes a Config to a JSON file, creating or truncating it.
//
// ObjectiveFunc, Rand and the constraint function slices are not written --
// they cannot be serialized -- so a file written here always needs its
// ObjectiveFunc restored in code after LoadConfig. Every serializable field is
// emitted, including a weight left at WeightAuto and a weight deliberately
// pinned to zero, which are distinct settings and stay distinct on the way
// back in.
func SaveConfig(config *Config, path string) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	writeErr := os.WriteFile(path, data, 0o600)
	if writeErr != nil {
		return fmt.Errorf("failed to write config file: %w", writeErr)
	}

	return nil
}

// ValidateConfig checks a configuration and reports the first problem it finds
// as an error naming the offending field by its JSON name.
//
// It is the exported face of the same checks Optimize runs, so a caller can
// fail fast -- when reading a file, when accepting configuration from a user,
// or in a test -- instead of discovering the problem at the start of a run.
// A configuration that passes here is accepted by Optimize.
func ValidateConfig(config *Config) error {
	return validateConfig(config)
}

// validateWithoutObjective runs ValidateConfig's checks against everything a
// JSON file can carry.
//
// ObjectiveFunc is `json:"-"`, so a freshly loaded configuration never has one
// and would fail the required-field check for a reason that says nothing about
// the file. Validating a shallow copy carrying a stand-in function exercises
// every other check without duplicating them here.
func validateWithoutObjective(config *Config) error {
	if config == nil {
		return errors.New("config must not be nil")
	}

	probe := *config
	if probe.ObjectiveFunc == nil {
		probe.ObjectiveFunc = placeholderObjective
	}

	return validateConfig(&probe)
}

// placeholderObjective stands in for a real objective while a deserialized
// configuration is validated. It is never evaluated.
func placeholderObjective(_ []float64) float64 {
	return 0
}

// NewPresetConfig builds a fresh configuration from a named preset.
//
// Each call returns a newly allocated Config, so a caller may mutate the
// result freely; presets are never shared. You must still set ObjectiveFunc,
// ProblemSize, LowerBound and UpperBound, exactly as with the factory
// functions the presets name.
func NewPresetConfig(preset ConfigPreset) (*Config, error) {
	switch preset {
	case PresetDefault:
		return NewDefaultConfig(), nil
	case PresetHighDimensional:
		return NewHighDimensionalConfig(), nil
	case PresetFastConvergence:
		return NewFastConvergenceConfig(), nil
	case PresetBinary:
		return NewBinaryConfig(), nil
	default:
		return nil, fmt.Errorf("unknown preset %q (known presets: %s)",
			string(preset), strings.Join(PresetNames(), ", "))
	}
}

// ListPresets returns every available preset with a one-line description.
func ListPresets() map[ConfigPreset]string {
	return map[ConfigPreset]string{
		PresetDefault:         "Standard continuous DA with every weight on its adaptive schedule",
		PresetHighDimensional: "Larger swarm, longer run and slower radius growth for many dimensions",
		PresetFastConvergence: "Short run on a cheap objective: converges early, explores less",
		PresetBinary:          "BDA: binary positions on [0,1] with the v3 transfer function",
	}
}

// PresetNames returns the known preset names in a stable alphabetical order.
func PresetNames() []string {
	presets := ListPresets()
	names := make([]string, 0, len(presets))

	for preset := range presets {
		names = append(names, string(preset))
	}

	sort.Strings(names)

	return names
}

// PrintPresets writes every available preset and its description to standard
// output, for a command-line front end that offers a --list-presets flag.
func PrintPresets() {
	descriptions := ListPresets()

	fmt.Println("Available Configuration Presets:")
	fmt.Println(strings.Repeat("=", 80))

	for _, name := range PresetNames() {
		fmt.Printf("  %-20s : %s\n", name, descriptions[ConfigPreset(name)])
	}

	fmt.Println(strings.Repeat("=", 80))
}

// Auto-tuning thresholds. They are deliberately coarse: the point is to keep a
// caller who only knows their problem's dimensionality out of the worst
// defaults, not to replace a tuning study.
const (
	// autoTuneSmallDims is the dimensionality below which the default swarm is
	// larger than the problem needs.
	autoTuneSmallDims = 10
	// autoTuneLargeDims is the dimensionality above which the default swarm and
	// run length stop being enough to cover the search space.
	autoTuneLargeDims = 50
)

// AutoTuneConfig adjusts swarm size and run length to the configured
// ProblemSize, in place. It is a handful of coarse heuristics, not a search:
// a tuned configuration for a particular objective will beat it.
//
// Set ProblemSize (and ideally the bounds and ObjectiveFunc) before calling.
// A nil config, or one whose ProblemSize is not yet positive, is left alone --
// there is nothing to tune against. Nothing here touches the five swarming
// weights: leaving them at WeightAuto keeps the paper's schedules, and pinning
// one is a deliberate choice a heuristic must not override.
func AutoTuneConfig(config *Config) {
	if config == nil || config.ProblemSize <= 0 {
		return
	}

	autoTuneSwarmSize(config)
	autoTuneRunLength(config)
	autoTuneRadius(config)

	// The neighborhood radius schedule is bound-relative -- it is written in
	// units of (ub-lb), not of the coordinate values -- so it needs no scaling
	// with ProblemSize, and MaxStepRatio is likewise a fraction of the box.
	// Only how fast the radius grows depends on dimensionality, which
	// autoTuneRadius handles.
}

// autoTuneSwarmSize scales NPop with dimensionality. The paper's 40 is tuned
// for the classic 10-to-30-dimensional benchmarks; below that a smaller swarm
// spends fewer evaluations per iteration for the same coverage, and above it a
// fixed swarm thins out badly as the volume grows.
func autoTuneSwarmSize(config *Config) {
	switch {
	case config.ProblemSize < autoTuneSmallDims:
		config.NPop = 30
	case config.ProblemSize >= autoTuneLargeDims:
		config.NPop = 100
	default:
		config.NPop = 40
	}
}

// autoTuneRunLength scales MaxIterations with dimensionality, on the same
// reasoning: a low-dimensional problem converges long before 1000 iterations,
// and a high-dimensional one is still exploring at that point. The result is
// never reduced below what a small problem needs to leave the initial swarm
// behind.
func autoTuneRunLength(config *Config) {
	switch {
	case config.ProblemSize < autoTuneSmallDims:
		config.MaxIterations = 500
	case config.ProblemSize >= autoTuneLargeDims:
		config.MaxIterations = 3000
	default:
		config.MaxIterations = 1000
	}
}

// autoTuneRadius slows the radius growth in high dimensions, for the reason
// NewHighDimensionalConfig gives: a radius that reaches the whole box early
// makes every dragonfly a neighbor of every other, which collapses the swarm
// onto the food source before it has explored.
func autoTuneRadius(config *Config) {
	if config.ProblemSize >= autoTuneLargeDims {
		config.RadiusGrowth = 1.0
	}
}
