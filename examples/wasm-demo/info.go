//go:build js && wasm

package main

import (
	"runtime"
	"syscall/js"

	"github.com/CWBudde/dragonfly"
)

// jsInfo is the capability table every page builds its controls from.
//
// Almost nothing in it is restated here: the variants come from
// GetAllVariants(), the transfer functions from TransferFunctionNames(), and
// each benchmark's modality and landscape from BenchmarkCharacteristics(). The
// <select> elements in the static HTML ship empty and are filled as soon as
// this call returns, so adding a benchmark to functions.go or a transfer
// function to binary.go puts it in the UI without anyone editing markup.
func jsInfo(_ js.Value) any {
	return map[string]any{
		"goVersion":  runtime.Version(),
		"goos":       runtime.GOOS,
		"goarch":     runtime.GOARCH,
		"variants":   infoVariants(),
		"benchmarks": infoBenchmarks(),
		"multi":      infoMultiBenchmarks(),
		"transfers":  infoTransfers(),
		"problems":   infoBinaryProblems(),
		"contenders": infoContenders(),
		"boundaries": infoBoundaries(),

		"maxDimensions":  maxDimensions,
		"maxIterations":  maxIterations,
		"maxPopulation":  maxPopulation,
		"maxGrid":        maxGrid,
		"maxRuns":        maxCompareRuns,
		"maxArchiveSize": maxArchiveSize,
		"maxNGrid":       maxNGrid,
	}
}

func infoVariants() []any {
	variants := variantOrder()
	list := make([]any, 0, len(variants))

	for _, variant := range variants {
		list = append(list, map[string]any{
			"key":            variantKey(variant),
			"name":           variant.Name(),
			"fullName":       variant.FullName(),
			"description":    variant.Description(),
			"recommendedFor": stringsToJS(variant.RecommendedFor()),
			"overhead":       jsNumber(variant.EstimatedOverhead()),
			"multiObjective": variant.IsMultiObjective(),
		})
	}

	return list
}

func infoBenchmarks() []any {
	names := benchmarkNames()
	list := make([]any, 0, len(names))

	for _, name := range names {
		spec := benchmarks[name]
		modality, landscape := classify(name)

		list = append(list, map[string]any{
			"name":      spec.name,
			"blurb":     spec.blurb,
			"modality":  modality,
			"landscape": landscape,
			"lower":     jsNumber(spec.lower),
			"upper":     jsNumber(spec.upper),

			// Reported for two dimensions specifically, because Michalewicz's
			// minimum is only tabulated for a few and the page must not print
			// a 2-D reference value beside a 10-D run.
			"optimum2d": optionalNumber(spec.optimumValue(2)),
		})
	}

	return list
}

func infoMultiBenchmarks() []any {
	names := multiBenchmarkNames()
	list := make([]any, 0, len(names))

	for _, name := range names {
		spec := multiBenchmarks[name]
		modality, landscape := classify(name)

		list = append(list, map[string]any{
			"name":       spec.name,
			"blurb":      spec.blurb,
			"objectives": stringsToJS(spec.objectives),
			"modality":   modality,
			"landscape":  landscape,
			"lower":      jsNumber(spec.lower),
			"upper":      jsNumber(spec.upper),
		})
	}

	return list
}

// infoTransfers lists BDA's transfer functions in the library's order, with the
// paper's default flagged rather than assumed by position.
func infoTransfers() []any {
	names := dragonfly.TransferFunctionNames()
	list := make([]any, 0, len(names))

	for _, name := range names {
		list = append(list, map[string]any{
			"key":     string(name),
			"family":  transferFamily(name),
			"default": name == dragonfly.DefaultTransferFunction,
		})
	}

	return list
}

// transferFamily splits the eight into the two shapes the paper distinguishes:
// V-shaped functions flip a bit on the magnitude of the step regardless of its
// sign, S-shaped ones drive it toward one or zero.
func transferFamily(name dragonfly.TransferFunction) string {
	if len(name) > 0 && (name[0] == 'v' || name[0] == 'V') {
		return "v-shaped"
	}

	return "s-shaped"
}

// infoBoundaries lists the boundary policies. Wrap is named as the paper's
// default here because a visitor arriving from PSO or GA will read wrapping as
// a bug otherwise.
func infoBoundaries() []any {
	return []any{
		map[string]any{
			"key": string(dragonfly.BoundaryWrap), "default": true,
			"description": "Teleport to the opposite bound and reset that step component to a fresh random draw. The paper's behavior.",
		},
		map[string]any{
			"key": string(dragonfly.BoundaryClamp), "default": false,
			"description": "Pin the coordinate to the bound it crossed. What most other swarm libraries do.",
		},
		map[string]any{
			"key": string(dragonfly.BoundaryReflect), "default": false,
			"description": "Mirror the overshoot back into the feasible interval.",
		},
	}
}

// infoBinaryProblems lists the binary page's objectives. Unlike the continuous
// benchmarks these are the demo's own -- the library's suite is defined over
// real vectors -- so the table lives in binary.go and is merely reported here.
func infoBinaryProblems() []any {
	names := binaryProblemNames()
	list := make([]any, 0, len(names))

	for _, name := range names {
		problem := binaryProblems[name]
		list = append(list, map[string]any{
			"name":  problem.name,
			"blurb": problem.blurb,
		})
	}

	return list
}

// infoContenders lists the shootout's configurations. See compare.go for why
// the shootout compares configurations rather than the library's two
// single-objective variants.
func infoContenders() []any {
	list := make([]any, 0, len(contenders))

	for _, entry := range contenders {
		list = append(list, map[string]any{
			"key":         entry.key,
			"label":       entry.label,
			"description": entry.description,
		})
	}

	return list
}
