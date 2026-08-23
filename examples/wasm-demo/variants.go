//go:build js && wasm

package main

import (
	"strings"

	"github.com/CWBudde/dragonfly"
)

// variantKey is the name JavaScript uses for a variant: the library's own
// name, lowercased. The library accepts either case through NewVariant, and a
// single convention on the JS side keeps the option objects comparable.
func variantKey(variant dragonfly.AlgorithmVariant) string {
	return strings.ToLower(variant.Name())
}

// variantOrder returns every variant in the library's canonical order, which
// GetAllVariants already guarantees. The demo does not impose its own: the
// order a reader sees in the dropdown should be the order the documentation
// lists.
func variantOrder() []dragonfly.AlgorithmVariant {
	return dragonfly.GetAllVariants()
}

func stringsToJS(values []string) []any {
	items := make([]any, len(values))
	for i, value := range values {
		items[i] = value
	}

	return items
}
