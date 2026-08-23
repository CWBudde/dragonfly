//go:build js && wasm

package main

import "math/rand"

// rngFor is the single place a run's random source is built, so that every
// export seeds the same way and a seed quoted by one page means the same thing
// on another.
func rngFor(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
