//go:build js && wasm

package main

// The caps every export clamps its options to.
//
// They exist because the page is a public URL and a call into Go blocks the
// thread's event loop for its whole duration: there is no way to interrupt a
// running optimization from JavaScript, so an accidental extra zero in an
// input box would freeze the tab rather than merely take a while. Clamping is
// preferred to rejecting — a request slightly over the line still gets an
// answer, and info() publishes these numbers so the controls can bound
// themselves before it comes to that.
const (
	maxDimensions  = 30
	maxIterations  = 1000
	maxPopulation  = 120
	maxGrid        = 320
	maxCompareRuns = 25
	maxArchiveSize = 200
	maxNGrid       = 30
)
