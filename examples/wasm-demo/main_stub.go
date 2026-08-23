//go:build !js || !wasm

package main

// This stub keeps the package buildable for non-WASM targets, so `just lint`
// and `go build ./...` cover the demo module like any other.
func main() {}
