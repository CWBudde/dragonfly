//go:build js && wasm

// Command wasm-demo is the browser front end for github.com/CWBudde/dragonfly.
//
// The rule this package exists to enforce: no optimization logic lives in
// JavaScript. Every number the pages draw is computed by the library compiled
// to js/wasm, so what a visitor sees is the same code a Go caller would get and
// not a reimplementation that has drifted from it.
package main

import "syscall/js"

// exports lists every function the demo publishes, by its name on the
// namespaced globalThis.dragonfly object. Each one is wrapped by guard, which
// is the single rule this bridge has: nothing reaches JavaScript without a
// recover() in front of it (see bridge.go).
var exports = map[string]func(js.Value) any{
	"info":      jsInfo,
	"run":       jsRun,
	"landscape": jsLandscape,
	"pareto":    jsPareto,
	"binary":    jsBinary,
	"compare":   jsCompare,
}

// live keeps the js.Func values referenced so they are never released.
var live []js.Func

func main() {
	namespace := js.Global().Get("Object").New()

	for name, fn := range exports {
		wrapped := guard(name, fn)
		live = append(live, wrapped)
		namespace.Set(name, wrapped)
	}

	js.Global().Set("dragonfly", namespace)

	// main must not return: the Go runtime tears the instance down when it
	// does, taking every exported function with it. The JavaScript side knows
	// this and never awaits go.run().
	select {}
}
