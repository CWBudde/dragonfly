/*
 * bench-worker.js — the shootout's wasm instance, off the main thread.
 *
 * CLASSIC worker, deliberately. wasm_exec.js is a classic script that assigns
 * globalThis.Go, and module workers cannot importScripts(), so a
 * type:"module" worker has no way to load it short of rewriting it.
 *
 * Two rules this file exists to hold:
 *
 *   1. go.run() must NOT be awaited. The demo's main() ends in select{}, so the
 *      promise it returns never resolves. Start it and poll for self.dragonfly.
 *   2. A call into Go blocks this worker's event loop, so a "cancel" cannot be
 *      dispatched while a Go call is in flight. The chunking IS the
 *      cancellation: one benchmark per call, with a yield in between.
 */
"use strict";

importScripts("wasm_exec.js");

const READY_TIMEOUT_MS = 30000;
const READY_POLL_MS = 10;

let api = null;
let cancelRequested = -1;

function post(message) {
  self.postMessage(message);
}

function yieldToLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function boot() {
  const go = new Go();
  const response = await fetch("dragonfly.wasm");
  if (!response.ok) throw new Error(`fetch dragonfly.wasm: ${response.status}`);

  const result = await WebAssembly.instantiate(await response.arrayBuffer(), go.importObject);

  go.run(result.instance).catch((err) => {
    post({ type: "fatal", error: `go runtime exited: ${err && err.message}` });
  });

  const deadline = Date.now() + READY_TIMEOUT_MS;

  while (!self.dragonfly) {
    if (Date.now() > deadline) {
      throw new Error("timed out waiting for the wasm module to publish dragonfly");
    }

    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }

  api = self.dragonfly;
  post({ type: "ready", info: api.info() });
}

async function runSweep(request) {
  const runId = request.runId;
  const benchmarks = request.benchmarks || [];

  let completed = 0;
  let cancelled = false;

  post({ type: "runStarted", runId, total: benchmarks.length });

  for (let i = 0; i < benchmarks.length; i += 1) {
    if (cancelRequested === runId) {
      cancelled = true;
      break;
    }

    const benchmark = benchmarks[i];
    post({ type: "jobStarted", runId, benchmark, index: i });

    const result = api.compare({
      benchmark,
      contenders: request.contenders,
      dimensions: request.dimensions,
      runs: request.runs,
      iterations: request.iterations,
      seed: request.seed,
    });

    if (!result || result.error) {
      const panicked = Boolean(result && result.panic);

      post({
        type: "jobError",
        runId,
        benchmark,
        error: (result && result.error) || "compare returned nothing",
        panic: panicked,
      });

      // A panic has aborted the whole instance; every later call would fail
      // the same way, so stop rather than grind through the rest.
      if (panicked) {
        cancelled = true;
        break;
      }
    } else {
      completed += 1;
      post({ type: "jobResult", runId, benchmark, result });
    }

    post({ type: "jobProgress", runId, completed, total: benchmarks.length, index: i });

    // The yield that makes Stop work.
    await yieldToLoop();
  }

  post({ type: "runDone", runId, completed, cancelled });
}

self.onmessage = async (event) => {
  const message = event.data || {};

  if (message.type === "cancel") {
    cancelRequested = message.runId;
    return;
  }

  if (message.type !== "run") return;

  if (!api) {
    post({ type: "fatal", error: "worker is not ready" });
    return;
  }

  try {
    await runSweep(message);
  } catch (err) {
    post({ type: "fatal", error: String((err && err.message) || err) });
  }
};

boot().catch((err) => post({ type: "fatal", error: String((err && err.message) || err) }));
