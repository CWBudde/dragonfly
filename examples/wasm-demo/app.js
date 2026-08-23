/*
 * app.js — the Swarm Lab controller.
 *
 * It owns the wasm instance on the main thread, calls dragonfly.run() once per
 * run, and then replays the recorded history against a scrubber. Nothing here
 * computes anything about the algorithm; see render.js for the matching rule
 * about drawing.
 */
"use strict";

(function () {
  const { $, setStatus, call, cacheSinks, option, Transport } = window.Demo;

  const el = {
    status: $("status"),
    stage: $("stage"),
    convergence: $("convergence"),
    diversity: $("diversity"),
    branches: $("branches"),
    projectionNote: $("projectionNote"),

    benchmark: $("benchmark"),
    boundary: $("boundary"),
    dimensions: $("dimensions"),
    npop: $("npop"),
    iterations: $("iterations"),
    seed: $("seed"),
    lower: $("lower"),
    upper: $("upper"),
    levy: $("levy"),
    focus: $("focus"),
    showOptimum: $("showOptimum"),
    optimumKey: $("optimumKey"),

    run: $("run"),
    rerun: $("rerun"),
    newSeed: $("newSeed"),


    benchNote: $("benchNote"),
    boundaryNote: $("boundaryNote"),
    buildInfo: $("buildInfo"),

    tBest: $("tBest"),
    tOptimum: $("tOptimum"),
    tEvals: $("tEvals"),
    tIterations: $("tIterations"),
    tTermination: $("tTermination"),
    tSeed: $("tSeed"),
    tDiversity: $("tDiversity"),
    tRadius: $("tRadius"),
  };

  const state = {
    info: null,
    run: null,
    heatmap: null,
    // [x, y] of the known minimiser on the plotted plane, or null when the
    // function has none for this dimension count. Set by paintLandscape.
    optimum: null,
    branchMix: null,
    transport: null,
    frame: 0,
    // Reused Float32Array views, handed back to Go as opts.out so a re-run
    // allocates nothing. See marshal.go for why the buffers are JS-owned.
    sinks: {},
  };

  // ── options ─────────────────────────────────────────────────────────────

  function populate(info) {
    for (const spec of info.benchmarks) {
      el.benchmark.appendChild(option(spec.name, spec.name));
    }

    for (const spec of info.boundaries) {
      el.boundary.appendChild(option(spec.key, spec.key));
      if (spec.default) el.boundary.value = spec.key;
    }

    el.dimensions.max = info.maxDimensions;
    el.iterations.max = info.maxIterations;
    el.npop.max = info.maxPopulation;

    el.benchmark.value = "Rastrigin";
    if (!el.benchmark.value) el.benchmark.value = info.benchmarks[0].name;

    el.buildInfo.textContent = `${info.goVersion} · ${info.goos}/${info.goarch}`;

    syncBenchmark();
    syncBoundary();
  }

  function currentBenchmark() {
    return state.info.benchmarks.find((b) => b.name === el.benchmark.value);
  }

  function syncBenchmark() {
    const spec = currentBenchmark();
    if (!spec) return;

    el.lower.value = spec.lower;
    el.upper.value = spec.upper;

    const shape = [spec.modality, spec.landscape].filter(Boolean).join(", ");
    el.benchNote.textContent = shape ? `${spec.blurb} (${shape})` : spec.blurb;
  }

  function syncBoundary() {
    const spec = state.info.boundaries.find((b) => b.key === el.boundary.value);
    el.boundaryNote.textContent = spec ? spec.description : "";
  }

  function syncFocus(count) {
    const previous = el.focus.value;
    el.focus.textContent = "";
    el.focus.appendChild(option("-1", "none"));

    for (let i = 0; i < count; i += 1) {
      el.focus.appendChild(option(String(i), `#${i}`));
    }

    el.focus.value = previous && Number(previous) < count ? previous : "0";
  }

  // ── the run ─────────────────────────────────────────────────────────────

  function readOptions() {
    const spec = currentBenchmark();

    return {
      benchmark: el.benchmark.value,
      dimensions: Number(el.dimensions.value) || 2,
      iterations: Number(el.iterations.value) || 200,
      npop: Number(el.npop.value) || 30,
      seed: Number(el.seed.value) || 0,
      lower: Number(el.lower.value),
      upper: Number(el.upper.value),
      boundary: el.boundary.value,
      levy: el.levy.checked,
      axisX: 0,
      axisY: 1,
      spec,
    };
  }

  const RUN_ARRAYS = [
    "convergence",
    "swarm",
    "cost",
    "foodTrail",
    "enemyTrail",
    "radius",
    "diversity",
    "branch",
    "neighbors",
  ];

  function paintLandscape(options) {
    const result = call("landscape", {
      benchmark: options.benchmark,
      dimensions: options.dimensions,
      lower: options.lower,
      upper: options.upper,
      axisX: options.axisX,
      axisY: options.axisY,
      mode: "rank",
    });

    if (!result) return;

    state.heatmap = Render.heatmap(result.values, result.width, result.height);

    /*
     * Go reports the minimiser only when one is actually tabulated for this
     * dimension count; Michalewicz above 2-D has none. Absent means the marker
     * and its legend entry both disappear rather than pointing at the middle
     * of the domain the slice fell back to.
     */
    const known =
      typeof result.optimumX === "number" && typeof result.optimumY === "number";

    state.optimum = known ? [result.optimumX, result.optimumY] : null;
    el.optimumKey.hidden = !known;
    el.showOptimum.disabled = !known;

    if (result.projected && !result.throughOptimum) {
      el.projectionNote.textContent =
        `${result.dimensions}-D slice: no minimiser is known for this function above two ` +
        `dimensions, so the plane is taken through the middle of the domain and does not ` +
        `pass through the optimum.`;
    } else if (result.projected) {
      el.projectionNote.textContent =
        `${result.dimensions}-D slice through the known minimiser: the other ` +
        `${result.dimensions - 2} axes are pinned there.`;
    } else {
      el.projectionNote.textContent = "";
    }
  }

  function execute() {
    const options = readOptions();

    if (!(options.lower < options.upper)) {
      setStatus("the lower bound must be below the upper bound", "error");
      return;
    }

    setStatus("Running…", "loading");
    paintLandscape(options);

    const result = call("run", Object.assign({ out: state.sinks }, options));
    if (!result) return;

    cacheSinks(state.sinks, result, RUN_ARRAYS);

    state.run = result;
    state.branchMix = branchMix(result);
    state.frame = 0;

    syncFocus(result.npop);

    fillTelemetry(result);
    state.transport.reset(result.iterations);

    setStatus(
      `${result.benchmark}, ${result.dimensions}-D, ${result.npop} dragonflies, ` +
        `seed ${result.seed} — replaying ${result.iterations} iterations`,
      "ready",
    );
  }

  /*
   * The per-iteration share of each branch. It is a pure re-reading of what Go
   * already reported per dragonfly, not a second opinion about the algorithm:
   * the classification itself is made in run.go against the library's own
   * radius schedule and neighbour test.
   */
  function branchMix(result) {
    const mix = [];

    for (let t = 0; t < result.iterations; t += 1) {
      const counts = [0, 0, 0];

      for (let i = 0; i < result.npop; i += 1) {
        const branch = result.branch[t * result.npop + i];
        if (branch >= 0 && branch < 3) counts[branch] += 1;
      }

      mix.push([
        counts[0] / result.npop,
        counts[1] / result.npop,
        counts[2] / result.npop,
      ]);
    }

    return mix;
  }

  function fillTelemetry(result) {
    el.tBest.textContent = Render.format(result.bestCost);
    el.tOptimum.textContent = Render.format(result.optimum);
    el.tEvals.textContent = result.evaluations.toLocaleString();
    el.tIterations.textContent = String(result.iterations);
    el.tTermination.textContent = result.terminationReason.replace(/_/g, " ");
    el.tSeed.textContent = String(result.seed);
  }

  // ── replay ──────────────────────────────────────────────────────────────

  function draw() {
    const result = state.run;
    if (!result) return;

    const t = Math.max(0, Math.min(state.frame, result.iterations - 1));
    const npop = result.npop;

    const positions = result.swarm.subarray(t * npop * 2, (t + 1) * npop * 2);
    const branchesAt = result.branch.subarray(t * npop, (t + 1) * npop);

    Render.stage(el.stage, {
      lower: result.lower,
      upper: result.upper,
      heatmap: state.heatmap,
      positions,
      branches: branchesAt,
      count: npop,
      index: t,
      focus: Number(el.focus.value),
      radius: result.radius[t],
      food: [result.foodTrail[t * 2], result.foodTrail[t * 2 + 1]],
      enemy: [result.enemyTrail[t * 2], result.enemyTrail[t * 2 + 1]],
      foodTrail: result.foodTrail,
      enemyTrail: result.enemyTrail,
      optimum: el.showOptimum.checked ? state.optimum : null,
    });

    Render.convergence(el.convergence, result.convergence, t);
    Render.line(el.diversity, result.diversity, t, "mean distance to centroid");
    Render.branches(el.branches, state.branchMix, t);

    el.tDiversity.textContent = Render.format(result.diversity[t]);
    el.tRadius.textContent = Render.format(result.radius[t]);
  }

  function setFrame(value) {
    state.frame = value;
    draw();
  }

  // ── boot ────────────────────────────────────────────────────────────────

  function bind() {
    el.benchmark.addEventListener("change", () => {
      syncBenchmark();
      execute();
    });

    el.boundary.addEventListener("change", () => {
      syncBoundary();
      execute();
    });

    el.focus.addEventListener("change", draw);

    el.showOptimum.addEventListener("change", draw);

    el.run.addEventListener("click", execute);

    el.rerun.addEventListener("click", execute);

    el.newSeed.addEventListener("click", () => {
      el.seed.value = String(Math.floor(Math.random() * 1e6));
      execute();
    });

    window.addEventListener("resize", () => {
      if (state.run) draw();
    });
  }

  Demo.start((info) => {
    state.info = info;
    state.transport = Transport(
      { play: "play", scrub: "scrub", speed: "speed", readout: "frameReadout" },
      setFrame,
    );

    populate(info);
    bind();
    execute();
  });
})();
