/*
 * pareto.js — the MODA page controller.
 *
 * The archive is variable-length, so unlike the swarm it cannot be indexed
 * arithmetically: pareto.go returns one concatenation of points plus a count
 * per frame, and this file walks a running offset. The cells array is the same
 * idea, each frame prefixed by how many occupied hypercubes it holds.
 */
"use strict";

(function () {
  const { $, setStatus, call, cacheSinks, option, Transport } = window.Demo;

  const el = {
    stage: $("stage"),
    archiveCurve: $("archiveCurve"),
    cellHistogram: $("cellHistogram"),

    benchmark: $("benchmark"),
    dimensions: $("dimensions"),
    npop: $("npop"),
    iterations: $("iterations"),
    archiveSize: $("archiveSize"),
    nGrid: $("nGrid"),
    seed: $("seed"),

    run: $("run"),
    rerun: $("rerun"),
    newSeed: $("newSeed"),

    benchNote: $("benchNote"),

    tArchive: $("tArchive"),
    tFinal: $("tFinal"),
    tCells: $("tCells"),
    tCrowd: $("tCrowd"),
    tEvals: $("tEvals"),
    tIterations: $("tIterations"),
    tTermination: $("tTermination"),
    tSeed: $("tSeed"),
  };

  const state = {
    info: null,
    run: null,
    frames: null,
    transport: null,
    sinks: {},
  };

  const ARRAYS = ["points", "counts", "grid", "cells", "archiveCurve"];

  function populate(info) {
    for (const spec of info.multi) {
      el.benchmark.appendChild(option(spec.name, spec.name));
    }

    el.benchmark.value = info.multi[0].name;
    el.dimensions.max = info.maxDimensions;
    el.iterations.max = info.maxIterations;
    el.npop.max = info.maxPopulation;
    el.archiveSize.max = info.maxArchiveSize;
    el.nGrid.max = info.maxNGrid;

    syncBenchmark();
  }

  function currentBenchmark() {
    return state.info.multi.find((b) => b.name === el.benchmark.value);
  }

  function syncBenchmark() {
    const spec = currentBenchmark();
    if (!spec) return;

    el.benchNote.textContent = spec.blurb;
  }

  /*
   * index() turns the two flat concatenations into per-frame views, once per
   * run rather than once per animation frame. Walking the offsets on every
   * repaint would make scrubbing backwards cost O(t) work for no reason.
   */
  function index(result) {
    const frames = [];

    let pointOffset = 0;
    let cellOffset = 0;

    for (let t = 0; t < result.iterations; t += 1) {
      const count = result.counts[t];
      const points = result.points.subarray(pointOffset, pointOffset + count * 2);
      pointOffset += count * 2;

      const cellCount = result.cells[cellOffset];
      cellOffset += 1;

      const cells = [];
      let maxCount = 0;

      for (let c = 0; c < cellCount; c += 1) {
        const base = cellOffset + c * 3;
        const cell = {
          ix: result.cells[base],
          iy: result.cells[base + 1],
          count: result.cells[base + 2],
        };
        maxCount = Math.max(maxCount, cell.count);
        cells.push(cell);
      }

      cellOffset += cellCount * 3;

      frames.push({
        points,
        cells,
        maxCount,
        grid: result.grid.subarray(t * 4, t * 4 + 4),
      });
    }

    return frames;
  }

  function execute() {
    setStatus("Running…", "loading");

    const options = {
      benchmark: el.benchmark.value,
      dimensions: Number(el.dimensions.value) || 10,
      npop: Number(el.npop.value) || 50,
      iterations: Number(el.iterations.value) || 100,
      archiveSize: Number(el.archiveSize.value) || 100,
      nGrid: Number(el.nGrid.value) || 10,
      seed: Number(el.seed.value) || 0,
      out: state.sinks,
    };

    const result = call("pareto", options);
    if (!result) return;

    cacheSinks(state.sinks, result, ARRAYS);

    state.run = result;
    state.frames = index(result);

    el.tFinal.textContent = String(result.finalArchive);
    el.tEvals.textContent = result.evaluations.toLocaleString();
    el.tIterations.textContent = String(result.iterations);
    el.tTermination.textContent = result.terminationReason.replace(/_/g, " ");
    el.tSeed.textContent = String(result.seed);

    state.transport.reset(result.iterations);

    setStatus(
      `${result.benchmark}, ${result.dimensions}-D, archive ${result.archiveSize} ` +
        `over a ${result.nGrid}×${result.nGrid} grid, seed ${result.seed}`,
      "ready",
    );
  }

  function draw(t) {
    const result = state.run;
    if (!result || !state.frames) return;

    const frame = state.frames[Math.min(t, state.frames.length - 1)];

    // The grid's own extent moves with the archive, so a cell's objective-space
    // rectangle is derived per frame from the bounds MODA used that iteration.
    const [gl0, gl1, gu0, gu1] = frame.grid;
    const stepX = (gu0 - gl0) / result.nGrid || 1;
    const stepY = (gu1 - gl1) / result.nGrid || 1;

    const cells = frame.cells.map((cell) => ({
      count: cell.count,
      x0: gl0 + cell.ix * stepX,
      x1: gl0 + (cell.ix + 1) * stepX,
      y0: gl1 + cell.iy * stepY,
      y1: gl1 + (cell.iy + 1) * stepY,
    }));

    Render.front(el.stage, {
      points: frame.points,
      cells,
      maxCount: frame.maxCount,
      grid: [gl0, gl1, gu0, gu1],
      nGrid: result.nGrid,
      frame: result.frame,
      labels: result.objectives,
    });

    Render.line(el.archiveCurve, result.archiveCurve, t, "archive members", Render.COLOR.swarm);
    histogram(frame);

    el.tArchive.textContent = String(frame.points.length / 2);
    el.tCells.textContent = String(frame.cells.length);
    el.tCrowd.textContent = String(frame.maxCount);
  }

  /*
   * The occupancy histogram, sorted descending: the leftmost bar is the cell
   * the enemy and the next eviction are drawn from, the rightmost the cell the
   * food source comes from. Sorting rather than laying the cells out spatially
   * is the point -- the draw weights depend only on the counts.
   */
  function histogram(frame) {
    const counts = frame.cells.map((c) => c.count).sort((a, b) => b - a);
    const { ctx, width, height } = Render.fit(el.cellHistogram);

    ctx.fillStyle = "#061010";
    ctx.fillRect(0, 0, width, height);

    if (counts.length === 0) return;

    const pad = { top: 16, right: 12, bottom: 20, left: 12 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const barWidth = plotWidth / counts.length;
    const max = counts[0] || 1;

    for (let i = 0; i < counts.length; i += 1) {
      const barHeight = (counts[i] / max) * plotHeight;

      ctx.fillStyle =
        counts[i] === max
          ? Render.COLOR.enemy
          : counts[i] === counts[counts.length - 1]
            ? Render.COLOR.food
            : Render.COLOR.swarm;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(
        pad.left + i * barWidth,
        height - pad.bottom - barHeight,
        Math.max(1, barWidth - 1),
        barHeight,
      );
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = Render.COLOR.inkFaint;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`${counts.length} occupied cells · most crowded holds ${max}`, pad.left, pad.top - 5);
  }

  function bind() {
    el.benchmark.addEventListener("change", () => {
      syncBenchmark();
      execute();
    });

    for (const node of [el.run, el.rerun]) {
      node.addEventListener("click", execute);
    }

    el.newSeed.addEventListener("click", () => {
      el.seed.value = String(Math.floor(Math.random() * 1e6));
      execute();
    });

    window.addEventListener("resize", () => draw(state.transport.frame()));
  }

  Demo.start((info) => {
    state.info = info;
    state.transport = Transport(
      { play: "play", scrub: "scrub", speed: "speed", readout: "frameReadout" },
      draw,
    );

    populate(info);
    bind();
    execute();
  });
})();
