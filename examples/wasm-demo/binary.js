/*
 * binary.js — the BDA page controller.
 *
 * The bit matrix is laid out iteration x dragonfly x bit, so one frame is a
 * single subarray. Everything drawn here was produced by the library: the bits
 * by OptimizeBinaryContext, the transfer curve by LookupTransferFunction.
 */
"use strict";

(function () {
  const { $, setStatus, call, cacheSinks, option, Transport } = window.Demo;

  const el = {
    stage: $("stage"),
    transferPlot: $("transferPlot"),
    convergence: $("convergence"),
    agreement: $("agreement"),

    problem: $("problem"),
    transfer: $("transfer"),
    bits: $("bits"),
    npop: $("npop"),
    iterations: $("iterations"),
    seed: $("seed"),

    run: $("run"),
    rerun: $("rerun"),
    newSeed: $("newSeed"),

    problemNote: $("problemNote"),
    transferNote: $("transferNote"),

    tBest: $("tBest"),
    tOptimum: $("tOptimum"),
    tSet: $("tSet"),
    tSettled: $("tSettled"),
    tEvals: $("tEvals"),
    tIterations: $("tIterations"),
    tTermination: $("tTermination"),
    tSeed: $("tSeed"),
  };

  const state = { info: null, run: null, transport: null, sinks: {} };

  const ARRAYS = ["convergence", "matrix", "setCount", "agreement", "transferCurve"];

  const FAMILY = {
    "v-shaped":
      "Symmetric about zero: the flip probability depends on the magnitude of the step, " +
      "not its sign. A large step means “change”, whichever way the bit currently is.",
    "s-shaped":
      "Monotone: a large positive step drives the bit toward one and a large negative " +
      "step toward zero. Commits earlier than a V-shaped function, and is harder to " +
      "talk out of a decision.",
  };

  function populate(info) {
    for (const spec of info.problems) {
      el.problem.appendChild(option(spec.name, spec.name));
    }

    for (const spec of info.transfers) {
      el.transfer.appendChild(
        option(spec.key, spec.default ? `${spec.key} (paper default)` : spec.key),
      );
      if (spec.default) el.transfer.value = spec.key;
    }

    el.problem.value = info.problems[0].name;
    el.bits.max = info.maxDimensions;
    el.iterations.max = info.maxIterations;
    el.npop.max = info.maxPopulation;

    syncProblem();
    syncTransfer();
  }

  function syncProblem() {
    const spec = state.info.problems.find((p) => p.name === el.problem.value);
    el.problemNote.textContent = spec ? spec.blurb : "";
  }

  function syncTransfer() {
    const spec = state.info.transfers.find((t) => t.key === el.transfer.value);
    el.transferNote.textContent = spec ? `${spec.family}. ${FAMILY[spec.family] || ""}` : "";
  }

  function execute() {
    setStatus("Running…", "loading");

    const result = call("binary", {
      problem: el.problem.value,
      transfer: el.transfer.value,
      bits: Number(el.bits.value) || 24,
      npop: Number(el.npop.value) || 30,
      iterations: Number(el.iterations.value) || 150,
      seed: Number(el.seed.value) || 0,
      out: state.sinks,
    });

    if (!result) return;

    cacheSinks(state.sinks, result, ARRAYS);
    state.run = result;

    el.tBest.textContent = Render.format(result.bestCost);
    el.tOptimum.textContent = Render.format(result.optimum);
    el.tEvals.textContent = result.evaluations.toLocaleString();
    el.tIterations.textContent = String(result.iterations);
    el.tTermination.textContent = result.terminationReason.replace(/_/g, " ");
    el.tSeed.textContent = String(result.seed);

    state.transport.reset(result.iterations);

    setStatus(
      `${result.problem}, ${result.bits} bits, ${result.npop} dragonflies, ` +
        `T = ${result.transfer}, seed ${result.seed}`,
      "ready",
    );
  }

  function draw(t) {
    const result = state.run;
    if (!result) return;

    const cells = result.npop * result.bits;
    const matrix = result.matrix.subarray(t * cells, (t + 1) * cells);
    const agreement = result.agreement.subarray(t * result.bits, (t + 1) * result.bits);

    // The best solution's bits, highlighted in the matrix so "the swarm agreed
    // on the right answer" and "the swarm agreed" are distinguishable.
    const best = result.bestBits.map((b) => b >= 0.5);

    Render.bits(el.stage, matrix, result.bits, result.npop, best);
    Render.convergence(el.convergence, result.convergence, t);
    Render.transferCurve(el.transferPlot, result.transferCurve, result.transferSpan);
    bars(agreement);

    el.tSet.textContent = `${result.setCount[t]} / ${result.bits}`;
    el.tSettled.textContent = `${settled(agreement)} / ${result.bits}`;
  }

  // A bit counts as settled when at least 90% of the swarm agrees on it. The
  // threshold is a reading aid, not an algorithm parameter, and it is stated
  // here rather than buried so nobody mistakes it for one.
  function settled(agreement) {
    let count = 0;

    for (let i = 0; i < agreement.length; i += 1) {
      if (agreement[i] >= 0.9 || agreement[i] <= 0.1) count += 1;
    }

    return count;
  }

  function bars(agreement) {
    const { ctx, width, height } = Render.fit(el.agreement);

    ctx.fillStyle = "#061010";
    ctx.fillRect(0, 0, width, height);

    const pad = { top: 16, right: 12, bottom: 20, left: 12 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const barWidth = plotWidth / agreement.length;

    // The midline is the interesting reference: a bar near it is a bit the
    // swarm is still split on.
    ctx.strokeStyle = Render.COLOR.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, height - pad.bottom - plotHeight / 2);
    ctx.lineTo(width - pad.right, height - pad.bottom - plotHeight / 2);
    ctx.stroke();

    for (let i = 0; i < agreement.length; i += 1) {
      const value = agreement[i];
      const barHeight = value * plotHeight;

      ctx.fillStyle =
        value >= 0.9 || value <= 0.1 ? Render.COLOR.food : Render.COLOR.swarm;
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
    ctx.fillText("share of the swarm with each bit set · amber = settled", pad.left, pad.top - 5);
  }

  function bind() {
    el.problem.addEventListener("change", () => {
      syncProblem();
      execute();
    });

    el.transfer.addEventListener("change", () => {
      syncTransfer();
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
