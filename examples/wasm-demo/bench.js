/*
 * bench.js — the shootout controller.
 *
 * Unlike the other three pages, the wasm instance for this one lives in a
 * worker (bench-worker.js) and the sweep loop lives in JavaScript. That is not
 * a stylistic choice: a call into Go blocks its thread's event loop for its
 * whole duration, so a Stop button cannot be serviced while one is in flight.
 * Chunking the sweep at one benchmark per call, with a yield in between, is
 * what makes Stop possible at all.
 *
 * This page therefore does NOT load wasm_exec.js itself; the worker does.
 */
"use strict";

(function () {
  const { $, setStatus } = window.Demo;

  const el = {
    chart: $("chart"),
    chartLegend: $("chartLegend"),
    contenders: $("contenders"),
    benchmarks: $("benchmarks"),
    dimensions: $("dimensions"),
    iterations: $("iterations"),
    runs: $("runs"),
    seed: $("seed"),
    start: $("start"),
    stop: $("stop"),
    progress: $("progress"),
    progressNote: $("progressNote"),
    statsTable: $("statsTable"),
    wilcoxonTable: $("wilcoxonTable"),
    friedman: $("friedman"),
    budget: $("budget"),
  };

  // The benchmarks pre-selected on load: the five classics, in rising
  // difficulty. Every one of the fifteen is offered, but a sweep over all of
  // them at the default budget takes long enough to look like a hang.
  const DEFAULT_BENCHMARKS = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "Griewank"];

  const state = {
    info: null,
    worker: null,
    runId: 0,
    running: false,
    groups: [],
    names: [],
    lastResult: null,
  };

  function checkboxList(container, items, isChecked) {
    container.textContent = "";

    for (const item of items) {
      const label = document.createElement("label");
      label.style.display = "block";
      label.style.marginBottom = "0.35rem";
      label.style.fontFamily = "var(--mono)";
      label.style.fontSize = "0.78rem";
      label.title = item.description || "";

      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = item.key;
      box.checked = isChecked(item);
      box.style.width = "auto";
      box.style.marginRight = "0.45rem";
      box.addEventListener("change", updateBudget);

      label.appendChild(box);
      label.appendChild(document.createTextNode(item.label));
      container.appendChild(label);
    }
  }

  function checked(container) {
    return Array.from(container.querySelectorAll("input:checked")).map((box) => box.value);
  }

  function populate(info) {
    checkboxList(
      el.contenders,
      info.contenders.map((c) => ({ key: c.key, label: c.label, description: c.description })),
      () => true,
    );

    checkboxList(
      el.benchmarks,
      info.benchmarks.map((b) => ({ key: b.name, label: b.name, description: b.blurb })),
      (item) => DEFAULT_BENCHMARKS.includes(item.key),
    );

    el.dimensions.max = info.maxDimensions;
    el.iterations.max = info.maxIterations;
    el.runs.max = info.maxRuns;

    updateBudget();
  }

  /*
   * The budget line exists because this page is the one that can take minutes.
   * Stating the run count before the user presses the button is cheaper than
   * explaining afterwards why the tab is busy.
   */
  function updateBudget() {
    const contenders = checked(el.contenders).length;
    const benchmarks = checked(el.benchmarks).length;
    const runs = Number(el.runs.value) || 0;
    const total = contenders * benchmarks * runs;

    el.budget.textContent = total
      ? `${total.toLocaleString()} optimizations: ${benchmarks} benchmarks × ` +
        `${contenders} contenders × ${runs} runs.`
      : "Select at least one contender and one benchmark.";
  }

  function setRunning(running) {
    state.running = running;
    el.start.disabled = running;
    el.stop.disabled = !running;
  }

  // ── results ─────────────────────────────────────────────────────────────

  function record(result) {
    state.names = result.contenders;

    const means = result.statistics.map((row) => row.mean);
    let best = 0;

    for (let i = 1; i < means.length; i += 1) {
      if (Number.isFinite(means[i]) && (!Number.isFinite(means[best]) || means[i] < means[best])) {
        best = i;
      }
    }

    state.groups.push({ name: result.benchmark, values: means, best });
    state.lastResult = result;

    BenchChart.draw(el.chart, state.groups, state.names);
    drawLegend();
    appendStats(result);
    drawTests(result);
  }

  function drawLegend() {
    el.chartLegend.textContent = "";

    state.names.forEach((name, index) => {
      const item = document.createElement("li");
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = BenchChart.color(index);
      item.appendChild(key);
      item.appendChild(document.createTextNode(name));
      el.chartLegend.appendChild(item);
    });
  }

  function cell(row, text, className) {
    const node = document.createElement("td");
    node.textContent = text;
    if (className) node.className = className;
    row.appendChild(node);
  }

  function appendStats(result) {
    for (const stats of result.statistics) {
      const row = document.createElement("tr");
      cell(row, result.benchmark);
      cell(row, stats.name);
      cell(row, String(stats.rank), stats.rank === 1 ? "win" : "");
      cell(row, Render.format(stats.mean));
      cell(row, Render.format(stats.median));
      cell(row, Render.format(stats.stddev));
      cell(row, Render.format(stats.best));
      cell(
        row,
        result.hasTarget ? `${Render.format(stats.successRate)}%` : "—",
        result.hasTarget ? "" : "tie",
      );
      el.statsTable.appendChild(row);
    }
  }

  /*
   * The pairwise tests are shown for the most recent benchmark only. A row per
   * pair per benchmark is a table nobody reads; the per-benchmark ranks above
   * carry the sweep, and this answers "was that difference real" for the one
   * in front of you.
   */
  function drawTests(result) {
    el.wilcoxonTable.textContent = "";

    for (const test of result.wilcoxon) {
      const row = document.createElement("tr");
      cell(row, test.a);
      cell(row, test.b);
      cell(row, test.winner, test.significant ? "win" : "tie");
      cell(row, Render.format(test.w));
      cell(row, Render.format(test.p), test.significant ? "win" : "tie");
      el.wilcoxonTable.appendChild(row);
    }

    const friedman = result.friedman;

    if (!friedman) {
      el.friedman.textContent = "";
      return;
    }

    el.friedman.textContent =
      `${result.benchmark}: Friedman χ² = ${Render.format(friedman.chiSquare)} ` +
      `on ${friedman.df} d.f., p = ${Render.format(friedman.p)} — ` +
      (friedman.significant
        ? "the contenders are not all equivalent."
        : "no significant difference between the contenders.") +
      (result.runs < 10
        ? " Note that below ten runs the p-values are a normal approximation and should not be leaned on."
        : "");
  }

  // ── the sweep ───────────────────────────────────────────────────────────

  function startSweep() {
    const contenders = checked(el.contenders);
    const benchmarks = checked(el.benchmarks);

    if (contenders.length === 0 || benchmarks.length === 0) {
      setStatus("select at least one contender and one benchmark", "error");
      return;
    }

    state.runId += 1;
    state.groups = [];
    el.statsTable.textContent = "";
    el.wilcoxonTable.textContent = "";
    el.friedman.textContent = "";
    BenchChart.draw(el.chart, [], []);

    setRunning(true);

    state.worker.postMessage({
      type: "run",
      runId: state.runId,
      contenders,
      benchmarks,
      dimensions: Number(el.dimensions.value) || 10,
      iterations: Number(el.iterations.value) || 200,
      runs: Number(el.runs.value) || 10,
      seed: Number(el.seed.value) || 0,
    });
  }

  function stopSweep() {
    state.worker.postMessage({ type: "cancel", runId: state.runId });
    setStatus("stopping after the benchmark in flight…", "loading");
  }

  function onMessage(event) {
    const message = event.data || {};

    switch (message.type) {
      case "ready":
        state.info = message.info;
        populate(message.info);

        {
          const build = $("buildInfo");
          if (build) {
            build.textContent = `${message.info.goVersion} · ${message.info.goos}/${message.info.goarch}`;
          }
        }

        setStatus("ready — pick contenders and benchmarks, then run the sweep", "ready");
        el.start.disabled = false;
        break;

      case "runStarted":
        el.progress.max = message.total;
        el.progress.value = 0;
        el.progressNote.textContent = `0 of ${message.total} benchmarks`;
        break;

      case "jobStarted":
        setStatus(`running ${message.benchmark}…`, "loading");
        break;

      case "jobResult":
        if (message.runId === state.runId) record(message.result);
        break;

      case "jobError":
        setStatus(`${message.benchmark}: ${message.error}`, "error");
        break;

      case "jobProgress":
        el.progress.value = message.completed;
        el.progressNote.textContent = `${message.completed} of ${message.total} benchmarks`;
        break;

      case "runDone":
        setRunning(false);
        setStatus(
          message.cancelled
            ? `stopped after ${message.completed} benchmarks`
            : `done — ${message.completed} benchmarks`,
          message.cancelled ? "" : "ready",
        );
        break;

      case "fatal":
        setRunning(false);
        setStatus(message.error, "error");
        break;

      default:
        break;
    }
  }

  function boot() {
    el.start.disabled = true;
    el.start.addEventListener("click", startSweep);
    el.stop.addEventListener("click", stopSweep);

    for (const node of [el.runs, el.dimensions, el.iterations]) {
      node.addEventListener("input", updateBudget);
    }

    window.addEventListener("resize", () => BenchChart.draw(el.chart, state.groups, state.names));

    // Classic, not a module: wasm_exec.js assigns globalThis.Go and a module
    // worker cannot importScripts() it.
    state.worker = new Worker("bench-worker.js");
    state.worker.onmessage = onMessage;
    state.worker.onerror = (err) => setStatus(`worker: ${err.message}`, "error");
  }

  boot();
})();
