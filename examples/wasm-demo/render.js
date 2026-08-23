/*
 * render.js — every pixel this demo draws.
 *
 * It is deliberately the only file that touches a canvas, and it knows nothing
 * about the optimizer: it is handed plain typed arrays and told where to put
 * them. The division matters because the one rule this demo has is that no
 * optimization logic lives in JavaScript, and a renderer that started deriving
 * quantities would be the first place that rule quietly broke.
 *
 * Published as window.Render (a classic script, no modules — see index.html).
 */
"use strict";

window.Render = (function () {
  const css = getComputedStyle(document.documentElement);

  function token(name, fallback) {
    const value = css.getPropertyValue(name).trim();
    return value || fallback;
  }

  const COLOR = {
    ink: token("--ink", "#e6f2f0"),
    inkDim: token("--ink-dim", "#94b0ac"),
    inkFaint: token("--ink-faint", "#5d7b78"),
    rule: token("--rule", "#1d3634"),
    swarm: token("--swarm", "#3fd0c9"),
    food: token("--food", "#f0a63c"),
    enemy: token("--enemy", "#e0568a"),
    ok: token("--ok", "#59d98b"),
    branch: [
      token("--branch-food", "#f0a63c"),
      token("--branch-swarm", "#3fd0c9"),
      token("--branch-levy", "#a98cf0"),
    ],
  };

  /*
   * Canvases are sized in CSS and backed at devicePixelRatio, so a line one
   * unit wide is one device pixel and text is not resampled. fit() returns the
   * context already scaled into CSS-pixel coordinates, so every draw function
   * below can work in the units the layout uses.
   */
  function fit(canvas) {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height || width));

    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    return { ctx, width, height };
  }

  /*
   * The heatmap ramp: deep ground for the low-cost basins the swarm is hunting,
   * lifting through petrol to a pale crest. Low is dark on purpose — the swarm
   * is drawn bright, so the eye separates "where the answer is" from "where the
   * dragonflies are" without either fighting the other.
   */
  const RAMP = [
    [4, 12, 14],
    [10, 40, 46],
    [16, 78, 82],
    [40, 120, 118],
    [110, 160, 148],
    [200, 205, 186],
  ];

  function ramp(t) {
    const clamped = Math.max(0, Math.min(1, t));
    const scaled = clamped * (RAMP.length - 1);
    const index = Math.min(RAMP.length - 2, Math.floor(scaled));
    const frac = scaled - index;
    const a = RAMP[index];
    const b = RAMP[index + 1];

    return [
      Math.round(a[0] + (b[0] - a[0]) * frac),
      Math.round(a[1] + (b[1] - a[1]) * frac),
      Math.round(a[2] + (b[2] - a[2]) * frac),
    ];
  }

  /*
   * The landscape is painted once per objective change into an offscreen canvas
   * at the sample grid's own resolution, then stretched. Repainting 25,600
   * samples on every animation frame would dominate the frame budget for a
   * picture that never changes between runs.
   */
  function heatmap(values, width, height) {
    const off = document.createElement("canvas");
    off.width = width;
    off.height = height;

    const ctx = off.getContext("2d");
    const image = ctx.createImageData(width, height);

    for (let i = 0; i < values.length; i += 1) {
      const [r, g, b] = ramp(values[i]);
      const p = i * 4;
      image.data[p] = r;
      image.data[p + 1] = g;
      image.data[p + 2] = b;
      image.data[p + 3] = 255;
    }

    ctx.putImageData(image, 0, 0);

    return off;
  }

  // A projector from problem coordinates into canvas pixels. Row 0 of the
  // heatmap is the high y value, so y is flipped here and only here.
  function projector(lower, upper, width, height) {
    const span = upper - lower || 1;

    return {
      x: (value) => ((value - lower) / span) * width,
      y: (value) => height - ((value - lower) / span) * height,
      scale: (value) => (value / span) * width,
    };
  }

  function dot(ctx, x, y, radius, color, alpha) {
    ctx.globalAlpha = alpha === undefined ? 1 : alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  /*
   * The stage: heatmap, the food and enemy trails, the focused dragonfly's
   * neighbourhood, then the swarm on top coloured by which branch of the step
   * update it is about to take.
   */
  function stage(canvas, frame) {
    const { ctx, width, height } = fit(canvas);
    const p = projector(frame.lower, frame.upper, width, height);

    if (frame.heatmap) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(frame.heatmap, 0, 0, width, height);
    } else {
      ctx.fillStyle = "#061010";
      ctx.fillRect(0, 0, width, height);
    }

    trail(ctx, p, frame.foodTrail, frame.index, COLOR.food);
    trail(ctx, p, frame.enemyTrail, frame.index, COLOR.enemy);

    /*
     * The known minimiser, drawn under the swarm so a dragonfly sitting on it
     * is not hidden by it. It is a cross rather than a ringed glyph because
     * the two ringed glyphs are spoken for: X⁺ and X⁻ are results of the run,
     * this is the answer the run is being watched against.
     */
    if (frame.optimum) {
      cross(ctx, p.x(frame.optimum[0]), p.y(frame.optimum[1]), COLOR.ok);
    }

    /*
     * The neighbourhood is a square because the neighbour test is
     * per-dimension: all(|a_k - b_k| <= r). Drawing a circle here would be a
     * confident picture of the wrong algorithm.
     */
    if (frame.focus >= 0 && frame.focus < frame.count) {
      const fx = frame.positions[frame.focus * 2];
      const fy = frame.positions[frame.focus * 2 + 1];
      const side = p.scale(frame.radius) * 2;

      ctx.strokeStyle = COLOR.swarm;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(p.x(fx) - side / 2, p.y(fy) - side / 2, side, side);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    for (let i = 0; i < frame.count; i += 1) {
      const x = p.x(frame.positions[i * 2]);
      const y = p.y(frame.positions[i * 2 + 1]);
      const branch = frame.branches ? frame.branches[i] : 1;
      const color = COLOR.branch[branch] || COLOR.swarm;

      if (i === frame.focus) {
        ctx.strokeStyle = COLOR.ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.stroke();
      }

      dot(ctx, x, y, 3.2, color, 0.92);
    }

    marker(ctx, p.x(frame.food[0]), p.y(frame.food[1]), COLOR.food, "+");
    marker(ctx, p.x(frame.enemy[0]), p.y(frame.enemy[1]), COLOR.enemy, "-");
  }

  function cross(ctx, x, y, color) {
    const arm = 6;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.moveTo(x - arm, y - arm);
    ctx.lineTo(x + arm, y + arm);
    ctx.moveTo(x + arm, y - arm);
    ctx.lineTo(x - arm, y + arm);
    ctx.stroke();
    ctx.restore();
  }

  // The food source and the enemy are drawn as ringed glyphs rather than
  // larger dots: at a glance they must not read as "a big dragonfly".
  function marker(ctx, x, y, color, glyph) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = "#08100f";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = "700 11px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(glyph, x, y + 0.5);
    ctx.restore();
  }

  function trail(ctx, p, points, upto, color) {
    if (!points || upto < 1) return;

    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1.25;
    ctx.beginPath();

    for (let i = 0; i <= upto; i += 1) {
      const x = p.x(points[i * 2]);
      const y = p.y(points[i * 2 + 1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // ── plots ───────────────────────────────────────────────────────────────

  const PAD = { top: 20, right: 12, bottom: 22, left: 58 };

  // The caption sits above the plot area, right-aligned, because the y-axis
  // tick labels are wide (a log-scaled cost can be "0.00044") and a caption
  // inside the plot collided with them at every canvas size that mattered.
  function plotFrame(ctx, width, height, yLabel) {
    ctx.strokeStyle = COLOR.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, height - PAD.bottom);
    ctx.lineTo(width - PAD.right, height - PAD.bottom);
    ctx.stroke();

    if (yLabel) {
      ctx.fillStyle = COLOR.inkFaint;
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(yLabel, width - PAD.right, PAD.top - 7);
    }
  }

  // Axis ticks get their own, terser formatting: two significant figures is
  // enough to read a scale from, and the full precision belongs in the
  // telemetry row where there is room for it.
  function tick(value) {
    if (!Number.isFinite(value)) return "—";
    if (value === 0) return "0";

    const magnitude = Math.abs(value);
    if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(1);
    if (magnitude >= 100) return value.toFixed(0);
    if (magnitude >= 1) return value.toFixed(2);

    return value.toPrecision(2);
  }

  function axisText(ctx, text, x, y, align) {
    ctx.fillStyle = COLOR.inkFaint;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = align || "right";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
  }

  /*
   * A convergence curve spans many orders of magnitude, so it is plotted on a
   * log scale whenever every value is positive. A linear axis on Sphere shows
   * one drop in the first few iterations and a flat line for the rest, which
   * hides exactly the part worth looking at.
   */
  function convergence(canvas, values, marker) {
    const { ctx, width, height } = fit(canvas);
    if (!values || values.length === 0) {
      plotFrame(ctx, width, height, "");
      return;
    }

    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < values.length; i += 1) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      plotFrame(ctx, width, height, "no finite costs");
      return;
    }

    const logScale = min > 0 && max / min > 100;
    const toY = (value) => {
      const lo = logScale ? Math.log10(min) : min;
      const hi = logScale ? Math.log10(max) : max;
      const v = logScale ? Math.log10(Math.max(value, min)) : value;
      const span = hi - lo || 1;
      return height - PAD.bottom - ((v - lo) / span) * (height - PAD.top - PAD.bottom);
    };

    plotFrame(ctx, width, height, logScale ? "best cost (log)" : "best cost");

    axisText(ctx, tick(max), PAD.left - 5, PAD.top + 4);
    axisText(ctx, tick(min), PAD.left - 5, height - PAD.bottom - 4);
    axisText(ctx, String(values.length), width - PAD.right, height - PAD.bottom + 10, "right");

    const plotWidth = width - PAD.left - PAD.right;
    const toX = (i) => PAD.left + (i / Math.max(1, values.length - 1)) * plotWidth;

    ctx.strokeStyle = COLOR.swarm;
    ctx.lineWidth = 1.6;
    ctx.beginPath();

    for (let i = 0; i < values.length; i += 1) {
      const x = toX(i);
      const y = toY(values[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();

    if (marker >= 0 && marker < values.length) {
      playhead(ctx, toX(marker), height);
      dot(ctx, toX(marker), toY(values[marker]), 3, COLOR.food);
    }
  }

  function line(canvas, values, marker, label, color) {
    const { ctx, width, height } = fit(canvas);
    plotFrame(ctx, width, height, label);

    if (!values || values.length === 0) return;

    let max = -Infinity;
    for (let i = 0; i < values.length; i += 1) {
      if (Number.isFinite(values[i]) && values[i] > max) max = values[i];
    }

    if (!Number.isFinite(max) || max <= 0) max = 1;

    const plotWidth = width - PAD.left - PAD.right;
    const plotHeight = height - PAD.top - PAD.bottom;
    const toX = (i) => PAD.left + (i / Math.max(1, values.length - 1)) * plotWidth;
    const toY = (v) => height - PAD.bottom - (v / max) * plotHeight;

    axisText(ctx, tick(max), PAD.left - 5, PAD.top + 4);
    axisText(ctx, "0", PAD.left - 5, height - PAD.bottom - 4);

    ctx.strokeStyle = color || COLOR.swarm;
    ctx.lineWidth = 1.6;
    ctx.beginPath();

    for (let i = 0; i < values.length; i += 1) {
      const x = toX(i);
      const y = toY(values[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();

    if (marker >= 0 && marker < values.length) {
      playhead(ctx, toX(marker), height);
      dot(ctx, toX(marker), toY(values[marker]), 3, COLOR.food);
    }
  }

  /*
   * The branch mix: what fraction of the swarm took each branch, iteration by
   * iteration, as a stacked area. It is the clearest single picture of the
   * two-branch update at work — early on the radius is small and most of the
   * swarm is out of range of the food, and the bands invert as r grows.
   */
  function branches(canvas, mix, marker) {
    const { ctx, width, height } = fit(canvas);
    plotFrame(ctx, width, height, "share of swarm");

    if (!mix || mix.length === 0) return;

    const plotWidth = width - PAD.left - PAD.right;
    const plotHeight = height - PAD.top - PAD.bottom;
    const toX = (i) => PAD.left + (i / Math.max(1, mix.length - 1)) * plotWidth;

    // Drawn bottom-up: food share, then swarming, then Levy on top.
    let base = new Float32Array(mix.length);

    for (let band = 0; band < 3; band += 1) {
      ctx.fillStyle = COLOR.branch[band];
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      ctx.moveTo(toX(0), height - PAD.bottom - base[0] * plotHeight);

      for (let i = 0; i < mix.length; i += 1) {
        ctx.lineTo(toX(i), height - PAD.bottom - (base[i] + mix[i][band]) * plotHeight);
      }

      for (let i = mix.length - 1; i >= 0; i -= 1) {
        ctx.lineTo(toX(i), height - PAD.bottom - base[i] * plotHeight);
      }

      ctx.closePath();
      ctx.fill();

      for (let i = 0; i < mix.length; i += 1) base[i] += mix[i][band];
    }

    ctx.globalAlpha = 1;

    axisText(ctx, "100%", PAD.left - 5, PAD.top + 4);
    axisText(ctx, "0", PAD.left - 5, height - PAD.bottom - 4);

    if (marker >= 0 && marker < mix.length) playhead(ctx, toX(marker), height);
  }

  function playhead(ctx, x, height) {
    ctx.strokeStyle = COLOR.inkFaint;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, PAD.top);
    ctx.lineTo(x, height - PAD.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  /*
   * Scatter in objective space, for the Pareto page. Points are drawn over the
   * hypercube grid the archive actually uses, because "which cell is crowded"
   * is the quantity MODA's food and enemy draws turn on.
   */
  function front(canvas, snapshot) {
    const { ctx, width, height } = fit(canvas);
    plotFrame(ctx, width, height, snapshot.labels ? snapshot.labels[1] : "f2");

    const points = snapshot.points || [];
    if (points.length === 0) return;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < points.length; i += 2) {
      minX = Math.min(minX, points[i]);
      maxX = Math.max(maxX, points[i]);
      minY = Math.min(minY, points[i + 1]);
      maxY = Math.max(maxY, points[i + 1]);
    }

    // A fixed frame across the whole replay, so the front does not appear to
    // wander while it is only being rescaled.
    if (snapshot.frame) {
      minX = snapshot.frame[0];
      minY = snapshot.frame[1];
      maxX = snapshot.frame[2];
      maxY = snapshot.frame[3];
    }

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const plotWidth = width - PAD.left - PAD.right;
    const plotHeight = height - PAD.top - PAD.bottom;
    const toX = (v) => PAD.left + ((v - minX) / spanX) * plotWidth;
    const toY = (v) => height - PAD.bottom - ((v - minY) / spanY) * plotHeight;

    if (snapshot.grid && snapshot.nGrid > 0) {
      const [gl0, gl1, gu0, gu1] = snapshot.grid;
      ctx.strokeStyle = COLOR.rule;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.8;

      for (let b = 0; b <= snapshot.nGrid; b += 1) {
        const t = b / snapshot.nGrid;
        const x = toX(gl0 + (gu0 - gl0) * t);
        const y = toY(gl1 + (gu1 - gl1) * t);

        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, height - PAD.bottom);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(width - PAD.right, y);
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
    }

    // Occupancy shading: the sparse cells are where the food source is drawn
    // from, the crowded ones where the enemy and the eviction come from.
    if (snapshot.cells) {
      for (const cell of snapshot.cells) {
        ctx.fillStyle = cell.count >= snapshot.maxCount ? COLOR.enemy : COLOR.food;
        ctx.globalAlpha = cell.count >= snapshot.maxCount ? 0.13 : 0.09;
        ctx.fillRect(toX(cell.x0), toY(cell.y1), toX(cell.x1) - toX(cell.x0), toY(cell.y0) - toY(cell.y1));
      }
      ctx.globalAlpha = 1;
    }

    for (let i = 0; i < points.length; i += 2) {
      dot(ctx, toX(points[i]), toY(points[i + 1]), 3, COLOR.swarm, 0.9);
    }

    axisText(ctx, tick(maxY), PAD.left - 5, PAD.top + 4);
    axisText(ctx, tick(minY), PAD.left - 5, height - PAD.bottom - 4);
    axisText(ctx, snapshot.labels ? snapshot.labels[0] : "f1", width - PAD.right, height - PAD.bottom + 10, "right");
  }

  /*
   * The bit matrix, for the binary page: one column per dragonfly, one row per
   * decision variable, a filled cell for a set bit. It is the only view in
   * which "the swarm agreed on this feature" is visible at a glance.
   */
  function bits(canvas, matrix, rows, columns, selected) {
    const { ctx, width, height } = fit(canvas);

    ctx.fillStyle = "#061010";
    ctx.fillRect(0, 0, width, height);

    if (!matrix || rows === 0 || columns === 0) return;

    const cellWidth = width / columns;
    const cellHeight = height / rows;

    for (let c = 0; c < columns; c += 1) {
      for (let r = 0; r < rows; r += 1) {
        if (matrix[c * rows + r] < 0.5) continue;

        ctx.fillStyle = selected && selected[r] ? COLOR.food : COLOR.swarm;
        ctx.globalAlpha = selected && selected[r] ? 0.95 : 0.65;
        ctx.fillRect(
          c * cellWidth + 0.5,
          r * cellHeight + 0.5,
          Math.max(1, cellWidth - 1),
          Math.max(1, cellHeight - 1),
        );
      }
    }

    ctx.globalAlpha = 1;
  }

  /*
   * The transfer function's own curve, so "why did that bit flip" has a visible
   * answer: the height of this curve at Δx is the flip probability.
   *
   * The samples come from Go, through the library's LookupTransferFunction --
   * a curve transcribed into JavaScript could disagree with the function that
   * actually produced the flips, which would make it worse than no curve.
   */
  function transferCurve(canvas, samples, span, marker) {
    const { ctx, width, height } = fit(canvas);
    plotFrame(ctx, width, height, "P(flip)");

    if (!samples || samples.length === 0) return;

    const plotWidth = width - PAD.left - PAD.right;
    const plotHeight = height - PAD.top - PAD.bottom;
    const toX = (i) => PAD.left + (i / Math.max(1, samples.length - 1)) * plotWidth;
    const toY = (v) => height - PAD.bottom - Math.max(0, Math.min(1, v)) * plotHeight;

    // Δx = 0 is where the two families differ most: a V-shaped function has a
    // zero there and an S-shaped one has a half, so the axis earns its line.
    const zero = PAD.left + plotWidth / 2;
    ctx.strokeStyle = COLOR.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(zero, PAD.top);
    ctx.lineTo(zero, height - PAD.bottom);
    ctx.stroke();

    ctx.strokeStyle = COLOR.swarm;
    ctx.lineWidth = 1.8;
    ctx.beginPath();

    for (let i = 0; i < samples.length; i += 1) {
      const x = toX(i);
      const y = toY(samples[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();

    if (Number.isFinite(marker)) {
      const clamped = Math.max(-span, Math.min(span, marker));
      const index = ((clamped + span) / (span * 2)) * (samples.length - 1);
      dot(ctx, toX(index), toY(samples[Math.round(index)]), 3.5, COLOR.food);
    }

    axisText(ctx, "1", PAD.left - 5, PAD.top + 4);
    axisText(ctx, "0", PAD.left - 5, height - PAD.bottom - 4);
    axisText(ctx, `Δx = ±${span}`, width - PAD.right, height - PAD.bottom + 10, "right");
  }

  /*
   * format is shared by every readout on every page, so a number means the same
   * thing wherever it appears. Non-finite values render as an em dash rather
   * than "NaN": the Go side already maps them to null for exactly this reason.
   */
  function format(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    if (value === 0) return "0";

    const magnitude = Math.abs(value);
    if (magnitude >= 1e5 || magnitude < 1e-4) return value.toExponential(3);
    if (magnitude >= 100) return value.toFixed(1);
    if (magnitude >= 1) return value.toFixed(3);

    return value.toPrecision(4);
  }

  return {
    COLOR,
    fit,
    heatmap,
    stage,
    convergence,
    line,
    branches,
    front,
    bits,
    transferCurve,
    format,
    tick,
  };
})();
