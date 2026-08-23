/*
 * bench-chart.js — the shootout's grouped bar chart.
 *
 * One group per benchmark, one bar per contender. Bars are normalised within
 * their group, because the benchmarks span many orders of magnitude and a
 * single shared scale would render every group but the largest as a flat line.
 * The chart answers "which contender won here", not "how big is this number";
 * the table below it answers the second question.
 *
 * Published as window.BenchChart.
 */
"use strict";

window.BenchChart = (function () {
  // One hue per contender, stable across groups so the eye can track a
  // contender down the chart. Six is the number of contenders compare.go
  // ships; a seventh would wrap, which is why the table carries the names too.
  const SERIES = [
    "#3fd0c9",
    "#f0a63c",
    "#a98cf0",
    "#e0568a",
    "#59d98b",
    "#7fb2f0",
  ];

  function color(index) {
    return SERIES[index % SERIES.length];
  }

  /*
   * Normalising within a group needs care: mean costs can be zero (a solved
   * function) and can span from 1e-12 to 1e3 inside one group. Rank-based
   * height would throw away the margin entirely, so the bars are scaled by
   * log1p of the value against the group's own maximum, and a group whose
   * values are all equal draws full-height bars rather than dividing by zero.
   */
  function heights(values) {
    const finite = values.filter((v) => Number.isFinite(v));
    if (finite.length === 0) return values.map(() => 0);

    const max = Math.max(...finite);
    const min = Math.min(...finite);

    if (!(max > min)) return values.map((v) => (Number.isFinite(v) ? 1 : 0));

    const shift = min < 0 ? -min : 0;
    const top = Math.log1p(max + shift);

    return values.map((v) => {
      if (!Number.isFinite(v)) return 0;
      return top > 0 ? Math.log1p(v + shift) / top : 1;
    });
  }

  function draw(canvas, groups, names) {
    const { ctx, width, height } = Render.fit(canvas);

    ctx.fillStyle = "#061010";
    ctx.fillRect(0, 0, width, height);

    if (!groups || groups.length === 0 || !names || names.length === 0) {
      ctx.fillStyle = "#5d7b78";
      ctx.font = "12px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText("no results yet", width / 2, height / 2);
      return;
    }

    const pad = { top: 14, right: 12, bottom: 46, left: 12 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const groupWidth = plotWidth / groups.length;
    const barWidth = Math.max(2, (groupWidth * 0.78) / names.length);

    for (let g = 0; g < groups.length; g += 1) {
      const group = groups[g];
      const scaled = heights(group.values);
      const originX = pad.left + g * groupWidth + groupWidth * 0.11;

      for (let s = 0; s < scaled.length; s += 1) {
        // A bar is always at least a pixel tall, so "won this benchmark" reads
        // as a short bar rather than as a missing one.
        const barHeight = Math.max(1, scaled[s] * plotHeight);

        ctx.fillStyle = color(s);
        ctx.globalAlpha = group.best === s ? 1 : 0.6;
        ctx.fillRect(
          originX + s * barWidth,
          height - pad.bottom - barHeight,
          Math.max(1, barWidth - 1),
          barHeight,
        );
      }

      ctx.globalAlpha = 1;
      ctx.save();
      ctx.translate(pad.left + g * groupWidth + groupWidth / 2, height - pad.bottom + 6);
      ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = "#94b0ac";
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(group.name, 0, 0);
      ctx.restore();
    }

    ctx.fillStyle = "#5d7b78";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("mean cost, log-scaled within each benchmark · solid = winner", pad.left, pad.top - 2);
  }

  return { draw, color };
})();
