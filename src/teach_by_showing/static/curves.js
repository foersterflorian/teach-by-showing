// Speed, swing and command over the time of the current transport, the
// agent's against the teacher's (DECISIONS.md §20). Display only: every
// value, scale and label comes from the server; this file collects the
// points of one transport and draws them.
"use strict";

const CURVE_KEYS = ["v", "theta", "command"];
const CURVE_LABEL_W = 0.12; // share of the width for the labels on the left

let curveSegment = null;
let curvePoints = []; // {t, agent, teacher} of the current transport

// Called with every state message as it arrives, not with the interpolated
// view: the curves need every point, not the one being drawn.
function recordCurves(message) {
  const c = message.curves;
  if (!c) {
    curveSegment = null;
    curvePoints = [];
    return;
  }
  // A new transport, or the first state after a reconnect.
  if (c.segment !== curveSegment) {
    curveSegment = c.segment;
    curvePoints = [];
  }
  curvePoints.push({ t: c.t, agent: c.agent, teacher: c.teacher });
}

function drawCurves(ctx, width, height, curves) {
  ctx.clearRect(0, 0, width, height);
  const rowH = height / CURVE_KEYS.length;
  const x0 = width * CURVE_LABEL_W;
  const px = (t) => x0 + (t / curves.seconds) * (width - x0);
  // Sized by the window, like the page's text, not by the strips: from
  // standing distance a label must be as legible as the rest (§20).
  const root = getComputedStyle(document.documentElement);
  const vh = parseFloat(root.getPropertyValue("--text-curve-vh"));
  ctx.font = `${Math.round((window.innerHeight * vh) / 100)}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";

  CURVE_KEYS.forEach((key, row) => {
    const mid = row * rowH + rowH / 2;
    const half = rowH * 0.4;
    const py = (value) => mid - Math.max(-1, Math.min(1, value / curves.scales[key])) * half;

    ctx.fillStyle = COLORS.label;
    ctx.textAlign = "left";
    ctx.fillText(curves.labels[key], 0, mid);
    ctx.fillStyle = COLORS.girder;
    ctx.fillRect(x0, mid, width - x0, 1);

    // The teacher behind, translucent like the ghost; the agent in front.
    ctx.lineWidth = Math.max(2, rowH * 0.025);
    for (const [who, color, alpha] of [
      ["teacher", COLORS.human, GHOST_ALPHA + 0.25],
      ["agent", COLORS.agent, 1],
    ]) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      let drawing = false;
      for (const p of curvePoints) {
        const values = p[who];
        if (!values) {
          drawing = false;
          continue;
        }
        if (drawing) ctx.lineTo(px(p.t), py(values[key]));
        else ctx.moveTo(px(p.t), py(values[key]));
        drawing = true;
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  });

  // Every five seconds along the bottom; the label at the right edge is
  // aligned to it, or half of it would be cut off.
  ctx.fillStyle = COLORS.label;
  ctx.textBaseline = "bottom";
  for (let t = 5; t <= curves.seconds; t += 5) {
    ctx.textAlign = t === curves.seconds ? "right" : "center";
    ctx.fillText(`${t} s`, px(t), height);
  }
}
