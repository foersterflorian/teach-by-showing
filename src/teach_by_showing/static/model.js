// The learned model: one neuron with its three inputs, and its learning as
// a falling curve (DECISIONS.md §21). Display only: weights, labels, texts
// and the curve's positions all come from the server.
"use strict";

// Left margin of the loss curve; it has no labels on the left, unlike the
// curves it replaces while the agent learns.
const LOSS_LEFT = 0.12;

const MODEL_COLORS = {
  positive: "#60a5fa",
  negative: "#c084fc",
  node: "#e6e6e6",
};

// The same text size as the curves' labels, from the stylesheet (§20).
function textPixels() {
  const root = getComputedStyle(document.documentElement);
  const vh = parseFloat(root.getPropertyValue("--text-curve-vh"));
  return Math.round((window.innerHeight * vh) / 100);
}

function drawModel(ctx, width, height, model) {
  ctx.clearRect(0, 0, width, height);
  const font = textPixels();
  ctx.font = `${font}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const r = Math.min(height * 0.07, font * 0.5);
  const xIn = width * 0.45;
  const xSum = width * 0.8;
  const xOut = width * 0.96;
  const ySum = height / 2;
  const rows = model.labels.map((_, k) => height * (0.17 + 0.33 * k));

  // Connections first, so that the nodes cover their ends. Thickness is the
  // weight's size, colour its sign.
  model.w.forEach((w, k) => {
    const share = Math.min(1, Math.abs(w) / model.weight_scale);
    ctx.strokeStyle = w < 0 ? MODEL_COLORS.negative : MODEL_COLORS.positive;
    ctx.lineWidth = 1 + share * r * 1.25;
    ctx.beginPath();
    ctx.moveTo(xIn, rows[k]);
    ctx.lineTo(xSum, ySum);
    ctx.stroke();
    // The weight beside its input, where the connections are still apart.
    ctx.fillStyle = ctx.strokeStyle;
    ctx.textAlign = "left";
    ctx.fillText(model.w_text[k], xIn + r * 2, rows[k] + (ySum - rows[k]) * 0.1 - font * 0.55);
  });
  ctx.strokeStyle = COLORS.agent;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(xSum, ySum);
  ctx.lineTo(xOut, ySum);
  ctx.stroke();

  // Inputs: brighter the further the value is from zero, while the agent
  // drives; plain while it learns.
  model.labels.forEach((label, k) => {
    const value = model.inputs ? Math.min(1, Math.abs(model.inputs[k])) : 0;
    ctx.fillStyle = MODEL_COLORS.node;
    ctx.globalAlpha = model.inputs ? 0.25 + 0.75 * value : 0.5;
    ctx.beginPath();
    ctx.arc(xIn, rows[k], r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = COLORS.label;
    ctx.textAlign = "right";
    ctx.fillText(label, xIn - r * 1.6, rows[k]);
  });

  // The neuron, and its output: the command, as in the readout below.
  ctx.fillStyle = "#2a2a2a";
  ctx.strokeStyle = MODEL_COLORS.node;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(xSum, ySum, r * 1.6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = MODEL_COLORS.node;
  ctx.textAlign = "center";
  ctx.fillText("Σ", xSum, ySum);

  const out = model.output ?? 0;
  ctx.fillStyle = COLORS.agent;
  ctx.globalAlpha = model.output === null ? 0.5 : 0.25 + 0.75 * Math.min(1, Math.abs(out));
  ctx.beginPath();
  ctx.arc(xOut, ySum, r, 0, 2 * Math.PI);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = COLORS.label;
  ctx.textAlign = "right";
  ctx.fillText(model.output_label, xOut + r, ySum - r - font * 0.8);
}

// The loss over the learning so far, in the place of the curves (§21). The
// positions arrive in 0..1; this only scales them.
function drawLoss(ctx, width, height, loss) {
  ctx.clearRect(0, 0, width, height);
  const font = textPixels();
  ctx.font = `${font}px system-ui, sans-serif`;
  const x0 = width * LOSS_LEFT;
  const top = font * 1.6;
  const bottom = height - font * 1.25;
  const px = (x) => x0 + x * (width - x0);
  const py = (y) => bottom - y * (bottom - top);

  ctx.fillStyle = COLORS.girder;
  ctx.fillRect(x0, bottom, width - x0, 1);
  ctx.fillRect(x0, top, 1, bottom - top);
  ctx.fillStyle = COLORS.label;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(loss.label, x0 + font * 0.5, 0);
  ctx.textBaseline = "bottom";
  ctx.textAlign = "right";
  ctx.fillText(loss.axis, width, height);

  ctx.strokeStyle = COLORS.agent;
  ctx.lineWidth = Math.max(2, font * 0.12);
  ctx.beginPath();
  loss.points.forEach(([x, y], k) => {
    if (k === 0) ctx.moveTo(px(x), py(y));
    else ctx.lineTo(px(x), py(y));
  });
  ctx.stroke();
}
