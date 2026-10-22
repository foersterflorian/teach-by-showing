// Draws the crane in side view. Display only: every position comes from the
// server, every dimension from the config message (DECISIONS.md §9).
"use strict";

// The colours, read from the stylesheet's tokens (DECISIONS.md §25), so that
// the theme is one set of CSS variables and the scripts hold no colour. The
// load is coloured by who drives it (§19): a person orange, a learned agent
// green, the naive driver red, the gallery's opponent violet (§23). Success
// shows in the target zone and the banner's words, not in a colour.
const COLOR_TOKENS = {
  girder: "--girder",
  trolley: "--trolley",
  rope: "--rope",
  human: "--human",
  agent: "--agent",
  naive: "--naive",
  rival: "--rival",
  target: "--target",
  label: "--label",
  weightPos: "--weight-pos",
  weightNeg: "--weight-neg",
  weightDash: "--weight-dash",
  node: "--node",
  nodeFill: "--node-fill",
};
const COLORS = {};

// Again whenever the theme changes (main.js).
function readColors() {
  const root = getComputedStyle(document.documentElement);
  for (const [key, token] of Object.entries(COLOR_TOKENS)) {
    COLORS[key] = root.getPropertyValue(token).trim();
  }
}
readColors();

// Sizes of the drawn parts in metres, so that they scale with the crane.
const TROLLEY_W = 0.6;
const TROLLEY_H = 0.25;
const LOAD_W = 0.55;
const LOAD_H = 0.4;
const OVERHANG = 0.5; // girder beyond each end stop
const FLOOR_GAP = 0.3; // between the hanging load and the floor
const GHOST_ALPHA = 0.35;

// Map metres to pixels: fit the track horizontally and the rope plus load
// vertically, one scale for both so that angles look right.
function layout(width, height, crane) {
  const trackLen = crane.track_max - crane.track_min;
  const scale = Math.min(
    width / (trackLen + 2 * OVERHANG + 0.6),
    height / (TROLLEY_H + crane.rope_length + LOAD_H + FLOOR_GAP + 1.0),
  );
  const left = (width - trackLen * scale) / 2;
  // Girder to floor: trolley, rope, load, and a gap below the hanging load.
  const depth = TROLLEY_H + crane.rope_length + LOAD_H + FLOOR_GAP;
  const girderY = (height - depth * scale) / 2;
  return {
    scale,
    girderY,
    floorY: girderY + depth * scale,
    px: (x) => left + (x - crane.track_min) * scale,
  };
}

function drawScene(ctx, width, height, config, view) {
  const crane = config.crane;
  const L = layout(width, height, crane);
  const s = L.scale;
  ctx.clearRect(0, 0, width, height);
  ctx.lineCap = "round";

  // Girder with end stops and a metre scale.
  ctx.fillStyle = COLORS.girder;
  const g0 = L.px(crane.track_min - OVERHANG);
  const g1 = L.px(crane.track_max + OVERHANG);
  ctx.fillRect(g0, L.girderY - 0.12 * s, g1 - g0, 0.12 * s);
  for (const end of [crane.track_min, crane.track_max]) {
    const side = end === crane.track_min ? -1 : 1;
    ctx.fillRect(L.px(end) + side * (TROLLEY_W / 2 + 0.05) * s - 0.04 * s,
                 L.girderY - 0.2 * s, 0.08 * s, 0.2 * s + TROLLEY_H * s);
  }
  ctx.fillStyle = COLORS.label;
  ctx.font = `${Math.round(0.16 * s)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  for (let m = Math.ceil(crane.track_min); m <= crane.track_max; m++) {
    ctx.fillRect(L.px(m) - 1, L.girderY - 0.2 * s, 2, 0.06 * s);
    ctx.fillText(`${m} m`, L.px(m), L.girderY - 0.26 * s);
  }

  // Floor and target zone; the zone fills up while the load hangs calm.
  ctx.fillStyle = COLORS.girder;
  ctx.fillRect(g0, L.floorY, g1 - g0, 2);
  const tol = config.task.position_tolerance;
  for (const target of view.target) {
    const zx = L.px(target - tol);
    const zw = 2 * tol * s;
    ctx.fillStyle = COLORS.target;
    ctx.globalAlpha = 0.25;
    ctx.fillRect(zx, L.floorY - 0.5 * s, zw, 0.5 * s);
    ctx.globalAlpha = 1;
    // Filled in the target's own colour, not green: green is the agent (§25).
    ctx.fillRect(zx, L.floorY - 0.5 * s * view.calm, zw, 0.5 * s * view.calm);
    // Worded by the server, in the page (DECISIONS.md §24).
    ctx.fillText(document.body.dataset.target, L.px(target), L.floorY + 0.3 * s);
  }

  // In a replay the teacher's recorded drive runs behind, translucent
  // (DECISIONS.md §17); a person drove it. Then the other cranes, such as
  // the naive driver, and in front the crane the readouts describe.
  if (view.ghost) {
    ctx.globalAlpha = GHOST_ALPHA;
    for (let i = 0; i < view.ghost.x.length; i++) {
      drawCrane(ctx, L, crane, view.ghost.x[i], view.ghost.theta[i], COLORS.human);
    }
    ctx.globalAlpha = 1;
  }
  for (const other of view.others ?? []) {
    for (let i = 0; i < other.x.length; i++) {
      drawCrane(ctx, L, crane, other.x[i], other.theta[i], COLORS[other.kind]);
    }
  }
  const load = COLORS[view.driver] ?? COLORS.human;
  for (let i = 0; i < view.x.length; i++) {
    drawCrane(ctx, L, crane, view.x[i], view.theta[i], load);
  }
}

// One trolley with its rope and load.
function drawCrane(ctx, L, crane, x, theta, loadColor) {
  const s = L.scale;
  const tx = L.px(x);
  const hookX = tx + Math.sin(theta) * crane.rope_length * s;
  const hookY = L.girderY + TROLLEY_H * s + Math.cos(theta) * crane.rope_length * s;

  ctx.fillStyle = COLORS.trolley;
  ctx.fillRect(tx - (TROLLEY_W / 2) * s, L.girderY, TROLLEY_W * s, TROLLEY_H * s);

  ctx.strokeStyle = COLORS.rope;
  ctx.lineWidth = Math.max(2, 0.025 * s);
  ctx.beginPath();
  ctx.moveTo(tx, L.girderY + TROLLEY_H * s);
  ctx.lineTo(hookX, hookY);
  ctx.stroke();

  // The load hangs from the hook and tilts with the rope.
  ctx.save();
  ctx.translate(hookX, hookY);
  ctx.rotate(-theta);
  ctx.fillStyle = loadColor;
  ctx.fillRect(-(LOAD_W / 2) * s, 0, LOAD_W * s, LOAD_H * s);
  ctx.restore();
}
