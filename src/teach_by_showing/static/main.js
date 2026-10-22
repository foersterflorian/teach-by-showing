// Connection, render loop and readouts. Holds no logic of the crane: it shows
// the states the server sends and sends back the command (DECISIONS.md §9).
"use strict";

// How far behind the newest state the picture runs, in ms. States arrive
// every 16.7 ms with about ±1 ms jitter; drawing one interval behind lets the
// renderer always interpolate between two received states instead of showing
// one state twice and skipping the next. `?delay=0` draws the newest state as
// it arrives, to compare latency against smoothness on the stand.
const DELAY_MS = Number(new URLSearchParams(location.search).get("delay") ?? 21);
const RECONNECT_MS = 1000;
const BUFFER_SECONDS = 2;

const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const curvesCanvas = document.getElementById("curves");
const curvesCtx = curvesCanvas.getContext("2d");
const modelCanvas = document.getElementById("model");
const modelCtx = modelCanvas.getContext("2d");
const el = (id) => document.getElementById(id);
// The words and the number style the server put into the page, in the
// screen's language (DECISIONS.md §24).
const WORDS = document.body.dataset;
const seconds = (value) => `${value.toFixed(1).replace(".", WORDS.decimal)} s`;

let socket = null;
let config = null;
let states = []; // received state messages, oldest first, with arrival time
let lastFrameTime = performance.now();

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onopen = () => { el("offline").hidden = true; };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "config") {
      config = message;
      states = [];
      // A replay brings its own headline; the live page keeps the one in index.html.
      showTexts(config);
    } else if (message.type === "state") {
      // The staff switched the language: the page comes again from the
      // server, its fixed words and the funding notice with it (§24).
      if (message.language && message.language !== document.documentElement.lang) {
        location.reload();
        return;
      }
      message.arrival = performance.now();
      states.push(message);
      recordCurves(message);
      const keep = BUFFER_SECONDS * config.hz;
      if (states.length > keep) states.splice(0, states.length - keep);
    }
  };
  // The server may restart (container healthcheck); keep trying.
  socket.onclose = () => {
    el("offline").hidden = false;
    setTimeout(connect, RECONNECT_MS);
  };
}

// The state to draw now: newest frame minus DELAY_MS, interpolated between the
// two received states around it. Sim time is mapped to browser time through
// the earliest arrival in the buffer, which is the least delayed one.
function viewAt(now) {
  const newest = states[states.length - 1];
  if (DELAY_MS <= 0 || states.length < 2) return newest;
  const msPerFrame = 1000 / config.hz;
  let offset = Infinity;
  for (const s of states) offset = Math.min(offset, s.arrival - s.frame * msPerFrame);
  const frame = (now - DELAY_MS - offset) / msPerFrame;
  if (frame >= newest.frame) return newest;
  for (let i = states.length - 1; i > 0; i--) {
    const a = states[i - 1];
    const b = states[i];
    if (a.frame <= frame) {
      const t = (frame - a.frame) / (b.frame - a.frame);
      const mix = (p, q) => p.map((v, k) => v + (q[k] - v) * t);
      const view = { ...a, x: mix(a.x, b.x), theta: mix(a.theta, b.theta) };
      if (a.ghost && b.ghost) {
        const g = { x: mix(a.ghost.x, b.ghost.x), theta: mix(a.ghost.theta, b.ghost.theta) };
        view.ghost = { ...a.ghost, ...g };
      }
      // Other cranes are matched by position in the list; a list that
      // changed between the two states is shown as it was, not blended.
      if (a.others && b.others && a.others.length === b.others.length) {
        view.others = a.others.map((o, k) => ({
          ...o,
          x: mix(o.x, b.others[k].x),
          theta: mix(o.theta, b.others[k].theta),
        }));
      }
      return view;
    }
  }
  return states[0];
}

// Texts worded on the server, by the id of the element that shows them. The
// stand sends them with every state, as its steps change; a replay once,
// with the config. A text that is null hides its element.
const TEXT_IDS = {
  headline: "headline",
  hint: "hint",
  prompt: "prompt",
  elapsed_label: "elapsed-label",
  ghost_label: "ghost-label",
  command_label: "command-label",
  ghost_command_label: "ghost-command-label",
  formula: "formula",
  explain: "explain",
};

function showTexts(texts) {
  for (const [key, id] of Object.entries(TEXT_IDS)) {
    if (!(key in texts)) continue;
    const node = el(id);
    const text = texts[key] ?? "";
    // Only on change: rewriting the same text every frame costs a layout.
    if (node.textContent !== text) node.textContent = text;
    node.hidden = texts[key] === null;
  }
}

// Sizes a canvas's pixels to its size on the page. Never larger than the
// window: a canvas whose stylesheet gives it no size takes its size from its
// pixels, and each frame would enlarge it until the browser stops drawing.
// That froze the page under a stale stylesheet (DECISIONS.md §20).
function fitCanvas(target, context) {
  const ratio = window.devicePixelRatio || 1;
  const box = target.getBoundingClientRect();
  const rect = {
    width: Math.min(box.width, window.innerWidth),
    height: Math.min(box.height, window.innerHeight),
  };
  if (
    target.width !== Math.round(rect.width * ratio) ||
    target.height !== Math.round(rect.height * ratio)
  ) {
    target.width = Math.round(rect.width * ratio);
    target.height = Math.round(rect.height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

// The band under the scene: the curves while an agent drives, the falling
// loss while it learns, and the model beside them (§20, §21).
function showBand(newest) {
  if (el("band").hidden) return;
  const c = fitCanvas(curvesCanvas, curvesCtx);
  if (newest.curves) drawCurves(curvesCtx, c.width, c.height, newest.curves);
  else if (newest.model && newest.model.loss) drawLoss(curvesCtx, c.width, c.height, newest.model.loss);
  else curvesCtx.clearRect(0, 0, c.width, c.height);
  if (newest.model) {
    const m = fitCanvas(modelCanvas, modelCtx);
    drawModel(modelCtx, m.width, m.height, newest.model);
    showTexts({ formula: newest.model.formula, explain: newest.model.explain });
  }
}

// The style profile as the server words it (§22): a title, rows of cells,
// a note. Rebuilt only when a cell changes, which is once per transport.
let shownProfile = "";
function showProfile(profile) {
  const text = JSON.stringify(profile);
  if (text === shownProfile) return;
  shownProfile = text;
  el("profile-title").textContent = profile.title;
  el("profile-note").textContent = profile.note;
  const table = el("profile");
  table.replaceChildren();
  const head = table.insertRow();
  for (const name of profile.columns) {
    const th = document.createElement("th");
    th.textContent = name;
    head.appendChild(th);
  }
  profile.rows.forEach((cells, k) => {
    const row = table.insertRow();
    // The visitor's own agent among the gallery's (§23).
    if (k === profile.highlight) row.className = "own";
    for (const cell of cells) row.insertCell().textContent = cell;
  });
}

// A command of −1..1 as a bar from the centre: left negative, right positive.
function showCommand(fill, command) {
  fill.style.left = `${50 + Math.min(0, command) * 50}%`;
  fill.style.width = `${Math.abs(command) * 50}%`;
}

function showReadout(view) {
  el("elapsed").textContent = seconds(view.elapsed);
  el("transports").textContent = view.transports;
  showCommand(el("command-fill"), view.command);
  if (view.texts) showTexts(view.texts);

  el("ghost-readout").hidden = !view.ghost;
  el("ghost-command-readout").hidden = !view.ghost;
  if (view.ghost) {
    el("ghost-elapsed").textContent = seconds(view.ghost.elapsed);
    showCommand(el("ghost-command-fill"), view.ghost.command);
  }

  const banner = el("banner");
  banner.className = view.result;
  banner.hidden = view.result === "running";
  if (view.banner !== undefined) {
    // A replay words its banner on the server, once both have finished.
    banner.hidden = view.banner === null;
    banner.textContent = view.banner ?? "";
  } else if (view.result === "success") {
    banner.textContent = `${WORDS.reached} – ${seconds(view.elapsed)}`;
  } else if (view.result === "timeout") {
    banner.textContent = WORDS.timedOut;
  }
}

// Runs once per displayed frame. Sending the command every frame doubles as
// a heartbeat: when the tab is hidden, the browser stops calling this, the
// server's input timeout fires, and the crane brakes.
function tick(now) {
  const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
  lastFrameTime = now;
  const command = readCommand(dt);
  const buttons = readButtons();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(
      JSON.stringify({
        type: "input",
        u: command,
        a: buttons.a,
        b: buttons.b,
        reset: buttons.reset,
        language: buttons.language,
      }),
    );
  }
  if (config && states.length > 0) {
    const view = viewAt(now);
    // Shown or hidden first, so that the scene is sized to what is left.
    const newest = states[states.length - 1];
    el("band").hidden = !newest.curves && !newest.model;
    el("model-panel").hidden = !newest.model;
    el("profile-panel").hidden = !newest.profile;
    if (newest.profile) showProfile(newest.profile);
    const rect = fitCanvas(canvas, ctx);
    drawScene(ctx, rect.width, rect.height, config, view);
    showBand(newest);
    showReadout(view);
  }
  requestAnimationFrame(tick);
}

connect();
requestAnimationFrame(tick);
