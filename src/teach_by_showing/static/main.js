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
const el = (id) => document.getElementById(id);

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
      message.arrival = performance.now();
      states.push(message);
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

function fitCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(rect.width * ratio)) {
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

// A command of −1..1 as a bar from the centre: left negative, right positive.
function showCommand(fill, command) {
  fill.style.left = `${50 + Math.min(0, command) * 50}%`;
  fill.style.width = `${Math.abs(command) * 50}%`;
}

function showReadout(view) {
  el("elapsed").textContent = `${view.elapsed.toFixed(1).replace(".", ",")} s`;
  el("transports").textContent = view.transports;
  showCommand(el("command-fill"), view.command);
  if (view.texts) showTexts(view.texts);

  el("ghost-readout").hidden = !view.ghost;
  el("ghost-command-readout").hidden = !view.ghost;
  if (view.ghost) {
    el("ghost-elapsed").textContent = `${view.ghost.elapsed.toFixed(1).replace(".", ",")} s`;
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
    banner.textContent = `Ziel erreicht – ${view.elapsed.toFixed(1).replace(".", ",")} s`;
  } else if (view.result === "timeout") {
    banner.textContent = "Zeit abgelaufen";
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
    socket.send(JSON.stringify({ type: "input", u: command, a: buttons.a, b: buttons.b }));
  }
  if (config && states.length > 0) {
    const view = viewAt(now);
    const rect = fitCanvas();
    drawScene(ctx, rect.width, rect.height, config, view);
    showReadout(view);
  }
  requestAnimationFrame(tick);
}

connect();
requestAnimationFrame(tick);
