// Reads the input device and returns a command in -1..1. Input only: what the
// command does to the crane is decided in Python (DECISIONS.md §9).
//
// For now the keyboard, for development. A key is all or nothing, so it is
// ramped like a stick being pushed; without the ramp every key press would
// be a full-throttle jolt no visitor's gamepad could produce. The gamepad
// joins here in the spike's second half (concept §16).
"use strict";

const KEY_RAMP_PER_SECOND = 2.5; // 0 to full in 0.4 s

const keysDown = new Set();
window.addEventListener("keydown", (e) => keysDown.add(e.key));
window.addEventListener("keyup", (e) => keysDown.delete(e.key));
// Releasing a key while the window has no focus never sends keyup.
window.addEventListener("blur", () => keysDown.clear());

let keyCommand = 0;

function readCommand(dtSeconds) {
  let want = 0;
  if (keysDown.has("ArrowLeft") || keysDown.has("a") || keysDown.has("A")) want -= 1;
  if (keysDown.has("ArrowRight") || keysDown.has("d") || keysDown.has("D")) want += 1;
  const step = KEY_RAMP_PER_SECOND * dtSeconds;
  keyCommand += Math.max(-step, Math.min(step, want - keyCommand));
  return keyCommand;
}
