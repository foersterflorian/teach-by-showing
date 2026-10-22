// Reads the input device and returns a command in -1..1. Input only: what the
// command does to the crane is decided in Python (DECISIONS.md §9).
//
// The gamepad's left stick, or the keyboard for development and staff.
"use strict";

// Indices in the Gamepad API's "standard" mapping, which Firefox reports for
// the GameSir G7 SE (DECISIONS.md §12). The calibration dialog of concept
// §7.1 will replace these constants for devices that report otherwise.
const STICK_AXIS = 0; // left stick, horizontal
const BUTTON_A = 0;
const BUTTON_B = 1;
// The stick reports 8 bits and rests one step off zero; a small dead zone
// keeps that from creeping into the recorded demonstrations.
const DEAD_ZONE = 0.05;

const KEY_RAMP_PER_SECOND = 2.5; // 0 to full in 0.4 s

const keysDown = new Set();
// Keys pressed since the last read: a tap shorter than one frame would
// otherwise be released before it is ever seen as held.
const keysTapped = new Set();
// The staff's keys, all listed in README.md: Ctrl+Alt+G, the gallery from
// now on (DECISIONS.md §23), not Ctrl+R, which reloads the page; Ctrl+Alt+L,
// the other language (§24); Ctrl+Alt+H, light or dark (§25).
let staffTapped = false;
let languageTapped = false;
let themeTapped = false;
window.addEventListener("keydown", (e) => {
  keysDown.add(e.key);
  keysTapped.add(e.key);
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "g") staffTapped = true;
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "l") languageTapped = true;
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "h") themeTapped = true;
});
window.addEventListener("keyup", (e) => keysDown.delete(e.key));
// Releasing a key while the window has no focus never sends keyup.
window.addEventListener("blur", () => keysDown.clear());

let keyCommand = 0;

// The deflection of whichever connected gamepad is pushed furthest, so that
// a spare pad plugged in beside the main one needs no selection. Rescaled
// past the dead zone, so the command starts at 0, not with a step.
function readStick() {
  let best = 0;
  for (const pad of navigator.getGamepads()) {
    if (!pad || pad.axes.length <= STICK_AXIS) continue;
    const v = pad.axes[STICK_AXIS];
    if (Math.abs(v) > Math.abs(best)) best = v;
  }
  if (Math.abs(best) <= DEAD_ZONE) return 0;
  return Math.sign(best) * (Math.abs(best) - DEAD_ZONE) / (1 - DEAD_ZONE);
}

// A key is all or nothing, so it is ramped like a stick being pushed; without
// the ramp every key press would be a full-throttle jolt no gamepad produces.
function readKeys(dtSeconds) {
  let want = 0;
  if (keysDown.has("ArrowLeft") || keysDown.has("a") || keysDown.has("A")) want -= 1;
  if (keysDown.has("ArrowRight") || keysDown.has("d") || keysDown.has("D")) want += 1;
  const step = KEY_RAMP_PER_SECOND * dtSeconds;
  keyCommand += Math.max(-step, Math.min(step, want - keyCommand));
  return keyCommand;
}

// The stick wins whenever it is deflected; at rest the keyboard drives.
function readCommand(dtSeconds) {
  const keys = readKeys(dtSeconds);
  const stick = readStick();
  return stick !== 0 ? stick : keys;
}

// Whether A and B are held on any gamepad, or Enter and Backspace on the
// keyboard. Held, not pressed: when a press counts is decided in Python.
function readButtons() {
  const held = (index) => {
    for (const pad of navigator.getGamepads()) {
      if (pad && pad.buttons.length > index && pad.buttons[index].pressed) return true;
    }
    return false;
  };
  const key = (name) => keysDown.has(name) || keysTapped.has(name);
  const buttons = {
    a: held(BUTTON_A) || key("Enter"),
    b: held(BUTTON_B) || key("Backspace"),
    reset: staffTapped,
    language: languageTapped,
    theme: themeTapped,
  };
  keysTapped.clear();
  staffTapped = false;
  languageTapped = false;
  themeTapped = false;
  return buttons;
}
