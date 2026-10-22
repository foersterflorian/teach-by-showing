"""The model display: one neuron, its learning in slow motion, and the
stand's learning step that waits for A (DECISIONS.md §21)."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest

from teach_by_showing.cloning import LinearPolicy, Policy, clone, dataset, train
from teach_by_showing.features import features
from teach_by_showing.live import Settings, StandParams
from teach_by_showing.model_view import LearningTrace, formula, learn, model_message
from teach_by_showing.recording import Run
from teach_by_showing.stand import Phase, StandSession
from teach_by_showing.texts import ENGLISH, GERMAN
from tests.support import ROOT
from tests.teachers import quick, teach
from tests.test_stand import SETTINGS, demonstrate, press, showcase

STATIC = ROOT / "src" / "teach_by_showing" / "static"


@cache
def taught() -> Run:
    return teach(quick, transports=2)


@cache
def learnt() -> tuple[Policy, LearningTrace | None]:
    return learn([taught()], np.random.default_rng(0))


def stand_in_learning(settings: Settings = SETTINGS) -> StandSession:
    s: StandSession = StandSession.start(
        np.random.default_rng(0), settings, showcase(), lambda: "test", lambda _: None
    )
    press(s, a=True)
    demonstrate(s)
    assert s.phase is Phase.LEARN
    return s


def test_training_shows_every_epoch_and_the_trained_policy_last() -> None:
    f, u = dataset([taught()])
    seen: list[int] = []
    last: list[LinearPolicy] = []

    def on_epoch(epoch: int, policy: LinearPolicy) -> None:
        seen.append(epoch)
        last[:] = [policy]

    policy, _ = train(
        LinearPolicy.initial(np.random.default_rng(0)), f, u, 50, on_epoch=on_epoch
    )
    assert seen == list(range(51)) and last[0] is policy


def test_learning_keeps_the_same_agent_and_snapshots_on_a_log_scale() -> None:
    """The snapshots change nothing about the agent: the same as clone()."""
    policy, trace = learnt()
    cloned, losses = clone([taught()], np.random.default_rng(0))
    assert isinstance(policy, LinearPolicy) and isinstance(cloned, LinearPolicy)
    assert policy.w.tolist() == cloned.w.tolist()
    assert trace is not None
    assert trace.epochs[0] == 0 and trace.epochs[-1] == 2000
    assert trace.policies[-1].w.tolist() == policy.w.tolist()
    assert trace.losses[0] == losses[0] and trace.losses[-1] <= losses[-1] * 1.001
    # Dense at the start, where a cautious drive's weights already settle:
    # more than a third within 50 of 2000 epochs, where an even spacing
    # would have one.
    assert sum(e <= 50 for e in trace.epochs) > len(trace.epochs) / 3
    assert trace.seconds > 0


def test_a_network_is_not_shown() -> None:
    from teach_by_showing import model_view

    original = model_view.clone

    def as_network(runs: list[Run], rng: np.random.Generator, **kwargs: Any) -> Any:
        return original(runs, rng, kind="mlp", on_epoch=kwargs.get("on_epoch"))

    model_view.clone = as_network  # type: ignore[assignment]
    try:
        _, trace = learn([taught()], np.random.default_rng(0), epochs=20)
    finally:
        model_view.clone = original
    assert trace is None


def test_the_formula_in_each_languages_decimals_with_real_minus_signs() -> None:
    """§21, §24: German decimal commas, English decimal points."""
    p: LinearPolicy = LinearPolicy(np.array([1.8, 0.1, -0.13]), np.array(-0.011))
    assert formula(p, GERMAN) == (
        "Stellgröße = tanh(1,80 · Abstand + 0,10 · Winkel − 0,13 · Winkelgeschw. − 0,01)"
    )
    assert formula(p, ENGLISH) == (
        "Command = tanh(1.80 · Distance + 0.10 · Angle − 0.13 · Ang. speed − 0.01)"
    )
    q: LinearPolicy = LinearPolicy(np.array([-0.3, 0.0, 0.02]), np.array(0.0))
    assert formula(q, GERMAN).startswith("Stellgröße = tanh(−0,30 · Abstand + 0,00 · Winkel")


def test_an_offset_learnt_to_zero_is_shown_as_plus_zero() -> None:
    """§21, decided 2026-10-22: mirrored, the offset ends a few 1e-4 from
    zero; it stays in the formula, as "+ 0,00", never "− 0,00", and so does
    a weight that rounds to zero."""
    p: LinearPolicy = LinearPolicy(np.array([0.52, -0.001, 0.01]), np.array(-0.0002))
    assert formula(p, GERMAN) == (
        "Stellgröße = tanh(0,52 · Abstand + 0,00 · Winkel + 0,01 · Winkelgeschw. + 0,00)"
    )
    assert model_message(p, ENGLISH)["w_text"] == ["+0.52", "+0.00", "+0.01"]
    q: LinearPolicy = LinearPolicy(np.array([-0.004, 0.1, 0.0]), np.array(-0.006))
    assert formula(q, ENGLISH) == (
        "Command = tanh(0.00 · Distance + 0.10 · Angle + 0.00 · Ang. speed − 0.01)"
    )


def test_the_loss_curve_grows_with_the_slow_motion_in_unit_coordinates() -> None:
    policy, trace = learnt()
    assert isinstance(policy, LinearPolicy) and trace is not None
    m: dict[str, Any] = model_message(policy, ENGLISH, trace=trace, shown=10)
    points: list[list[float]] = m["loss"]["points"]
    assert len(points) == 11
    xs: list[float] = [x for x, _ in points]
    assert xs == sorted(xs) and xs[0] == 0.0
    assert all(0.0 <= y <= 1.0 for _, y in points)
    assert m["w_text"] == [("−" if w < 0 else "+") + f"{abs(w):.2f}" for w in policy.w]


def test_the_learning_step_replays_the_weights_then_waits_for_a() -> None:
    s: StandSession = stand_in_learning()
    p: StandParams = SETTINGS.stand
    assert s.trace is not None
    first: dict[str, Any] = s.state_message()
    assert first["model"]["w"] == s.trace.policies[0].w.tolist()
    assert first["texts"]["prompt"] is None and "slow motion" in first["texts"]["hint"]
    for _ in range(p.frames(p.learn_replay)):
        s.tick()
    done: dict[str, Any] = s.state_message()
    assert done["model"]["w"] == s.trace.policies[-1].w.tolist()
    assert len(done["model"]["loss"]["points"]) == s.trace.points
    assert done["texts"]["prompt"] == ENGLISH.learn_prompt
    for _ in range(p.frames(p.learn_wait) - 2):
        s.tick()
    assert s.phase is Phase.LEARN
    press(s, a=True)
    assert s.phase is Phase.AGENT


def test_without_a_button_the_stand_goes_on_after_the_wait() -> None:
    s: StandSession = stand_in_learning()
    p: StandParams = SETTINGS.stand
    for _ in range(p.frames(p.learn_replay + p.learn_wait)):
        s.tick()
    assert s.phase is Phase.AGENT


def test_b_ends_the_learning_step() -> None:
    s: StandSession = stand_in_learning()
    press(s, b=True)
    assert s.phase is Phase.IDLE


def test_the_durations_come_from_the_settings_file() -> None:
    quick_steps: Settings = Settings.from_dict(
        {"input": {"expo": 0.0}, "stand": {"learn_replay": 1.0, "learn_wait": 0.5}}
    )
    s: StandSession = stand_in_learning(quick_steps)
    for _ in range(quick_steps.stand.frames(1.5)):
        s.tick()
    assert s.phase is Phase.AGENT
    with pytest.raises(TypeError):
        Settings.from_dict({"stand": {"learn_slowly": 3.0}})


def test_while_the_agent_drives_the_neuron_works_on_its_live_inputs() -> None:
    s: StandSession = stand_in_learning()
    press(s, a=True)
    for _ in range(90):
        s.tick()
    m: dict[str, Any] = s.state_message()["model"]
    r = s.replay
    assert r is not None and r.state is not None and r.transport is not None
    assert isinstance(s.showcase.policy, LinearPolicy)
    assert m["w"] == s.showcase.policy.w.tolist() and m["loss"] is None
    assert m["inputs"] == features(r.state, r.transport.target, r.run.crane)[0].tolist()
    assert m["output"] == r.command
    assert ENGLISH.model_short[0] in m["formula"] and m["explain"]


def test_the_waiting_stand_shows_no_model() -> None:
    s: StandSession = StandSession.start(
        np.random.default_rng(0), SETTINGS, showcase(), lambda: "test", lambda _: None
    )
    assert "model" not in s.state_message()


def test_the_model_is_drawn_from_the_message_with_no_wording_of_its_own() -> None:
    model: str = (STATIC / "model.js").read_text(encoding="utf-8")
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    assert "drawModel(" in main and "drawLoss(" in main
    assert "model.w_text[k]" in model and "model.labels" in model
    # English words occur in the comments; only a quoted one would be a text.
    for words, quote in ((GERMAN, ""), (ENGLISH, '"')):
        for wording in (*words.model_short, words.command_label, words.loss_axis):
            assert f"{quote}{wording}{quote}" not in model, wording
