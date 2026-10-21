"""The gallery: every visitor's agent joins it, and drives against one of the
others in the stand's fourth step (concept §5.6, DECISIONS.md §23).

Every agent is measured once, as it joins, on the same ten reference
targets, with the style profile of §22; the opponent and the table are
chosen from those figures. A filter "from T on" hides older agents without
deleting them. The two models drive from the start and are never hidden.

Agents and the filter are JSON files under the data directory. The module
knows nothing about HTTP or the frontend; `main()` is the staff's command
for the filter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np

from teach_by_showing.cloning import Driving, Policy, Rollout, policy_from_dict, rollout
from teach_by_showing.crane import CraneParams, CraneState, FloatArray
from teach_by_showing.profile import ProfileSummary, profiles, summarize
from teach_by_showing.task import TaskParams, sample_transports

SCHEMA: Final[int] = 1
# The same targets for every agent, so that their figures compare.
REFERENCE_TARGETS: Final[int] = 10
REFERENCE_SEED: Final[int] = 2310
# Own agent, opponent and the two most recent others: six rows made the
# scene visibly smaller (decided 2026-10-21).
TABLE_ROWS: Final[int] = 4
VISITOR_NAME: Final[str] = "Kranführer·in #{}"


def reference_targets(crane: CraneParams, task: TaskParams) -> tuple[FloatArray, FloatArray]:
    return sample_transports(
        np.random.default_rng(REFERENCE_SEED), crane, task, REFERENCE_TARGETS
    )


def reference_driving(policy: Policy, crane: CraneParams, task: TaskParams) -> Driving:
    """The agent setting off on the reference targets; the stand drives it a
    few steps per frame (§23)."""
    start, target = reference_targets(crane, task)
    return Driving.begin(policy, CraneState.at_rest(start), target, crane, task)


def reference_summary(driving: Driving) -> ProfileSummary:
    """The style profile (§22) of a finished reference drive."""
    assert driving.done
    r: Rollout = driving.result()
    return summarize(profiles(r.transport, r.tally))


def reference_profile(policy: Policy, crane: CraneParams, task: TaskParams) -> ProfileSummary:
    """The agent on the reference targets in one go, by the same rollout()
    and style profile as the evaluation (§16, §22)."""
    start, target = reference_targets(crane, task)
    r: Rollout = rollout(policy, CraneState.at_rest(start), target, crane, task)
    return summarize(profiles(r.transport, r.tally))


@dataclass(frozen=True, slots=True)
class Agent:
    """One member of the gallery. `number` is None for the models, which are
    shipped rather than taught at the stand."""

    number: int | None
    name: str
    # Stamp of the run it learnt from, as in the run's file name; the
    # filter compares these as strings, which sort as times do.
    created: str
    run: str | None
    policy: Policy
    reference: ProfileSummary
    # The parameters the reference was driven under; another set drives the
    # reference again rather than show stale figures.
    crane: CraneParams
    task: TaskParams

    @property
    def seconds(self) -> float:
        """Mean time on the reference targets; the time limit if it reached
        none, so that such an agent still compares."""
        return (
            self.reference.seconds
            if self.reference.seconds is not None
            else self.task.time_limit
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "number": self.number,
            "name": self.name,
            "created": self.created,
            "run": self.run,
            "policy": self.policy.to_dict(),
            "reference": asdict(self.reference),
            "crane": self.crane.to_dict(),
            "task": self.task.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"agent has schema {data.get('schema')!r}, expected {SCHEMA}")
        return cls(
            data["number"],
            data["name"],
            data["created"],
            data["run"],
            policy_from_dict(data["policy"]),
            ProfileSummary(**data["reference"]),
            CraneParams.from_dict(data["crane"]),
            TaskParams.from_dict(data["task"]),
        )

    def measured_under(self, crane: CraneParams, task: TaskParams) -> Agent:
        """This agent with its reference driven under `crane` and `task`."""
        if crane == self.crane and task == self.task:
            return self
        return Agent(
            self.number,
            self.name,
            self.created,
            self.run,
            self.policy,
            reference_profile(self.policy, crane, task),
            crane,
            task,
        )


@dataclass(slots=True)
class Joining:
    """An agent on its way into the gallery, its reference driven a few
    steps per frame by the stand (§23)."""

    number: int
    name: str
    created: str
    run: str | None
    driving: Driving

    def agent(self) -> Agent:
        d: Driving = self.driving
        return Agent(
            self.number,
            self.name,
            self.created,
            self.run,
            d.policy,
            reference_summary(d),
            d.crane,
            d.task,
        )


def make_agent(
    number: int | None,
    name: str,
    created: str,
    run: str | None,
    policy: Policy,
    crane: CraneParams,
    task: TaskParams,
) -> Agent:
    return Agent(
        number,
        name,
        created,
        run,
        policy,
        reference_profile(policy, crane, task),
        crane,
        task,
    )


@dataclass(frozen=True, slots=True)
class GalleryStore:
    """Agents as files under `directory`/agents/, the filter in
    `directory`/gallery.json. Read on every call, so that a filter set with
    the staff's command takes effect with the next visitor, without a
    restart; the files are small."""

    directory: Path
    models: tuple[Agent, ...] = ()

    @property
    def _agents_dir(self) -> Path:
        return self.directory / "agents"

    @property
    def _filter_file(self) -> Path:
        return self.directory / "gallery.json"

    def agents(self) -> list[Agent]:
        """Every visitor's agent ever stored, oldest first."""
        if not self._agents_dir.is_dir():
            return []
        found: list[Agent] = [
            Agent.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self._agents_dir.glob("agent-*.json"))
        ]
        return sorted(found, key=lambda a: a.number or 0)

    def next_number(self) -> int:
        """Never reused, not even for hidden agents: a name stays unique."""
        return max((a.number or 0 for a in self.agents()), default=0) + 1

    def add(self, agent: Agent) -> Path:
        assert agent.number is not None, "the models are shipped, not stored"
        self._agents_dir.mkdir(parents=True, exist_ok=True)
        path: Path = self._agents_dir / f"agent-{agent.number:04d}.json"
        partial: Path = path.with_suffix(".partial")
        partial.write_text(json.dumps(agent.to_dict()), encoding="utf-8")
        os.replace(partial, path)
        return path

    def since(self) -> str | None:
        if not self._filter_file.is_file():
            return None
        since: str | None = json.loads(self._filter_file.read_text(encoding="utf-8"))["since"]
        return since

    def set_since(self, stamp: str | None) -> None:
        """Hide the agents created before `stamp`; None shows them all."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._filter_file.write_text(json.dumps({"since": stamp}), encoding="utf-8")

    def visible(self, crane: CraneParams, task: TaskParams) -> list[Agent]:
        """The models and the agents from the filter on, each measured under
        the given parameters."""
        since: str | None = self.since()
        shown: list[Agent] = list(self.models) + [
            a for a in self.agents() if since is None or a.created >= since
        ]
        return [a.measured_under(crane, task) for a in shown]


def opponent(own: Agent, others: list[Agent]) -> Agent | None:
    """The agent whose reference time differs most from the own agent's, as
    a ratio, so that a cautious visitor meets a quick one and the other way
    round; between equals the most recent."""
    candidates: list[Agent] = [a for a in others if a.name != own.name]
    if not candidates:
        return None
    return max(
        reversed(candidates),
        key=lambda a: abs(math.log(own.seconds / a.seconds)),
    )


def table_agents(own: Agent, rival: Agent, others: list[Agent]) -> list[Agent]:
    """At most TABLE_ROWS: the own agent, the opponent and the most recent
    others, sorted by reference time."""
    rest: list[Agent] = [a for a in others if a.name not in (own.name, rival.name)]
    recent: list[Agent] = sorted(rest, key=lambda a: a.created, reverse=True)
    rows: list[Agent] = [own, rival, *recent[: TABLE_ROWS - 2]]
    return sorted(rows, key=lambda a: a.seconds)


def since_argument(value: str, store: GalleryStore, now: str) -> str | None:
    if value == "all":
        return None
    if value == "now":
        return now
    if value.startswith("#"):
        number: int = int(value[1:])
        for agent in store.agents():
            if agent.number == number:
                return agent.created
        raise SystemExit(f"no agent #{number}")
    return value


def main() -> None:
    """The staff's command: list the gallery, or set its filter."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="List the gallery's agents or set from when on they are shown (§23)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "data")),
        help="the server's data directory; default: $DATA_DIR or ./data",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list", help="every stored agent, with its date and whether it is shown"
    )
    since = commands.add_parser("since", help="show only the agents from then on")
    since.add_argument("when", help="now, all, a stamp such as 20261103-090000, or #n")
    args: argparse.Namespace = parser.parse_args()

    store: GalleryStore = GalleryStore(args.data_dir)
    if args.command == "since":
        stamp: str | None = since_argument(args.when, store, time.strftime("%Y%m%d-%H%M%S"))
        store.set_since(stamp)
        print(f"shown from {stamp}" if stamp else "all shown")
        return
    current: str | None = store.since()
    print(f"filter: from {current}" if current else "filter: none, all shown")
    for agent in store.agents():
        shown: bool = current is None or agent.created >= current
        print(
            f"#{agent.number:<4} {agent.created}  {'shown ' if shown else 'hidden'}"
            f"  {agent.seconds:5.1f} s  {agent.name}"
        )


if __name__ == "__main__":
    main()
