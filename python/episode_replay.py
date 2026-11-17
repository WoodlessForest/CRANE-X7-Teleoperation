# Validate recorded actions and schedule accepted targets for replay.
# Preserve recorded timing gaps without depending on video availability.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

from robot_control import ArmCommand, SIDES, finite_number


@dataclass(frozen=True)
class ReplayStep:
    index: int
    timestamp_ns: int
    commands: dict[str, ArmCommand]


@dataclass(frozen=True)
class ReplayPlan:
    path: Path
    metadata: dict
    sides: tuple[str, ...]
    steps: tuple[ReplayStep, ...]
    samples_read: int
    skipped_actions: int

    @property
    def duration(self):
        return (self.steps[-1].timestamp_ns - self.steps[0].timestamp_ns) / 1e9

    def first_commands(self):
        targets = {side: {} for side in self.sides}
        for step in self.steps:
            for side, command in step.commands.items():
                for name in ("joints", "gripper"):
                    value = getattr(command, name)
                    if value is not None:
                        targets[side].setdefault(name, value)
        return {side: ArmCommand(**values) for side, values in targets.items()}


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _index(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _target(entry, kind, label):
    if entry is None:
        return None
    _object(entry, label)
    if entry.get("interface_accepted") is not True:
        if entry.get("status") == "accepted":
            raise ValueError(f"{label}: accepted status and interface_accepted are inconsistent")
        return None
    if entry.get("submitted") is not True or entry.get("status") != "accepted":
        raise ValueError(f"{label}: action acceptance state is inconsistent")
    if entry.get("unit") != "rad":
        raise ValueError(f"{label}: action unit must be rad")
    if entry.get("target") is None:
        raise ValueError(f"{label}: accepted action is missing target")
    if kind == "joint":
        return ArmCommand(joints=entry["target"]).joints
    return ArmCommand(gripper=entry["target"]).gripper


def load_episode(path, side="auto"):

    if side not in ("auto", "both", *SIDES):
        raise ValueError("side must be auto, both, left, or right")
    path = Path(path).expanduser().resolve()
    metadata = _object(json.loads((path / "metadata.json").read_text(encoding="utf-8")),
                       "metadata")
    if metadata.get("schema_version") != "1.0":
        raise ValueError("only recording schema_version=1.0 is supported")
    if metadata.get("joint_order", [f"joint{i}" for i in range(1, 8)]) != [
            f"joint{i}" for i in range(1, 8)]:
        raise ValueError("joint_order must be joint1..joint7")
    if metadata.get("gripper_dynamixel_id", 9) != 9:
        raise ValueError("the current control interface requires gripper ID 9")
    selected = SIDES if side in ("auto", "both") else (side,)
    steps, active = [], set()
    previous_index = previous_stamp = -1
    samples_read = skipped = 0
    with (path / "samples.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                sample = _object(json.loads(line), "sample")
                index = _index(sample.get("step_index"), "step_index")
                stamp = _index(sample.get("decision_timestamp_ns"), "decision_timestamp_ns")
                if index <= previous_index or stamp <= previous_stamp:
                    raise ValueError("step_index and decision_timestamp_ns must be strictly increasing")
                previous_index, previous_stamp = index, stamp
                actions = _object(sample.get("action"), "action")
                groups = {k: _object(actions.get(k), f"action.{k}")
                          for k in ("joint", "gripper")}
                commands = {}
                for which in selected:
                    targets = {}
                    for kind in ("joint", "gripper"):
                        value = _target(groups[kind].get(which), kind,
                                        f"action.{kind}.{which}")
                        if value is None:
                            skipped += 1
                        else:
                            targets["joints" if kind == "joint" else kind] = value
                    if targets:
                        commands[which] = ArmCommand(**targets)
                        active.add(which)
                if commands:
                    steps.append(ReplayStep(index, stamp, commands))
                samples_read += 1
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path / 'samples.jsonl'} line {line_number}: {exc}") from exc
    if not steps:
        raise ValueError("no accepted actions are available for replay on the selected arms")
    if side != "auto" and set(selected) - active:
        raise ValueError(f"no replayable actions for the selected arms: {', '.join(sorted(set(selected) - active))}")
    return ReplayPlan(path, metadata, tuple(s for s in selected if s in active),
                      tuple(steps), samples_read, skipped)


def home_pose(metadata):
    mapping = metadata.get("mapping")
    if mapping is None:
        mapping = {"home_position_m": [0.35, 0.0, 0.30],
                   "home_rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}
    _object(mapping, "mapping")
    position, rotation = mapping.get("home_position_m"), mapping.get("home_rotation")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("mapping.home_position_m must be a 3D position")
    if (not isinstance(rotation, list) or len(rotation) != 3 or
            any(not isinstance(row, list) or len(row) != 3 for row in rotation)):
        raise ValueError("mapping.home_rotation must be a 3x3 rotation matrix")
    pose = [[finite_number(rotation[i][j], "home_rotation") for j in range(3)] +
            [finite_number(position[i], "home_position_m")] for i in range(3)]
    return pose + [[0.0, 0.0, 0.0, 1.0]]


def wait_until(deadline, *, clock=time.monotonic, sleep=time.sleep):
    while (remaining := deadline - clock()) > 0:
        sleep(remaining)


def initialize(robot, plan, mode="first-action", seconds=3.0, *,
               clock=time.monotonic, sleep=time.sleep):
    seconds = finite_number(seconds, "init_seconds")
    if seconds <= 0:
        raise ValueError("init_seconds must be greater than 0")
    if mode == "none":
        return
    if mode == "first-action":
        robot.submit(plan.first_commands())
        wait_until(clock() + seconds, clock=clock, sleep=sleep)
    elif mode == "home":
        pose = home_pose(plan.metadata)
        deadline = clock() + seconds
        while True:
            started = clock()
            robot.submit_home(pose)
            wait_until(min(deadline, started + 1 / 30), clock=clock, sleep=sleep)
            if clock() >= deadline:
                break
    else:
        raise ValueError(f"unknown initialization mode: {mode}")


def replay(plan, robot, *, speed=1.0, clock=time.monotonic, sleep=time.sleep,
           progress=None):
    speed = finite_number(speed, "speed")
    if speed <= 0:
        raise ValueError("speed must be greater than 0")
    previous_stamp, sent_at = plan.steps[0].timestamp_ns, clock()
    for number, step in enumerate(plan.steps, 1):
        interval = (step.timestamp_ns - previous_stamp) / 1e9 / speed
        wait_until(sent_at + interval, clock=clock, sleep=sleep)
        sent_at = clock()
        try:
            robot.submit(step.commands)
        except Exception as exc:
            raise RuntimeError(f"failed to submit replay step_index={step.index}: {exc}") from exc
        previous_stamp = step.timestamp_ns
        if progress is not None:
            progress(number, step)
