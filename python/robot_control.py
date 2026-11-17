# Load the local py_x7 extension and manage selected arm controllers.
# Validate commands in radians before submitting joint and gripper targets.

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import sysconfig


SIDES = ("left", "right")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HW_CONFIG = ROOT / "cpp/config/crane-x7.yaml"
DEFAULT_LINKS_CONFIG = ROOT / "cpp/config/crane-x7_links.csv"


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class ArmCommand:

    joints: tuple[float, ...] | None = None
    gripper: float | None = None

    def __post_init__(self):
        if self.joints is not None:
            if not isinstance(self.joints, (list, tuple)) or len(self.joints) != 7:
                raise ValueError("joint targets must contain 7 values in radians for joint1..joint7")
            object.__setattr__(self, "joints", tuple(
                finite_number(v, f"joint{i + 1}") for i, v in enumerate(self.joints)))
        if self.gripper is not None:
            object.__setattr__(self, "gripper", finite_number(self.gripper, "gripper"))
        if self.joints is None and self.gripper is None:
            raise ValueError("an action requires at least one joint or gripper target")


def load_binding():
    path = ROOT / "python" / f"py_x7{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not path.is_file():
        raise RuntimeError(
            f"no extension matching the current Python interpreter was found at {path}; "
            "use the interpreter used to build the existing .so (.venv/bin/python for this project).")
    try:
        spec = importlib.util.spec_from_file_location("py_x7", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"failed to load the existing control extension {path}: {exc}") from exc


class X7Robot:

    def __init__(self):
        self._controllers = {}

    def connect(self, ports, *, hw_config=DEFAULT_HW_CONFIG,
                links_config=DEFAULT_LINKS_CONFIG):
        if self._controllers:
            raise RuntimeError("robot arms are already connected")
        if not ports or set(ports) - set(SIDES):
            raise ValueError("ports must specify left and/or right")
        hw_config = Path(hw_config).expanduser().resolve()
        links_config = Path(links_config).expanduser().resolve()
        for path in (hw_config, links_config):
            if not path.is_file():
                raise ValueError(f"robot arm configuration file does not exist: {path}")
        canonical_ports = [Path(p).expanduser().resolve() for p in ports.values()]
        if len(set(canonical_ports)) != len(canonical_ports):
            raise ValueError("left and right arms cannot use the same serial port (including symlinks to the same device)")
        for path in canonical_ports:
            if not path.exists() or not os.access(path, os.R_OK | os.W_OK):
                raise ValueError(f"robot arm serial port does not exist or is not readable and writable: {path}")
        binding = load_binding()
        print(f"[CONTROL] Using existing extension: {binding.__file__}")
        try:
            for side in SIDES:
                if side not in ports:
                    continue
                self._controllers[side] = binding.Controller()
                port = str(Path(ports[side]).expanduser().resolve())
                if not self._controllers[side].connect(
                        port, 3000000, str(hw_config), str(links_config)):
                    raise RuntimeError(f"{side} arm connection failed: {port}")
                print(f"[CONTROL] {side} connected: {port}")
        except BaseException:
            self.close()
            raise
        return self

    def submit(self, commands: dict[str, ArmCommand]):

        for side, command in commands.items():
            if side not in self._controllers:
                raise ValueError(f"{side} arm is not connected")
            if not isinstance(command, ArmCommand):
                raise ValueError("use ArmCommand to submit actions")
        for side in SIDES:
            if side not in commands:
                continue
            command = commands[side]
            if command.gripper is not None:
                accepted = self._controllers[side].write_gripper(
                    9, math.degrees(command.gripper))
                if not accepted:
                    raise RuntimeError(f"{side} write_gripper returned False")
            if command.joints is not None:
                if not self._controllers[side].set_positions(list(command.joints)):
                    raise RuntimeError(f"{side} set_positions returned False")

    def submit_home(self, pose):
        import numpy as np

        target = np.asarray(pose, dtype=np.float64)
        for side in self._controllers:
            q = self._controllers[side].solve_ik(target)
            self.submit({side: ArmCommand(joints=q)})

    def close(self):
        self._controllers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
