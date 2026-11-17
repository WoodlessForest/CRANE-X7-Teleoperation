# Provide the command-line entry point for validation and robot replay.
# Dry-run mode validates actions without loading the controller or opening ports.

from __future__ import annotations

import argparse
import signal
import sys
import time

from episode_replay import home_pose, initialize, load_episode, replay
from robot_control import DEFAULT_HW_CONFIG, DEFAULT_LINKS_CONFIG, X7Robot, finite_number


def positive(value):
    try:
        result = finite_number(float(value), "argument")
        if result <= 0:
            raise ValueError("argument must be greater than 0")
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def make_parser():
    parser = argparse.ArgumentParser(description=(
        "Replay accepted actions from samples.jsonl using the existing py_x7.so, without camera, VR, or video dependencies."))
    parser.add_argument("episode", help="Episode directory containing metadata.json and samples.jsonl")
    parser.add_argument("--side", choices=("auto", "left", "right", "both"), default="auto",
                        help="Default auto: connect only arms with accepted recorded actions")
    parser.add_argument("--left_port", default="/dev/ttyUSB0")
    parser.add_argument("--right_port", default="/dev/ttyUSB1")
    parser.add_argument("--hw_config", default=str(DEFAULT_HW_CONFIG))
    parser.add_argument("--links_config", default=str(DEFAULT_LINKS_CONFIG))
    parser.add_argument("--speed", type=positive, default=1.0,
                        help="Playback speed: 0.5 is half speed; target magnitudes are unchanged (default: 1)")
    parser.add_argument("--init", choices=("first-action", "home", "none"), default="first-action",
                        help="Initialization: first recorded target / recorded teleoperation home pose / none")
    parser.add_argument("--init_seconds", type=positive, default=3.0,
                        help="Initialization duration; home mode calls the existing IK at 30 Hz (default: 3 seconds)")
    parser.add_argument("--end_hold_seconds", type=positive, default=1.0,
                        help="Keep the connection open after the last command (default: 1 second)")
    parser.add_argument("--dry_run", "--dry-run", action="store_true",
                        help="Validate actions and print a summary without loading the .so, accessing devices, or waiting")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        plan = load_episode(args.episode, args.side)
        pose = home_pose(plan.metadata) if args.init == "home" else None
        print(f"[REPLAY] Dataset: {plan.path}")
        print(f"[REPLAY] Arms: {', '.join(plan.sides)}; read {plan.samples_read} samples; "
              f"replay {len(plan.steps)} action steps; skip {plan.skipped_actions} unaccepted/missing channels")
        print(f"[REPLAY] step {plan.steps[0].index} → {plan.steps[-1].index}; "
              f"recorded duration {plan.duration:.3f}s; approximately {plan.duration / args.speed:.3f}s at {args.speed:g}x speed")
        print(f"[REPLAY] Initialization: {args.init} ({args.init_seconds:g}s); "
              f"End hold: {args.end_hold_seconds:g}s; no additional target magnitude, velocity, or tracking-error thresholds")
        if pose is not None:
            print(f"[REPLAY] Teleoperation home pose: {pose}")
        if plan.metadata.get("status") != "complete":
            print(f"[REPLAY] Original recording status={plan.metadata.get('status')}; "
                  "replay only parsed accepted actions, regardless of video finalization status")
        ports = {side: getattr(args, f"{side}_port") for side in plan.sides}
        print(f"[REPLAY] Serial port mapping: {ports}")
        if args.dry_run:
            print("[DRY RUN] Action validation passed; no control extension loaded or devices accessed.")
            return 0

        last_log = time.monotonic()

        def progress(number, step):
            nonlocal last_log
            now = time.monotonic()
            if now - last_log >= 5 or number == len(plan.steps):
                print(f"[REPLAY] {number}/{len(plan.steps)}, step_index={step.index}")
                last_log = now

        def terminate(_signum, _frame):
            raise KeyboardInterrupt

        old_sigterm = signal.signal(signal.SIGTERM, terminate)
        try:
            with X7Robot() as robot:
                robot.connect(ports, hw_config=args.hw_config, links_config=args.links_config)
                initialize(robot, plan, args.init, args.init_seconds)
                print("[REPLAY] Initialization complete; submitting recorded actions.")
                replay(plan, robot, speed=args.speed, progress=progress)
                time.sleep(args.end_hold_seconds)
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
        print("[REPLAY] Action submission complete; control connections released.")
        return 0
    except KeyboardInterrupt:
        print("\n[REPLAY] Interrupted; stopping new targets and releasing control connections.")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
