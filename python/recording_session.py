# Connect the teleoperation loop to the episode recording schema.
# Capture runtime metadata, tracking references, feedback, and command outcomes.

from pathlib import Path
import hashlib
import subprocess
import time

import numpy as np

from episode_recording import (image_observations, json_value, no_action,
                               read_feedback, submit_action)


def config_snapshot(path):
    path = Path(path).resolve()
    try:
        content = path.read_bytes()
        return dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                    content=content.decode("utf-8"), status="available")
    except OSError as exc:
        return dict(path=str(path), content=None, status="unavailable", error=str(exc))


def code_version(root):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    files = ["python/run_teleop.py", "python/v4l2_capture.py", "python/episode_recording.py",
             "python/recording_session.py", "python/TeleVision.py", "python/Preprocessor_arm.py",
             "cpp/src/x7_controller.cpp", "cpp/src/rt_manipulators_ik.cpp",
             "cpp/bindings/pybind_x7.cpp", "cpp/include/ik_diagnostics.hpp",
             "third_party/rt_manipulators_lib/rt_manipulators_lib/src/hardware.cpp",
             "third_party/rt_manipulators_lib/rt_manipulators_lib/include/recording_feedback.hpp"]
    files += [str(p.relative_to(root)) for p in (root / "python").glob("py_x7*.so")]
    def digest(path):
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None
    return dict(git_commit=git("rev-parse", "HEAD"),
                tracked_worktree_changes=git("status", "--porcelain", "--untracked-files=no"),
                source_sha256={f: digest(root / f) for f in files})


def runtime_metadata(args, camera, arms, pre):
    from Preprocessor_arm import QUEST_YUP_TO_WORLD_ZUP, R_x7_from_quest_left, R_x7_from_quest_right
    root = Path(__file__).resolve().parent.parent
    hw_file, link_file = root / "cpp/config/crane-x7.yaml", root / "cpp/config/crane-x7_links.csv"
    hw = config_snapshot(hw_file)
    try:
        from py_x7 import inspect_hardware_config
        cfg = inspect_hardware_config(str(hw_file))
        hw["parse_status"] = "validated_by_hardware_library"
    except ImportError:
        cfg = {}
        hw["parse_status"] = "robot_binding_unavailable"
    groups = cfg.get("joint_groups", {})
    robot = {}
    for side, arm in arms.items():
        connected = arm is not None
        init_status = arm.initialization_status() if connected and hasattr(arm, "initialization_status") else {}
        communicating = connected and init_status.get("communication_thread_started", False)
        robot[side] = dict(connected=connected, port=getattr(args, f"{side}_port"),
            configuration_loaded=connected, configured_groups={}, active_sync_read={}, active_sync_write={},
            position_feedback_enabled={}, model_position_limits_rad=None)
        for group, definition in groups.items():
            names = definition.get("joints", [])
            robot[side]["configured_groups"][group] = dict(
                joint_order=names, dynamixel_ids=[cfg[n]["id"] for n in names],
                motors=[cfg[n].get("dynamixel") for n in names],
                operating_modes=[cfg[n]["operating_mode"] for n in names],
                declared_sync_read=definition.get("sync_read", []),
                declared_sync_write=definition.get("sync_write", []))
            robot[side]["active_sync_read"][group] = definition.get("sync_read", []) if communicating else []
            robot[side]["active_sync_write"][group] = definition.get("sync_write", []) if communicating else []
            robot[side]["position_feedback_enabled"][group] = (
                arm.feedback_snapshot(group)["enabled"] if connected and hasattr(arm, "feedback_snapshot") else False)
        if connected and hasattr(arm, "model_position_limits"):
            robot[side]["model_position_limits_rad"] = dict(
                min=arm.model_position_limits(False), max=arm.model_position_limits(True),
                source="controller model after hardware limit synchronization")
        robot[side]["gripper_position_limits_rad"] = (
            arm.gripper_position_limits() if connected and hasattr(arm, "gripper_position_limits") else None)
        if connected and hasattr(arm, "initialization_status"):
            robot[side]["initialization_interface_results"] = arm.initialization_status()
    mapping = None if pre is None else dict(
        home_position_m=pre.home_position, home_rotation=pre.home_rotation,
        translation_gain=pre.translation_gain,
        quest_to_world=QUEST_YUP_TO_WORLD_ZUP,
        world_to_base_left=R_x7_from_quest_left, world_to_base_right=R_x7_from_quest_right,
        grip_range_rad=np.deg2rad([pre.grip_min, pre.grip_max]),
        grip_input_distance_m=[0.02, 0.08], grip_missing_landmarks_distance_m=0.05,
        thumb_index=pre.thumb_idx, index_index=pre.index_idx)
    configs = dict(robot_yaml=hw, links_csv=config_snapshot(link_file))
    if getattr(camera, "calibration_path", None):
        configs["camera_calibration"] = config_snapshot(camera.calibration_path)
    return json_value(dict(robot_model="CRANE-X7", robots=robot, units=dict(joint="rad", gripper="rad", translation="m"),
        joint_order=[f"joint{j}" for j in range(1, 8)], gripper_dynamixel_id=9,
        code=code_version(root), configuration_files=configs, mapping=mapping,
        ik=dict(position_branch="analytic positive acos before axis sign inversion", joint3_fixed_rad=0.0,
            wrist_branch_selection="feasible minimum squared distance to command cache",
            singular_epsilon=1e-8, singular_s_threshold=0.17,
            max_step_rad=[0.08]*4+[0.06]*3, step_limit_basis="per solve, not per second",
            singular_continuity_enabled=True),
        controller=dict(baudrate=3000000, requested_background_cycle_ms=10,
            startup_profile_acceleration_rad_s2=np.pi, startup_profile_velocity_rad_s=np.pi,
            startup_position_pid=[800, 0, 0], startup_settings_readback=False,
            submission_semantics="updates library goal cache; no acknowledgement of physical execution",
            recording_added_serial_reads=0),
        camera=dict(device=camera.device, raw_sbs_size=[camera.width, camera.height],
            capture_backend=getattr(camera, "capture_backend", "unknown"),
            capture_checks=getattr(camera.capture, "stats", {}).get("payload_check"),
            raw_layout_requested=args.camera_layout, raw_layout_resolved=camera.resolved_layout,
            recorded_eye_order="physical left / physical right as resolved by camera split",
            rectified=camera.rectifier is not None, rectify_alpha=args.rectify_alpha,
            stream_eye_size=[args.stream_eye_width,args.stream_eye_height],
            resize="OpenCV INTER_AREA when reducing, INTER_LINEAR when enlarging",
            recording_before_sbs_padding=True, capture_device_timestamp_available=False,
            headset_display_time_measured=False),
        arguments=vars(args)))


class DecisionRecorder:
    def __init__(self, recorder, mode):
        self.recorder, self.mode = recorder, mode
        self.step = 0
        self.sequences = {}
        self.references = {}

    def begin(self, arms, capture, hands):
        observation = read_feedback(arms, self.sequences)
        now = time.monotonic_ns()
        for kind in observation.values():
            for entry in kind.values():
                if entry["timestamp_ns"] is not None:
                    entry["age_ns"] = now - entry["timestamp_ns"]
                    from episode_recording import FEEDBACK_MAX_AGE_NS
                    if entry["valid"] and entry["age_ns"] > FEEDBACK_MAX_AGE_NS:
                        entry.update(value=None, valid=False, status="stale")
        observation["image"] = image_observations(capture, now)
        sample = dict(step_index=self.step, decision_timestamp_ns=now, observation=observation,
            action={kind:{side:no_action("not_connected" if arms[side] is None else "no_command")
                          for side in ("left","right")} for kind in ("joint","gripper")},
            references={})
        if self.mode == "ik_analysis":
            sample["diagnostics"] = {}
            for side in ("left", "right"):
                h = hands.get(side, {})
                stamp = h.get("timestamp_ns")
                sample["diagnostics"][side] = dict(wrist_input_pose=h.get("wrist"),
                    input_timestamp_ns=stamp, input_sequence=h.get("sequence"),
                    input_age_ns=now-stamp if stamp is not None else None,
                    input_time_source="host HAND_MOVE receipt", target_ee_pose=None,
                    status="not_called", q_ik_raw=None, q_ik_processed=None,
                    action_path=f"action.joint.{side}")
        self.step += 1
        return sample

    def reference(self, sample, side, pre, command):
        info = pre.reference_info(side)
        if info is None:
            return
        previous = self.references.get(side)
        if previous is None or previous["version"] != info["version"]:
            info.update(effective_step_index=sample["step_index"],
                        effective_timestamp_ns=sample["decision_timestamp_ns"])
            self.references[side] = json_value(info)
        sample["references"][side] = self.references[side]
        if "diagnostics" in sample:
            sample["diagnostics"][side]["reference_version"] = info["version"]
            sample["diagnostics"][side]["target_ee_pose"] = command.T_ee if command is not None else None


def drive_recorded(arm, side, commands, sample, last_action_log):
    if arm is None or side not in commands:
        return
    command = commands[side]
    T = np.array(command.T_ee, dtype=np.float64, order="F")
    grip = submit_action(lambda: arm.write_gripper(9, command.grip_angle_deg),
                         float(command.grip_angle_deg * np.pi / 180.0))
    sample["action"]["gripper"][side] = grip
    if not grip["interface_accepted"]:
        print(f"write_gripper {grip['status']} for {side} arm")
    q = None
    start = time.monotonic_ns()
    try:
        if "diagnostics" in sample:
            q, diag = arm.solve_ik_with_diagnostics(T)
            sample["diagnostics"][side].update(diag)
        else:
            q = arm.solve_ik(T)
    except Exception as exc:
        print(f"solve_ik failed for {side} arm: {exc}")
        sample["action"]["joint"][side]["status"] = "ik_exception"
        if "diagnostics" in sample:
            sample["diagnostics"][side].update(status="exception", error=str(exc),
                q_ik_raw=None, q_ik_processed=None, started_ns=start, ended_ns=time.monotonic_ns())
    if q is not None:
        action = submit_action(lambda: arm.set_positions(q), q)
        sample["action"]["joint"][side] = action
        if not action["interface_accepted"]:
            print(f"set_positions {action['status']} for {side} arm")
            return
        now = time.monotonic()
        if now - last_action_log[side] >= 0.5:
            print(f"[ACTION] {side}: xyz={np.round(T[:3,3],3).tolist()}, "
                  f"grip={command.grip_angle_deg:.1f} deg, q={np.round(q,3).tolist()}")
            last_action_log[side] = now
