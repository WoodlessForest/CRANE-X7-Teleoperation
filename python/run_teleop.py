# Coordinate stereo capture, VR streaming, optional arm control, and recording.
# Robot connections are enabled only when --with_arms is supplied.

import argparse
import glob
import os
import platform
import queue
import time
from multiprocessing import Queue

import cv2
import numpy as np

from Preprocessor_arm import VuerArmPreprocessor
from zed_rectify import (
    ZEDStereoRectifier,
    detect_stereo_layout,
    find_or_download_calibration,
)


try:
    import py_x7
    X7_AVAILABLE = True
except Exception as e:
    X7_AVAILABLE = False
    X7_IMPORT_ERR = str(e)

def sbs_concat(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    assert left_bgr.shape == right_bgr.shape
    return np.concatenate([left_bgr, right_bgr], axis=1)


def align_video_frame(frame: np.ndarray, alignment: int = 16) -> np.ndarray:
    height, width = frame.shape[:2]
    bottom = (-height) % alignment
    right = (-width) % alignment
    if not bottom and not right:
        return np.ascontiguousarray(frame)
    return cv2.copyMakeBorder(
        frame,
        0,
        bottom,
        0,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def resize_eye_for_stream(
    eye_bgr: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if eye_bgr.shape[:2] == (height, width):
        return eye_bgr
    interpolation = (
        cv2.INTER_AREA
        if width < eye_bgr.shape[1] or height < eye_bgr.shape[0]
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        eye_bgr,
        (width, height),
        interpolation=interpolation,
    )


def put_latest(frame_queue: Queue, frame: np.ndarray):
    try:
        frame_queue.put_nowait(frame)
        return
    except queue.Full:
        pass

    try:
        frame_queue.get_nowait()
    except queue.Empty:
        return

    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass


def parse_camera_device(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def resolve_camera_device(device):
    if device != "auto":
        return device

    candidates = sorted(
        glob.glob("/dev/v4l/by-id/*ZED*video-index0")
    )
    if not candidates:
        candidates = sorted(glob.glob("/dev/video*"))

    selected = candidates[0]
    print(f"[INFO] Automatically selected camera: {selected}")
    return selected


class OpenCVStereoCamera:

    def __init__(
        self,
        device,
        width: int,
        height: int,
        fps: int,
        layout: str,
        fourcc: str,
        rectify: bool,
        zed_serial: str,
        calibration_file: str,
        rectify_alpha: float,
        record_timestamps: bool = False,
        capture_backend: str = "auto",
        filter_bad_frames: bool = False,
    ):
        self.device = resolve_camera_device(device)
        self.width = width
        self.height = height
        self.resolved_layout = None if layout == "auto" else layout
        self.rectifier = None
        self.calibration_path = None
        self.record_timestamps = record_timestamps
        self.capture_timestamp_ns = None

        if capture_backend not in ("auto", "v4l2", "opencv"):
            raise ValueError("capture_backend must be auto, v4l2, or opencv")
        use_checked = capture_backend == "v4l2" or (
            capture_backend == "auto" and platform.system() == "Linux" and fourcc in ("", "YUYV")
        )
        if filter_bad_frames and not use_checked:
            raise ValueError("--filter_bad_frames requires Linux YUYV with auto/v4l2 backend")
        self.capture_backend = "v4l2" if use_checked else "opencv"
        if use_checked:
            from v4l2_capture import CheckedV4L2Capture
            if fourcc not in ("", "YUYV"):
                raise ValueError("Checked V4L2 currently supports YUYV only")
            device_path = f"/dev/video{self.device}" if isinstance(self.device, int) else self.device
            self.capture = CheckedV4L2Capture(device_path, width, height, fps, filter_bad_frames=filter_bad_frames)
            actual_width, actual_height = self.capture.width, self.capture.height
            actual_fps, actual_fourcc = self.capture.fps, "YUYV"
        else:
            backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
            self.capture = cv2.VideoCapture(self.device, backend)
            if not self.capture.isOpened():
                self.capture.release()
                raise RuntimeError(f"OpenCV could not open camera {self.device!r}.")
            if fourcc:
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            fourcc_value = int(self.capture.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        print(
            f"[INFO] Camera opened (backend={self.capture_backend}): "
            f"device={self.device!r}, requested={width}x{height}@{fps}, "
            f"reported={actual_width}x{actual_height}@{actual_fps:.1f}, "
            f"fourcc={actual_fourcc!r}"
        )
        if rectify:
            calibration_path = find_or_download_calibration(
                zed_serial,
                calibration_file,
            )
            self.calibration_path = calibration_path
            self.rectifier = ZEDStereoRectifier(
                calibration_path,
                width=width // 2,
                height=height,
                alpha=rectify_alpha,
            )
            print(
                "[INFO] ZED OpenCV stereo rectification enabled: "
                f"SN={zed_serial}, mode={self.rectifier.resolution}, "
                f"alpha={rectify_alpha:.2f}, "
                f"calibration={calibration_path}"
            )
        else:
            print(
                "[WARN] ZED stereo rectification disabled; raw UVC eyes may "
                "show distortion or fail to fuse in VR."
            )

    def read(self):
        ok, frame = self.capture.read()
        if self.record_timestamps:
            self.capture_timestamp_ns = time.monotonic_ns()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read a frame from camera {self.device!r}.")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"camera returned an unsupported frame format: shape={frame.shape}")

        frame_height, frame_width = frame.shape[:2]
        if (frame_width, frame_height) != (self.width, self.height):
            raise RuntimeError(
                "camera did not use the requested ZED SBS mode: "
                f"expected {self.width}x{self.height}, got {frame_width}x{frame_height}."
            )
        if frame_width % 2:
            raise RuntimeError(f"SBS frame width must be even, got {frame_width}.")

        first, second = np.split(frame, 2, axis=1)
        if self.resolved_layout is None:
            self.resolved_layout, reason = detect_stereo_layout(first, second)
            print(
                "[INFO] Detected ZED UVC stereo layout: "
                f"{self.resolved_layout} ({reason})"
            )

        if self.resolved_layout == "right-left":
            right, left = first, second
        else:
            left, right = first, second
        if self.rectifier is not None:
            left, right = self.rectifier.rectify(left, right)
        return left, right

    def close(self):
        self.capture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--right_port", type=str, default="/dev/ttyUSB1")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested ZED UVC frame rate (positive integer, default: 30)",
    )
    parser.add_argument("--webrtc_port", type=int, default=8080)
    parser.add_argument(
        "--video_transport", choices=("webrtc", "websocket"), default="webrtc",
        help="webrtc: UDP video; websocket: JPEG over Vuer TCP/USB ADB, without ICE or public network access",
    )
    parser.add_argument("--jpeg_quality", type=int, default=80,
                        help="WebSocket JPEG quality, 1-100 (default: 80)")
    parser.add_argument("--filter_bad_frames", action="store_true",
                        help="Enable filtering of V4L2 error, short, and zero-filled frames (disabled by default)")
    parser.add_argument(
        "--stream_eye_width",
        type=int,
        default=640,
        help="WebRTC per-eye width (default: 640)",
    )
    parser.add_argument(
        "--stream_eye_height",
        type=int,
        default=360,
        help="WebRTC per-eye height (default: 360)",
    )
    parser.add_argument(
        "--stream_fps",
        type=int,
        default=30,
        help="Target video transmission frame rate (positive integer, default: 30)",
    )
    parser.add_argument(
        "--video_codec",
        choices=("H264", "VP8"),
        default="VP8",
        help="WebRTC video codec (default: VP8; H264 available for comparison)",
    )
    arm_mode = parser.add_mutually_exclusive_group()
    arm_mode.add_argument(
        "--no_arms",
        dest="no_arms",
        action="store_true",
        default=True,
        help="Stream video without connecting or controlling arms (default mode)",
    )
    arm_mode.add_argument(
        "--with_arms",
        dest="no_arms",
        action="store_false",
        help="Enable arm connections, hand tracking, and action computation",
    )
    parser.add_argument(
        "--arm_home_position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.35, 0.0, 0.30),
        help=(
            "Relative-control home position in the X7 base frame, in meters; "
            "X7 forward is +X (default: 0.35 0.0 0.30)"
        ),
    )
    parser.add_argument(
        "--arm_translation_gain",
        type=float,
        default=1.0,
        help="Scale from relative VR hand displacement to arm displacement (default: 1.0)",
    )
    parser.add_argument(
        "--camera_device",
        type=parse_camera_device,
        default="auto",
        help="OpenCV index, V4L2 path, or auto (default: auto, preferring ZED)",
    )
    parser.add_argument(
        "--camera_backend", choices=("auto", "v4l2", "opencv"), default="auto",
        help="auto: Linux YUYV backend with optional V4L2 frame checks; opencv: legacy backend for comparison",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=1344,
        help="Total stereo SBS width (default: 1344, 672 per eye)",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=376,
        help="Stereo SBS height (default: 376)",
    )
    parser.add_argument(
        "--camera_layout",
        choices=("auto", "right-left", "left-right"),
        default="auto",
        help="Raw UVC SBS layout (default: auto, detected from stereo disparity)",
    )
    parser.add_argument(
        "--camera_fourcc",
        type=str,
        default="",
        help="Optional V4L2 FOURCC, such as YUYV or MJPG",
    )
    rectify_mode = parser.add_mutually_exclusive_group()
    rectify_mode.add_argument(
        "--rectify",
        dest="rectify",
        action="store_true",
        default=True,
        help="Use ZED factory calibration for OpenCV stereo rectification (enabled by default)",
    )
    rectify_mode.add_argument(
        "--no_rectify",
        dest="rectify",
        action="store_false",
        help="Stream unrectified UVC stereo images for diagnostics",
    )
    parser.add_argument(
        "--zed_serial",
        type=str,
        default="10029943",
        help="ZED Mini serial number (default: 10029943)",
    )
    parser.add_argument(
        "--zed_calibration_file",
        type=str,
        default="",
        help="Path to ZED SN*.conf; leave empty to use the bundled factory calibration",
    )
    parser.add_argument(
        "--rectify_alpha",
        type=float,
        default=0.0,
        help="OpenCV rectification alpha: 0 removes black borders, 1 retains all pixels (default: 0)",
    )
    parser.add_argument(
        "--vr_display_aspect",
        type=float,
        default=16.0 / 9.0,
        help="Vuer per-eye plane aspect ratio (default: 16/9, matching TeleVision0730)",
    )
    parser.add_argument(
        "--vr_plane_height",
        type=float,
        default=2.4,
        help="Head-following video plane height (default: 2.4, matching TeleVision0730)",
    )
    parser.add_argument(
        "--vr_plane_distance",
        type=float,
        default=1.0,
        help="Head-following video plane distance (default: 1, matching TeleVision0730)",
    )
    parser.add_argument("--record_mode", choices=("off", "dataset", "ik_analysis"), default="off")
    parser.add_argument("--record_dir", help="Episode root directory, required when recording")
    parser.add_argument("--record_close_timeout", type=float, default=10.0,
                        help="Maximum recording drain/final verification wait in seconds")
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg_quality must be between 1 and 100")
    if args.record_mode != "off" and not args.record_dir:
        parser.error("--record_dir is required when recording is enabled")
    if not np.isfinite(args.record_close_timeout) or args.record_close_timeout <= 0:
        parser.error("--record_close_timeout must be a finite positive number")
    if args.record_mode != "off" and not args.no_arms and X7_AVAILABLE:
        if not hasattr(py_x7.Controller, "feedback_snapshot") or not hasattr(py_x7.Controller, "solve_ik_with_diagnostics"):
            parser.error("recording requires rebuilding the py_x7 extension; run scripts/build.sh first")

    if args.camera_fourcc and len(args.camera_fourcc) != 4:
        parser.error("--camera_fourcc must be empty or exactly 4 characters")
    if args.camera_width < 2 or args.camera_width % 2:
        parser.error("--camera_width must be a positive even number (total SBS width)")
    if args.camera_height < 1:
        parser.error("--camera_height must be greater than 0")
    if not 0.0 <= args.rectify_alpha <= 1.0:
        parser.error("--rectify_alpha must be between 0 and 1")
    if args.vr_display_aspect <= 0:
        parser.error("--vr_display_aspect must be greater than 0")
    if args.vr_plane_height <= 0:
        parser.error("--vr_plane_height must be greater than 0")
    if args.vr_plane_distance <= 0:
        parser.error("--vr_plane_distance must be greater than 0")
    if args.fps < 1:
        parser.error("--fps must be greater than 0")
    if args.stream_eye_width < 2 or args.stream_eye_width % 2:
        parser.error("--stream_eye_width must be a positive even number")
    if args.stream_eye_height < 2 or args.stream_eye_height % 2:
        parser.error("--stream_eye_height must be a positive even number")
    if args.stream_fps < 1:
        parser.error("--stream_fps must be greater than 0")
    if not np.isfinite(args.arm_home_position).all():
        parser.error("--arm_home_position must contain only finite numbers")
    if args.arm_translation_gain <= 0.0:
        parser.error("--arm_translation_gain must be greater than 0")

    recorder = None
    if args.record_mode != "off":
        from episode_recording import create_recorder
        from recording_session import DecisionRecorder, drive_recorded, runtime_metadata
        recorder = create_recorder(args.record_mode, args.record_dir,
            size=(args.stream_eye_width, args.stream_eye_height), fps=args.fps,
            metadata={"arguments": vars(args)})
    camera = tv = img_queue = None
    run_end_state, end_reason = "normal", "normal_exit"
    try:
        try:
            camera = OpenCVStereoCamera(
                device=args.camera_device,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.fps,
                layout=args.camera_layout,
                fourcc=args.camera_fourcc,
                rectify=args.rectify,
                zed_serial=args.zed_serial,
                calibration_file=args.zed_calibration_file,
                rectify_alpha=args.rectify_alpha,
                record_timestamps=recorder is not None,
                capture_backend=args.camera_backend,
                filter_bad_frames=args.filter_bad_frames,
            )
        except RuntimeError as exc:
            raise SystemExit(f"[FATAL] {exc}") from exc

        from TeleVision import OpenTeleVision

        img_queue: Queue = Queue(maxsize=2)
        encoded_width = (
            (args.stream_eye_width * 2 + 15) // 16
        ) * 16
        encoded_height = (
            (args.stream_eye_height + 15) // 16
        ) * 16
        if args.video_transport == "websocket":
            encoded_width = args.stream_eye_width * 2
            encoded_height = args.stream_eye_height
        codec_label = args.video_codec if args.video_transport == "webrtc" else f"JPEG quality={args.jpeg_quality}"
        print(
            f"[INFO] Vuer video transport={args.video_transport}: "
            f"SBS={args.stream_eye_width * 2}x{args.stream_eye_height}, "
            f"encoded={encoded_width}x{encoded_height}, "
            f"codec={codec_label}, target={args.stream_fps} FPS, "
            f"video aspect={args.stream_eye_width / args.stream_eye_height:.3f}, "
            f"VR display aspect={args.vr_display_aspect:.3f}"
        )

        tv = OpenTeleVision(
            img_shape=(encoded_height, encoded_width // 2, 3),
            queue=img_queue,
            webrtc_port=args.webrtc_port,
            stream_fps=args.stream_fps,
            video_codec=args.video_codec,
            enable_tracking=not args.no_arms,
            display_aspect=args.vr_display_aspect,
            plane_height=args.vr_plane_height,
            plane_distance=args.vr_plane_distance,
            record_inputs=recorder is not None,
            video_transport=args.video_transport,
            jpeg_quality=args.jpeg_quality,
        )

        is_windows = platform.system().lower().startswith("win")
        left_arm = None
        right_arm = None

        if args.no_arms:
            print("[INFO] --no_arms: arm connections and control are disabled.")
        elif X7_AVAILABLE and (not is_windows):
            def try_connect(port):
                if not os.path.exists(port):
                    print(f"[INFO] Port {port} does not exist; skipping connection.")
                    return None
                if not os.access(port, os.R_OK | os.W_OK):
                    print(
                        f"[ERROR] The current user lacks read/write permission for robot arm serial port {port}. "
                        f"Run first: sudo chmod 666 {port}"
                    )
                    return None
                try:
                    print(f"[INFO] Connecting to arm: {port} ...")
                    ctl = py_x7.Controller()
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    config_path = os.path.realpath(
                        os.path.join(base_path, "..", "cpp", "config", "crane-x7.yaml")
                    )
                    links_path = os.path.realpath(
                        os.path.join(
                            base_path,
                            "..",
                            "cpp",
                            "config",
                            "crane-x7_links.csv",
                        )
                    )
                    if not os.path.isfile(config_path) or not os.path.isfile(links_path):
                        print(
                            "[WARN] Robot arm configuration file does not exist: "
                            f"hw={config_path}, links={links_path}"
                        )
                        return None
                    print(
                        "[INFO] Arm configuration: "
                        f"hw={config_path}, links={links_path}"
                    )
                    connected = (ctl.connect(port, 3000000, config_path, links_path, record_feedback=True)
                                 if recorder is not None else ctl.connect(port, 3000000, config_path, links_path))
                    if connected:
                        print(f"[SUCCESS] Arm {port} connected successfully!")
                        return ctl
                    else:
                        print(f"[WARN] The C++ controller rejected the arm connection request for {port} (returned False).")

                except Exception as e:
                    print(f"[WARN] Failed to connect to {port}: {e}")
                return None

            left_arm  = try_connect(args.left_port)
            right_arm = try_connect(args.right_port)

            if left_arm is None:
                print("[WARN] Left arm is not connected")
            if right_arm is None:
                print("[WARN] Right arm is not connected")
        else:
            if not X7_AVAILABLE:
                print(f"[WARN] X7 control library is unavailable: {X7_IMPORT_ERR}")
            print("[WARN] Arm commands are disabled on this platform (Windows or missing control library)")

        pre = (
            None
            if args.no_arms
            else VuerArmPreprocessor(
                grip_min_deg=0.0,
                grip_max_deg=70.0,
                home_position=args.arm_home_position,
                translation_gain=args.arm_translation_gain,
            )
        )
        if pre is not None:
            print(
                "[RELATIVE] Relative pose control enabled: "
                "X7 forward=+X, "
                f"home xyz={np.round(pre.home_position, 3).tolist()} m, "
                f"translation gain={pre.translation_gain:.3f}. "
                "Position and rotation use wrist 4x4; landmarks control only the gripper. "
                "The first valid tracking frame for each hand is stored as its reference."
            )

        arms = {"left": left_arm, "right": right_arm}
        latest_capture = None
        if recorder is not None:
            try:
                recorder.update_metadata(runtime_metadata(args, camera, arms, pre))
            except Exception as exc:
                recorder._disable(f"runtime_metadata_failed: {exc}")
            decisions = DecisionRecorder(recorder, args.record_mode)

        last_dbg = 0.0
        last_action_log = {"left": 0.0, "right": 0.0}
        stream_stats_started = time.perf_counter()
        stream_stats_frames = 0
        stream_stats_delta = 0.0
        previous_stream_probe = None
        while True:
            left_frame, right_frame = camera.read()
            left_stream = resize_eye_for_stream(
                left_frame,
                args.stream_eye_width,
                args.stream_eye_height,
            )
            right_stream = resize_eye_for_stream(
                right_frame,
                args.stream_eye_width,
                args.stream_eye_height,
            )
            sbs = sbs_concat(left_stream, right_stream)
            stream_frame = align_video_frame(sbs) if args.video_transport == "webrtc" else sbs
            put_latest(img_queue, stream_frame)
            if recorder is not None:
                latest_capture = recorder.capture(left_stream, right_stream, camera.capture_timestamp_ns,
                    source=getattr(camera.capture, "last_frame_info", None))
                if recorder.metadata.get("resolved_camera_layout") != camera.resolved_layout:
                    recorder.update_metadata({"resolved_camera_layout": camera.resolved_layout,
                            "camera": {**recorder.metadata.get("camera", {}), "raw_layout_resolved": camera.resolved_layout}})

            stream_probe = left_stream[::16, ::16]
            if previous_stream_probe is not None:
                stream_stats_delta += float(
                    np.mean(
                        np.abs(
                            stream_probe.astype(np.int16)
                            - previous_stream_probe.astype(np.int16)
                        )
                    )
                )
            previous_stream_probe = stream_probe.copy()
            stream_stats_frames += 1
            stats_now = time.perf_counter()
            stats_elapsed = stats_now - stream_stats_started
            if stats_elapsed >= 5.0:
                comparisons = max(1, stream_stats_frames - 1)
                print(
                    "[STREAM] camera/input alive: "
                    f"{stream_stats_frames / stats_elapsed:.1f} FPS, "
                    f"frames={stream_stats_frames}, "
                    f"mean temporal delta="
                    f"{stream_stats_delta / comparisons:.2f}"
                )
                stream_stats_started = stats_now
                stream_stats_frames = 0
                stream_stats_delta = 0.0

            if args.no_arms:
                if recorder is not None:
                    recorder.sample(decisions.begin(arms, latest_capture, {}))
                continue

            if recorder is not None:
                hands = {side: tv.hand_snapshot(side) for side in ("left", "right")}
                Lw, Rw = hands["left"]["wrist"], hands["right"]["wrist"]
                Lk, Rk = hands["left"]["landmarks"], hands["right"]["landmarks"]
                sample = decisions.begin(arms, latest_capture, hands)
            else:
                Lw = tv.left_hand
                Rw = tv.right_hand
                Lk = tv.left_landmarks
                Rk = tv.right_landmarks
            detL = np.linalg.det(Lw[:3, :3]) if Lw is not None else 0.0
            detR = np.linalg.det(Rw[:3, :3]) if Rw is not None else 0.0
            sumL = float(np.sum(np.abs(Lk))) if Lk is not None else 0.0
            sumR = float(np.sum(np.abs(Rk))) if Rk is not None else 0.0

            now = time.time()
            if now - last_dbg > 1.0:
                print(f"[HAND DBG] detL={detL:.3e} detR={detR:.3e} | sumL={sumL:.3e} sumR={sumR:.3e}")
                last_dbg = now

            cmd = {}
            if Lw is not None:
                had_left_reference = pre.has_reference("left")
                left_cmd = pre.compute_side("left", Lw, Lk)
                if left_cmd is not None:
                    cmd["left"] = left_cmd
                if (
                    not had_left_reference
                    and pre.has_reference("left")
                ):
                    print(
                        "[RELATIVE] left hand reference pose locked; "
                        "the current hand pose corresponds to the arm home pose."
                    )
            if Rw is not None:
                had_right_reference = pre.has_reference("right")
                right_cmd = pre.compute_side("right", Rw, Rk)
                if right_cmd is not None:
                    cmd["right"] = right_cmd
                if (
                    not had_right_reference
                    and pre.has_reference("right")
                ):
                    print(
                        "[RELATIVE] right hand reference pose locked; "
                        "the current hand pose corresponds to the arm home pose."
                    )
            if recorder is not None:
                for side in ("left", "right"):
                    decisions.reference(sample, side, pre, cmd.get(side))
            if not cmd:
                if recorder is not None:
                    recorder.sample(sample)
                time.sleep(0.005)
                continue

            def drive(arm, which: str):
                if recorder is not None:
                    drive_recorded(arm, which, cmd, sample, last_action_log)
                    return
                if arm is None or which not in cmd:
                    return
                T = np.array(cmd[which].T_ee, dtype=np.float64, order="F")
                try:
                    gripper_ok = arm.write_gripper(
                        9,
                        cmd[which].grip_angle_deg,
                    )
                    if not gripper_ok:
                        print(f"write_gripper rejected for {which} arm")
                except Exception as e:
                    print(f"write_gripper failed for {which} arm: {e}")

                q = None
                try:
                    q = arm.solve_ik(T)
                except Exception as e:
                    print(f"solve_ik failed for {which} arm: {e}")

                if q is not None:
                    try:
                        position_ok = arm.set_positions(q)
                        if not position_ok:
                            print(f"set_positions rejected for {which} arm")
                            return
                        action_now = time.monotonic()
                        if action_now - last_action_log[which] >= 0.5:
                            print(
                                f"[ACTION] {which}: "
                                f"xyz={np.round(T[:3, 3], 3).tolist()}, "
                                f"grip={cmd[which].grip_angle_deg:.1f} deg, "
                                f"q={np.round(q, 3).tolist()}"
                            )
                            last_action_log[which] = action_now
                    except Exception as e:
                        print(f"set_positions failed for {which} arm: {e}")

            drive(left_arm,  "left")
            drive(right_arm, "right")
            if recorder is not None:
                recorder.sample(sample)

    except KeyboardInterrupt:
        run_end_state, end_reason = "interrupted", "KeyboardInterrupt"
    except BaseException as exc:
        run_end_state, end_reason = "exception", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if camera is not None:
                if recorder is not None:
                    recorder.update_metadata({"camera_capture_statistics": getattr(camera.capture, "stats", None)})
                camera.close()
            if tv is not None:
                tv.close()
            if img_queue is not None:
                img_queue.cancel_join_thread()
                img_queue.close()
        finally:
            if recorder is not None:
                recorder.close(run_end_state, end_reason, args.record_close_timeout)
        print("[INFO] Teleop stopped.")

if __name__ == "__main__":
    main()
