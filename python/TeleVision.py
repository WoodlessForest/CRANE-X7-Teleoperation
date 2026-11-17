# Run the Vuer interface and video services in separate processes.
# Share wrist poses and hand landmarks with the teleoperation loop.

import os
import asyncio
from time import monotonic_ns
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import Process, Array

import numpy as np

from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import Hands, ImageBackground, WebRTCStereoVideoPlane, DefaultScene

from vuer_hand import decode_hand_payload, payload_shapes
from zed_server import RTC, on_shutdown

DEFAULT_VUER_PORT = 8765
DEFAULT_WEBRTC_PORT = 8080
EXPECTED_VUER_VERSION = "0.0.61"

def _run_webrtc_app(
    img_shape,
    queue,
    webrtc_port,
    stream_fps,
    video_codec,
):
    from aiohttp import web
    import aiohttp_cors

    rtc = RTC(
        img_shape,
        queue,
        stream_fps,
        video_codec=video_codec,
    )
    app = web.Application()
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        },
    )
    app.on_shutdown.append(on_shutdown)
    cors.add(app.router.add_post("/offer", rtc.offer))

    web.run_app(app, host="0.0.0.0", port=int(webrtc_port))


def _run_vuer_process(
    vuer_port,
    webrtc_port,
    left_hand_shared,
    right_hand_shared,
    left_landmarks_shared,
    right_landmarks_shared,
    display_aspect,
    plane_height,
    plane_distance,
    enable_tracking,
    input_stamps=None,
    video_transport="webrtc",
    img_queue=None,
    stream_fps=30,
    jpeg_quality=80,
):
    debug_hand = os.environ.get("TELEVISION_DEBUG_HAND", "0") not in ("0", "false", "False", "")
    last_dbg_t = 0.0
    reported_hand_protocols = set()
    reported_invalid_hand_payloads = set()

    app = Vuer(
        domain=f"http://localhost:{int(vuer_port)}/",
        host="0.0.0.0",
        port=int(vuer_port),
        queries=dict(grid=False),
        queue_len=3,
    )
    jpeg_stream = None
    if video_transport == "websocket":
        from video_stream import LatestStereoJpeg
        jpeg_stream = LatestStereoJpeg(img_queue, stream_fps, jpeg_quality)

    def _write_mat16(shared_arr, mat16):
        arr = np.asarray(mat16, dtype=np.float64).flatten()
        if arr.size < 16:
            return
        with shared_arr.get_lock():
            for i in range(16):
                shared_arr[i] = float(arr[i])

    def _write_lms25(shared_arr, lms):
        pts = np.asarray(lms, dtype=np.float64).reshape(-1, 3)
        n = min(pts.shape[0], 25)
        flat = pts[:n].reshape(-1)
        with shared_arr.get_lock():
            for i in range(n * 3):
                shared_arr[i] = float(flat[i])
            for i in range(n * 3, 25 * 3):
                shared_arr[i] = 0.0

    @app.add_handler("HAND_MOVE")
    async def _hand_move_handler(event: ClientEvent, session):
        nonlocal last_dbg_t
        try:
            received_ns = monotonic_ns() if input_stamps is not None else None
            v = event.value if isinstance(event.value, dict) else {}
            left_sample = decode_hand_payload(v, "left")
            right_sample = decode_hand_payload(v, "right")

            for side, sample, wrist, landmarks in (
                ("left", left_sample, left_hand_shared, left_landmarks_shared),
                ("right", right_sample, right_hand_shared, right_landmarks_shared),
            ):
                if sample is None:
                    continue
                if input_stamps is None:
                    _write_mat16(wrist, sample.wrist_column_major)
                    _write_lms25(landmarks, sample.landmarks)
                else:
                    stamp = input_stamps[side]
                    with stamp.get_lock():
                        _write_mat16(wrist, sample.wrist_column_major)
                        _write_lms25(landmarks, sample.landmarks)
                        stamp[0] = received_ns
                        stamp[1] += 1

            for side, sample in (
                ("left", left_sample),
                ("right", right_sample),
            ):
                report_key = (side, getattr(sample, "source", None))
                if sample is not None and report_key not in reported_hand_protocols:
                    reported_hand_protocols.add(report_key)
                    print(
                        f"[TeleVision] {side} HAND_MOVE decoded as "
                        f"{sample.source}; wrist=column-major 4x4, "
                        "landmarks=25x3 metres"
                    )
                candidate_keys = (
                    side,
                    f"{side}Hand",
                    f"{side}_hand",
                    f"{side}Wrist",
                    f"{side}_wrist",
                    f"{side}Landmarks",
                    f"{side}_landmarks",
                )
                has_payload = any(v.get(key) is not None for key in candidate_keys)
                if (
                    sample is None
                    and has_payload
                    and side not in reported_invalid_hand_payloads
                ):
                    reported_invalid_hand_payloads.add(side)
                    print(
                        f"[TeleVision] WARNING: rejected {side} HAND_MOVE "
                        f"before IK; shapes={payload_shapes(v)}. Expected "
                        "Vuer 0.0.61: 25 column-major 4x4 matrices in metres."
                    )

            if debug_hand:
                import time

                now = time.time()
                if now - last_dbg_t > 2.0:
                    last_dbg_t = now
                    print(
                        "[HAND_MOVE protocol] "
                        f"shapes={payload_shapes(v)}, "
                        f"left={getattr(left_sample, 'source', None)}, "
                        f"right={getattr(right_sample, 'source', None)}"
                    )

        except Exception as e:
            if debug_hand:
                print("[HAND_MOVE handler error]", repr(e))

    @app.spawn(start=False)
    async def main(session):
        session.set @ DefaultScene(frameloop="always")
        if enable_tracking:
            session.upsert @ Hands(
                fps=60,
                stream=True,
                key="hands",
                disableLeft=False,
                disableRight=False,
                hideLeft=False,
                hideRight=False,
                scale=1.0,
            )
        if jpeg_stream is not None:
            sequence = 0
            while session.CURRENT_WS_ID in app.ws:
                current, images = await jpeg_stream.get()
                if images is not None and current != sequence:
                    session.upsert([
                        ImageBackground(
                            src=data, key=f"zed-{side}", layers=layer,
                            aspect=float(display_aspect), height=float(plane_height),
                            distanceToCamera=float(plane_distance),
                            position=[0, 0, 0], fixed=False,
                            interpolate=True, material={"toneMapped": False},
                        )
                        for data, side, layer in zip(images, ("left", "right"), (1, 2))
                    ], to="bgChildren")
                    sequence = current
                await asyncio.sleep(1.0 / stream_fps)
            return

        session.upsert(
            WebRTCStereoVideoPlane(
                src=f"http://localhost:{int(webrtc_port)}/offer",
                key="zed-bg",
                layout="stereo-left-right",
                aspect=float(display_aspect),
                distanceToCamera=float(plane_distance),
                height=float(plane_height),
                position=[0, 0, 0],
                fixed=False,
                side=2,
                iceServer=None,
                rtcOptions={"iceServers": []},
            ),
            to="bgChildren",
        )
        while True:
            await asyncio.sleep(1.0)

    app.run()


class OpenTeleVision:
    def __init__(
        self,
        img_shape=(960, 640, 3),
        queue=None,
        vuer_port: int = DEFAULT_VUER_PORT,
        webrtc_port: int = DEFAULT_WEBRTC_PORT,
        stream_fps: int = 30,
        video_codec: str = "VP8",
        enable_tracking: bool = False,
        display_aspect: float = 16.0 / 9.0,
        plane_height: float = 2.4,
        plane_distance: float = 1.0,
        record_inputs: bool = False,
        video_transport: str = "webrtc",
        jpeg_quality: int = 80,
    ):
        self.eye_h, self.eye_w = img_shape[0], img_shape[1]
        self.queue = queue
        self.vuer_port = vuer_port
        self.webrtc_port = webrtc_port
        self.stream_fps = max(1, int(stream_fps))
        self.video_codec = video_codec.upper()
        if video_transport not in ("webrtc", "websocket"):
            raise ValueError("video_transport must be 'webrtc' or 'websocket'")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.video_transport = video_transport
        if self.video_codec not in ("H264", "VP8"):
            raise ValueError("video_codec must be 'H264' or 'VP8'.")
        self.enable_tracking = bool(enable_tracking)
        self.display_aspect = float(display_aspect)
        self.plane_height = float(plane_height)
        self.plane_distance = float(plane_distance)
        try:
            self.vuer_version = version("vuer")
        except PackageNotFoundError:
            self.vuer_version = "unknown"
        if self.vuer_version != EXPECTED_VUER_VERSION:
            print(
                "[TeleVision] WARNING: expected Vuer "
                f"{EXPECTED_VUER_VERSION}, current={self.vuer_version}. "
                "HAND_MOVE and video schemas target the expected version."
            )

        self.left_hand_shared = Array("d", 16, lock=True)
        self.right_hand_shared = Array("d", 16, lock=True)
        self.left_landmarks_shared = Array("d", 25 * 3, lock=True)
        self.right_landmarks_shared = Array("d", 25 * 3, lock=True)

        self.input_stamps = {s: Array("q", 2, lock=True) for s in ("left", "right")} if record_inputs else None

        self.webrtc_process = None
        if video_transport == "webrtc":
            self.webrtc_process = Process(
                target=_run_webrtc_app,
                args=(
                    (self.eye_h, self.eye_w, 3), self.queue,
                    self.webrtc_port, self.stream_fps, self.video_codec,
                ),
            )
            self.webrtc_process.daemon = True
            self.webrtc_process.start()
            print(f"[TeleVision] webrtc server on http://0.0.0.0:{self.webrtc_port}")
        else:
            print("[TeleVision] stereo JPEG over Vuer WebSocket; USB adb reverse tcp:8765 tcp:8765")

        self.vuer_process = Process(
            target=_run_vuer_process,
            args=(
                self.vuer_port,
                self.webrtc_port,
                self.left_hand_shared,
                self.right_hand_shared,
                self.left_landmarks_shared,
                self.right_landmarks_shared,
                self.display_aspect,
                self.plane_height,
                self.plane_distance,
                self.enable_tracking,
                self.input_stamps,
                video_transport,
                self.queue if video_transport == "websocket" else None,
                self.stream_fps,
                jpeg_quality,
            ),
        )
        self.vuer_process.daemon = True
        self.vuer_process.start()
        print(f"[TeleVision] vuer frontend on http://0.0.0.0:{self.vuer_port}")
        print(
            "[TeleVision] immersive stereo plane: "
            f"aspect={self.display_aspect:.3f}, "
            f"height={self.plane_height:.1f}, "
            f"distance={self.plane_distance:.1f}"
        )
        print(
            f"[TeleVision] Quest local client (Vuer {self.vuer_version}, "
            f"transport={video_transport}): "
            f"http://localhost:{self.vuer_port}/"
            f"?ws=ws%3A%2F%2Flocalhost%3A{self.vuer_port}&grid=False"
        )

    @staticmethod
    def _mat_from_shared(shared_arr):
        with shared_arr.get_lock():
            arr = np.frombuffer(shared_arr.get_obj(), dtype="d").copy()
        if float(np.sum(np.abs(arr))) < 1e-9:
            return None
        return np.array(arr, dtype=np.float64).reshape((4, 4), order="F")

    @staticmethod
    def _lms_from_shared(shared_arr):
        with shared_arr.get_lock():
            arr = np.frombuffer(shared_arr.get_obj(), dtype="d").copy()
        if float(np.sum(np.abs(arr))) < 1e-9:
            return None
        return np.array(arr, dtype=np.float64).reshape((25, 3))

    @property
    def left_hand(self):
        try:
            return self._mat_from_shared(self.left_hand_shared)
        except Exception:
            return None

    @property
    def right_hand(self):
        try:
            return self._mat_from_shared(self.right_hand_shared)
        except Exception:
            return None

    @property
    def left_landmarks(self):
        try:
            return self._lms_from_shared(self.left_landmarks_shared)
        except Exception:
            return None

    @property
    def right_landmarks(self):
        try:
            return self._lms_from_shared(self.right_landmarks_shared)
        except Exception:
            return None

    def hand_snapshot(self, side):
        if self.input_stamps is None:
            return dict(wrist=getattr(self, f"{side}_hand"),
                        landmarks=getattr(self, f"{side}_landmarks"),
                        timestamp_ns=None, sequence=None)
        stamp = self.input_stamps[side]
        with stamp.get_lock():
            return dict(wrist=getattr(self, f"{side}_hand"),
                        landmarks=getattr(self, f"{side}_landmarks"),
                        timestamp_ns=stamp[0] or None, sequence=stamp[1] or None)

    def close(self):
        try:
            if getattr(self, "webrtc_process", None) and self.webrtc_process.is_alive():
                self.webrtc_process.terminate()
        except Exception:
            pass
        try:
            if getattr(self, "vuer_process", None) and self.vuer_process.is_alive():
                self.vuer_process.terminate()
        except Exception:
            pass

    def __del__(self):
        self.close()
