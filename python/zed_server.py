# Serve WebRTC signaling and stream queued stereo frames.
# Use the requested frame rate to schedule RTP timestamps.

import asyncio
import fractions
import json
import queue
import time

from aiohttp import web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame
import numpy as np

VIDEO_CLOCK_RATE = 90_000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class ZedVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, queue, fps):
        super().__init__()
        self.img_queue = queue
        self.fps = max(1, int(fps))
        self.frame_ticks = round(VIDEO_CLOCK_RATE / self.fps)
        self._timestamp = None
        self._started_at = None
        self._reported_first_frame = False

    async def _next_timestamp(self):
        if self._timestamp is None:
            self._timestamp = 0
            self._started_at = time.monotonic()
        else:
            self._timestamp += self.frame_ticks
            target = self._started_at + self._timestamp / VIDEO_CLOCK_RATE
            await asyncio.sleep(max(0.0, target - time.monotonic()))
        return self._timestamp, VIDEO_TIME_BASE

    async def _latest_frame(self):
        frame = None
        while frame is None:
            try:
                frame = self.img_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.002)

        while True:
            try:
                frame = self.img_queue.get_nowait()
            except queue.Empty:
                return frame

    async def recv(self):
        timestamp, time_base = await self._next_timestamp()
        frame = await self._latest_frame()
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24").reformat(
            format="yuv420p"
        )
        av_frame.pts = timestamp
        av_frame.time_base = time_base
        if not self._reported_first_frame:
            print(
                "[WebRTC] first camera frame: "
                f"shape={frame.shape}, strides={frame.strides}, "
                f"contiguous={frame.flags['C_CONTIGUOUS']}, "
                f"av={av_frame.width}x{av_frame.height}/"
                f"{av_frame.format.name}, "
                f"planes={[plane.line_size for plane in av_frame.planes]}, "
                f"fps={self.fps}"
            )
            self._reported_first_frame = True
        return av_frame


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


class RTC():
    def __init__(
        self,
        img_shape,
        img_queue,
        fps,
        video_codec="VP8",
    ) -> None:
        self.img_shape = img_shape
        self.img_queue = img_queue
        self.fps = fps
        self.video_codec = video_codec.upper()

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        print(
            "[WebRTC] offer received: "
            f"codec={self.video_codec}, frame={self.img_shape[1] * 2}x"
            f"{self.img_shape[0]}@{self.fps}"
        )

        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[WebRTC] peer connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
            if pc.connectionState in ("failed", "closed"):
                pcs.discard(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"[WebRTC] ICE state: {pc.iceConnectionState}")

        zed_track = ZedVideoTrack(self.img_queue, self.fps)
        video_sender = pc.addTrack(zed_track)
        force_codec(pc, video_sender, f"video/{self.video_codec}")

        try:
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception:
            await pc.close()
            pcs.discard(pc)
            raise

        candidates = [line for line in pc.localDescription.sdp.splitlines()
                      if line.startswith("a=candidate:")]
        print(f"[WebRTC] local ICE (no STUN/TURN): {candidates}")
        print("[WebRTC] media needs a reachable UDP route; adb reverse TCP does not tunnel it.")

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )


pcs = set()

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
