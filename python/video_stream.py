# Encode stereo images as paired JPEGs for WebSocket delivery.
# Consume the latest queued frame and share each encoded pair across sessions.

import asyncio
import queue
import time

import cv2
import numpy as np


def encode_stereo_jpeg(frame, quality):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[1] % 2:
        raise ValueError("Expected an even-width BGR stereo frame")
    images = []
    for eye in np.split(frame, 2, axis=1):
        ok, encoded = cv2.imencode(".jpg", eye, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Stereo JPEG encoding failed")
        images.append(encoded.tobytes())
    return tuple(images)


class LatestStereoJpeg:


    def __init__(self, frame_queue, fps, quality):
        self.queue = frame_queue
        self.period = 1 / fps
        self.quality = quality
        self.lock = asyncio.Lock()
        self.updated = float("-inf")
        self.sequence = 0
        self.images = None

    async def get(self):
        async with self.lock:
            if time.monotonic() - self.updated < self.period:
                return self.sequence, self.images
            frame = None
            while True:
                try:
                    frame = self.queue.get_nowait()
                except queue.Empty:
                    break
            if frame is not None:
                self.images = await asyncio.to_thread(encode_stereo_jpeg, frame, self.quality)
                self.sequence += 1
            self.updated = time.monotonic()
            return self.sequence, self.images
