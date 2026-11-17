# Capture YUYV frames through Linux V4L2 memory-mapped buffers.
# Optionally reject driver errors, short frames, and zero-filled image data.

import errno
import mmap
import os
import select
import struct
import sys
import time

import cv2
import numpy as np

V4L2_BUF_FLAG_ERROR = 0x40
YUYV = int.from_bytes(b"YUYV", "little")
ZERO_FILL_PROBE = bytes(1024)


def payload_integrity_issue(payload):

    offset = payload.find(ZERO_FILL_PROBE)
    if offset < 0:
        return None
    end = offset + len(ZERO_FILL_PROBE)
    while end < len(payload) and payload[end] == 0:
        end += 1
    return {"reason": "suspected_zero_filled_payload", "offset": offset,
            "zero_bytes": end - offset}


def _buffer(index=0):
    value = bytearray(88)
    struct.pack_into("=II", value, 0, index, 1)
    struct.pack_into("=I", value, 60, 1)
    return value


class CheckedV4L2Capture:


    def __init__(self, device, width, height, fps, timeout=2.0, filter_bad_frames=False):
        if not sys.platform.startswith("linux") or struct.calcsize("P") != 8 or sys.byteorder != "little":
            raise RuntimeError("Checked V4L2 requires a 64-bit little-endian Linux ABI")
        self.fd = None
        self.maps = []
        self.streaming = False
        self.filter_bad_frames = filter_bad_frames
        self.timeout = timeout
        self.received = self.discarded = 0
        self.discard_reasons = {}
        self.last_frame_info = None
        self.last_report = float("-inf")
        self.last_bad = None
        self.width, self.height = width, height
        self.fps = fps
        try:
            self.fd = os.open(str(device), os.O_RDWR | os.O_NONBLOCK)
            fmt = bytearray(208)
            struct.pack_into("=I", fmt, 0, 1)
            struct.pack_into("=IIII", fmt, 8, width, height, YUYV, 1)
            self._ioctl(5, fmt)
            actual_w, actual_h, fourcc, _, stride, size = struct.unpack_from("=6I", fmt, 8)
            if (actual_w, actual_h, fourcc) != (width, height, YUYV):
                raise RuntimeError(f"V4L2 rejected YUYV {width}x{height}: got {actual_w}x{actual_h}, fourcc={fourcc}")
            if stride < width * 2 or size < stride * height:
                raise RuntimeError(f"Invalid V4L2 frame layout: stride={stride}, size={size}")
            self.stride = stride
            self.expected_bytes = stride * height
            parm = bytearray(204)
            struct.pack_into("=I", parm, 0, 1)
            struct.pack_into("=II", parm, 12, 1, fps)
            self._ioctl(22, parm)
            numerator, denominator = struct.unpack_from("=II", parm, 12)
            self.fps = denominator / numerator if numerator else 0
            req = bytearray(20)
            struct.pack_into("=III", req, 0, 4, 1, 1)
            self._ioctl(8, req)
            count = struct.unpack_from("=I", req)[0]
            if not 1 <= count <= 32:
                raise RuntimeError(f"Invalid V4L2 buffer count: {count}")
            for index in range(count):
                buf = _buffer(index)
                self._ioctl(9, buf)
                offset = struct.unpack_from("=I", buf, 64)[0]
                length = struct.unpack_from("=I", buf, 72)[0]
                if length < self.expected_bytes:
                    raise RuntimeError("V4L2 buffer is shorter than a complete frame")
                self.maps.append(mmap.mmap(self.fd, length, flags=mmap.MAP_SHARED,
                                          prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=offset))
                self._ioctl(15, buf)
            self._ioctl(18, bytearray(struct.pack("=I", 1)), direction=1)
            self.streaming = True
        except Exception:
            self.release()
            raise

    def _ioctl(self, number, value, direction=3):
        import fcntl
        request = (direction << 30) | (len(value) << 16) | (ord("V") << 8) | number
        while True:
            try:
                fcntl.ioctl(self.fd, request, value, True)
                return
            except InterruptedError:
                continue

    @property
    def stats(self):
        return {"received": self.received, "discarded": self.discarded,
                "discard_reasons": dict(self.discard_reasons),
                "payload_check": "zero_fill_1024_v1" if self.filter_bad_frames else "disabled", "last_bad": self.last_bad}

    def _reject(self, info, issue):
        self.discarded += 1
        reason = issue["reason"]
        self.discard_reasons[reason] = self.discard_reasons.get(reason, 0) + 1
        self.last_bad = {**info, **issue}
        now = time.monotonic()
        if now - self.last_report >= 5:
            print(f"[UVC] discarded corrupt/suspect frame: {self.last_bad}; "
                  f"total={self.discarded}/{self.received}", flush=True)
            self.last_report = now

    def _take(self):
        buf = _buffer()
        self._ioctl(17, buf)
        index, _, used, flags = struct.unpack_from("=4I", buf)
        sequence = struct.unpack_from("=I", buf, 56)[0]
        if index >= len(self.maps):
            raise RuntimeError(f"V4L2 returned invalid buffer index {index}")
        self.received += 1
        info = {"sequence": sequence, "flags": hex(flags), "bytesused": used,
                "expected": self.expected_bytes, "received_timestamp_ns": time.monotonic_ns(),
                "payload_check": "zero_fill_1024_v1" if self.filter_bad_frames else "disabled"}
        try:
            if self.filter_bad_frames and (flags & V4L2_BUF_FLAG_ERROR or used != self.expected_bytes):
                self._reject(info, {"reason": "driver_error" if flags & V4L2_BUF_FLAG_ERROR else "payload_size"})
                return None
            payload = self.maps[index][:self.expected_bytes]
            issue = payload_integrity_issue(payload) if self.filter_bad_frames else None
            if issue is not None:
                self._reject(info, issue)
                return None
            self.last_frame_info = info
            return payload
        finally:
            self._ioctl(15, _buffer(index))

    def read(self):
        deadline = time.monotonic() + self.timeout
        latest = None
        drained = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"No complete UVC frame within {self.timeout}s; {self.stats}")
            wait = 0 if latest is not None else remaining
            try:
                ready, _, _ = select.select([self.fd], [], [], wait)
            except InterruptedError:
                continue
            if not ready:
                if latest is not None:
                    break
                raise RuntimeError(f"UVC capture timed out; {self.stats}")
            try:
                frame = self._take()
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    continue
                raise
            if frame is not None:
                latest = frame
            drained += 1
            if latest is not None and drained >= len(self.maps):
                break
        raw = np.frombuffer(latest, np.uint8).reshape(self.height, self.stride)
        packed = raw[:, :self.width * 2].reshape(self.height, self.width, 2)
        return True, cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUYV)

    def release(self):
        if self.fd is None:
            return
        try:
            if self.streaming:
                try:
                    self._ioctl(19, bytearray(struct.pack("=I", 1)), direction=1)
                except OSError:
                    pass
        finally:
            self.streaming = False
            for mapping in self.maps:
                mapping.close()
            self.maps.clear()
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
