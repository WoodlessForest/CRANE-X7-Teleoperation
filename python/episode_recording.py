# Write episode metadata, samples, stereo video, and frame indexes.
# Keep recording work separate from the control loop using bounded queues.

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import re
import signal
import sqlite3
import time

import numpy as np

SCHEMA_VERSION = "1.0"
EYES = ("left", "right")
FEEDBACK_MAX_AGE_NS = 100_000_000
IMAGE_MAX_AGE_NS = 250_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def dumps(value):
    return json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def atomic_metadata(path, metadata):
    temporary = path / "metadata.json.tmp"
    temporary.write_text(dumps(metadata) + "\n", encoding="utf-8")
    temporary.replace(path / "metadata.json")


def allocate_episode(root):
    root = Path(root).expanduser().absolute()
    try:
        root.mkdir(parents=True, exist_ok=True)
        maximum = 0
        for child in root.iterdir():
            match = re.fullmatch(r"episode_([0-9]{6,})", child.name)
            if match and int(match[1]) > 0 and child.name == f"episode_{int(match[1]):06d}":
                maximum = max(maximum, int(match[1]))
        number = maximum + 1
        while True:
            path = root / f"episode_{number:06d}"
            try:
                path.mkdir()
                return path, number
            except FileExistsError:
                number += 1
    except OSError as exc:
        raise OSError(f"Cannot create recording episode under {root}: {exc}") from exc


def missing_observation(status, snapshot_ns=None):
    return dict(value=None, unit="rad", valid=False, status=status,
                timestamp_ns=None, age_ns=None, snapshot_timestamp_ns=snapshot_ns,
                timestamp_source="host_monotonic_position_read_complete",
                sequence=None, repeated=None, last_attempt_timestamp_ns=None)


def position_observation(snapshot, now_ns, *, count, previous_sequence=None):
    out = missing_observation("unavailable", now_ns)
    if not snapshot.get("enabled"):
        out["status"] = "feedback_disabled"
        return out
    stamp = snapshot.get("update_ns") or None
    seq = snapshot.get("sequence") or None
    age = now_ns - stamp if stamp is not None else None
    out.update(timestamp_ns=stamp, age_ns=age, sequence=seq,
               repeated=seq == previous_sequence if seq is not None else None,
               last_attempt_timestamp_ns=snapshot.get("attempt_ns") or None)
    if not snapshot.get("last_read_ok") and snapshot.get("attempt_ns"):
        out["status"] = "read_failed"
    elif not snapshot.get("updated"):
        out["status"] = "not_updated"
    elif age is None or age < 0:
        out["status"] = "clock_error"
    elif age > FEEDBACK_MAX_AGE_NS:
        out["status"] = "stale"
    else:
        values = snapshot.get("positions", [])
        if len(values) != count or not np.isfinite(values).all():
            out["status"] = "invalid_feedback"
        else:
            out.update(value=list(values) if count > 1 else float(values[0]),
                       valid=True, status="ok")
    return out


def read_feedback(arms, previous_sequences):
    result = {"joint": {}, "gripper": {}}
    for side, arm in arms.items():
        for kind, group, count in (("joint", "arm", 7), ("gripper", "hand", 1)):
            if arm is None:
                observation = missing_observation("not_connected", time.monotonic_ns())
            elif not hasattr(arm, "feedback_snapshot"):
                observation = missing_observation("unsupported_binding", time.monotonic_ns())
            else:
                try:
                    snapshot = arm.feedback_snapshot(group)
                    observation = position_observation(snapshot, time.monotonic_ns(), count=count,
                        previous_sequence=previous_sequences.get((side, kind)))
                    previous_sequences[side, kind] = observation["sequence"]
                except Exception as exc:
                    observation = missing_observation("snapshot_failed", time.monotonic_ns())
                    observation["error"] = str(exc)
            result[kind][side] = observation
    return result


def no_action(reason="no_command"):
    return dict(target=None, unit="rad", submitted=False, status=reason,
                submission_timestamp_ns=None, return_timestamp_ns=None,
                interface_accepted=None)


def submit_action(call, target):
    item = no_action()
    item.update(target=json_value(target), submitted=True,
                submission_timestamp_ns=time.monotonic_ns())
    try:
        accepted = call()
        item.update(interface_accepted=bool(accepted),
                    status="accepted" if accepted else "rejected")
    except Exception as exc:
        item.update(status="exception", error=str(exc))
    item["return_timestamp_ns"] = time.monotonic_ns()
    return item


def image_observations(capture, decision_ns):
    result = {}
    for eye in EYES:
        if capture is None:
            result[eye] = dict(capture_id=None, eye=eye, valid=False,
                               status="unavailable", timestamp_ns=None, image_age_ns=None)
            continue
        age = decision_ns - capture["capture_timestamp_ns"]
        available = capture["available_timestamp_ns"] <= decision_ns
        result[eye] = dict(capture_id=capture["capture_id"], eye=eye,
            timestamp_ns=capture["capture_timestamp_ns"], image_age_ns=age,
            available_timestamp_ns=capture["available_timestamp_ns"],
            valid=None, status="resolve_frames_journal",
            fresh=available and 0 <= age <= IMAGE_MAX_AGE_NS,
            admission=capture["admission"])
    return result


def frame_hash(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


class EyeEncoder:
    def __init__(self, path, size, fps):
        import av
        self.av = av
        self.container = av.open(str(path), "w", format="matroska")
        self.stream = self.container.add_stream("ffv1", rate=fps)
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "bgr0"
        self.stream.codec_context.thread_count = 1
        self.stream.time_base = Fraction(1, 1000)
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.closed = False

    def write(self, array, pts_ms):
        frame = self.av.VideoFrame.from_ndarray(array, format="bgr24")
        frame.pts, frame.time_base = pts_ms, Fraction(1, 1000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.closed:
            return
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        finally:
            self.container.close()
            self.closed = True


def _frame_record(db, capture_id, final):
    cid, stamp, available, admission = db.execute(
        "SELECT * FROM captures WHERE id=?", (capture_id,)).fetchone()
    record = dict(capture_id=cid, capture_timestamp_ns=stamp,
                  available_timestamp_ns=available, final=final,
                  stereo_pair_valid=False, reason=None)
    source = db.execute("SELECT data FROM capture_sources WHERE id=?", (cid,)).fetchone()
    record["source"] = json.loads(source[0]) if source else None
    for eye in EYES:
        row = db.execute("SELECT status,frame_index,pts,tb_num,tb_den,reason,hash,input_pts FROM eyes WHERE id=? AND eye=?",
                         (cid, eye)).fetchone()
        record[eye] = dict(video_file=f"{eye}.mkv", status=row[0],
            video_frame_index=row[1], pts=row[2],
            time_base=[row[3], row[4]] if row[3] is not None else None,
            reason=row[5], sha256_bgr24=row[6], encoding_pts=row[7], encoding_time_base=[1, 1000])
    record["stereo_pair_valid"] = final and all(record[e]["status"] == "written" for e in EYES)
    if not record["stereo_pair_valid"]:
        record["reason"] = admission if admission != "accepted" else "stereo_pair_not_final_or_incomplete"
    return record


def _verify_eye(path, eye, db):
    import av
    count = 0
    with av.open(str(path / f"{eye}.mkv")) as video:
        for index, frame in enumerate(video.decode(video=0)):
            count += 1
            if frame.pts is None or frame.time_base is None:
                continue
            pts_ms = Fraction(frame.pts) * frame.time_base * 1000
            if pts_ms.denominator != 1:
                continue
            row = db.execute("SELECT id,hash,status FROM eyes WHERE eye=? AND input_pts=?",
                             (eye, int(pts_ms))).fetchone()
            if row is None:
                raise ValueError(f"{eye}: decoded PTS has no capture identity")
            digest = frame_hash(frame.to_ndarray(format="bgr24"))
            if row[1] != digest or row[2] == "written":
                raise ValueError(f"{eye}: content mismatch or duplicate PTS for capture {row[0]}")
            if row[2] == "pending":
                db.execute("UPDATE eyes SET status='written',frame_index=?,pts=?,tb_num=?,tb_den=? WHERE id=? AND eye=?",
                    (index, frame.pts, frame.time_base.numerator, frame.time_base.denominator, row[0], eye))
    return count


def _worker(path_string, metadata, events, frame_slots, ready, failed, encoder_factory):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    path = Path(path_string)
    encoders = {}
    errors = []
    error_count = 0
    samples_written = 0
    decoded = dict.fromkeys(EYES, 0)
    ending = None

    def error(message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 20:
            errors.append(str(message))
        print(f"[RECORD ERROR] {message}", flush=True)

    try:
        db = sqlite3.connect(path / "recording_index.sqlite3")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE captures(id INTEGER PRIMARY KEY, timestamp INTEGER, available INTEGER, admission TEXT)")
        db.execute("CREATE TABLE capture_sources(id INTEGER PRIMARY KEY, data TEXT)")
        db.execute("CREATE TABLE eyes(id INTEGER,eye TEXT,input_pts INTEGER,hash TEXT,status TEXT,frame_index INTEGER,pts INTEGER,tb_num INTEGER,tb_den INTEGER,reason TEXT,PRIMARY KEY(id,eye),UNIQUE(eye,input_pts))")
        samples = (path / "samples.jsonl").open("x", encoding="utf-8", buffering=1)
        frames = (path / "frames.jsonl").open("x", encoding="utf-8", buffering=1)
        for eye in EYES:
            encoders[eye] = encoder_factory(path / f"{eye}.mkv", tuple(metadata["video"]["size"]), metadata["video"]["nominal_fps"])
        ready.send(None)
        first_capture_ns = None
        previous_pts = -1
        while True:
            kind, payload = events.get()
            if kind == "close":
                ending = payload
                break
            if kind == "metadata":
                metadata.update(payload)
                atomic_metadata(path, metadata)
            elif kind == "sample":
                samples.write(dumps(payload) + "\n")
                samples_written += 1
                for side, ref in payload.get("references", {}).items():
                    if ref and side not in metadata["initial_references"]:
                        metadata["initial_references"][side] = ref
            elif kind == "capture":
                capture, pair = payload
                try:
                    cid = capture["capture_id"]
                    db.execute("INSERT INTO captures VALUES(?,?,?,?)", (cid,
                        capture["capture_timestamp_ns"], capture["available_timestamp_ns"], capture["admission"]))
                    if capture.get("source") is not None:
                        db.execute("INSERT INTO capture_sources VALUES(?,?)", (cid, dumps(capture["source"])))
                    if first_capture_ns is None:
                        first_capture_ns = capture["capture_timestamp_ns"]
                    pts = max(previous_pts + 1, (capture["capture_timestamp_ns"] - first_capture_ns) // 1_000_000)
                    previous_pts = pts
                    for eye in EYES:
                        db.execute("INSERT INTO eyes VALUES(?,?,?,?,?,NULL,NULL,NULL,NULL,?)", (cid, eye, pts,
                            frame_hash(pair[eye]) if pair else None,
                            "pending" if pair else "dropped",
                            None if pair else capture["admission"]))
                    db.commit()
                    frames.write(dumps(_frame_record(db, cid, pair is None)) + "\n")
                    if pair:
                        for eye in EYES:
                            try:
                                encoders[eye].write(pair[eye], pts)
                            except Exception as exc:
                                error(f"capture {cid} {eye} encode: {exc}")
                                db.execute("UPDATE eyes SET status='failed',reason=? WHERE id=? AND eye=?",
                                           (f"encode: {exc}", cid, eye))
                        db.commit()
                        frames.write(dumps(_frame_record(db, cid, False)) + "\n")
                finally:
                    if pair is not None:
                        frame_slots.release()

        for eye in EYES:
            try:
                encoders[eye].close()
            except Exception as exc:
                error(f"{eye} flush/close: {exc}")
                db.execute("UPDATE eyes SET status='failed',reason='encoder_close_failed' WHERE eye=? AND status='pending'", (eye,))
        for eye in EYES:
            try:
                decoded[eye] = _verify_eye(path, eye, db)
            except Exception as exc:
                error(f"{eye} final decode verification: {exc}")
                db.execute("UPDATE eyes SET status='failed',reason='decode_verification_failed' WHERE eye=? AND status IN ('pending','written')", (eye,))
        db.execute("UPDATE eyes SET status='failed',reason='not_found_in_final_video' WHERE status='pending'")
        db.commit()
        invalid_pairs = 0
        for (cid,) in db.execute("SELECT id FROM captures ORDER BY id"):
            record = _frame_record(db, cid, True)
            invalid_pairs += not record["stereo_pair_valid"]
            frames.write(dumps(record) + "\n")
        samples.close()
        frames.close()
        db.close()
        metadata.update(ending)
        metadata["statistics"].update(samples_written=samples_written,
            video_frames=decoded, invalid_stereo_pairs=invalid_pairs,
            recording_errors=error_count + bool(metadata["statistics"].get("fatal_recording_error")), errors=errors)
        metadata["mapping_status"] = "finalized"
        metadata["status"] = "complete" if (metadata["run_end_state"] in ("normal", "interrupted")
            and not error_count and not invalid_pairs and not metadata["statistics"]["samples_dropped"]
            and not metadata["statistics"].get("fatal_recording_error")) else "incomplete"
        atomic_metadata(path, metadata)
    except BaseException as exc:
        failed.set()
        failure = f"{type(exc).__name__}: {exc}"
        error(failure)
        try:
            ready.send(failure)
        except (OSError, BrokenPipeError):
            pass
        try:
            if ending:
                metadata.update(ending)
            metadata.update(status="incomplete", mapping_status="unfinalized",
                            recording_error=failure)
            atomic_metadata(path, metadata)
        except Exception:
            pass
    finally:
        for encoder in encoders.values():
            try:
                encoder.close()
            except Exception:
                pass


class EpisodeRecorder:
    def __init__(self, root, mode, *, size, fps, metadata=None,
                 frame_queue_size=8, event_queue_size=256, encoder_factory=EyeEncoder):
        if mode not in ("dataset", "ik_analysis"):
            raise ValueError("EpisodeRecorder requires dataset or ik_analysis")
        self.path, number = allocate_episode(root)
        self.metadata = dict(schema_version=SCHEMA_VERSION, episode_id=number,
            record_mode=mode, status="recording", mapping_status="pending",
            start_time_utc=utc_now(), start_timestamp_ns=time.monotonic_ns(),
            end_time_utc=None, end_timestamp_ns=None, run_end_state=None, end_reason=None,
            joint_order=[f"joint{j}" for j in range(1,8)],
            units=dict(joint="rad",gripper="rad",translation="m"),
            robots={s:dict(connected=None,configuration_loaded=False) for s in EYES},
            initial_references={}, statistics=dict(samples_written=None,
                video_frames=dict(left=None,right=None), recording_errors=None),
            clocks=dict(host="CLOCK_MONOTONIC / time.monotonic_ns", utc="human_readable_only",
                capture="host UVC read completion, before split/rectify/resize; not exposure",
                feedback="host complete successful group position extraction; not servo sampling",
                hand="host HAND_MOVE handler receipt; not headset sampling",
                video="unique millisecond PTS derived from capture intervals, not exposure or headset display"),
            video=dict(container="matroska", codec="ffv1", pixel_format="bgr0", lossless=True,
                size=list(size), nominal_fps=fps, input_time_base=[1, 1000],
                left_file="left.mkv", right_file="right.mkv", index="verified decode order",
                processing="stream eye images before SBS black padding"),
            validity=dict(feedback_max_age_ns=FEEDBACK_MAX_AGE_NS, image_max_age_ns=IMAGE_MAX_AGE_NS))
        self.metadata.update(json_value(metadata or {}))
        atomic_metadata(self.path, self.metadata)
        ctx = mp.get_context("spawn")
        self.events = ctx.Queue(maxsize=event_queue_size)
        self.slots = ctx.BoundedSemaphore(frame_queue_size)
        self.failed = ctx.Event()
        self.ready, child_ready = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_worker, args=(str(self.path), self.metadata,
            self.events, self.slots, child_ready, self.failed, encoder_factory), daemon=True)
        self.capture_index = 0
        self.last_capture_timestamp_ns = None
        self.stats = dict(captures_attempted=0, captures_accepted=0, captures_dropped=0,
                         samples_attempted=0, samples_enqueued=0, samples_dropped=0,
                         fatal_recording_error=None, unjournaled_capture_range=None,
                         dropped_step_range=None)
        self.closed = False
        self.disabled = False
        self.process.start()
        child_ready.close()
        if not self.ready.poll(15):
            self.close("exception", "recorder_start_timeout", timeout=1)
            raise RuntimeError(f"Recorder startup timed out: {self.path}")
        message = self.ready.recv()
        self.ready.close()
        if message is not None:
            self.close("exception", message, timeout=1)
            raise RuntimeError(f"Recorder startup failed at {self.path}: {message}")
        print(f"[RECORD] {mode}: {self.path}")

    def _disable(self, reason):
        if not self.disabled:
            self.disabled = True
            self.stats["fatal_recording_error"] = reason
            print(f"[RECORD ERROR] {reason}; recording disabled, control continues", flush=True)

    def _enqueue(self, event):
        if self.closed:
            return False
        if self.failed.is_set() or not self.process.is_alive():
            self._disable("writer_failed")
        if self.disabled:
            return False
        try:
            self.events.put_nowait(event)
            return True
        except queue.Full:
            self._disable("event_queue_full")
            return False

    @staticmethod
    def _extend_range(current, index):
        return [index, index] if current is None else [current[0], index]

    def capture(self, left, right, timestamp_ns, available_ns=None, *, source=None):
        cid = self.capture_index
        self.capture_index += 1
        self.stats["captures_attempted"] += 1
        available_ns = time.monotonic_ns() if available_ns is None else available_ns
        capture = dict(capture_id=cid, capture_timestamp_ns=int(timestamp_ns),
                       available_timestamp_ns=int(available_ns), admission="accepted",
                       source=json_value(source))
        pair = None
        acquired = False
        try:
            if self.last_capture_timestamp_ns is not None and timestamp_ns <= self.last_capture_timestamp_ns:
                raise ValueError("capture timestamps must increase")
            self.last_capture_timestamp_ns = timestamp_ns
            if available_ns < timestamp_ns:
                raise ValueError("image available before capture")
            expected = (self.metadata["video"]["size"][1], self.metadata["video"]["size"][0], 3)
            if any(a.shape != expected or a.dtype != np.uint8 for a in (left, right)):
                raise ValueError("recorded eyes must match configured size and uint8 BGR")
            acquired = not self.disabled and self.slots.acquire(block=False)
            if acquired:
                pair = {"left": np.array(left, copy=True, order="C"),
                        "right": np.array(right, copy=True, order="C")}
            else:
                capture["admission"] = "recorder_disabled" if self.disabled else "frame_queue_full"
        except Exception as exc:
            capture["admission"] = f"capture_invalid: {exc}"
            if acquired:
                self.slots.release()
                acquired = False
        enqueued = self._enqueue(("capture", (capture.copy(), pair)))
        if not enqueued:
            if acquired:
                self.slots.release()
            capture["admission"] = "unjournaled_recording_failure"
            self.stats["unjournaled_capture_range"] = self._extend_range(
                self.stats["unjournaled_capture_range"], cid)
        if enqueued and pair is not None:
            self.stats["captures_accepted"] += 1
        else:
            self.stats["captures_dropped"] += 1
        return capture

    def sample(self, sample):
        self.stats["samples_attempted"] += 1
        if self._enqueue(("sample", json_value(sample))):
            self.stats["samples_enqueued"] += 1
            return True
        self.stats["samples_dropped"] += 1
        self.stats["dropped_step_range"] = self._extend_range(self.stats["dropped_step_range"], sample["step_index"])
        return False

    def update_metadata(self, fields):
        self.metadata.update(json_value(fields))
        return self._enqueue(("metadata", json_value(fields)))

    def close(self, state="normal", reason="normal_exit", timeout=10.0):
        if self.closed:
            return
        self.closed = True
        ending = dict(run_end_state=state, end_reason=reason, end_time_utc=utc_now(),
                      end_timestamp_ns=time.monotonic_ns(), statistics=self.stats.copy())
        deadline = time.monotonic() + timeout
        try:
            if self.process.is_alive():
                self.events.put(("close", ending), timeout=max(0, deadline - time.monotonic()))
        except (queue.Full, OSError):
            pass
        self.process.join(max(0, deadline - time.monotonic()))
        forced = self.process.is_alive()
        if forced:
            self.process.terminate()
            self.process.join(1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(1)
        self.events.cancel_join_thread()
        self.events.close()
        try:
            current = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
            if forced or current.get("mapping_status") != "finalized":
                current.update(ending, status="incomplete", mapping_status="unfinalized")
                current["statistics"].update(samples_written=None, video_frames=dict(left=None,right=None),
                                             recording_errors=None)
                current["end_reason"] = reason
                current["finalization_error"] = "close_timeout" if forced else "writer_failed"
                atomic_metadata(self.path, current)
                print(f"[RECORD ERROR] incomplete episode: {self.path} ({current['finalization_error']})")
        except Exception as exc:
            print(f"[RECORD ERROR] could not finalize metadata at {self.path}: {exc}")


def create_recorder(mode, root=None, **kwargs):
    if mode == "off":
        return None
    if not root:
        raise ValueError("--record_dir is required when recording is enabled")
    return EpisodeRecorder(root, mode, **kwargs)
