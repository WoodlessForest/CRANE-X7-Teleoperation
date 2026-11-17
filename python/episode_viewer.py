# Validate recorded episodes and read observations without connecting hardware.
# Build temporary indexes for inspection and optional journal recovery.

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from fractions import Fraction
from pathlib import Path
import sqlite3
import tempfile

from episode_recording import SCHEMA_VERSION, EYES, dumps, frame_hash


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_quantity(value, count):
    return (finite_number(value) if count == 1 else
            isinstance(value, list) and len(value) == count and all(finite_number(v) for v in value))


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(text):
    return json.loads(text, parse_constant=_reject_constant)


class Episode:
    def __init__(self, path, *, recover=False):
        self.path = Path(path).resolve()
        self.metadata = read_json((self.path / "metadata.json").read_text(encoding="utf-8"))
        self._temp = tempfile.TemporaryDirectory(prefix="episode-view-")
        self.db = sqlite3.connect(Path(self._temp.name) / "index.sqlite3")
        self.db.execute("CREATE TABLE frames(id INTEGER PRIMARY KEY,data TEXT)")
        self.db.execute("CREATE TABLE expected(eye TEXT,frame_index INTEGER,capture_id INTEGER,data TEXT,verified INTEGER DEFAULT 0,PRIMARY KEY(eye,frame_index))")
        self.issues = Counter()
        self.examples = []
        self.quality_exclusions = {}
        self.quality_review_invalid = False
        review_path = self.path / "quality_exclusions.json"
        if review_path.exists():
            try:
                review = read_json(review_path.read_text(encoding="utf-8"))
                if review.get("schema_version") != "1.0" or not isinstance(review.get("excluded_captures"), list):
                    raise ValueError("unsupported quality review schema")
                for item in review["excluded_captures"]:
                    cid = item["capture_id"]
                    if type(cid) is not int or cid < 0 or not isinstance(item.get("reason"), str):
                        raise ValueError("invalid quality exclusion")
                    self.quality_exclusions[cid] = item["reason"]
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.quality_review_invalid = True
                self.issue("invalid_quality_review", str(exc))
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            self.issue("unsupported_schema_version")
        if self.metadata.get("record_mode") not in ("dataset", "ik_analysis"):
            self.issue("invalid_record_mode")
        if self.metadata.get("mapping_status") != "finalized":
            self.issue("unfinalized_episode")
        if self.metadata.get("status") != "complete":
            self.issue("incomplete_episode")
        for row in self.rows("frames.jsonl"):
            try:
                cid = row["capture_id"]
                if (not isinstance(cid, int) or cid < 0 or
                    not isinstance(row["capture_timestamp_ns"], int) or
                    not isinstance(row.get("available_timestamp_ns"), int) or
                    not isinstance(row.get("final"), bool) or
                    not isinstance(row.get("stereo_pair_valid"), bool) or
                    any(not isinstance(row.get(e),dict) for e in EYES)):
                    raise ValueError("invalid capture identity, timestamp or frame schema")
                previous = self.frame(cid)
                if previous and previous.get("final") and previous != row:
                    self.issue("conflicting_final_frame", str(cid))
                self.db.execute("INSERT OR REPLACE INTO frames VALUES(?,?)", (cid, dumps(row)))
            except (KeyError, TypeError, ValueError) as exc:
                self.issue("invalid_frame_schema", str(exc))
        self.recovered = False
        if recover:
            self._recover_mappings()
        last_stamp = -1
        for row in self.frames():
            if row["capture_timestamp_ns"] <= last_stamp:
                self.issue("capture_timestamp_order", str(row["capture_id"]))
            last_stamp = row["capture_timestamp_ns"]
            if row.get("available_timestamp_ns", last_stamp) < last_stamp:
                self.issue("image_available_before_capture", str(row["capture_id"]))
            for eye in EYES:
                info = row.get(eye, {})
                if row.get("final") and info.get("status") == "written":
                    try:
                        index = info["video_frame_index"]
                        if not isinstance(index, int) or index < 0:
                            raise ValueError("invalid video frame index")
                        self.db.execute("INSERT INTO expected(eye,frame_index,capture_id,data) VALUES(?,?,?,?)",
                            (eye, index, row["capture_id"], dumps(info)))
                    except (sqlite3.IntegrityError, KeyError, ValueError) as exc:
                        self.issue("invalid_or_duplicate_video_index", str(exc))
        self.db.commit()
        self.video_checked = False

    def _recover_mappings(self):

        import av
        self.db.execute("CREATE TABLE recovery(eye TEXT,pts INTEGER,capture_id INTEGER,hash TEXT,PRIMARY KEY(eye,pts))")
        for record in self.frames():
            for eye in EYES:
                info = record.get(eye, {})
                if (info.get("status") in ("pending", "written") and
                    isinstance(info.get("encoding_pts"), int) and info.get("sha256_bgr24")):
                    self.db.execute("INSERT INTO recovery VALUES(?,?,?,?)", (eye,
                        info["encoding_pts"], record["capture_id"], info["sha256_bgr24"]))
                    info.update(status="pending", video_frame_index=None, pts=None, time_base=None)
            record.update(final=False, stereo_pair_valid=False)
            self.db.execute("UPDATE frames SET data=? WHERE id=?", (dumps(record), record["capture_id"]))
        for eye in EYES:
            try:
                with av.open(str(self._video_path(eye))) as video:
                    for index, frame in enumerate(video.decode(video=0)):
                        if frame.pts is None or frame.time_base is None:
                            continue
                        pts = Fraction(frame.pts) * frame.time_base * 1000
                        if pts.denominator != 1:
                            continue
                        candidate = self.db.execute("SELECT capture_id,hash FROM recovery WHERE eye=? AND pts=?",
                                                    (eye,int(pts))).fetchone()
                        if candidate is None:
                            continue
                        record = self.frame(candidate[0])
                        info = record[eye]
                        if info["status"] != "pending" or frame_hash(frame.to_ndarray(format="bgr24")) != candidate[1]:
                            info.update(status="failed",reason="recovery_content_or_duplicate_pts_error")
                        else:
                            info.update(status="written",video_frame_index=index,pts=frame.pts,
                                time_base=[frame.time_base.numerator,frame.time_base.denominator],
                                reason=None,recovered_read_only=True)
                        self.db.execute("UPDATE frames SET data=? WHERE id=?", (dumps(record),candidate[0]))
            except Exception as exc:
                self.issue("recovery_decode_error", f"{eye}: {exc}")
        for record in self.frames():
            for eye in EYES:
                if record.get(eye,{}).get("status") == "pending":
                    record[eye].update(status="failed",reason="not_found_during_read_only_recovery")
            record.update(final=True, stereo_pair_valid=all(record.get(e,{}).get("status")=="written" for e in EYES),
                          recovered_read_only=True)
            self.db.execute("UPDATE frames SET data=? WHERE id=?", (dumps(record),record["capture_id"]))
        self.db.commit()
        self.recovered = True

    def issue(self, name, detail=None):
        self.issues[name] += 1
        if detail is not None and len(self.examples) < 20:
            self.examples.append(dict(issue=name, detail=detail))

    def rows(self, filename):
        try:
            with (self.path / filename).open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    try:
                        row = read_json(line)
                        if not isinstance(row, dict):
                            raise ValueError("JSON object required")
                        yield row
                    except (ValueError, TypeError) as exc:
                        self.issue("invalid_jsonl", f"{filename}:{line_number}: {exc}")
        except OSError as exc:
            self.issue("missing_file", str(exc))

    def frame(self, capture_id):
        row = self.db.execute("SELECT data FROM frames WHERE id=?", (capture_id,)).fetchone()
        return read_json(row[0]) if row else None

    def frames(self):
        for (data,) in self.db.execute("SELECT data FROM frames ORDER BY id"):
            yield read_json(data)

    def _video_path(self, eye):
        filename = self.metadata.get("video", {}).get(f"{eye}_file", f"{eye}.mkv")
        if filename != f"{eye}.mkv":
            raise ValueError(f"unexpected {eye} video filename")
        return self.path / filename

    def check_videos(self):
        import av
        counts = {}
        self.db.execute("UPDATE expected SET verified=0")
        for eye in EYES:
            counts[eye] = 0
            try:
                with av.open(str(self._video_path(eye))) as video:
                    for index, decoded in enumerate(video.decode(video=0)):
                        counts[eye] += 1
                        row = self.db.execute("SELECT data FROM expected WHERE eye=? AND frame_index=?", (eye, index)).fetchone()
                        if row is None:
                            self.issue("unmapped_decoded_frame", f"{eye}:{index}")
                            continue
                        expected = read_json(row[0])
                        actual_tb = [decoded.time_base.numerator, decoded.time_base.denominator] if decoded.time_base else None
                        if (expected.get("pts") != decoded.pts or expected.get("time_base") != actual_tb or
                            expected.get("sha256_bgr24") != frame_hash(decoded.to_ndarray(format="bgr24"))):
                            self.issue("video_content_or_pts_mismatch", f"{eye}:{index}")
                        else:
                            self.db.execute("UPDATE expected SET verified=1 WHERE eye=? AND frame_index=?", (eye, index))
                if counts[eye] != self.metadata.get("statistics", {}).get("video_frames", {}).get(eye):
                    self.issue("video_frame_count_mismatch", eye)
            except Exception as exc:
                self.issue("video_decode_failed", f"{eye}: {exc}")
        missing = self.db.execute("SELECT COUNT(*) FROM expected WHERE verified=0").fetchone()[0]
        if missing:
            self.issues["unverified_final_frames"] += missing
        self.video_checked = True
        return counts

    def resolve(self, sample, required_sides=EYES):
        try:
            return self._resolve_sample(sample, required_sides)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            return dict(valid=False, visual_valid=False, images={},
                        reasons=["invalid_sample_schema"], schema_error=str(exc))

    def _resolve_sample(self, sample, required_sides):
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            return dict(valid=False, visual_valid=False, images={}, reasons=["unsupported_schema_version"])
        reasons = []
        refs = sample.get("observation", {}).get("image", {})
        images = {}
        ids = [refs.get(e, {}).get("capture_id") for e in EYES]
        if self.quality_review_invalid:
            reasons.append("invalid_quality_review")
        if any(cid in self.quality_exclusions for cid in ids if isinstance(cid, int)):
            reasons.append("review_excluded_capture")
        decision = sample.get("decision_timestamp_ns")
        record = self.frame(ids[0]) if ids[0] is not None and ids[0] == ids[1] else None
        if self.metadata.get("mapping_status") != "finalized" and not self.recovered:
            reasons.append("mapping_unfinalized")
        if record is None:
            reasons.append("missing_stereo_capture")
        else:
            if not record.get("final") or not record.get("stereo_pair_valid"):
                reasons.append("stereo_pair_invalid")
            for eye in EYES:
                ref, info = refs.get(eye, {}), record.get(eye, {})
                images[eye] = info
                if (ref.get("eye") != eye or info.get("video_file") != f"{eye}.mkv" or
                    info.get("status") != "written" or not isinstance(info.get("video_frame_index"), int)):
                    reasons.append(f"{eye}_not_written")
                stamp, available = record["capture_timestamp_ns"], record.get("available_timestamp_ns")
                if (not isinstance(decision, int) or available is None or available > decision or
                    stamp > decision or ref.get("timestamp_ns") != stamp or
                    ref.get("image_age_ns") != decision - stamp):
                    reasons.append(f"{eye}_image_timing_invalid")
                elif decision - stamp > self.metadata["validity"]["image_max_age_ns"]:
                    reasons.append(f"{eye}_image_stale")
                if self.video_checked:
                    verified = self.db.execute("SELECT verified FROM expected WHERE eye=? AND frame_index=? AND capture_id=?",
                        (eye, info.get("video_frame_index"), ids[0])).fetchone()
                    if not verified or not verified[0]:
                        reasons.append(f"{eye}_video_not_verified")
        visual_valid = not reasons
        for side in required_sides:
            for kind in ("joint", "gripper"):
                obs = sample.get("observation", {}).get(kind, {}).get(side, {})
                action = sample.get("action", {}).get(kind, {}).get(side, {})
                values = obs.get("value")
                if not obs.get("valid") or obs.get("status") != "ok" or values is None or obs.get("unit") != "rad":
                    reasons.append(f"{side}_{kind}_observation_missing")
                elif not valid_quantity(values, 7 if kind == "joint" else 1):
                    reasons.append(f"{side}_{kind}_quantity_invalid")
                stamp, snap = obs.get("timestamp_ns"), obs.get("snapshot_timestamp_ns")
                if obs.get("valid") and (not isinstance(stamp, int) or not isinstance(snap, int) or
                    not isinstance(decision, int) or stamp > snap or snap > decision or
                    obs.get("age_ns") != decision-stamp or
                    decision-stamp > self.metadata["validity"]["feedback_max_age_ns"]):
                    reasons.append(f"{side}_{kind}_observation_timing_invalid")
                if not action.get("submitted") or not action.get("interface_accepted") or action.get("target") is None:
                    reasons.append(f"{side}_{kind}_action_missing")
                if action.get("submitted"):
                    if action.get("unit") != "rad" or not valid_quantity(action.get("target"), 7 if kind == "joint" else 1):
                        reasons.append(f"{side}_{kind}_target_invalid")
                    start, end = action.get("submission_timestamp_ns"), action.get("return_timestamp_ns")
                    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(decision, int) or start < decision or end < start:
                        reasons.append(f"{side}_{kind}_submission_timing_invalid")
                elif action.get("target") is not None:
                    reasons.append(f"{side}_{kind}_unsubmitted_target")
        return dict(valid=not reasons, visual_valid=visual_valid, reasons=reasons, images=images)

    def samples(self, valid_only=False, required_sides=EYES):
        if valid_only and not self.video_checked:
            self.check_videos()
        for sample in self.rows("samples.jsonl"):
            if not valid_only or self.resolve(sample, required_sides)["valid"]:
                yield sample

    def read_frame(self, eye, frame_info):
        import av
        if frame_info.get("status") != "written":
            raise ValueError("Video reference is not finalized/written")
        index = frame_info["video_frame_index"]
        with av.open(str(self._video_path(eye))) as video:
            seeked = self.video_checked and frame_info.get("pts") is not None
            if seeked:
                stream = video.streams.video[0]
                tb = Fraction(*frame_info["time_base"])
                video.seek(int(frame_info["pts"] * tb / stream.time_base), stream=stream, backward=True)
            for i, frame in enumerate(video.decode(video=0)):
                match = (frame.pts is not None and frame.pts * frame.time_base ==
                         frame_info["pts"] * Fraction(*frame_info["time_base"])) if seeked else i == index
                if match:
                    array = frame.to_ndarray(format="bgr24")
                    tb = [frame.time_base.numerator, frame.time_base.denominator] if frame.time_base else None
                    if (frame_hash(array) != frame_info["sha256_bgr24"] or
                        frame.pts != frame_info.get("pts") or tb != frame_info.get("time_base")):
                        raise ValueError("Video content or PTS differs from final mapping")
                    return array
        raise ValueError("Video frame does not exist")

    def step(self, step_index, decode=False):
        for sample in self.samples():
            if sample.get("step_index") == step_index:
                resolved = self.resolve(sample)
                result = dict(sample=sample, resolution=resolved)
                if decode and resolved["visual_valid"]:
                    result["images"] = {eye:self.read_frame(eye, resolved["images"][eye]) for eye in EYES}
                return result
        raise KeyError(f"step_index {step_index} is absent (possibly dropped)")

    def validate(self):
        counts = self.check_videos()
        samples, valid, invalid_visual, previous_step, previous_time = 0, 0, 0, -1, -1
        missing = Counter()
        gaps = 0
        for sample in self.samples():
            samples += 1
            step, stamp = sample.get("step_index"), sample.get("decision_timestamp_ns")
            if not isinstance(step, int) or not isinstance(stamp, int):
                self.issue("invalid_sample_schema")
                continue
            if step <= previous_step or stamp <= previous_time:
                self.issue("sample_time_or_step_order")
            gaps += max(0, step - previous_step - 1)
            previous_step, previous_time = step, stamp
            if self.metadata.get("record_mode") == "ik_analysis" and "diagnostics" not in sample:
                self.issue("missing_diagnostics")
            resolved = self.resolve(sample)
            valid += resolved["valid"]
            invalid_visual += not resolved["visual_valid"]
            missing.update(resolved["reasons"])
            diagnostics = sample.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                self.issue("invalid_diagnostics_schema", str(step))
                diagnostics = {}
            for side, diag in diagnostics.items():
                if not isinstance(diag, dict):
                    self.issue("invalid_diagnostics_schema", str(step))
                    continue
                try:
                    target = sample.get("action", {}).get("joint", {}).get(side, {}).get("target")
                except (TypeError, AttributeError):
                    self.issue("invalid_action_schema", str(step))
                    continue
                if target is not None and diag.get("q_final") != target:
                    self.issue("diagnostics_action_mismatch", str(step))
        frame_counts = Counter()
        for frame in self.frames():
            frame_counts["captures"] += 1
            frame_counts["invalid_stereo_pairs"] += not frame.get("stereo_pair_valid", False)
            for eye in EYES:
                frame_counts[f"{eye}_missing"] += frame.get(eye, {}).get("status") != "written"
        if samples != self.metadata.get("statistics", {}).get("samples_written"):
            self.issue("sample_count_mismatch")
        return dict(episode_id=self.metadata.get("episode_id"), status=self.metadata.get("status"),
            recovered_read_only=self.recovered,
            schema_version=self.metadata.get("schema_version"), decodable_video_frames=counts,
            samples=samples, valid_samples=valid, invalid_samples=samples-valid,
            invalid_visual_samples=invalid_visual, sample_gaps=gaps, frames=dict(frame_counts),
            missing=dict(missing), recorded_statistics=self.metadata.get("statistics", {}),
            issues=dict(self.issues), examples=self.examples)

    def close(self):
        self.db.close()
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description='Read-only episode validation and lazy observation/action access.\n\nNo hardware imports. Videos are decoded one frame at a time; indexes are built\nin a temporary directory, never in the original episode. Pending journals can\nbe inspected, but are deliberately excluded from training.\n')
    parser.add_argument("episode")
    parser.add_argument("--step", type=int, help="Inspect one step and its final stereo references")
    parser.add_argument("--recover", action="store_true", help="Verify pending journals into a temporary read-only recovery index")
    args = parser.parse_args()
    with Episode(args.episode, recover=args.recover) as episode:
        report = episode.validate()
        print(dumps(report))
        if args.step is not None:
            result = episode.step(args.step, decode=True)
            result.pop("images", None)
            print(dumps(result))
        return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
