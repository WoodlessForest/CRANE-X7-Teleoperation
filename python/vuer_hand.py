# Normalize supported Vuer hand payloads into wrist matrices and landmarks.
# Reject malformed tracking data before it reaches arm preprocessing.

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


WEBXR_HAND_JOINT_KEYS = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)


@dataclass(frozen=True)
class DecodedHand:

    wrist_column_major: np.ndarray
    landmarks: np.ndarray
    source: str


def _float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result).all():
        return None
    return result


def _column_major_matrix(value: Any) -> Optional[np.ndarray]:
    array = _float_array(value)
    if array is None:
        return None
    if array.shape == (4, 4):
        flat = array.flatten(order="F")
    elif array.size == 16:
        flat = array.reshape(16)
    else:
        return None

    matrix = flat.reshape((4, 4), order="F")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-4):
        return None
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not 0.5 <= determinant <= 1.5:
        return None
    return flat.copy()


def _joint_matrices(value: Any) -> Optional[np.ndarray]:
    array = _float_array(value)
    if array is None:
        return None

    joint_count = len(WEBXR_HAND_JOINT_KEYS)
    if array.shape == (joint_count, 4, 4):
        matrices = np.stack(
            [matrix.flatten(order="F") for matrix in array], axis=0
        )
    elif array.size == joint_count * 16:
        matrices = array.reshape((joint_count, 16))
    else:
        return None

    if any(_column_major_matrix(matrix) is None for matrix in matrices):
        return None
    return matrices.copy()


def _legacy_landmarks(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, Mapping):
        positions = []
        for joint_name in WEBXR_HAND_JOINT_KEYS:
            joint = value.get(joint_name)
            array = _float_array(joint)
            if array is None:
                return None
            if array.size == 3:
                positions.append(array.reshape(3))
                continue
            matrix = _column_major_matrix(array)
            if matrix is None:
                return None
            positions.append(matrix[12:15])
        return np.asarray(positions, dtype=np.float64)

    array = _float_array(value)
    if array is None or array.size != len(WEBXR_HAND_JOINT_KEYS) * 3:
        return None
    return array.reshape((len(WEBXR_HAND_JOINT_KEYS), 3)).copy()


def _first_present(payload: Mapping[str, Any], names) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _geometry_is_sane(wrist: np.ndarray, landmarks: np.ndarray) -> bool:

    if landmarks.shape != (len(WEBXR_HAND_JOINT_KEYS), 3):
        return False
    wrist_position = wrist[12:15]
    wrist_gap = float(np.linalg.norm(landmarks[0] - wrist_position))
    hand_span = float(np.max(np.linalg.norm(landmarks - landmarks[0], axis=1)))
    return wrist_gap <= 0.08 and 0.03 <= hand_span <= 0.45


def decode_hand_payload(
    payload: Mapping[str, Any], side: str
) -> Optional[DecodedHand]:


    if side not in ("left", "right") or not isinstance(payload, Mapping):
        return None

    legacy_wrist = _first_present(
        payload,
        (
            f"{side}Hand",
            f"{side}_hand",
            f"{side}Wrist",
            f"{side}_wrist",
        ),
    )
    legacy_points = _first_present(
        payload, (f"{side}Landmarks", f"{side}_landmarks")
    )
    wrist = _column_major_matrix(legacy_wrist)
    landmarks = _legacy_landmarks(legacy_points)
    if wrist is not None and landmarks is not None:
        if not _geometry_is_sane(wrist, landmarks):
            return None
        return DecodedHand(wrist, landmarks, "vuer-0.0.32rc7")

    matrices = _joint_matrices(payload.get(side))
    if matrices is None:
        return None
    wrist = matrices[0]
    landmarks = matrices[:, 12:15]
    if not _geometry_is_sane(wrist, landmarks):
        return None
    return DecodedHand(wrist, landmarks, "vuer-0.0.61")


def payload_shapes(payload: Mapping[str, Any]) -> Mapping[str, Any]:

    result = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            result[key] = "mapping"
            continue
        try:
            result[key] = tuple(np.asarray(value).shape)
        except Exception:
            result[key] = type(value).__name__
    return result


__all__ = [
    "DecodedHand",
    "WEBXR_HAND_JOINT_KEYS",
    "decode_hand_payload",
    "payload_shapes",
]
