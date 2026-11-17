# Map tracked wrist poses to relative targets in the X7 base frame.
# Keep an independent reference for each hand and derive gripper commands from landmarks.

from dataclasses import dataclass
from typing import Optional

import numpy as np

QUEST_YUP_TO_WORLD_ZUP = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

R_x7_from_quest_right = np.eye(3, dtype=np.float32)
R_x7_from_quest_left = np.eye(3, dtype=np.float32)

X7_FORWARD_HOME_POSITION = np.array(
    [0.35, 0.0, 0.30],
    dtype=np.float32,
)
X7_FORWARD_HOME_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

THUMB_TIP = 4
INDEX_TIP = 9


def _mat4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


@dataclass
class EECommand:
    T_ee: np.ndarray
    grip_angle_deg: float


class VuerArmPreprocessor:
    def __init__(self, grip_min_deg: float = 0.0, grip_max_deg: float = 70.0,
                 thumb_idx: int = THUMB_TIP, index_idx: int = INDEX_TIP,
                 home_position=None, translation_gain: float = 1.0):
        self.grip_min = float(grip_min_deg)
        self.grip_max = float(grip_max_deg)
        self.thumb_idx = int(thumb_idx)
        self.index_idx = int(index_idx)
        self.translation_gain = float(translation_gain)
        if self.translation_gain <= 0.0:
            raise ValueError("translation_gain must be greater than zero")
        self.home_position = np.array(
            X7_FORWARD_HOME_POSITION
            if home_position is None
            else home_position,
            dtype=np.float32,
        ).reshape(3,)
        self.home_rotation = X7_FORWARD_HOME_ROTATION.copy()
        self._reference_wrist_world = {"left": None, "right": None}
        self._reference_version = {"left": 0, "right": 0}

    @staticmethod
    def _questYup_to_worldZup(T_quest: np.ndarray) -> np.ndarray:
        return QUEST_YUP_TO_WORLD_ZUP @ T_quest

    @staticmethod
    def _quest_ptsYup_to_worldZup(pts_quest: np.ndarray) -> Optional[np.ndarray]:
        if pts_quest is None or len(pts_quest) < 1:
            return None
        pts = np.asarray(pts_quest, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return None
        homo = np.concatenate([pts.T, np.ones((1, pts.shape[0]), dtype=np.float32)], axis=0)
        pts_world = (QUEST_YUP_TO_WORLD_ZUP @ homo)[:3, :].T
        return pts_world

    def _grip_map(self, dist_m: float, d_min: float = 0.02, d_max: float = 0.08) -> float:
        x = np.clip((dist_m - d_min) / (d_max - d_min), 0.0, 1.0)
        return float((1.0 - x) * self.grip_min + x * self.grip_max)

    def _thumb_index_distance(self, landmarks_quest: Optional[np.ndarray]) -> float:
        max_idx = max(self.thumb_idx, self.index_idx)
        if landmarks_quest is None or len(landmarks_quest) <= max_idx:
            return 0.05
        pts_world = VuerArmPreprocessor._quest_ptsYup_to_worldZup(landmarks_quest)
        if pts_world is None:
            return 0.05
        thumb = pts_world[self.thumb_idx, :]
        index = pts_world[self.index_idx, :]
        return float(np.linalg.norm(thumb - index))

    @staticmethod
    def _project_to_rotation(R: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
        result = U @ Vt
        if np.linalg.det(result) < 0.0:
            U[:, -1] *= -1.0
            result = U @ Vt
        return result.astype(np.float32)

    @classmethod
    def _validated_wrist_pose(cls, wrist_quest: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if wrist_quest is None:
            return None
        try:
            wrist = np.array(wrist_quest, dtype=np.float64, copy=True)
        except (TypeError, ValueError):
            return None
        if wrist.shape != (4, 4) or not np.isfinite(wrist).all():
            return None
        if not np.allclose(wrist[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-4):
            return None

        R = wrist[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=1e-3):
            return None
        if np.linalg.det(R) <= 0.0:
            return None
        return _mat4(cls._project_to_rotation(R), wrist[:3, 3])

    def has_reference(self, side: str) -> bool:
        if side not in self._reference_wrist_world:
            raise ValueError("side must be 'left' or 'right'")
        return self._reference_wrist_world[side] is not None

    def reference_info(self, side: str):
        if not self.has_reference(side):
            return None
        return dict(version=self._reference_version[side],
                    wrist_world_pose=self._reference_wrist_world[side].copy(),
                    robot_initial_target_pose=_mat4(self.home_rotation, self.home_position))

    def _relative_target(self, side: str, wrist_world: np.ndarray) -> np.ndarray:
        reference = self._reference_wrist_world[side]
        if reference is None:
            reference = np.array(wrist_world, dtype=np.float32, copy=True)
            self._reference_wrist_world[side] = reference
            self._reference_version[side] += 1

        R_map = (
            R_x7_from_quest_right
            if side == "right"
            else R_x7_from_quest_left
        )

        delta_world = wrist_world[:3, 3] - reference[:3, 3]
        target_position = (
            self.home_position
            + self.translation_gain * (R_map @ delta_world)
        )

        delta_R_world = (
            wrist_world[:3, :3]
            @ reference[:3, :3].T
        )
        delta_R_base = R_map @ delta_R_world @ R_map.T
        target_rotation = self._project_to_rotation(
            delta_R_base @ self.home_rotation
        )
        return _mat4(target_rotation, target_position)

    def compute_side(
        self,
        side: str,
        wrist_quest: Optional[np.ndarray],
        landmarks_quest: Optional[np.ndarray] = None,
    ) -> Optional[EECommand]:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        wrist = self._validated_wrist_pose(wrist_quest)
        if wrist is None:
            return None
        wrist_world = self._questYup_to_worldZup(wrist)
        ee_x7 = self._relative_target(side, wrist_world)
        grip = self._grip_map(
            self._thumb_index_distance(landmarks_quest)
        )
        return EECommand(T_ee=ee_x7, grip_angle_deg=grip)


__all__ = [
    "VuerArmPreprocessor",
    "EECommand",
    "QUEST_YUP_TO_WORLD_ZUP",
    "X7_FORWARD_HOME_POSITION",
    "X7_FORWARD_HOME_ROTATION",
]
