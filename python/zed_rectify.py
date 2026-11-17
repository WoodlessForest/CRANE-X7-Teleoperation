# Locate ZED factory calibration and prepare OpenCV stereo rectification maps.
# Infer the raw stereo layout when automatic eye ordering is requested.

from __future__ import annotations

import configparser
import os
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


_RESOLUTION_SECTIONS = {
    (2208, 1242): "2K",
    (1920, 1080): "FHD",
    (1280, 720): "HD",
    (672, 376): "VGA",
}


def _get_float(section, key: str, default: float = 0.0) -> float:
    value = section.get(key, fallback=None)
    return default if value is None else float(value)


def find_or_download_calibration(
    serial_number: str,
    explicit_path: str = "",
) -> Path:

    serial = str(serial_number).strip()
    if not serial:
        raise ValueError("ZED serial number is required for stereo rectification")

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ZED calibration file not found: {path}")
        return path

    filename = f"SN{serial}.conf"
    module_dir = Path(__file__).resolve().parent
    candidates = [
        module_dir / "calibration" / filename,
        Path("/usr/local/zed/settings") / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path

    cache_path = candidates[0]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://calib.stereolabs.com/?SN={serial}"
    try:
        with urlopen(url, timeout=10) as response:
            data = response.read()
    except Exception as exc:
        raise RuntimeError(
            f"cannot download ZED calibration for SN{serial}: {exc}"
        ) from exc

    text = data.decode("utf-8", errors="replace")
    if "[STEREO]" not in text or "LEFT_CAM_" not in text:
        raise RuntimeError(
            f"invalid calibration response for ZED SN{serial}"
        )

    temporary = cache_path.with_suffix(".conf.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, cache_path)
    return cache_path


class ZEDStereoRectifier:

    def __init__(
        self,
        calibration_file: Path,
        width: int,
        height: int,
        alpha: float = 0.0,
    ):
        resolution = _RESOLUTION_SECTIONS.get((int(width), int(height)))
        if resolution is None:
            supported = ", ".join(
                f"{w}x{h}" for w, h in _RESOLUTION_SECTIONS
            )
            raise ValueError(
                f"unsupported per-eye ZED resolution {width}x{height}; "
                f"supported: {supported}"
            )

        config = configparser.ConfigParser()
        loaded = config.read(calibration_file)
        if not loaded:
            raise RuntimeError(
                f"cannot read ZED calibration file: {calibration_file}"
            )

        left_name = f"LEFT_CAM_{resolution}"
        right_name = f"RIGHT_CAM_{resolution}"
        for section_name in ("STEREO", left_name, right_name):
            if section_name not in config:
                raise RuntimeError(
                    f"missing [{section_name}] in {calibration_file}"
                )

        stereo = config["STEREO"]
        left = config[left_name]
        right = config[right_name]

        left_matrix = np.array(
            [
                [_get_float(left, "fx"), 0.0, _get_float(left, "cx")],
                [0.0, _get_float(left, "fy"), _get_float(left, "cy")],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        right_matrix = np.array(
            [
                [_get_float(right, "fx"), 0.0, _get_float(right, "cx")],
                [0.0, _get_float(right, "fy"), _get_float(right, "cy")],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        left_distortion = np.array(
            [
                _get_float(left, "k1"),
                _get_float(left, "k2"),
                _get_float(left, "p1"),
                _get_float(left, "p2"),
                _get_float(left, "k3"),
            ],
            dtype=np.float64,
        )
        right_distortion = np.array(
            [
                _get_float(right, "k1"),
                _get_float(right, "k2"),
                _get_float(right, "p1"),
                _get_float(right, "p2"),
                _get_float(right, "k3"),
            ],
            dtype=np.float64,
        )

        rotation_vector = np.array(
            [
                _get_float(stereo, f"RX_{resolution}"),
                _get_float(
                    stereo,
                    f"CV_{resolution}",
                    _get_float(stereo, f"RY_{resolution}"),
                ),
                _get_float(stereo, f"RZ_{resolution}"),
            ],
            dtype=np.float64,
        )
        rotation, _ = cv2.Rodrigues(rotation_vector)
        translation = np.array(
            [
                [-_get_float(stereo, "Baseline")],
                [
                    _get_float(
                        stereo,
                        f"TY_{resolution}",
                        _get_float(stereo, "TY"),
                    )
                ],
                [
                    _get_float(
                        stereo,
                        f"TZ_{resolution}",
                        _get_float(stereo, "TZ"),
                    )
                ],
            ],
            dtype=np.float64,
        )

        size = (int(width), int(height))
        (
            left_rectification,
            right_rectification,
            left_projection,
            right_projection,
            _,
            _,
            _,
        ) = cv2.stereoRectify(
            cameraMatrix1=left_matrix,
            distCoeffs1=left_distortion,
            cameraMatrix2=right_matrix,
            distCoeffs2=right_distortion,
            imageSize=size,
            R=rotation,
            T=translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
            newImageSize=size,
        )
        self.left_map = cv2.initUndistortRectifyMap(
            left_matrix,
            left_distortion,
            left_rectification,
            left_projection,
            size,
            cv2.CV_16SC2,
        )
        self.right_map = cv2.initUndistortRectifyMap(
            right_matrix,
            right_distortion,
            right_rectification,
            right_projection,
            size,
            cv2.CV_16SC2,
        )
        self.resolution = resolution

    def rectify(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        left = cv2.remap(
            left_bgr,
            self.left_map[0],
            self.left_map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        right = cv2.remap(
            right_bgr,
            self.right_map[0],
            self.right_map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return left, right


def detect_stereo_layout(
    first_bgr: np.ndarray,
    second_bgr: np.ndarray,
) -> tuple[str, str]:

    scale = min(1.0, 640.0 / max(first_bgr.shape[1], 1))
    if scale < 1.0:
        size = (
            max(1, int(first_bgr.shape[1] * scale)),
            max(1, int(first_bgr.shape[0] * scale)),
        )
        first_bgr = cv2.resize(first_bgr, size, interpolation=cv2.INTER_AREA)
        second_bgr = cv2.resize(second_bgr, size, interpolation=cv2.INTER_AREA)

    first_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    keypoints_a, descriptors_a = detector.detectAndCompute(first_gray, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(second_gray, None)
    if descriptors_a is None or descriptors_b is None:
        return "left-right", "insufficient features; using ZED native default"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    disparities = []
    max_vertical = max(8.0, first_gray.shape[0] * 0.04)
    for pair in pairs:
        if len(pair) != 2:
            continue
        best, second_best = pair
        if best.distance >= 0.72 * second_best.distance:
            continue
        point_a = keypoints_a[best.queryIdx].pt
        point_b = keypoints_b[best.trainIdx].pt
        if abs(point_a[1] - point_b[1]) <= max_vertical:
            disparities.append(point_a[0] - point_b[0])

    if len(disparities) < 12:
        return "left-right", (
            f"only {len(disparities)} reliable matches; "
            "using ZED native default"
        )
    median = float(np.median(disparities))
    layout = "left-right" if median >= 0.0 else "right-left"
    return layout, (
        f"{len(disparities)} matches, median first-minus-second "
        f"disparity={median:.2f}px"
    )
