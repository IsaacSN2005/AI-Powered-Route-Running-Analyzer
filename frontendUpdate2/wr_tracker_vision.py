"""
WR vision tracker with field-plane calibration.

Built for wide-receiver route clips where a single athlete needs to be
tracked precisely enough to estimate speed and acceleration. The core idea is:

1. Detect people with YOLO pose so we have stable candidate boxes.
2. Let the user lock the target on the first frame.
3. Refine the player footprint inside each box with a pixel mask.
4. Project the foot-contact point into field coordinates with a homography.
5. Compute speed / acceleration in world space rather than raw pixels.

Install:
    pip install ultralytics opencv-python numpy pandas

Run:
    python wr_tracker_vision.py --video SlantRoute/CodySlant.mov
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


DEFAULT_SIDE_VIEW_MODEL = "yolov8x-pose.pt"
DEFAULT_DRONE_MODEL = "yolov8x.pt"
DEFAULT_DETECT_EVERY = 3
DEFAULT_CONFIDENCE = 0.25
DEFAULT_TRACK_WINDOW = 140
DEFAULT_AUTO_FIELD_LENGTH_YARDS = 40.0
DEFAULT_AUTO_FIELD_WIDTH_YARDS = 18.0
DEFAULT_DRONE_FIELD_LENGTH_YARDS = 53.3
DEFAULT_DRONE_FIELD_WIDTH_YARDS = 26.7
SMOOTH_ALPHA = 0.35
ACCEL_ALPHA = 0.4
WORLD_POINT_ALPHA = 0.3
SPEED_MEDIAN_WINDOW = 7
SUSTAINED_PEAK_PERCENTILE = 95
MAX_FOOT_JUMP_RATIO = 0.12
PROGRESS_PREVIEW_EVERY_N_FRAMES = 1
SOFT_SPEED_THRESHOLD_MPH = 19.0
MAX_REASONABLE_SPEED_MPH = 23.0
REP_START_SPEED_MPH = 3.0
REP_END_SPEED_MPH = 1.75
REP_START_MIN_FRAMES = 1
REP_END_MIN_FRAMES = 6
REP_MIN_DISTANCE_YARDS = 0.5
REP_START_BACKTRACK_FRAMES = 6
MIN_HIP_BOX_HEIGHT_PX = 120.0
MIN_HIP_VALID_FRAMES = 4
MIN_CUT_ANGLE_DEG = 18.0
MIN_MAJOR_CUT_ANGLE_DEG = 28.0
MIN_CUT_SPEED_DROP_MPH = 1.25
HEADING_LAG_FRAMES = 3
SPEED_LOOKBACK_FRAMES = 4
KINEMATICS_POSITION_WINDOW = 9
KINEMATICS_DERIVATIVE_HALF_WINDOW = 3

KP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

POSE_SKELETON_EDGES = [
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]

C = {
    "green": (0, 220, 80),
    "cyan": (0, 220, 220),
    "yellow": (0, 220, 255),
    "orange": (0, 140, 255),
    "white": (255, 255, 255),
    "dim": (150, 150, 150),
    "bg": (10, 10, 10),
    "red": (50, 50, 255),
}


def smooth(new_value: float, old_value: float | None, alpha: float) -> float:
    if old_value is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * old_value


def yards_per_second_to_mph(yards_per_second: float) -> float:
    return yards_per_second * 3600.0 / 1760.0


def image_to_world(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    src = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    xw, yw = dst[0, 0]
    return float(xw), float(yw)


def midpoint(p1: tuple[float, float] | None, p2: tuple[float, float] | None) -> tuple[float, float] | None:
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def kp_point(kpts_xy: np.ndarray, kpts_conf: np.ndarray | None, index: int, conf_threshold: float = 0.25):
    x, y = float(kpts_xy[index][0]), float(kpts_xy[index][1])
    if x <= 1.0 and y <= 1.0:
        return None
    if kpts_conf is not None and float(kpts_conf[index]) < conf_threshold:
        return None
    return (x, y)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_center(box: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def box_xyxy_to_xywh(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1, y1, max(2.0, x2 - x1), max(2.0, y2 - y1))


def box_xywh_to_xyxy(box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def clamp_box_to_frame(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(0.0, x1), max(0.0, width - 2.0))
    y1 = min(max(0.0, y1), max(0.0, height - 2.0))
    x2 = min(max(x1 + 2.0, x2), float(width - 1))
    y2 = min(max(y1 + 2.0, y2), float(height - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def create_box_tracker():
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    raise RuntimeError("OpenCV MIL tracker is not available in this build.")


def wrap_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def acute_angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    if theta > 180.0:
        theta = 360.0 - theta
    return min(theta, 180.0 - theta if theta > 90.0 else theta)


def full_turn_angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    if theta > 180.0:
        theta = 360.0 - theta
    return theta


def signed_turn_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    angle = full_turn_angle_between_deg(v1, v2)
    cross_z = float(v1[0] * v2[1] - v1[1] * v2[0])
    if abs(cross_z) <= 1e-6:
        return angle
    return angle if cross_z > 0 else -angle


def smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values.copy()
    half = window // 2
    out = np.zeros_like(values, dtype=float)
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out[idx] = float(np.mean(values[lo:hi]))
    return out


def heading_profile_deg(points: np.ndarray, lag: int = HEADING_LAG_FRAMES) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    lag = max(1, int(lag))
    headings = np.full(len(points), np.nan, dtype=float)
    for idx in range(lag, len(points)):
        delta = points[idx] - points[idx - lag]
        if float(np.linalg.norm(delta)) <= 1e-6:
            continue
        headings[idx] = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    return headings


def circular_mean_deg(angles_deg: np.ndarray) -> float | None:
    valid = np.asarray(angles_deg, dtype=float)
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return None
    radians = np.deg2rad(valid)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    if abs(mean_sin) <= 1e-6 and abs(mean_cos) <= 1e-6:
        return None
    return math.degrees(math.atan2(mean_sin, mean_cos))


def angle_delta_deg(a_deg: float, b_deg: float) -> float:
    return abs(wrap_angle_deg(a_deg - b_deg))


def fit_segment_direction(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 2:
        return None
    centered = points - points.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) <= 1e-6:
        return None
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    return direction / norm


def homography_rms_error(
    image_points: np.ndarray,
    world_points: np.ndarray,
    homography: np.ndarray,
) -> float:
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_points, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def calibration_confidence_from_rms(rms_error_yards: float) -> float:
    """
    Convert homography RMS error into a 0..1 confidence score.

    0.0 yd RMS -> 1.0 confidence
    2.0+ yd RMS -> trends toward 0.0 confidence
    """
    score = 1.0 - (rms_error_yards / 2.0)
    return max(0.0, min(1.0, score))


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmax(d)],
        pts[np.argmax(s)],
        pts[np.argmin(d)],
    ], dtype=np.float32)


def robust_peak_speed_mph(speed_values: np.ndarray) -> float:
    if len(speed_values) == 0:
        return 0.0
    return float(np.percentile(speed_values, SUSTAINED_PEAK_PERCENTILE))


def soft_clip_speed_yps(speed_yps: float) -> float:
    """
    Compress unrealistic top-end speeds without flattening them to one value.

    Below the threshold, leave the value untouched. Above it, apply a smooth
    saturation curve that still preserves relative differences.
    """
    threshold = SOFT_SPEED_THRESHOLD_MPH * 1760.0 / 3600.0
    hard_max = MAX_REASONABLE_SPEED_MPH * 1760.0 / 3600.0
    if speed_yps <= threshold:
        return speed_yps
    excess = speed_yps - threshold
    span = max(hard_max - threshold, 1e-6)
    compressed = threshold + span * (1.0 - math.exp(-excess / span))
    return min(compressed, hard_max)


def recompute_frame_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate speed and acceleration for every frame from the final tracked
    world positions, instead of relying only on the online tracker state.

    This makes mph/accel less sensitive to one-frame tracking jitter and keeps
    the per-frame CSV aligned to the final smoothed path.
    """
    if df.empty or len(df) < 3:
        return df.copy()

    work = df.copy().reset_index(drop=True)
    x = smooth_series(work["field_x_yd"].to_numpy(dtype=float), window=KINEMATICS_POSITION_WINDOW)
    y = smooth_series(work["field_y_yd"].to_numpy(dtype=float), window=KINEMATICS_POSITION_WINDOW)
    t = work["time_s"].to_numpy(dtype=float)
    half = max(1, int(KINEMATICS_DERIVATIVE_HALF_WINDOW))
    n = len(work)

    speed_yps = np.zeros(n, dtype=float)
    for idx in range(n):
        if idx < half:
            lo = idx
            hi = min(n - 1, idx + half)
        elif idx > n - 1 - half:
            lo = max(0, idx - half)
            hi = idx
        else:
            lo = max(0, idx - half)
            hi = min(n - 1, idx + half)
        dt = float(t[hi] - t[lo])
        if dt <= 1e-6:
            continue
        dist = math.hypot(float(x[hi] - x[lo]), float(y[hi] - y[lo]))
        speed_yps[idx] = dist / dt

    if n >= 5:
        speed_yps = smooth_series(speed_yps, window=5)
    edge_pad = min(half, n - 1)
    if edge_pad > 0:
        speed_yps[:edge_pad] = np.minimum.accumulate(speed_yps[:edge_pad][::-1])[::-1]
        speed_yps[n - edge_pad:] = np.minimum.accumulate(speed_yps[n - edge_pad:])
    speed_yps = np.array([soft_clip_speed_yps(float(v)) for v in speed_yps], dtype=float)

    accel_yps2 = np.zeros(n, dtype=float)
    for idx in range(n):
        if idx < half:
            lo = idx
            hi = min(n - 1, idx + half)
        elif idx > n - 1 - half:
            lo = max(0, idx - half)
            hi = idx
        else:
            lo = max(0, idx - half)
            hi = min(n - 1, idx + half)
        dt = float(t[hi] - t[lo])
        if dt <= 1e-6:
            continue
        accel_yps2[idx] = float(speed_yps[hi] - speed_yps[lo]) / dt

    if n >= 5:
        accel_yps2 = smooth_series(accel_yps2, window=5)

    work["speed_yards_per_sec"] = np.round(speed_yps, 3)
    work["speed_mph"] = np.round([yards_per_second_to_mph(v) for v in speed_yps], 3)
    work["accel_yards_per_sec2"] = np.round(accel_yps2, 3)
    work["accel_mph_per_sec"] = np.round([yards_per_second_to_mph(v) for v in accel_yps2], 3)
    return work


def compute_route_phase_metrics(
    df: pd.DataFrame,
    cut_frame_override: int | None = None,
    include_pose_metrics: bool = True,
) -> dict:
    """
    Derive football-specific route metrics from world-space tracking.

    - Acceleration off the line: average acceleration from first movement
      through the first 3 yards of path distance.
    - Deceleration into the cut: speed drop leading into the first meaningful
      direction change.
    - Cut time: how long the athlete spends changing direction.
    """
    if df.empty or len(df) < 8:
        return {}

    x = smooth_series(df["field_x_yd"].to_numpy(dtype=float), window=5)
    y = smooth_series(df["field_y_yd"].to_numpy(dtype=float), window=5)
    t = df["time_s"].to_numpy(dtype=float)
    speed_mph = smooth_series(df["speed_mph"].to_numpy(dtype=float), window=5)

    step_dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cumulative_dist = np.concatenate([[0.0], np.cumsum(step_dist)])

    off_line = {
        "off_line_start_frame": None,
        "off_line_3yd_time_s": None,
        "off_line_3yd_accel_mph_per_sec": None,
        "off_line_end_speed_mph": None,
    }
    release_start_idx = 0
    early_scan_end = min(len(df) - 1, max(10, len(df) // 4))
    for idx in range(early_scan_end + 1):
        dist_from_here = cumulative_dist[min(len(cumulative_dist) - 1, early_scan_end)] - cumulative_dist[idx]
        if speed_mph[idx] <= 2.0 and dist_from_here >= 0.5:
            release_start_idx = idx
            break

    off_line["off_line_start_frame"] = int(df["frame"].iloc[release_start_idx])
    off_line_idx = int(np.searchsorted(cumulative_dist, cumulative_dist[release_start_idx] + 3.0, side="left"))
    if release_start_idx < off_line_idx < len(df):
        dt = t[off_line_idx] - t[release_start_idx]
        if dt > 1e-6:
            off_line["off_line_3yd_time_s"] = round(float(dt), 3)
            off_line["off_line_end_speed_mph"] = round(float(speed_mph[off_line_idx]), 3)
            off_line["off_line_3yd_accel_mph_per_sec"] = round(float((speed_mph[off_line_idx] - speed_mph[release_start_idx]) / dt), 3)

    cut = {
        "cut_frame": None,
        "cut_time_s": None,
        "cut_entry_speed_mph": None,
        "cut_exit_speed_mph": None,
        "cut_decel_mph_per_sec": None,
        "cut_direction_change_deg": None,
        "actual_path_cut_angle_deg": None,
        "full_turn_angle_deg": None,
        "signed_turn_angle_deg": None,
        "idealized_cut_angle_deg": None,
        "break_style": None,
        "break_side": None,
        "cut_confidence": "low",
        "hip_drop_px": None,
        "hip_drop_pct_body_height": None,
        "hip_drop_time_s": None,
        "hip_drop_confidence": "low",
        "cut_point_field_x_yd": None,
        "cut_point_field_y_yd": None,
        "cut_window_start_frame": None,
        "cut_window_end_frame": None,
        "stem_start_field_x_yd": None,
        "stem_start_field_y_yd": None,
        "stem_end_field_x_yd": None,
        "stem_end_field_y_yd": None,
        "break_start_field_x_yd": None,
        "break_start_field_y_yd": None,
        "break_end_field_x_yd": None,
        "break_end_field_y_yd": None,
    }
    points = np.column_stack([x, y])
    if len(points) >= 16:
        stem_seed_end = min(len(points) - 8, max(8, len(points) // 5))
        stem_seed = points[: stem_seed_end + 1]
        stem_dir = fit_segment_direction(stem_seed)
        headings = heading_profile_deg(points, lag=HEADING_LAG_FRAMES)
        stem_heading = circular_mean_deg(headings[HEADING_LAG_FRAMES: stem_seed_end + 1])
        min_break_distance_yd = min(4.0, max(2.5, cumulative_dist[-1] * 0.25))
        best = None
        frame_values = df["frame"].to_numpy(dtype=int)
        forced_cut_idx = None
        if cut_frame_override is not None:
            forced_hits = np.where(frame_values >= int(cut_frame_override))[0]
            if len(forced_hits):
                forced_cut_idx = int(forced_hits[0])

        start_idx = max(8, stem_seed_end)
        end_idx = len(points) - 8
        if forced_cut_idx is not None:
            start_idx = max(start_idx, forced_cut_idx - 2)
            end_idx = min(end_idx, forced_cut_idx + 2)

        for idx in range(start_idx, end_idx):
            if forced_cut_idx is None and cumulative_dist[idx] < min_break_distance_yd:
                continue
            pre_points = points[max(0, idx - 8): idx + 1]
            post_points = points[idx: idx + 9]
            pre_dir = fit_segment_direction(pre_points)
            if pre_dir is None:
                pre_dir = stem_dir
            post_dir = fit_segment_direction(post_points)
            if pre_dir is None or post_dir is None or stem_dir is None:
                continue
            pre_heading = circular_mean_deg(headings[max(0, idx - 5): idx + 1])
            post_heading = circular_mean_deg(headings[idx: min(len(headings), idx + 6)])
            if stem_heading is not None and post_heading is not None:
                turn_deg = angle_delta_deg(post_heading, stem_heading)
                if turn_deg > 180.0:
                    turn_deg = 360.0 - turn_deg
                if turn_deg > 90.0:
                    turn_deg = 180.0 - turn_deg
            else:
                turn_deg = acute_angle_between_deg(stem_dir, post_dir)
            actual_turn = acute_angle_between_deg(pre_dir, post_dir)
            pre_axis = np.array([1.0, 0.0]) if abs(pre_dir[0]) >= abs(pre_dir[1]) else np.array([0.0, 1.0])
            post_axis = np.array([1.0, 0.0]) if abs(post_dir[0]) >= abs(post_dir[1]) else np.array([0.0, 1.0])
            idealized_turn = acute_angle_between_deg(pre_axis, post_axis)
            speed_window = speed_mph[max(0, idx - 6): min(len(speed_mph), idx + 7)]
            if len(speed_window) < 8 or max(turn_deg, actual_turn, idealized_turn) < MIN_CUT_ANGLE_DEG:
                continue
            pre_span = float(np.linalg.norm(pre_points[-1] - pre_points[0]))
            post_span = float(np.linalg.norm(post_points[-1] - post_points[0]))
            if pre_span < 1.5 or post_span < 1.5:
                continue
            entry_speed = float(np.max(speed_mph[max(0, idx - 6): idx + 1]))
            exit_speed = float(np.min(speed_mph[idx: min(len(speed_mph), idx + 7)]))
            speed_drop = max(0.0, entry_speed - exit_speed)
            speed_factor = max(float(speed_window.max()), 1.0)
            early_bias = 1.0 / (1.0 + idx * 0.015)
            angle_support = 1.0
            reverse_component = 0.0
            pre_norm = float(np.linalg.norm(pre_dir))
            post_norm = float(np.linalg.norm(post_dir))
            if pre_norm > 1e-6 and post_norm > 1e-6:
                reverse_component = float(np.dot(pre_dir / pre_norm, post_dir / post_norm))
            if idealized_turn >= 75.0:
                angle_support += 0.65
            elif actual_turn >= 45.0:
                angle_support += 0.35
            elif turn_deg >= MIN_MAJOR_CUT_ANGLE_DEG:
                angle_support += 0.15
            if reverse_component <= -0.2:
                angle_support += 0.65
            elif reverse_component <= 0.1 and speed_drop >= MIN_CUT_SPEED_DROP_MPH:
                angle_support += 0.2
            drop_support = 1.0 + min(speed_drop / 8.0, 0.5)
            score = max(turn_deg, actual_turn) * speed_factor * min(pre_span, post_span) * early_bias * angle_support * drop_support
            candidate = {
                "idx": idx,
                "turn_deg": turn_deg,
                "actual_turn": actual_turn,
                "idealized_turn": idealized_turn,
                "reverse_component": reverse_component,
                "score": score,
                "pre_dir": pre_dir,
                "post_dir": post_dir,
                "pre_points": pre_points.copy(),
                "post_points": post_points.copy(),
                "entry_speed": entry_speed,
                "exit_speed": exit_speed,
                "speed_drop": speed_drop,
            }

            # Prefer the first meaningful break off the stem rather than a later
            # shallow redirect. Only replace an earlier candidate if the new one
            # is materially stronger.
            if best is None:
                best = candidate
                continue

            significantly_earlier = (
                idx < best["idx"] - 4
                and (
                    turn_deg >= best["turn_deg"] * 0.8
                    or speed_drop >= best.get("speed_drop", 0.0) + 0.75
                    or reverse_component <= best.get("reverse_component", 1.0) - 0.2
                )
            )
            materially_stronger = score > best["score"] * 1.25
            if significantly_earlier or materially_stronger:
                best = {
                    **candidate,
                }

        if best is not None:
            idx = best["idx"]
            pre_points = best["pre_points"]
            post_points = best["post_points"]
            entry_start = max(0, idx - 6)
            exit_end = min(len(df) - 1, idx + 6)
            entry_speed = best["entry_speed"]
            exit_speed = best["exit_speed"]
            dt = float(t[exit_end] - t[entry_start])
            if dt > 1e-6:
                stem_vec = stem_seed[-1] - stem_seed[0] if len(stem_seed) >= 2 else pre_points[-1] - pre_points[0]
                break_vec = post_points[-1] - post_points[0]
                actual_turn_deg = acute_angle_between_deg(stem_vec, break_vec)
                full_turn_deg = full_turn_angle_between_deg(stem_vec, break_vec)
                signed_turn_deg = signed_turn_angle_deg(stem_vec, break_vec)
                cross_z = float(stem_vec[0] * break_vec[1] - stem_vec[1] * break_vec[0])
                break_side = "straight"
                if cross_z > 1e-6:
                    break_side = "left"
                elif cross_z < -1e-6:
                    break_side = "right"

                stem_norm = float(np.linalg.norm(stem_vec))
                break_norm = float(np.linalg.norm(break_vec))
                reverse_component = 0.0
                lateral_component = 0.0
                if stem_norm > 1e-6 and break_norm > 1e-6:
                    stem_unit = stem_vec / stem_norm
                    break_unit = break_vec / break_norm
                    reverse_component = float(np.dot(stem_unit, break_unit))
                    lateral_component = float(stem_unit[0] * break_unit[1] - stem_unit[1] * break_unit[0])

                # "Idealized" break angle uses axis-aligned stem/break directions,
                # which is more stable when the athlete rounds the cut a bit.
                stem_axis = np.array([1.0, 0.0]) if abs(stem_vec[0]) >= abs(stem_vec[1]) else np.array([0.0, 1.0])
                break_axis = np.array([1.0, 0.0]) if abs(break_vec[0]) >= abs(break_vec[1]) else np.array([0.0, 1.0])
                axis_turn_deg = acute_angle_between_deg(stem_axis, break_axis)
                if reverse_component <= -0.8:
                    idealized_turn_deg = 180.0
                elif reverse_component <= -0.25:
                    idealized_turn_deg = 135.0
                elif axis_turn_deg >= 75.0:
                    idealized_turn_deg = 90.0
                elif 20.0 <= actual_turn_deg <= 70.0:
                    idealized_turn_deg = 45.0
                else:
                    idealized_turn_deg = round(float(axis_turn_deg), 3)

                cut["cut_frame"] = int(df["frame"].iloc[idx])
                cut["cut_time_s"] = round(dt, 3)
                cut["cut_entry_speed_mph"] = round(entry_speed, 3)
                cut["cut_exit_speed_mph"] = round(exit_speed, 3)
                cut["cut_decel_mph_per_sec"] = round((exit_speed - entry_speed) / dt, 3)
                cut["cut_direction_change_deg"] = round(float(actual_turn_deg), 3)
                cut["actual_path_cut_angle_deg"] = round(float(actual_turn_deg), 3)
                cut["full_turn_angle_deg"] = round(float(full_turn_deg), 3)
                cut["signed_turn_angle_deg"] = round(float(signed_turn_deg), 3)
                cut["idealized_cut_angle_deg"] = round(float(idealized_turn_deg), 3)
                cut["break_side"] = break_side
                if full_turn_deg >= 150.0 or actual_turn_deg >= max(70.0, axis_turn_deg * 0.8):
                    cut["break_style"] = "sharp"
                else:
                    cut["break_style"] = "rounded"
                if idealized_turn_deg >= 90.0 and (actual_turn_deg >= 45.0 or full_turn_deg >= 120.0) and best["speed_drop"] >= MIN_CUT_SPEED_DROP_MPH:
                    cut["cut_confidence"] = "high"
                elif actual_turn_deg >= 25.0 or full_turn_deg >= 110.0 or abs(cut["cut_decel_mph_per_sec"]) >= 5.0 or best["speed_drop"] >= MIN_CUT_SPEED_DROP_MPH:
                    cut["cut_confidence"] = "medium"
                cut["cut_point_field_x_yd"] = round(float(x[idx]), 3)
                cut["cut_point_field_y_yd"] = round(float(y[idx]), 3)
                cut["cut_window_start_frame"] = int(df["frame"].iloc[entry_start])
                cut["cut_window_end_frame"] = int(df["frame"].iloc[exit_end])
                cut["stem_start_field_x_yd"] = round(float(pre_points[0][0]), 3)
                cut["stem_start_field_y_yd"] = round(float(pre_points[0][1]), 3)
                cut["stem_end_field_x_yd"] = round(float(pre_points[-1][0]), 3)
                cut["stem_end_field_y_yd"] = round(float(pre_points[-1][1]), 3)
                cut["break_start_field_x_yd"] = round(float(post_points[0][0]), 3)
                cut["break_start_field_y_yd"] = round(float(post_points[0][1]), 3)
                cut["break_end_field_x_yd"] = round(float(post_points[-1][0]), 3)
                cut["break_end_field_y_yd"] = round(float(post_points[-1][1]), 3)

                if include_pose_metrics and "hip_mid_y_px" in df.columns:
                    hip_series = df["hip_mid_y_px"].to_numpy(dtype=float)
                    box_h_series = df["box_height_px"].to_numpy(dtype=float) if "box_height_px" in df.columns else None
                    baseline_start = max(0, idx - 8)
                    baseline_end = max(baseline_start + 1, idx - 3)
                    sink_start = max(0, idx - 4)
                    sink_end = min(len(df) - 1, idx + 1)

                    baseline_hips = hip_series[baseline_start:baseline_end]
                    baseline_hips = baseline_hips[~np.isnan(baseline_hips)]
                    sink_hips = hip_series[sink_start:sink_end + 1]
                    sink_hips = sink_hips[~np.isnan(sink_hips)]
                    if len(baseline_hips) >= MIN_HIP_VALID_FRAMES and len(sink_hips) >= 2:
                        baseline_hip_y = float(np.mean(baseline_hips))
                        max_sink_hip_y = float(np.max(sink_hips))
                        hip_drop_px = max(0.0, max_sink_hip_y - baseline_hip_y)
                        avg_body_h = None
                        if box_h_series is not None:
                            body_heights = box_h_series[baseline_start:sink_end + 1]
                            body_heights = body_heights[~np.isnan(body_heights)]
                            if len(body_heights) > 0:
                                avg_body_h = float(np.mean(body_heights))
                        if (
                            avg_body_h is not None
                            and avg_body_h >= MIN_HIP_BOX_HEIGHT_PX
                            and hip_drop_px > 0.0
                            and cut["cut_confidence"] in {"medium", "high"}
                        ):
                            cut["hip_drop_px"] = round(hip_drop_px, 3)
                            cut["hip_drop_pct_body_height"] = round((hip_drop_px / avg_body_h) * 100.0, 3)
                            sink_slice = hip_series[sink_start:sink_end + 1]
                            sink_idx_local = int(np.nanargmax(sink_slice))
                            sink_idx = sink_start + sink_idx_local
                            cut["hip_drop_time_s"] = round(float(t[sink_idx] - t[baseline_end - 1]), 3)
                            if cut["cut_confidence"] == "high" and cut["hip_drop_pct_body_height"] >= 2.0:
                                cut["hip_drop_confidence"] = "high"
                            elif cut["hip_drop_pct_body_height"] >= 1.0:
                                cut["hip_drop_confidence"] = "medium"

    return {**off_line, **cut}


def guess_route_from_phase_metrics(phase_metrics: dict) -> dict:
    actual_angle = phase_metrics.get("actual_path_cut_angle_deg")
    full_turn_angle = phase_metrics.get("full_turn_angle_deg")
    ideal_angle = phase_metrics.get("idealized_cut_angle_deg")
    decel = phase_metrics.get("cut_decel_mph_per_sec")
    cut_confidence = phase_metrics.get("cut_confidence")
    break_style = phase_metrics.get("break_style")
    break_side = phase_metrics.get("break_side")

    stem_start_x = phase_metrics.get("stem_start_field_x_yd")
    stem_end_x = phase_metrics.get("stem_end_field_x_yd")
    stem_start_y = phase_metrics.get("stem_start_field_y_yd")
    stem_end_y = phase_metrics.get("stem_end_field_y_yd")
    break_start_x = phase_metrics.get("break_start_field_x_yd")
    break_end_x = phase_metrics.get("break_end_field_x_yd")
    break_start_y = phase_metrics.get("break_start_field_y_yd")
    break_end_y = phase_metrics.get("break_end_field_y_yd")

    result = {
        "route_guess": "Unknown",
        "route_confidence": "low",
        "route_reason": "Not enough route geometry.",
    }

    needed = [
        actual_angle, full_turn_angle, ideal_angle, decel,
        stem_start_x, stem_end_x, stem_start_y, stem_end_y,
        break_start_x, break_end_x, break_start_y, break_end_y,
    ]
    if any(v is None for v in needed):
        return result
    if cut_confidence == "low":
        result["route_reason"] = "Cut geometry confidence is low."
        return result

    stem_dx = float(stem_end_x - stem_start_x)
    stem_dy = float(stem_end_y - stem_start_y)
    break_dx = float(break_end_x - break_start_x)
    break_dy = float(break_end_y - break_start_y)

    stem_horizontal = abs(stem_dx) >= abs(stem_dy)
    break_horizontal = abs(break_dx) >= abs(break_dy)
    stem_vec = np.array([stem_dx, stem_dy], dtype=float)
    break_vec = np.array([break_dx, break_dy], dtype=float)
    stem_norm = float(np.linalg.norm(stem_vec))
    break_norm = float(np.linalg.norm(break_vec))
    reverse_component = 0.0
    if stem_norm > 1e-6 and break_norm > 1e-6:
        reverse_component = float(np.dot(stem_vec / stem_norm, break_vec / break_norm))

    stem_axis_idx = 0 if abs(stem_dx) >= abs(stem_dy) else 1
    stem_axis_sign = 1.0 if (stem_dx if stem_axis_idx == 0 else stem_dy) >= 0 else -1.0
    break_along_stem = ((break_dx if stem_axis_idx == 0 else break_dy) * stem_axis_sign)
    break_cross_stem = (break_dy if stem_axis_idx == 0 else break_dx)
    normalized_break_along = break_along_stem / max(break_norm, 1e-6)
    normalized_break_cross = abs(break_cross_stem) / max(break_norm, 1e-6)

    if ideal_angle >= 170 or full_turn_angle >= 155 or (normalized_break_along <= -0.8 and normalized_break_cross <= 0.45):
        result["route_guess"] = "Curl"
        result["route_confidence"] = "high" if decel <= -5 else "medium"
        result["route_reason"] = "The route reverses sharply back toward the stem."
    elif (
        ideal_angle >= 125
        or full_turn_angle >= 115
        or reverse_component <= -0.2
        or (normalized_break_along <= -0.2 and normalized_break_cross >= 0.2)
    ):
        result["route_guess"] = "Comeback"
        result["route_confidence"] = "high" if decel <= -5 else "medium"
        if break_style == "rounded":
            result["route_reason"] = "The route breaks back on a rounded angle instead of reversing straight back."
        else:
            result["route_reason"] = "The route breaks back toward the stem on an angled return."
    elif ideal_angle >= 75:
        if stem_horizontal != break_horizontal:
            result["route_guess"] = "Out / Dig / Corner Family"
            result["route_confidence"] = "medium"
            result["route_reason"] = "Stem and break are close to perpendicular."
            if break_horizontal and abs(break_dx) > abs(break_dy):
                result["route_guess"] = "Out Route"
                result["route_confidence"] = "high" if actual_angle >= 50 else "medium"
                result["route_reason"] = "Perpendicular break with a lateral finish."
            elif not break_horizontal and abs(break_dy) > abs(break_dx):
                result["route_guess"] = "Dig / Square-In"
                result["route_reason"] = "Perpendicular break with an inside vertical-field finish."
    elif (20 <= ideal_angle <= 70 or 20 <= actual_angle <= 70) and normalized_break_along > -0.15:
        result["route_guess"] = "Slant"
        result["route_confidence"] = "medium"
        turn_label = f"{break_side} break" if break_side in {"left", "right"} else "angled break"
        result["route_reason"] = f"Moderate angled change suggests a slant-type {turn_label}."

    return result


def save_route_debug_plot(df: pd.DataFrame, phase_metrics: dict, route_guess: dict, out_path: Path):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0b0f12")
    ax.set_facecolor("#0b0f12")

    ax.plot(
        df["field_x_yd"],
        df["field_y_yd"],
        color="#5dd39e",
        linewidth=2.0,
        label="Tracked path",
    )
    ax.scatter(
        df["field_x_yd"],
        df["field_y_yd"],
        c=df["speed_mph"],
        cmap="viridis",
        s=14,
        alpha=0.8,
    )

    if phase_metrics.get("stem_start_field_x_yd") is not None:
        ax.plot(
            [phase_metrics["stem_start_field_x_yd"], phase_metrics["stem_end_field_x_yd"]],
            [phase_metrics["stem_start_field_y_yd"], phase_metrics["stem_end_field_y_yd"]],
            color="#ffd166",
            linewidth=3,
            label="Fitted stem",
        )

    if phase_metrics.get("break_start_field_x_yd") is not None:
        ax.plot(
            [phase_metrics["break_start_field_x_yd"], phase_metrics["break_end_field_x_yd"]],
            [phase_metrics["break_start_field_y_yd"], phase_metrics["break_end_field_y_yd"]],
            color="#ef476f",
            linewidth=3,
            label="Fitted break",
        )

    if phase_metrics.get("cut_point_field_x_yd") is not None:
        ax.scatter(
            [phase_metrics["cut_point_field_x_yd"]],
            [phase_metrics["cut_point_field_y_yd"]],
            color="#ffffff",
            s=80,
            marker="x",
            linewidths=2,
            label="Detected cut",
        )

    title = "Route Debug View (Trimmed Rep)"
    if phase_metrics.get("cut_direction_change_deg") is not None:
        title += f"  |  Cut {phase_metrics['cut_direction_change_deg']:.1f} deg"
    if route_guess.get("route_guess") not in (None, "Unknown"):
        title += f"  |  {route_guess['route_guess']}"
    ax.set_title(title, color="white", fontsize=14)
    ax.set_xlabel("Field X (yd)", color="white")
    ax.set_ylabel("Field Y (yd)", color="white")
    ax.tick_params(colors="#b8c1cc")
    for spine in ax.spines.values():
        spine.set_color("#33404d")
    ax.grid(color="#24303b", alpha=0.35)
    ax.legend(facecolor="#10161b", edgecolor="#33404d", labelcolor="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_clean_metrics_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "time_s",
                "field_x_yd",
                "field_y_yd",
                "speed_mph",
                "accel_mph_per_sec",
            ]
        )
    return df[
        [
            "frame",
            "time_s",
            "field_x_yd",
            "field_y_yd",
            "speed_mph",
            "accel_mph_per_sec",
        ]
    ].copy()


def build_pose_points_csv(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["frame", "time_s", "image_x_px", "image_y_px", "field_x_yd", "field_y_yd"]
    pose_cols: list[str] = []
    for name in KP:
        pose_cols.extend([f"{name}_x_px", f"{name}_y_px", f"{name}_conf"])
    existing_cols = [col for col in base_cols + pose_cols if col in df.columns]
    if not existing_cols:
        return pd.DataFrame(columns=base_cols)
    return df[existing_cols].copy()


def trim_to_rep_window(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty or len(df) < max(REP_START_MIN_FRAMES, REP_END_MIN_FRAMES) + 2:
        return df.copy(), {"rep_start_frame": None, "rep_end_frame": None, "rep_duration_s": None}

    work = df.reset_index(drop=True).copy()
    speed = work["speed_mph"].to_numpy(dtype=float)
    x = work["field_x_yd"].to_numpy(dtype=float)
    y = work["field_y_yd"].to_numpy(dtype=float)
    time_s = work["time_s"].to_numpy(dtype=float)

    step_dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cumulative = np.concatenate([[0.0], np.cumsum(step_dist)])

    start_idx = 0
    consecutive = 0
    candidate_start = None
    for idx, mph in enumerate(speed):
        dist_from_start = cumulative[idx] - cumulative[0]
        if mph >= REP_START_SPEED_MPH or dist_from_start >= REP_MIN_DISTANCE_YARDS:
            if candidate_start is None:
                candidate_start = idx
            consecutive += 1
        else:
            consecutive = 0
            candidate_start = None
        if consecutive >= REP_START_MIN_FRAMES:
            start_idx = max(0, candidate_start if candidate_start is not None else idx - REP_START_MIN_FRAMES + 1)
            start_idx = max(0, start_idx - REP_START_BACKTRACK_FRAMES)
            break

    end_idx = len(work) - 1
    consecutive = 0
    peak_idx = int(np.argmax(speed))
    search_from = max(start_idx + 1, peak_idx)
    for idx in range(search_from, len(work)):
        mph = speed[idx]
        if mph <= REP_END_SPEED_MPH:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= REP_END_MIN_FRAMES:
            end_idx = idx - REP_END_MIN_FRAMES + 1
            break

    if end_idx <= start_idx:
        end_idx = len(work) - 1

    trimmed = work.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    rep_info = {
        "rep_start_frame": int(work["frame"].iloc[start_idx]),
        "rep_end_frame": int(work["frame"].iloc[end_idx]),
        "rep_duration_s": round(float(time_s[end_idx] - time_s[start_idx]), 3),
    }
    return trimmed, rep_info


def trim_to_manual_window(
    df: pd.DataFrame,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df.copy(), {"rep_start_frame": None, "rep_end_frame": None, "rep_duration_s": None, "rep_trim_mode": "manual"}

    work = df.reset_index(drop=True).copy()
    frames = work["frame"].to_numpy(dtype=int)

    if start_frame is None:
        start_idx = 0
    else:
        candidates = np.where(frames >= int(start_frame))[0]
        start_idx = int(candidates[0]) if len(candidates) else 0

    if end_frame is None:
        end_idx = len(work) - 1
    else:
        candidates = np.where(frames <= int(end_frame))[0]
        end_idx = int(candidates[-1]) if len(candidates) else len(work) - 1

    if end_idx < start_idx:
        end_idx = start_idx

    trimmed = work.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    rep_info = {
        "rep_start_frame": int(work["frame"].iloc[start_idx]),
        "rep_end_frame": int(work["frame"].iloc[end_idx]),
        "rep_duration_s": round(float(work["time_s"].iloc[end_idx] - work["time_s"].iloc[start_idx]), 3),
        "rep_trim_mode": "manual",
    }
    return trimmed, rep_info


def build_summary_row(
    df: pd.DataFrame,
    phase_metrics: dict,
    route_guess: dict,
    calibration_rms_error_yards: float | None,
    calibration_confidence: float | None,
    rep_info: dict,
) -> pd.DataFrame:
    row = {
        "frames_tracked": len(df),
        "peak_speed_mph": round(robust_peak_speed_mph(df["speed_mph"].to_numpy(dtype=float)), 3) if not df.empty else 0.0,
        "instant_peak_speed_mph": round(float(df["speed_mph"].max()), 3) if not df.empty else 0.0,
        "avg_speed_mph": round(float(df["speed_mph"].mean()), 3) if not df.empty else 0.0,
        "peak_accel_mph_per_sec": round(float(df["accel_mph_per_sec"].max()), 3) if not df.empty else 0.0,
        "calibration_rms_error_yards": calibration_rms_error_yards,
        "calibration_confidence": calibration_confidence,
    }
    row.update(rep_info)
    row.update(phase_metrics)
    row.update(route_guess)
    return pd.DataFrame([row])


@dataclass
class Detection:
    box: np.ndarray
    confidence: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray | None


class FieldCalibrator:
    """
    Build an image->field homography from at least 4 known points.

    The user clicks a point in the frame and enters its field coordinates
    in yards. Coordinates can be any consistent 2D field plane system.
    """

    def __init__(self):
        self.image_points: list[tuple[float, float]] = []
        self.world_points: list[tuple[float, float]] = []
        self.last_auto_rms_error: float | None = None
        self.last_auto_confidence: float | None = None
        self.last_auto_field_length_yards: float | None = None
        self.last_auto_field_width_yards: float | None = None

    def try_auto(
        self,
        frame: np.ndarray,
        field_length_yards: float = DEFAULT_AUTO_FIELD_LENGTH_YARDS,
        field_width_yards: float = DEFAULT_AUTO_FIELD_WIDTH_YARDS,
    ) -> np.ndarray | None:
        """
        Try to auto-calibrate from the visible turf polygon.

        This is intentionally conservative: it only succeeds when the turf
        region produces a plausible 4-corner field shape.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 25, 25], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_auto_rms_error = None
            self.last_auto_confidence = 0.0
            return None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        frame_area = float(frame.shape[0] * frame.shape[1])
        if area < frame_area * 0.12:
            self.last_auto_rms_error = None
            self.last_auto_confidence = 0.0
            return None

        hull = cv2.convexHull(largest)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        if len(approx) >= 4:
            pts = approx.reshape(-1, 2).astype(np.float32)
        else:
            pts = hull.reshape(-1, 2).astype(np.float32)

        if len(pts) < 4:
            self.last_auto_rms_error = None
            self.last_auto_confidence = 0.0
            return None

        quad = order_quad_points(
            np.array(
                [
                    pts[np.argmin(pts[:, 0] + pts[:, 1])],
                    pts[np.argmax(pts[:, 0] - pts[:, 1])],
                    pts[np.argmax(pts[:, 0] + pts[:, 1])],
                    pts[np.argmin(pts[:, 0] - pts[:, 1])],
                ],
                dtype=np.float32,
            )
        )

        top_w = float(np.linalg.norm(quad[1] - quad[0]))
        bot_w = float(np.linalg.norm(quad[2] - quad[3]))
        left_h = float(np.linalg.norm(quad[3] - quad[0]))
        right_h = float(np.linalg.norm(quad[2] - quad[1]))
        if min(top_w, bot_w, left_h, right_h) < 40.0:
            self.last_auto_rms_error = None
            self.last_auto_confidence = 0.0
            return None

        image_pts = quad.astype(np.float32)
        world_pts = np.array(
            [
                [0.0, field_width_yards],
                [field_length_yards, field_width_yards],
                [field_length_yards, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(image_pts, world_pts)
        rms = homography_rms_error(image_pts, world_pts, homography)

        # Confidence blends reprojection consistency with how much of the frame
        # looks like turf. Auto-calibration is helpful, but we still treat it as
        # lower-trust than explicit user points.
        coverage = min(1.0, area / max(frame_area * 0.45, 1.0))
        geom_balance = min(top_w, bot_w) / max(max(top_w, bot_w), 1.0)
        confidence = max(0.0, min(0.85, 0.45 + 0.25 * coverage + 0.15 * geom_balance))

        self.image_points = [tuple(p) for p in image_pts]
        self.world_points = [tuple(p) for p in world_pts]
        self.last_auto_rms_error = rms
        self.last_auto_confidence = confidence
        self.last_auto_field_length_yards = field_length_yards
        self.last_auto_field_width_yards = field_width_yards
        return homography

    def run(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        win = "FIELD CALIBRATION - click field landmarks, press C when done"
        state = {"pending_click": None}

        def redraw():
            canvas = display.copy()
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), C["bg"], -1)
            cv2.putText(
                canvas,
                "Click 4+ field landmarks, then enter each point's field X,Y yards in the terminal.",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                C["yellow"],
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Examples: near hash intersections, yard-line corners, sideline markers. C=compute, R=reset, Q=quit",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                C["dim"],
                1,
                cv2.LINE_AA,
            )
            for index, (img_pt, world_pt) in enumerate(zip(self.image_points, self.world_points), start=1):
                x, y = int(img_pt[0]), int(img_pt[1])
                cv2.circle(canvas, (x, y), 6, C["cyan"], -1)
                cv2.putText(
                    canvas,
                    f"{index}: ({world_pt[0]:.1f}, {world_pt[1]:.1f})",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    C["white"],
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(win, canvas)

        def on_click(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["pending_click"] = (float(x), float(y))

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_click)

        while True:
            redraw()
            key = cv2.waitKey(30) & 0xFF
            if state["pending_click"] is not None:
                click = state["pending_click"]
                state["pending_click"] = None
                print(f"\nClicked image point: ({click[0]:.1f}, {click[1]:.1f})")
                raw = input("Enter field coordinates in yards as X,Y (example 40,20): ").strip()
                try:
                    world_x, world_y = [float(v.strip()) for v in raw.split(",", 1)]
                except Exception:
                    print("Invalid coordinate format, skipping point.")
                    continue
                self.image_points.append(click)
                self.world_points.append((world_x, world_y))
            if key == ord("r"):
                self.image_points.clear()
                self.world_points.clear()
            elif key == ord("q"):
                cv2.destroyWindow(win)
                raise RuntimeError("Calibration cancelled.")
            elif key == ord("c"):
                if len(self.image_points) < 4:
                    print("Need at least 4 point pairs for calibration.")
                    continue
                image_pts = np.array(self.image_points, dtype=np.float32)
                world_pts = np.array(self.world_points, dtype=np.float32)
                homography, mask = cv2.findHomography(image_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=6.0)
                if homography is None:
                    print("Homography estimation failed. Add better-distributed points and try again.")
                    continue
                inliers = int(mask.sum()) if mask is not None else len(self.image_points)
                rms_error = homography_rms_error(image_pts, world_pts, homography)
                print(f"Calibration complete with {inliers}/{len(self.image_points)} inliers.")
                print(f"Calibration RMS error: {rms_error:.2f} yards")
                if rms_error > 1.0:
                    print("Warning: calibration quality is weak. Use more spread-out field points for better speed and cut metrics.")
                cv2.destroyWindow(win)
                return homography

    def from_points(self, calibration_points: list[dict]) -> np.ndarray:
        if len(calibration_points) < 4:
            raise RuntimeError("Need at least 4 calibration points.")

        self.image_points = [
            (float(point["image_x"]), float(point["image_y"]))
            for point in calibration_points
        ]
        self.world_points = [
            (float(point["field_x"]), float(point["field_y"]))
            for point in calibration_points
        ]

        image_pts = np.array(self.image_points, dtype=np.float32)
        world_pts = np.array(self.world_points, dtype=np.float32)
        homography, mask = cv2.findHomography(image_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if homography is None:
            raise RuntimeError("Manual in-page calibration failed. Try more spread-out field points.")
        inliers = int(mask.sum()) if mask is not None else len(self.image_points)
        rms_error = homography_rms_error(image_pts, world_pts, homography)
        print(f"Browser calibration complete with {inliers}/{len(self.image_points)} inliers.")
        print(f"Browser calibration RMS error: {rms_error:.2f} yards")
        self.last_auto_rms_error = rms_error
        self.last_auto_confidence = calibration_confidence_from_rms(rms_error)
        return homography


class PlayerPixelRefiner:
    """
    Refine a coarse person box into a more stable foot-contact point.

    This uses a small local segmentation step so speed is driven by player
    pixels rather than the full detection box. If that fails, it falls back to
    pose ankles, then box bottom center.
    """

    def __init__(self, search_pad: int = DEFAULT_TRACK_WINDOW):
        self.search_pad = search_pad

    def refine(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        keypoints_xy: np.ndarray | None,
        keypoints_conf: np.ndarray | None,
    ) -> tuple[tuple[float, float], np.ndarray | None]:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            center = box_center(box)
            return (center[0], float(y2)), None

        foot_point = None
        full_mask = None

        if keypoints_xy is not None:
            la = kp_point(keypoints_xy, keypoints_conf, KP["l_ankle"])
            ra = kp_point(keypoints_xy, keypoints_conf, KP["r_ankle"])
            foot_point = midpoint(la, ra)

        if foot_point is None:
            mask = self._segment_player(roi)
            if mask is not None:
                full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask
                ys, xs = np.where(mask > 0)
                if len(xs) > 24:
                    bottom_y = int(np.max(ys))
                    near_bottom = ys >= (bottom_y - max(4, (y2 - y1) // 10))
                    bottom_x = float(np.mean(xs[near_bottom])) + x1
                    foot_point = (bottom_x, float(bottom_y + y1))

        if foot_point is None:
            foot_point = ((x1 + x2) * 0.5, float(y2))

        return foot_point, full_mask

    def _segment_player(self, roi: np.ndarray) -> np.ndarray | None:
        h, w = roi.shape[:2]
        if h < 20 or w < 12:
            return None

        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        inner_x = max(1, int(w * 0.18))
        inner_y = max(1, int(h * 0.08))
        rect = (inner_x, inner_y, max(2, w - 2 * inner_x), max(2, h - 2 * inner_y))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None

        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        if int(binary.sum()) == 0:
            return None
        return binary


class VisionWRTracker:
    def __init__(
        self,
        video_path: str,
        model_path: str | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        detect_every: int = DEFAULT_DETECT_EVERY,
        mode: str = "side-view",
    ):
        self.video_path = video_path
        self.mode = mode
        self.use_pose = mode == "side-view"
        self.model_path = model_path or (DEFAULT_SIDE_VIEW_MODEL if self.use_pose else DEFAULT_DRONE_MODEL)
        self.confidence = confidence
        self.detect_every = 1 if mode == "drone" else max(1, detect_every)
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.refiner = PlayerPixelRefiner()

        self.prev_detection_box: np.ndarray | None = None
        self.prev_world: tuple[float, float] | None = None
        self.prev_smoothed_world: tuple[float, float] | None = None
        self.prev_smoothed_speed: float | None = None
        self.prev_smoothed_accel: float | None = None
        self.prev_image_point: tuple[float, float] | None = None
        self.target_box_seed: np.ndarray | None = None
        self.target_appearance_hist: np.ndarray | None = None
        self.prev_target_appearance_hist: np.ndarray | None = None
        self._box_tracker = None
        self._raw_speed_buffer: deque[float] = deque(maxlen=SPEED_MEDIAN_WINDOW)
        self._world_history: deque[tuple[float, float]] = deque(maxlen=max(SPEED_MEDIAN_WINDOW, SPEED_LOOKBACK_FRAMES + 2))
        self.calibration_rms_error_yards: float | None = None
        self.calibration_confidence: float | None = None

        self.metrics: list[dict] = []

    def read_first_frame(self) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read first frame.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame

    def set_calibration_quality(self, rms_error_yards: float | None, confidence_override: float | None = None):
        self.calibration_rms_error_yards = rms_error_yards
        if confidence_override is not None:
            self.calibration_confidence = max(0.0, min(1.0, float(confidence_override)))
        elif rms_error_yards is None:
            self.calibration_confidence = None
        else:
            self.calibration_confidence = calibration_confidence_from_rms(rms_error_yards)

    def detect_people(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        result = results[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints_xy = None
        keypoints_conf = None

        if self.use_pose and result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.cpu().numpy()

        for index, box in enumerate(boxes):
            cls_id = int(result.boxes.cls[index].item())
            if cls_id != 0:
                continue
            detections.append(
                Detection(
                    box=box.astype(np.float32),
                    confidence=float(scores[index]),
                    keypoints_xy=keypoints_xy[index] if keypoints_xy is not None else None,
                    keypoints_conf=keypoints_conf[index] if keypoints_conf is not None else None,
                )
            )

        return detections

    def select_target(self, frame: np.ndarray, detections: list[Detection], auto_bottom: bool = False) -> Detection:
        if not detections:
            raise RuntimeError("No players detected on the first frame.")

        if auto_bottom:
            return self.select_bottom_target(detections)

        display = frame.copy()
        chosen = {"detection": None}

        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection.box.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), C["cyan"], 2)
            cv2.putText(
                display,
                f"{index}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                C["cyan"],
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(display, (0, 0), (display.shape[1], 50), C["bg"], -1)
        cv2.putText(
            display,
            "Click the receiver you want to track. Q cancels.",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            C["yellow"],
            2,
            cv2.LINE_AA,
        )

        def on_click(event, x, y, flags, _):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for detection in detections:
                x1, y1, x2, y2 = detection.box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    chosen["detection"] = detection
                    break

        win = "TARGET SELECTION"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_click)
        while chosen["detection"] is None:
            cv2.imshow(win, display)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                cv2.destroyWindow(win)
                raise RuntimeError("Target selection cancelled.")
        cv2.destroyWindow(win)
        self.target_appearance_hist = self._compute_appearance_hist(frame, chosen["detection"].box)
        return chosen["detection"]

    def select_target_from_point(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        target_point: tuple[float, float],
    ) -> Detection:
        if not detections:
            raise RuntimeError("No players detected on the first frame.")

        x, y = float(target_point[0]), float(target_point[1])
        containing: list[Detection] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            if x1 <= x <= x2 and y1 <= y <= y2:
                containing.append(detection)

        if containing:
            chosen = min(containing, key=lambda det: box_area(det.box))
        else:
            chosen = min(
                detections,
                key=lambda det: math.hypot(box_center(det.box)[0] - x, box_center(det.box)[1] - y),
            )
        self.target_appearance_hist = self._compute_appearance_hist(frame, chosen.box)
        return chosen

    def select_bottom_target(self, detections: list[Detection]) -> Detection:
        """
        Pick the player closest to the bottom of the frame.

        For route clips like CodySlant, the featured receiver is usually the
        widest isolated player aligned nearest the bottom edge of the screen.
        We bias toward large, low boxes so sideline spectators and tiny distant
        detections do not win by accident.
        """
        if not detections:
            raise RuntimeError("No players detected on the first frame.")

        best_detection = None
        best_score = -1e9
        frame_height = max(1.0, float(self.height))
        frame_width = max(1.0, float(self.width))

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            width = max(1.0, float(x2 - x1))
            height = max(1.0, float(y2 - y1))
            area = width * height
            center_x = (x1 + x2) * 0.5
            bottom_bias = y2 / frame_height
            size_bias = min(area / (frame_width * frame_height * 0.02), 1.0)
            sideline_bias = 1.0 - abs((center_x / frame_width) - 0.5)
            score = bottom_bias * 3.5 + size_bias * 1.5 + sideline_bias * 0.35 + detection.confidence * 0.5
            if score > best_score:
                best_score = score
                best_detection = detection

        return best_detection

    def track(
        self,
        homography: np.ndarray,
        target_detection: Detection,
        save_video_path: str | None = None,
        show_preview: bool = True,
        progress_image_path: str | None = None,
    ) -> pd.DataFrame:
        if not self.use_pose:
            return self._track_drone(
                homography,
                target_detection,
                save_video_path,
                show_preview=show_preview,
                progress_image_path=progress_image_path,
            )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.metrics = []
        self.prev_detection_box = target_detection.box.copy()
        self.target_box_seed = target_detection.box.copy()
        first_frame = self.read_first_frame()
        if self.target_appearance_hist is None:
            self.target_appearance_hist = self._compute_appearance_hist(first_frame, target_detection.box.copy())
        self.prev_target_appearance_hist = self.target_appearance_hist.copy() if self.target_appearance_hist is not None else None
        self.prev_world = None
        self.prev_smoothed_world = None
        self.prev_smoothed_speed = None
        self.prev_smoothed_accel = None
        self.prev_image_point = None
        self._raw_speed_buffer.clear()
        self._world_history: deque[tuple[float, float]] = deque(maxlen=max(SPEED_MEDIAN_WINDOW, SPEED_LOOKBACK_FRAMES + 2))
        self._init_box_tracker(first_frame, target_detection.box.copy())
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        writer = None
        if save_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_video_path, fourcc, self.fps, (self.width, self.height))

        frame_index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            tracker_box, tracker_ok = self._update_box_tracker(frame)
            ran_detector = frame_index % self.detect_every == 0 or not tracker_ok
            detections = self.detect_people(frame) if ran_detector else []
            target = self._choose_target_candidate_with_frame(frame, detections, tracker_box)
            previous_point = self.prev_image_point

            if target is None:
                if tracker_ok and tracker_box is not None:
                    target = Detection(
                        box=tracker_box.copy(),
                        confidence=0.0,
                        keypoints_xy=None,
                        keypoints_conf=None,
                    )
                else:
                    annotated = frame.copy()
                    self._draw_missing_target(annotated, frame_index)
                    if writer is not None:
                        writer.write(annotated)
                    if progress_image_path and frame_index % PROGRESS_PREVIEW_EVERY_N_FRAMES == 0:
                        cv2.imwrite(progress_image_path, annotated)
                    if show_preview:
                        cv2.imshow("WR Vision Tracker", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    frame_index += 1
                    continue

            if target is not None:
                foot_image_xy, mask = self.refiner.refine(frame, target.box, target.keypoints_xy, target.keypoints_conf)
                foot_image_xy = self._stabilize_foot_point(foot_image_xy, target.box)
                world_xy = image_to_world(foot_image_xy, homography)
                hip_mid_y_px = None
                box_h_px = float(target.box[3] - target.box[1])
                if target.keypoints_xy is not None:
                    lh = kp_point(target.keypoints_xy, target.keypoints_conf, KP["l_hip"])
                    rh = kp_point(target.keypoints_xy, target.keypoints_conf, KP["r_hip"])
                    hip_mid = midpoint(lh, rh)
                    if hip_mid is not None:
                        hip_mid_y_px = float(hip_mid[1])
                frame_metrics = self._compute_metrics(
                    frame_index,
                    foot_image_xy,
                    world_xy,
                    hip_mid_y_px,
                    box_h_px,
                    keypoints_xy=target.keypoints_xy,
                    keypoints_conf=target.keypoints_conf,
                )
                self.prev_detection_box = target.box.copy()
                self._init_box_tracker(frame, target.box.copy())

                annotated = frame.copy()
                self._draw_frame_overlay(annotated, target, foot_image_xy, mask, frame_metrics, previous_point)
                if writer is not None:
                    writer.write(annotated)
                if progress_image_path and frame_index % PROGRESS_PREVIEW_EVERY_N_FRAMES == 0:
                    cv2.imwrite(progress_image_path, annotated)
                if show_preview:
                    cv2.imshow("WR Vision Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
                continue

        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        return pd.DataFrame(self.metrics)

    def _track_drone(
        self,
        homography: np.ndarray,
        target_detection: Detection,
        save_video_path: str | None = None,
        show_preview: bool = True,
        progress_image_path: str | None = None,
    ) -> pd.DataFrame:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.metrics = []
        self.prev_detection_box = target_detection.box.copy()
        self.target_box_seed = target_detection.box.copy()
        first_frame = self.read_first_frame()
        self.target_appearance_hist = self._compute_appearance_hist(first_frame, target_detection.box.copy())
        self.prev_target_appearance_hist = self.target_appearance_hist.copy() if self.target_appearance_hist is not None else None
        self.prev_world = None
        self.prev_smoothed_world = None
        self.prev_smoothed_speed = None
        self.prev_smoothed_accel = None
        self.prev_image_point = None
        self._raw_speed_buffer.clear()
        self._world_history = deque(maxlen=max(SPEED_MEDIAN_WINDOW, SPEED_LOOKBACK_FRAMES + 2))
        self._init_box_tracker(first_frame, target_detection.box.copy())
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        writer = None
        if save_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_video_path, fourcc, self.fps, (self.width, self.height))

        frame_index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            tracker_box, tracker_ok = self._update_box_tracker(frame)
            if tracker_ok and tracker_box is not None:
                target_box = tracker_box.copy()
            elif self.prev_detection_box is not None:
                target_box = self.prev_detection_box.copy()
                self._init_box_tracker(frame, target_box)
            else:
                annotated = frame.copy()
                self._draw_missing_target(annotated, frame_index)
                if writer is not None:
                    writer.write(annotated)
                if progress_image_path and frame_index % PROGRESS_PREVIEW_EVERY_N_FRAMES == 0:
                    cv2.imwrite(progress_image_path, annotated)
                if show_preview:
                    cv2.imshow("WR Vision Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_index += 1
                continue

            target = Detection(
                box=clamp_box_to_frame(target_box, self.width, self.height),
                confidence=1.0 if tracker_ok else 0.0,
                keypoints_xy=None,
                keypoints_conf=None,
            )
            previous_point = self.prev_image_point
            foot_image_xy, mask = self.refiner.refine(frame, target.box, None, None)
            foot_image_xy = self._stabilize_foot_point(foot_image_xy, target.box)
            world_xy = image_to_world(foot_image_xy, homography)
            box_h_px = float(target.box[3] - target.box[1])
            frame_metrics = self._compute_metrics(frame_index, foot_image_xy, world_xy, None, box_h_px)
            self.prev_detection_box = target.box.copy()
            self._init_box_tracker(frame, target.box.copy())

            annotated = frame.copy()
            self._draw_frame_overlay(annotated, target, foot_image_xy, mask, frame_metrics, previous_point)
            if writer is not None:
                writer.write(annotated)
            if progress_image_path and frame_index % PROGRESS_PREVIEW_EVERY_N_FRAMES == 0:
                cv2.imwrite(progress_image_path, annotated)
            if show_preview:
                cv2.imshow("WR Vision Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1

        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        return pd.DataFrame(self.metrics)

    def _choose_target_candidate(self, detections: list[Detection], tracker_box: np.ndarray | None) -> Detection | None:
        if not detections:
            return None
        if self.prev_detection_box is None and tracker_box is None:
            return detections[0]

        best_score = -1e9
        best_detection = None
        reference_box = tracker_box if tracker_box is not None else self.prev_detection_box
        prev_center = box_center(reference_box)

        for detection in detections:
            iou = compute_iou(reference_box, detection.box)
            cx, cy = box_center(detection.box)
            center_distance = math.hypot(cx - prev_center[0], cy - prev_center[1])
            score = iou * 4.0 - center_distance * 0.01 + detection.confidence
            if self.target_box_seed is not None:
                seed_iou = compute_iou(self.target_box_seed, detection.box)
                score += seed_iou * 0.6
            if self.prev_detection_box is not None:
                score += compute_iou(self.prev_detection_box, detection.box) * 1.5
            if score > best_score:
                best_score = score
                best_detection = detection

        return best_detection

    def _choose_target_candidate_with_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracker_box: np.ndarray | None,
    ) -> Detection | None:
        if not detections:
            return None
        if self.use_pose:
            return self._choose_target_candidate(detections, tracker_box)

        if self.prev_detection_box is None and tracker_box is None:
            return detections[0]

        best_score = -1e9
        best_detection = None
        reference_box = tracker_box if tracker_box is not None else self.prev_detection_box
        prev_center = box_center(reference_box)
        ref_w = max(1.0, float(reference_box[2] - reference_box[0]))
        ref_h = max(1.0, float(reference_box[3] - reference_box[1]))
        max_center_distance = max(ref_w, ref_h) * 1.35

        for detection in detections:
            iou = compute_iou(reference_box, detection.box)
            cx, cy = box_center(detection.box)
            center_distance = math.hypot(cx - prev_center[0], cy - prev_center[1])
            seed_iou = compute_iou(self.target_box_seed, detection.box) if self.target_box_seed is not None else 0.0
            if center_distance > max_center_distance and max(iou, seed_iou) < 0.08:
                continue

            score = iou * 3.0 - center_distance * 0.012 + detection.confidence * 1.2
            if self.target_box_seed is not None:
                score += seed_iou * 1.4
            if self.prev_detection_box is not None:
                score += compute_iou(self.prev_detection_box, detection.box) * 2.5
            hist_similarity = None
            if self.target_appearance_hist is not None:
                det_hist = self._compute_appearance_hist(frame, detection.box)
                if det_hist is not None:
                    hist_distance = cv2.compareHist(self.target_appearance_hist, det_hist, cv2.HISTCMP_BHATTACHARYYA)
                    hist_similarity = 1.0 - hist_distance
                    score += hist_similarity * 3.0
                    if self.prev_target_appearance_hist is not None:
                        recent_distance = cv2.compareHist(self.prev_target_appearance_hist, det_hist, cv2.HISTCMP_BHATTACHARYYA)
                        recent_similarity = 1.0 - recent_distance
                        score += recent_similarity * 1.8
                    if hist_similarity < 0.25 and max(iou, seed_iou) < 0.12:
                        continue
            if score > best_score:
                best_score = score
                best_detection = detection

        if best_detection is not None:
            det_hist = self._compute_appearance_hist(frame, best_detection.box)
            if det_hist is not None:
                self.prev_target_appearance_hist = det_hist
        return best_detection

    def _compute_appearance_hist(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in clamp_box_to_frame(box, self.width, self.height)]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        if hist is None:
            return None
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist

    def _init_box_tracker(self, frame: np.ndarray, box: np.ndarray):
        box = clamp_box_to_frame(box, self.width, self.height)
        try:
            tracker = create_box_tracker()
            tracker.init(frame, box_xyxy_to_xywh(box))
            self._box_tracker = tracker
        except Exception:
            self._box_tracker = None

    def _update_box_tracker(self, frame: np.ndarray) -> tuple[np.ndarray | None, bool]:
        if self._box_tracker is None:
            return None, False
        ok, tracked = self._box_tracker.update(frame)
        if not ok:
            return None, False
        box = clamp_box_to_frame(box_xywh_to_xyxy(tracked), self.width, self.height)
        return box, True

    def _stabilize_foot_point(self, foot_image_xy: tuple[float, float], box: np.ndarray) -> tuple[float, float]:
        if self.prev_image_point is None:
            return foot_image_xy

        x1, y1, x2, y2 = [float(v) for v in box]
        threshold = min(28.0, max(10.0, max(x2 - x1, y2 - y1) * MAX_FOOT_JUMP_RATIO))
        dx = foot_image_xy[0] - self.prev_image_point[0]
        dy = foot_image_xy[1] - self.prev_image_point[1]
        dist = math.hypot(dx, dy)
        if dist <= threshold or dist <= 1e-6:
            return foot_image_xy

        scale = threshold / dist
        return (
            self.prev_image_point[0] + dx * scale,
            self.prev_image_point[1] + dy * scale,
        )

    def _compute_metrics(
        self,
        frame_index: int,
        foot_image_xy: tuple[float, float],
        world_xy: tuple[float, float],
        hip_mid_y_px: float | None = None,
        box_height_px: float | None = None,
        keypoints_xy: np.ndarray | None = None,
        keypoints_conf: np.ndarray | None = None,
    ) -> dict:
        dt = 1.0 / max(self.fps, 1e-6)
        smoothed_world_xy = (
            smooth(world_xy[0], self.prev_smoothed_world[0] if self.prev_smoothed_world is not None else None, WORLD_POINT_ALPHA),
            smooth(world_xy[1], self.prev_smoothed_world[1] if self.prev_smoothed_world is not None else None, WORLD_POINT_ALPHA),
        )

        raw_speed_samples: list[float] = []
        history_list = list(self._world_history)
        max_lag = min(SPEED_LOOKBACK_FRAMES, len(history_list))
        for lag in range(1, max_lag + 1):
            prev_world_xy = history_list[-lag]
            lag_dt = dt * lag
            if lag_dt <= 1e-6:
                continue
            raw_speed_samples.append(math.dist(smoothed_world_xy, prev_world_xy) / lag_dt)

        raw_speed_yps = float(np.median(np.array(raw_speed_samples, dtype=float))) if raw_speed_samples else 0.0
        self._raw_speed_buffer.append(raw_speed_yps)
        filtered_speed_yps = float(np.median(np.array(self._raw_speed_buffer, dtype=float)))
        filtered_speed_yps = soft_clip_speed_yps(filtered_speed_yps)

        smoothed_speed_yps = smooth(filtered_speed_yps, self.prev_smoothed_speed, SMOOTH_ALPHA)
        raw_accel_yps2 = 0.0
        if self.prev_smoothed_speed is not None:
            raw_accel_yps2 = (smoothed_speed_yps - self.prev_smoothed_speed) / dt
        smoothed_accel_yps2 = smooth(raw_accel_yps2, self.prev_smoothed_accel, ACCEL_ALPHA)

        self.prev_world = world_xy
        self.prev_smoothed_world = smoothed_world_xy
        self.prev_smoothed_speed = smoothed_speed_yps
        self.prev_smoothed_accel = smoothed_accel_yps2
        self.prev_image_point = foot_image_xy
        self._world_history.append(smoothed_world_xy)

        metrics = {
            "frame": frame_index + 1,
            "time_s": round((frame_index + 1) / self.fps, 4),
            "image_x_px": round(foot_image_xy[0], 2),
            "image_y_px": round(foot_image_xy[1], 2),
            "field_x_yd": round(world_xy[0], 3),
            "field_y_yd": round(world_xy[1], 3),
            "speed_yards_per_sec": round(smoothed_speed_yps, 3),
            "speed_mph": round(yards_per_second_to_mph(smoothed_speed_yps), 3),
            "accel_yards_per_sec2": round(smoothed_accel_yps2, 3),
            "accel_mph_per_sec": round(yards_per_second_to_mph(smoothed_accel_yps2), 3),
            "hip_mid_y_px": round(float(hip_mid_y_px), 3) if hip_mid_y_px is not None else np.nan,
            "box_height_px": round(float(box_height_px), 3) if box_height_px is not None else np.nan,
        }
        for name, idx in KP.items():
            point = None
            conf = np.nan
            if keypoints_xy is not None:
                point = kp_point(keypoints_xy, keypoints_conf, idx, conf_threshold=0.0)
                if keypoints_conf is not None:
                    conf = float(keypoints_conf[idx])
            metrics[f"{name}_x_px"] = round(float(point[0]), 3) if point is not None else np.nan
            metrics[f"{name}_y_px"] = round(float(point[1]), 3) if point is not None else np.nan
            metrics[f"{name}_conf"] = round(float(conf), 4) if not np.isnan(conf) else np.nan
        self.metrics.append(metrics)
        return metrics

    def _draw_frame_overlay(
        self,
        frame: np.ndarray,
        detection: Detection,
        foot_image_xy: tuple[float, float],
        mask: np.ndarray | None,
        metrics: dict,
        previous_point: tuple[float, float] | None,
    ):
        x1, y1, x2, y2 = detection.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C["green"], 2)
        cv2.circle(frame, (int(foot_image_xy[0]), int(foot_image_xy[1])), 6, C["yellow"], -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "tracked foot point",
            (int(foot_image_xy[0]) + 8, int(foot_image_xy[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            C["yellow"],
            1,
            cv2.LINE_AA,
        )

        if mask is not None:
            overlay = frame.copy()
            overlay[mask > 0] = (0, 180, 255)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)

        if detection.keypoints_xy is not None:
            self._draw_pose_overlay(frame, detection.keypoints_xy, detection.keypoints_conf)

        cv2.rectangle(frame, (0, 0), (430, 110), C["bg"], -1)
        lines = [
            f"Frame: {metrics['frame']} / {self.total_frames}",
            f"Field: ({metrics['field_x_yd']:.2f}, {metrics['field_y_yd']:.2f}) yd",
            f"Speed: {metrics['speed_mph']:.2f} mph",
            f"Accel: {metrics['accel_mph_per_sec']:.2f} mph/s",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (14, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                C["white"] if index < 2 else C["green"],
                2 if index >= 2 else 1,
                cv2.LINE_AA,
            )

        if previous_point is not None:
            cv2.line(
                frame,
                (int(previous_point[0]), int(previous_point[1])),
                (int(foot_image_xy[0]), int(foot_image_xy[1])),
                C["cyan"],
                2,
                cv2.LINE_AA,
            )

    def _draw_pose_overlay(
        self,
        frame: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray | None,
    ):
        def point_for(name: str) -> tuple[int, int] | None:
            idx = KP[name]
            point = kp_point(keypoints_xy, keypoints_conf, idx, conf_threshold=0.15)
            if point is None:
                return None
            return (int(point[0]), int(point[1]))

        for a_name, b_name in POSE_SKELETON_EDGES:
            a = point_for(a_name)
            b = point_for(b_name)
            if a is None or b is None:
                continue
            cv2.line(frame, a, b, (255, 120, 60), 2, cv2.LINE_AA)

        for name, idx in KP.items():
            point = kp_point(keypoints_xy, keypoints_conf, idx, conf_threshold=0.15)
            if point is None:
                continue
            conf = float(keypoints_conf[idx]) if keypoints_conf is not None else 1.0
            radius = 5 if conf >= 0.5 else 3
            color = (90, 240, 255) if conf >= 0.5 else (120, 150, 180)
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)

        hip_mid = midpoint(
            kp_point(keypoints_xy, keypoints_conf, KP["l_hip"], conf_threshold=0.15),
            kp_point(keypoints_xy, keypoints_conf, KP["r_hip"], conf_threshold=0.15),
        )
        if hip_mid is not None:
            cv2.circle(frame, (int(hip_mid[0]), int(hip_mid[1])), 6, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "hip mid",
                (int(hip_mid[0]) + 8, int(hip_mid[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_missing_target(self, frame: np.ndarray, frame_index: int):
        cv2.rectangle(frame, (0, 0), (430, 70), C["bg"], -1)
        cv2.putText(
            frame,
            f"Frame: {frame_index + 1} / {self.total_frames}",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            C["white"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Target missing this frame",
            (14, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            C["orange"],
            2,
            cv2.LINE_AA,
        )

    def release(self):
        self.cap.release()


def save_outputs(
    df: pd.DataFrame,
    output_stem: Path,
    calibration_rms_error_yards: float | None = None,
    calibration_confidence: float | None = None,
    manual_start_frame: int | None = None,
    manual_end_frame: int | None = None,
    cut_frame_override: int | None = None,
    include_pose_metrics: bool = True,
    analysis_mode: str = "side-view",
):
    df = recompute_frame_kinematics(df)
    csv_path = output_stem.with_name(output_stem.name + "_metrics.csv")
    clean_csv_path = output_stem.with_name(output_stem.name + "_clean_metrics.csv")
    pose_csv_path = output_stem.with_name(output_stem.name + "_pose_points.csv")
    rep_clean_csv_path = output_stem.with_name(output_stem.name + "_rep_clean_metrics.csv")
    summary_csv_path = output_stem.with_name(output_stem.name + "_route_summary.csv")
    summary_path = output_stem.with_name(output_stem.name + "_summary.txt")
    debug_plot_path = output_stem.with_name(output_stem.name + "_route_debug.png")
    df.to_csv(csv_path, index=False)

    if manual_start_frame is not None or manual_end_frame is not None:
        rep_df, rep_info = trim_to_manual_window(df, manual_start_frame, manual_end_frame)
    else:
        rep_df, rep_info = trim_to_rep_window(df)
        rep_info["rep_trim_mode"] = "auto"
    summary_df_source = rep_df if not rep_df.empty else df

    peak_speed = robust_peak_speed_mph(summary_df_source["speed_mph"].to_numpy(dtype=float)) if not summary_df_source.empty else 0.0
    instant_peak_speed = float(summary_df_source["speed_mph"].max()) if not summary_df_source.empty else 0.0
    avg_speed = float(summary_df_source["speed_mph"].mean()) if not summary_df_source.empty else 0.0
    peak_accel = float(summary_df_source["accel_mph_per_sec"].max()) if not summary_df_source.empty else 0.0
    phase_metrics = compute_route_phase_metrics(
        summary_df_source,
        cut_frame_override=cut_frame_override,
        include_pose_metrics=include_pose_metrics,
    )
    route_guess = guess_route_from_phase_metrics(phase_metrics)
    clean_df = build_clean_metrics_csv(df)
    pose_df = build_pose_points_csv(df)
    rep_clean_df = build_clean_metrics_csv(summary_df_source)
    summary_df = build_summary_row(
        summary_df_source,
        phase_metrics,
        route_guess,
        calibration_rms_error_yards,
        calibration_confidence,
        rep_info,
    )
    clean_df.to_csv(clean_csv_path, index=False)
    pose_df.to_csv(pose_csv_path, index=False)
    rep_clean_df.to_csv(rep_clean_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    summary_lines = [
        f"frames_tracked={len(df)}",
        f"rep_start_frame={rep_info.get('rep_start_frame')}",
        f"rep_end_frame={rep_info.get('rep_end_frame')}",
        f"rep_duration_s={rep_info.get('rep_duration_s')}",
        f"rep_trim_mode={rep_info.get('rep_trim_mode')}",
        f"analysis_mode={analysis_mode}",
        f"cut_frame_override={cut_frame_override}",
        f"peak_speed_mph={peak_speed:.3f}",
        f"instant_peak_speed_mph={instant_peak_speed:.3f}",
        f"avg_speed_mph={avg_speed:.3f}",
        f"peak_accel_mph_per_sec={peak_accel:.3f}",
    ]
    summary_lines.append(f"calibration_rms_error_yards={calibration_rms_error_yards}")
    summary_lines.append(f"calibration_confidence={calibration_confidence}")
    if calibration_confidence is not None:
        if calibration_confidence < 0.5:
            summary_lines.append("confidence_warning=LOW_CONFIDENCE_CALIBRATION")
            summary_lines.append("confidence_warning_detail=Speed, accel, and cut angle may be materially distorted by poor field geometry.")
        elif calibration_confidence < 0.75:
            summary_lines.append("confidence_warning=MEDIUM_CONFIDENCE_CALIBRATION")
            summary_lines.append("confidence_warning_detail=Metrics are usable for rough comparison, but exact mph and cut angle may still be off.")
    for key, value in phase_metrics.items():
        summary_lines.append(f"{key}={value}")
    for key, value in route_guess.items():
        summary_lines.append(f"{key}={value}")
    summary_lines.append(f"route_debug_plot={debug_plot_path}")
    summary_lines.append(f"clean_csv={clean_csv_path}")
    summary_lines.append(f"pose_points_csv={pose_csv_path}")
    summary_lines.append(f"rep_clean_csv={rep_clean_csv_path}")
    summary_lines.append(f"summary_csv={summary_csv_path}")
    summary_lines.append(f"csv={csv_path}")
    summary = "\n".join(summary_lines)
    summary_path.write_text(summary + "\n", encoding="utf-8")
    save_route_debug_plot(summary_df_source, phase_metrics, route_guess, debug_plot_path)
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved clean metrics CSV: {clean_csv_path}")
    print(f"Saved pose points CSV: {pose_csv_path}")
    print(f"Saved rep clean metrics CSV: {rep_clean_csv_path}")
    print(f"Saved route summary CSV: {summary_csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved route debug plot: {debug_plot_path}")


def run(
    video_path: str,
    model_path: str | None,
    confidence: float,
    detect_every: int,
    auto_calibrate: bool,
    start_frame: int | None,
    end_frame: int | None,
    cut_frame: int | None,
    mode: str,
    auto_field_length: float | None,
    auto_field_width: float | None,
    calibration_json: str | None,
    target_point: str | None,
    headless: bool,
    progress_image_path: str | None,
):
    tracker = VisionWRTracker(
        video_path=video_path,
        model_path=model_path,
        confidence=confidence,
        detect_every=detect_every,
        mode=mode,
    )
    try:
        first_frame = tracker.read_first_frame()
        print("\nStep 1: field calibration")
        calibrator = FieldCalibrator()
        homography = None
        manual_calibration_points = None
        if calibration_json:
            manual_calibration_points = json.loads(calibration_json)
            homography = calibrator.from_points(manual_calibration_points)
            if len(calibrator.image_points) >= 4:
                image_pts = np.array(calibrator.image_points, dtype=np.float32)
                world_pts = np.array(calibrator.world_points, dtype=np.float32)
                tracker.set_calibration_quality(homography_rms_error(image_pts, world_pts, homography))
        elif auto_calibrate:
            field_length = auto_field_length
            field_width = auto_field_width
            if field_length is None:
                field_length = DEFAULT_DRONE_FIELD_LENGTH_YARDS if mode == "drone" else DEFAULT_AUTO_FIELD_LENGTH_YARDS
            if field_width is None:
                field_width = DEFAULT_DRONE_FIELD_WIDTH_YARDS if mode == "drone" else DEFAULT_AUTO_FIELD_WIDTH_YARDS
            homography = calibrator.try_auto(
                first_frame,
                field_length_yards=field_length,
                field_width_yards=field_width,
            )
            if homography is not None:
                tracker.set_calibration_quality(
                    calibrator.last_auto_rms_error,
                    confidence_override=calibrator.last_auto_confidence,
                )
                print(
                    "Auto-calibration succeeded "
                    f"(confidence={calibrator.last_auto_confidence:.2f}, rms={calibrator.last_auto_rms_error:.2f} yd)."
                )
                if (calibrator.last_auto_confidence or 0.0) < 0.65:
                    print("Auto-calibration confidence is low; falling back to manual calibration.")
                    homography = None

        if homography is None:
            if headless:
                raise RuntimeError("Headless analysis needs either browser calibration points or auto-calibration to succeed.")
            homography = calibrator.run(first_frame)
            if len(calibrator.image_points) >= 4:
                image_pts = np.array(calibrator.image_points, dtype=np.float32)
                world_pts = np.array(calibrator.world_points, dtype=np.float32)
                tracker.set_calibration_quality(homography_rms_error(image_pts, world_pts, homography))

        print("\nStep 2: receiver selection")
        detections = tracker.detect_people(first_frame)
        if target_point:
            target_x, target_y = [float(v.strip()) for v in target_point.split(",", 1)]
            target_detection = tracker.select_target_from_point(first_frame, detections, (target_x, target_y))
        elif headless:
            raise RuntimeError("Headless analysis needs a browser-selected target point.")
        else:
            target_detection = tracker.select_target(first_frame, detections, auto_bottom=False)
        print(
            "Target selected at box "
            f"{tuple(int(v) for v in target_detection.box)}"
            + " (manual/UI-selected)."
        )

        output_stem = Path(video_path).with_suffix("")
        overlay_path = str(output_stem.with_name(output_stem.name + "_overlay.mp4"))
        print("\nStep 3: tracking")
        df = tracker.track(
            homography,
            target_detection,
            save_video_path=overlay_path,
            show_preview=not headless,
            progress_image_path=progress_image_path,
        )
        print(f"Saved overlay video: {overlay_path}")
        save_outputs(
            df,
            output_stem,
            calibration_rms_error_yards=tracker.calibration_rms_error_yards,
            calibration_confidence=tracker.calibration_confidence,
            manual_start_frame=start_frame,
            manual_end_frame=end_frame,
            cut_frame_override=cut_frame,
            include_pose_metrics=tracker.use_pose,
            analysis_mode=mode,
        )
    finally:
        tracker.release()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WR vision tracker with homography-based speed estimation.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--mode", choices=["side-view", "drone"], default="side-view", help="Analysis mode: side-view uses pose, drone uses detector-only tracking")
    parser.add_argument("--model", default=None, help="Ultralytics model path or name. Defaults by mode: side-view uses pose, drone uses detector-only.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--detect-every", type=int, default=DEFAULT_DETECT_EVERY, help="Run detector every N frames")
    parser.add_argument("--auto-calibrate", action="store_true", help="Try vision-based field calibration before falling back to manual points")
    parser.add_argument("--auto-field-length", type=float, default=None, help="Optional visible field length in yards for auto-calibration scaling")
    parser.add_argument("--auto-field-width", type=float, default=None, help="Optional visible field width in yards for auto-calibration scaling")
    parser.add_argument("--start-frame", type=int, default=None, help="Optional manual rep start frame for trimming outputs")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional manual rep end frame for trimming outputs")
    parser.add_argument("--cut-frame", type=int, default=None, help="Optional manual cut frame override for route-phase metrics")
    parser.add_argument("--calibration-json", default=None, help="JSON array of browser-collected calibration points with image_x,image_y,field_x,field_y")
    parser.add_argument("--target-point", default=None, help="Browser-selected target point as image-space x,y")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV calibration/selection/tracking preview windows")
    parser.add_argument("--progress-image-path", default=None, help="Optional path to periodically write the latest annotated analysis frame")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(
        video_path=args.video,
        model_path=args.model,
        confidence=args.conf,
        detect_every=args.detect_every,
        auto_calibrate=args.auto_calibrate,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        cut_frame=args.cut_frame,
        mode=args.mode,
        auto_field_length=args.auto_field_length,
        auto_field_width=args.auto_field_width,
        calibration_json=args.calibration_json,
        target_point=args.target_point,
        headless=args.headless,
        progress_image_path=args.progress_image_path,
    )
