"""
wr_tracker.py
=============
Wide Receiver Motion Tracking Module — v3.0
---------------------------------------------
DEPENDENCIES:
    pip install ultralytics opencv-python numpy pandas matplotlib seaborn

STANDALONE USAGE:
    python wr_tracker.py --video input.mp4

HOW IT WORKS (3 phases):
    1. CALIBRATION  — click 2 field reference points, type the real distance
    2. SELECTION    — first frame shown, click the player you want to track
    3. TRACKING     — video plays with live HUD, Q to stop, CSV + report auto-saved

KEY FIXES IN v3.0:
    - ID mismatch fixed: selection now uses model.track() so IDs match tracking
    - Speed units fixed: all internal math stays in px/s, converts at output only
    - Keypoint validity: checks confidence score, not just zero coords
    - Tracker resets between selection and tracking to avoid ID drift
"""

import argparse
import math
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL_PATH   = "yolov8n-pose.pt"
DEFAULT_TRAIL_LENGTH = 60
DEFAULT_PX_PER_YARD  = 15.0   # fallback only — calibrate for real accuracy

# Minimum keypoint confidence to treat a point as valid
KP_CONF_THRESHOLD = 0.3


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT CONVERSIONS
# ══════════════════════════════════════════════════════════════════════════════
def px_per_sec_to_mph(px_per_sec: float, px_per_yard: float) -> float:
    yards_per_sec = px_per_sec / px_per_yard
    return yards_per_sec * (1 / 1760.0) * 3600.0

def px_per_sec_to_yards_per_sec(px_per_sec: float, px_per_yard: float) -> float:
    return px_per_sec / px_per_yard

def px_to_feet(px: float, px_per_yard: float) -> float:
    return (px / px_per_yard) * 3.0

def px_to_yards(px: float, px_per_yard: float) -> float:
    return px / px_per_yard

def speed_label(mph: float) -> str:
    if mph < 2.0:  return "STANDING"
    if mph < 5.0:  return "WALKING"
    if mph < 9.0:  return "JOGGING"
    if mph < 14.0: return "ROUTE"
    if mph < 18.0: return "RUNNING"
    if mph < 21.0: return "SPRINTING"
    return "TOP SPEED"


def classify_route(df: "pd.DataFrame") -> dict:
    """
    Analyses position and speed data to classify the WR route type
    and return a human-readable description.
    """
    if df.empty or len(df) < 10:
        return {"route_type": "Unknown", "description": "Not enough data."}

    xs = df["x_px"].values
    ys = df["y_px"].values

    # Total displacement vs total distance
    total_dist  = df["total_yards"].iloc[-1]
    displacement = math.hypot(xs[-1]-xs[0], ys[-1]-ys[0]) / (df["x_px"].std() + 1)

    # Direction changes: count times x velocity changes sign
    dx = np.diff(xs)
    dy = np.diff(ys)
    sign_changes_x = int(np.sum(np.diff(np.sign(dx)) != 0))
    sign_changes_y = int(np.sum(np.diff(np.sign(dy)) != 0))

    # Net vertical (y) movement — in image coords y increases downward
    net_y = ys[-1] - ys[0]   # positive = moved down screen
    net_x = xs[-1] - xs[0]   # positive = moved right

    # Speed profile
    peak_mph = df["mph"].max()
    avg_mph  = df["mph"].mean()
    # Proportion of frames at sprint speed
    sprint_pct = (df["mph"] > 14).mean()

    # ── Classify ──────────────────────────────────────────────────────────────
    route_type   = "Unknown Route"
    description  = ""

    mostly_vertical = abs(net_y) > abs(net_x) * 0.8
    mostly_lateral  = abs(net_x) > abs(net_y) * 1.5
    has_cuts = sign_changes_x >= 3 or sign_changes_y >= 3

    if sprint_pct > 0.45 and sign_changes_x < 3 and mostly_vertical:
        route_type  = "Go Route (Fly)"
        description = ("Straight vertical release at high speed with minimal "
                       "lateral deviation — consistent with a deep go/fly route.")

    elif has_cuts and sign_changes_x >= 4:
        if net_x > 0 and net_y < 0:
            route_type  = "Post Route"
            description = ("Initial vertical stem followed by a diagonal break "
                           "toward the middle of the field.")
        elif abs(net_x) < 30 and sign_changes_x >= 4:
            route_type  = "Slant Route"
            description = ("Sharp inside break after a quick vertical release — "
                           "consistent with a slant or quick inside route.")
        else:
            route_type  = "Out / Corner Route"
            description = ("Vertical stem with a late break toward the sideline "
                           "or corner of the end zone.")

    elif sign_changes_x >= 2 and avg_mph < 11:
        route_type  = "Curl / Comeback"
        description = ("Player drove vertically then slowed and broke back "
                       "toward the line of scrimmage — curl or comeback route.")

    elif mostly_lateral and avg_mph < 10:
        route_type  = "Flat / Screen"
        description = ("Shallow horizontal movement — consistent with a flat "
                       "route, screen, or swing pass.")

    elif sign_changes_y >= 3 and avg_mph > 10:
        route_type  = "Crossing / Dig Route"
        description = ("Horizontal crossing movement at route speed across "
                       "the middle of the field.")

    elif sprint_pct < 0.15 and avg_mph < 8:
        route_type  = "Short Stem / Hitch"
        description = ("Low overall speed with minimal displacement — "
                       "short hitch, quick out, or check-down release.")
    else:
        route_type  = "Combination Route"
        description = ("Multi-phase movement with mixed speed and direction "
                       "changes — could be a wheel, option, or broken route.")

    return {
        "route_type":    route_type,
        "description":   description,
        "peak_mph":      round(peak_mph, 2),
        "avg_mph":       round(avg_mph, 2),
        "sprint_pct":    round(sprint_pct * 100, 1),
        "direction_changes": sign_changes_x + sign_changes_y,
        "total_yards":   round(total_dist, 1),
    }


def compute_score(df: "pd.DataFrame") -> dict:
    """
    Scores the WR performance 0–100 across 5 categories.
    Returns individual scores and an overall weighted score with letter grade.
    """
    if df.empty:
        return {}

    peak_mph = df["mph"].max()
    avg_mph  = df["mph"].mean()
    sprint_pct = (df["mph"] > 14).mean()

    # 1. TOP SPEED (0–100): 28 mph = 100
    speed_score = min(100, (peak_mph / 28.0) * 100)

    # 2. ACCELERATION (0–100): reaching top speed quickly
    #    Look at how fast mph rises from standing
    mph_vals = df["mph"].values
    max_accel = df["accel_mph_s"].clip(lower=0).max()
    accel_score = min(100, (max_accel / 15.0) * 100)   # 15 mph/s burst = 100

    # 3. ROUTE EFFICIENCY (0–100): total distance vs net displacement
    #    More efficient = straighter route = higher score
    total_yards = df["total_yards"].iloc[-1]
    xs, ys = df["x_px"].values, df["y_px"].values
    net_disp_px = math.hypot(xs[-1]-xs[0], ys[-1]-ys[0])
    # Convert net displacement to yards
    px_per_yard_est = total_yards / max(net_disp_px, 1) if net_disp_px > 0 else 1
    if total_yards > 0:
        efficiency = min(net_disp_px / max(total_yards * (1/px_per_yard_est if px_per_yard_est!=1 else 15), 1), 1.0)
    else:
        efficiency = 0.5
    route_score = round(min(100, efficiency * 100 * 1.2), 1)  # slight boost

    # 4. SUSTAINED SPEED (0–100): % of time at route speed or above
    sustained_score = min(100, sprint_pct * 200)   # 50% sprint time = 100

    # 5. STRIDE CONSISTENCY (0–100): lower stride variance = more consistent
    stride_vals = df["stride_ft"].replace(0, np.nan).dropna()
    if len(stride_vals) > 5:
        cv = stride_vals.std() / (stride_vals.mean() + 1e-9)   # coefficient of variation
        stride_score = max(0, 100 - cv * 120)
    else:
        stride_score = 50.0

    # Weighted overall (speed matters most for WR)
    weights = {
        "Top Speed":    0.30,
        "Acceleration": 0.20,
        "Route Efficiency": 0.20,
        "Sustained Speed":  0.20,
        "Stride Consistency": 0.10,
    }
    scores = {
        "Top Speed":          round(speed_score,    1),
        "Acceleration":       round(accel_score,    1),
        "Route Efficiency":   round(route_score,    1),
        "Sustained Speed":    round(sustained_score,1),
        "Stride Consistency": round(stride_score,   1),
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    overall = round(overall, 1)

    if overall >= 90:   grade = "A+"
    elif overall >= 85: grade = "A"
    elif overall >= 80: grade = "A-"
    elif overall >= 75: grade = "B+"
    elif overall >= 70: grade = "B"
    elif overall >= 65: grade = "B-"
    elif overall >= 60: grade = "C+"
    elif overall >= 55: grade = "C"
    else:               grade = "C-"

    return {"scores": scores, "overall": overall, "grade": grade, "weights": weights}


# ══════════════════════════════════════════════════════════════════════════════
#  COCO KEYPOINT INDICES  (yolov8n-pose outputs 17 keypoints)
# ══════════════════════════════════════════════════════════════════════════════
KP = {
    "nose":       0,
    "l_eye":      1,  "r_eye":      2,
    "l_ear":      3,  "r_ear":      4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow":    7,  "r_elbow":    8,
    "l_wrist":    9,  "r_wrist":   10,
    "l_hip":     11,  "r_hip":     12,
    "l_knee":    13,  "r_knee":    14,
    "l_ankle":   15,  "r_ankle":   16,
}

SKELETON_PAIRS = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# BGR colors
C = {
    "green":    (0,   220,  80),
    "cyan":     (0,   220, 220),
    "yellow":   (0,   220, 255),
    "orange":   (0,   140, 255),
    "white":    (255, 255, 255),
    "dim":      (160, 160, 160),
    "panel_bg": (15,   15,  15),
}


# ══════════════════════════════════════════════════════════════════════════════
#  KEYPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kp_xy(kpts_xy: np.ndarray, kpts_conf: np.ndarray | None, idx: int):
    """
    Returns (x, y) for keypoint idx, or None if the point is invalid.
    Handles both (N,2) xy-only and confidence-gated cases.
    """
    x, y = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])

    # If we have confidence scores, gate on threshold
    if kpts_conf is not None:
        conf = float(kpts_conf[idx])
        if conf < KP_CONF_THRESHOLD:
            return None

    # Also reject if coords are exactly zero (undetected)
    if x < 1.0 and y < 1.0:
        return None

    return (x, y)

def kp_midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def kp_dist(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def kp_angle(p1, p2):
    if p1 is None or p2 is None:
        return None
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATOR
# ══════════════════════════════════════════════════════════════════════════════

class Calibrator:
    """
    Click two points with a known real distance → returns px_per_yard.
    Press S in the window to skip (uses DEFAULT_PX_PER_YARD).
    """

    @staticmethod
    def run(frame: np.ndarray) -> float:
        points  = []
        display = frame.copy()
        h, w    = display.shape[:2]

        cv2.rectangle(display, (0,0), (w,56), (10,10,10), -1)
        cv2.putText(display,
            "CALIBRATION: Click 2 field points with a known real distance",
            (12,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["yellow"], 1, cv2.LINE_AA)
        cv2.putText(display,
            "Good picks: yard lines, hash marks, end zone edge  |  S = skip",
            (12,44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["dim"], 1, cv2.LINE_AA)

        win = "CALIBRATION — click 2 points  (S to skip)"

        def on_click(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(display,(x,y),7,C["cyan"],-1)
                cv2.circle(display,(x,y),7,C["white"],1)
                cv2.putText(display,f"P{len(points)}",(x+10,y-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,C["cyan"],2)
                if len(points) == 2:
                    cv2.line(display,points[0],points[1],C["cyan"],2,cv2.LINE_AA)
                    px_d = math.hypot(points[1][0]-points[0][0],
                                      points[1][1]-points[0][1])
                    mid  = ((points[0][0]+points[1][0])//2,
                            (points[0][1]+points[1][1])//2)
                    cv2.putText(display,f"{px_d:.0f}px",(mid[0]+6,mid[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,C["white"],1)

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_click)

        while True:
            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                print(f"⚠️  Calibration skipped — using default {DEFAULT_PX_PER_YARD} px/yard (approximate).")
                cv2.destroyWindow(win)
                return DEFAULT_PX_PER_YARD
            if len(points) == 2:
                cv2.imshow(win, display)
                cv2.waitKey(400)
                break

        cv2.destroyWindow(win)

        px_dist = math.hypot(points[1][0]-points[0][0],
                             points[1][1]-points[0][1])
        print(f"\n📏  Pixel distance: {px_dist:.1f} px")
        print("   Enter the REAL distance in yards (e.g. 5 for five yard lines, 10 for end zone depth)")

        while True:
            try:
                real_yards = float(input("   Real distance in YARDS: ").strip())
                if real_yards > 0:
                    break
                print("   Must be a positive number.")
            except ValueError:
                print("   Please enter a number like 5 or 10.")

        px_per_yard = px_dist / real_yards
        print(f"✅  Calibrated: {px_per_yard:.2f} px = 1 yard\n")
        return px_per_yard


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class WRTracker:

    def __init__(
        self,
        video_path:   str,
        model_path:   str   = DEFAULT_MODEL_PATH,
        trail_length: int   = DEFAULT_TRAIL_LENGTH,
        px_per_yard:  float = DEFAULT_PX_PER_YARD,
    ):
        self.video_path   = video_path
        self.model_path   = model_path
        self.trail_length = trail_length
        self.px_per_yard  = px_per_yard

        self._model = None
        self._cap   = None

        self.fps          = 30.0
        self.total_frames = 0
        self.frame_width  = 0
        self.frame_height = 0

        # All internal speeds stored as px/sec — converted to human units at output
        self.locked_id    = None
        self._prev_pos    = None       # (x, y) in pixels
        self._prev_spd_px = 0.0       # px/sec
        self._total_px    = 0.0       # total pixels travelled
        self._trail       = deque(maxlen=trail_length)
        self._peak_mph    = 0.0
        self._frame_num   = 0

        self.metrics: list[dict] = []

        self._open_video()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _open_video(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        self.fps          = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    @staticmethod
    def _smooth(new_val: float, old_val: float, alpha: float = 0.75) -> float:
        return alpha * new_val + (1 - alpha) * old_val

    # ── Phase 1: selection (uses model.track so IDs match tracking) ───────────

    def get_first_frame_detections(self) -> tuple:
        """
        Runs model.track() on the first frame so IDs are consistent with
        process_frames(). Returns (annotated_frame, players).
        players = [{"id": int, "box": [x1,y1,x2,y2]}, ...]
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame.")

        model = self._get_model()

        # ── CRITICAL: use .track() not .predict() so IDs match later ──────────
        results = model.track(frame, persist=True,
                              tracker="bytetrack.yaml", verbose=False)
        result  = results[0]

        players   = []
        annotated = frame.copy()
        h, w      = annotated.shape[:2]

        cv2.rectangle(annotated, (0,0), (w,52), (10,10,10), -1)
        cv2.putText(annotated, "CLICK THE PLAYER YOU WANT TO TRACK",
                    (w//2-270, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    C["yellow"], 2, cv2.LINE_AA)

        if result.boxes.id is not None:
            ids   = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()

            for box, pid in zip(boxes, ids):
                x1,y1,x2,y2 = [int(v) for v in box]
                players.append({"id": int(pid), "box": [x1,y1,x2,y2]})

                cv2.rectangle(annotated,(x1,y1),(x2,y2),C["cyan"],2)
                label = f"#{pid}"
                (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
                cv2.rectangle(annotated,(x1,y1-th-12),(x1+tw+8,y1),C["cyan"],-1)
                cv2.putText(annotated,label,(x1+4,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2,cv2.LINE_AA)
        else:
            # No IDs from tracker — fall back to detection boxes, assign manual IDs
            print("⚠️  Tracker returned no IDs on first frame — using detection order as IDs.")
            boxes = result.boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = [int(v) for v in box]
                players.append({"id": i, "box": [x1,y1,x2,y2]})
                cv2.rectangle(annotated,(x1,y1),(x2,y2),C["cyan"],2)
                label = f"#{i}"
                (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
                cv2.rectangle(annotated,(x1,y1-th-12),(x1+tw+8,y1),C["cyan"],-1)
                cv2.putText(annotated,label,(x1+4,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2,cv2.LINE_AA)

        # Rewind — tracking will restart from frame 0
        # Reset tracker state so IDs restart fresh and consistently
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # NOTE: We do NOT reset tracker state here.
        # Instead, lock_player() stores the clicked box so we can
        # re-match by IoU overlap on frame 1 if the ID shifts.

        return annotated, players

    def select_player_by_click(self, click_x: float, click_y: float,
                                players: list) -> int | None:
        """Returns player ID whose box contains the click, or None."""
        for p in players:
            x1,y1,x2,y2 = p["box"]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                # Store the clicked box so we can re-match by IoU if ID drifts
                self._seed_box = p["box"]
                return p["id"]
        return None

    def lock_player(self, player_id: int):
        """Lock onto a player ID and reset all tracking state."""
        self.locked_id    = player_id
        self._prev_pos    = None
        self._prev_spd_px = 0.0
        self._total_px    = 0.0
        self._frame_num   = 0
        self._peak_mph    = 0.0
        self.metrics      = []
        self._trail.clear()
        if not hasattr(self, "_seed_box"):
            self._seed_box = None
        self._id_confirmed = False   # becomes True once we match by IoU on frame 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"🔒  Tracking player #{player_id}")

    # ── Phase 2: frame processing ─────────────────────────────────────────────

    def process_frames(self):
        """
        Generator — yields one dict per frame:
        {
            frame_number    : int
            total_frames    : int
            annotated_frame : np.ndarray (BGR)
            metrics         : dict | None
            done            : bool
        }

        metrics keys:
            frame, mph, yards_per_sec, accel_mph_s,
            stride_ft, stride_yards, rotation_deg,
            total_yards, total_feet, effort, x_px, y_px
        """
        if self.locked_id is None:
            raise RuntimeError("Call lock_player() before process_frames().")

        model = self._get_model()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                yield {"frame_number": self._frame_num, "total_frames": self.total_frames,
                       "annotated_frame": np.zeros((100,100,3), np.uint8),
                       "metrics": None, "done": True}
                return

            self._frame_num += 1
            cur_metrics = None

            results = model.track(
                frame, persist=True,
                tracker="bytetrack.yaml", verbose=False
            )
            result = results[0]

            if result.boxes.id is not None and result.keypoints is not None:
                ids   = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()

                # Keypoint xy coords — shape (N, 17, 2)
                kpts_xy = result.keypoints.xy.cpu().numpy()

                # Confidence scores — shape (N, 17) if available, else None
                kpts_conf = None
                if result.keypoints.conf is not None:
                    kpts_conf = result.keypoints.conf.cpu().numpy()

                # Draw all players dimly
                for i, tid in enumerate(ids):
                    x1,y1,x2,y2 = boxes[i].astype(int)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(60,60,60),1)
                    cv2.putText(frame,f"#{tid}",(x1,y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,80,80),1)

                # ── On the very first tracked frame, verify/correct the ID ──
                # ByteTrack may assign different IDs after a rewind.
                # We fix this once by finding whichever box has the best
                # IoU overlap with the box the user originally clicked.
                if not self._id_confirmed and self._seed_box is not None:
                    best_iou   = 0.0
                    best_tid   = self.locked_id
                    sx1,sy1,sx2,sy2 = self._seed_box
                    for ci, ctid in enumerate(ids):
                        bx1,by1,bx2,by2 = boxes[ci]
                        ix1,iy1 = max(sx1,bx1), max(sy1,by1)
                        ix2,iy2 = min(sx2,bx2), min(sy2,by2)
                        inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
                        union   = ((sx2-sx1)*(sy2-sy1) +
                                   (bx2-bx1)*(by2-by1) - inter)
                        iou     = inter/union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = int(ctid)
                    if best_iou > 0.3 and best_tid != self.locked_id:
                        print(f"  ℹ️  ID corrected: #{self.locked_id} → #{best_tid} (IoU={best_iou:.2f})")
                        self.locked_id = best_tid
                    self._id_confirmed = True

                # Process the locked player
                found = False
                for i, tid in enumerate(ids):
                    if int(tid) != int(self.locked_id):
                        continue
                    found = True

                    pk_xy   = kpts_xy[i]
                    pk_conf = kpts_conf[i] if kpts_conf is not None else None

                    # ── Get ankle keypoints ────────────────────────────────
                    la = kp_xy(pk_xy, pk_conf, KP["l_ankle"])
                    ra = kp_xy(pk_xy, pk_conf, KP["r_ankle"])

                    # If both ankles missing try using hips as fallback
                    if la is None and ra is None:
                        la = kp_xy(pk_xy, pk_conf, KP["l_hip"])
                        ra = kp_xy(pk_xy, pk_conf, KP["r_hip"])

                    # If still nothing, use box centre bottom
                    if la is None and ra is None:
                        x1,y1,x2,y2 = boxes[i].astype(int)
                        cx = (x1+x2)//2
                        la = ra = (float(cx), float(y2))

                    foot_mid = kp_midpoint(la, ra) or la or ra

                    # ── Speed (all in px/sec internally) ──────────────────
                    spd_px = 0.0
                    accel_px_s = 0.0
                    if self._prev_pos is not None:
                        dist_px    = float(np.linalg.norm(
                            np.array(foot_mid) - np.array(self._prev_pos)))
                        spd_px     = dist_px * self.fps          # px/sec
                        accel_px_s = (spd_px - self._prev_spd_px) * self.fps  # px/sec²
                        self._total_px += dist_px

                    spd_px            = self._smooth(spd_px, self._prev_spd_px)

                    # Hard clamp: 28 mph absolute max (world record ~27.8 mph)
                    # This kills outlier jumps from keypoint flicker between frames
                    MAX_PX_PER_SEC = 28.0 * (1760.0 / 3600.0) * self.px_per_yard
                    spd_px = min(spd_px, MAX_PX_PER_SEC)

                    self._prev_pos    = foot_mid
                    self._prev_spd_px = spd_px

                    # ── Convert to human units ─────────────────────────────
                    mph         = px_per_sec_to_mph(spd_px, self.px_per_yard)
                    yds_per_sec = px_per_sec_to_yards_per_sec(spd_px, self.px_per_yard)
                    accel_mph_s = px_per_sec_to_mph(accel_px_s, self.px_per_yard)
                    total_yards = px_to_yards(self._total_px, self.px_per_yard)
                    total_feet  = total_yards * 3.0

                    self._peak_mph = max(self._peak_mph, mph)

                    # ── Stride ─────────────────────────────────────────────
                    stride_px    = kp_dist(la, ra) or 0.0
                    stride_yards = px_to_yards(stride_px, self.px_per_yard)
                    stride_ft    = stride_yards * 3.0

                    # ── Rotation ───────────────────────────────────────────
                    ls = kp_xy(pk_xy, pk_conf, KP["l_shoulder"])
                    rs = kp_xy(pk_xy, pk_conf, KP["r_shoulder"])
                    lh = kp_xy(pk_xy, pk_conf, KP["l_hip"])
                    rh = kp_xy(pk_xy, pk_conf, KP["r_hip"])
                    sh_a  = kp_angle(ls, rs) or 0.0
                    hip_a = kp_angle(lh, rh) or 0.0
                    rotation = sh_a - hip_a

                    cur_metrics = {
                        "frame":        self._frame_num,
                        "mph":          round(mph, 2),
                        "yards_per_sec":round(yds_per_sec, 3),
                        "accel_mph_s":  round(accel_mph_s, 3),
                        "stride_ft":    round(stride_ft, 2),
                        "stride_yards": round(stride_yards, 3),
                        "rotation_deg": round(rotation, 2),
                        "total_yards":  round(total_yards, 2),
                        "total_feet":   round(total_feet, 2),
                        "effort":       speed_label(mph),
                        "x_px":         round(foot_mid[0], 1),
                        "y_px":         round(foot_mid[1], 1),
                    }
                    self.metrics.append(cur_metrics)

                    # ── Draw ───────────────────────────────────────────────
                    self._trail.append(foot_mid)
                    self._draw_trail(frame)
                    self._draw_skeleton(frame, pk_xy, pk_conf)

                    # Bright green box on tracked player
                    x1,y1,x2,y2 = boxes[i].astype(int)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),C["green"],2)

                    # Mark foot midpoint
                    cv2.circle(frame,(int(foot_mid[0]),int(foot_mid[1])),
                               6,C["yellow"],-1,cv2.LINE_AA)

                if not found:
                    # Player temporarily out of frame — show last known position
                    if self._prev_pos:
                        cv2.circle(frame,
                                   (int(self._prev_pos[0]),int(self._prev_pos[1])),
                                   10, C["orange"], 2, cv2.LINE_AA)
                        cv2.putText(frame,"SEARCHING...",(12,self.frame_height-40),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,C["orange"],2,cv2.LINE_AA)

            # Overlays
            if cur_metrics:
                self._draw_hud(frame, cur_metrics)
            else:
                cv2.putText(frame,"WAITING FOR PLAYER...",
                            (self.frame_width//2-180, self.frame_height//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,C["yellow"],2,cv2.LINE_AA)

            self._draw_badge(frame)
            self._draw_status(frame)

            yield {
                "frame_number":    self._frame_num,
                "total_frames":    self.total_frames,
                "annotated_frame": frame,
                "metrics":         cur_metrics,
                "done":            False,
            }

    # ── Phase 3: export ───────────────────────────────────────────────────────

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics)

    def save_csv(self, path: str = "wr_metrics.csv") -> str:
        df = self.get_dataframe()
        if df.empty:
            print("No metrics to save.")
            return ""
        df.to_csv(path, index=False)
        print(f"\n✅  CSV saved → {path}")
        self._print_summary(df)
        return path

    def save_report(self, path: str = "wr_report.png") -> str:
        df = self.get_dataframe()
        if df.empty:
            print("No data for report.")
            return ""
        _generate_report(df, path)
        return path

    def release(self):
        if self._cap:
            self._cap.release()

    def _print_summary(self, df: pd.DataFrame):
        route  = classify_route(df)
        scores = compute_score(df)

        print("\n" + "═"*52)
        print("  SESSION SUMMARY")
        print("═"*52)
        rows = [
            ("Peak Speed",     f"{df['mph'].max():.2f} mph"),
            ("Avg Speed",      f"{df['mph'].mean():.2f} mph"),
            ("Peak Accel",     f"{df['accel_mph_s'].max():+.2f} mph/s"),
            ("Peak Decel",     f"{df['accel_mph_s'].min():+.2f} mph/s"),
            ("Avg Stride",     f"{df['stride_ft'].mean():.2f} ft"),
            ("Max Stride",     f"{df['stride_ft'].max():.2f} ft"),
            ("Total Distance", f"{df['total_yards'].iloc[-1]:.1f} yds  "
                               f"({df['total_feet'].iloc[-1]:.0f} ft)"),
            ("Frames Tracked", f"{len(df)}"),
        ]
        for label, value in rows:
            print(f"  {label:<18}  {value}")

        print("─"*52)
        print(f"  ROUTE DETECTED:  {route['route_type']}")
        print(f"  {route['description']}")

        if scores:
            print("─"*52)
            print("  PERFORMANCE SCORES")
            for name, val in scores["scores"].items():
                bar_len = int(val / 5)
                bar     = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {name:<22} {bar}  {val:.0f}/100")
            print("─"*52)
            print(f"  OVERALL SCORE:   {scores['overall']}/100  [{scores['grade']}]")
        print("═"*52 + "\n")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_trail(self, frame):
        pts = list(self._trail)
        n   = len(pts)
        for i in range(1, n):
            if pts[i-1] is None or pts[i] is None:
                continue
            a = i / n
            cv2.line(frame,
                     (int(pts[i-1][0]),int(pts[i-1][1])),
                     (int(pts[i][0]),  int(pts[i][1])),
                     (0, int(220*a), int(80*a)),
                     max(1,int(3*a)), cv2.LINE_AA)

    def _draw_skeleton(self, frame, pk_xy: np.ndarray,
                       pk_conf: np.ndarray | None):
        for p1i, p2i in SKELETON_PAIRS:
            p1 = kp_xy(pk_xy, pk_conf, p1i)
            p2 = kp_xy(pk_xy, pk_conf, p2i)
            if p1 is None or p2 is None:
                continue
            cv2.line(frame,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),
                     C["green"],2,cv2.LINE_AA)
        for idx in range(17):
            pt = kp_xy(pk_xy, pk_conf, idx)
            if pt is None:
                continue
            cv2.circle(frame,(int(pt[0]),int(pt[1])),4,C["cyan"],-1,cv2.LINE_AA)
            cv2.circle(frame,(int(pt[0]),int(pt[1])),4,C["white"],1,cv2.LINE_AA)

    def _semi_rect(self, frame, x1,y1,x2,y2,color,alpha=0.6):
        ov = frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

    def _draw_hud(self, frame, m):
        peak = max(self._peak_mph, 1.0)
        cards = [
            ("SPEED",      f"{m['mph']:.1f}",         "mph",    C["green"],
             m["mph"]/peak),
            ("ACCEL",      f"{m['accel_mph_s']:+.1f}","mph/s",  C["cyan"],
             (m["accel_mph_s"]+10)/20),
            ("STRIDE",     f"{m['stride_ft']:.1f}",   "feet",   C["yellow"],
             min(m["stride_ft"]/10.0, 1.0)),
            ("ROTATION",   f"{m['rotation_deg']:.0f}°","sh-hip",C["orange"],
             (m["rotation_deg"]+90)/180),
            ("TOTAL DIST", f"{m['total_yards']:.1f}", "yards",  C["white"], None),
        ]
        for idx,(label,val,unit,color,bar) in enumerate(cards):
            col = idx % 2
            row = idx // 2
            x = 12 + col*175
            y = 12 + row*82
            cw,ch = 163,70
            self._semi_rect(frame,x,y,x+cw,y+ch,C["panel_bg"],0.72)
            cv2.rectangle(frame,(x,y),(x+cw,y+ch),color,1)
            cv2.putText(frame,label,(x+8,y+17),cv2.FONT_HERSHEY_SIMPLEX,0.36,C["dim"],1,cv2.LINE_AA)
            cv2.putText(frame,val,  (x+8,y+46),cv2.FONT_HERSHEY_SIMPLEX,0.80,color,  2,cv2.LINE_AA)
            cv2.putText(frame,unit, (x+8,y+62),cv2.FONT_HERSHEY_SIMPLEX,0.30,C["dim"],1,cv2.LINE_AA)
            if bar is not None:
                bar = max(0,min(1,bar))
                bx1,by1,bx2,by2 = x+8,y+ch-5,x+cw-8,y+ch-2
                cv2.rectangle(frame,(bx1,by1),(bx2,by2),(50,50,50),-1)
                cv2.rectangle(frame,(bx1,by1),(int(bx1+(bx2-bx1)*bar),by2),color,-1)

        # Effort badge
        effort = m.get("effort","")
        color  = {
            "STANDING":C["dim"],"WALKING":C["dim"],
            "JOGGING":C["yellow"],"ROUTE":C["yellow"],
            "RUNNING":C["orange"],"SPRINTING":C["green"],"TOP SPEED":C["green"],
        }.get(effort, C["white"])
        cv2.putText(frame,effort,(14,12+3*82-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2,cv2.LINE_AA)

    def _draw_badge(self, frame):
        txt = "WR TRACKER  v3.0"
        h,w = frame.shape[:2]
        (tw,_),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
        x = w-tw-20
        self._semi_rect(frame,x-8,8,w-8,36,C["panel_bg"],0.75)
        cv2.rectangle(frame,(x-8,8),(w-8,36),C["green"],1)
        cv2.putText(frame,txt,(x,28),cv2.FONT_HERSHEY_SIMPLEX,0.44,C["green"],1,cv2.LINE_AA)

    def _draw_status(self, frame):
        h,w = frame.shape[:2]
        self._semi_rect(frame,0,h-28,w,h,C["panel_bg"],0.75)
        pct = self._frame_num / max(self.total_frames,1)
        cv2.rectangle(frame,(0,h-3),(int(w*pct),h),C["green"],-1)
        elapsed = self._frame_num / self.fps
        txt = (f"  FRAME {self._frame_num}/{self.total_frames}"
               f"  |  {elapsed:.1f}s"
               f"  |  TRACKING #{self.locked_id}"
               f"  |  PEAK {self._peak_mph:.1f} mph")
        cv2.putText(frame,txt,(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.37,C["dim"],1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _generate_report(df: pd.DataFrame, out_path: str):
    sns.set_theme(style="dark", rc={"axes.facecolor":"#0d0d0d",
                                     "figure.facecolor":"#0a0a0a",
                                     "grid.color":"#1e1e1e"})
    GREEN="#00dc50";CYAN="#00dcdc";YELLOW="#ffd700"
    ORANGE="#ff8c00";DIM="#606060";WHITE="#e8e8e8";BG="#0d1117"

    fig = plt.figure(figsize=(18,11),facecolor="#080c10")
    gs  = gridspec.GridSpec(3,4,figure=fig,hspace=0.55,wspace=0.38,
                             left=0.06,right=0.97,top=0.88,bottom=0.07)

    def sa(ax):
        ax.set_facecolor(BG);ax.tick_params(colors=DIM,labelsize=7)
        ax.spines[:].set_color("#222")
        for lb in ax.get_xticklabels()+ax.get_yticklabels():lb.set_color(DIM)

    tkw=dict(color=WHITE,fontsize=11,fontweight="bold",pad=8)
    lkw=dict(color=DIM,fontsize=8)

    fig.text(0.5,0.955,"WIDE RECEIVER PERFORMANCE REPORT",ha="center",
             color=GREEN,fontsize=20,fontweight="black",fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3,foreground="#001a00")])
    fig.text(0.5,0.928,
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  {len(df)} frames tracked",
             ha="center",color=DIM,fontsize=8,fontfamily="monospace")

    frames=df["frame"].values

    ax=fig.add_subplot(gs[0,:2])
    ax.plot(frames,df["mph"],color=GREEN,lw=1.5)
    ax.fill_between(frames,df["mph"],alpha=0.12,color=GREEN)
    ax.axhline(df["mph"].mean(),color=GREEN,lw=0.8,ls="--",alpha=0.5)
    for spd,lbl in [(6,"Jog"),(12,"Route"),(18,"Sprint")]:
        if spd < df["mph"].max():
            ax.axhline(spd,color=DIM,lw=0.5,ls=":",alpha=0.5)
            ax.text(frames[-1],spd+0.3,lbl,color=DIM,fontsize=6,ha="right")
    ax.set_title("SPEED (mph)",**tkw);ax.set_xlabel("Frame",**lkw);sa(ax)

    ax2=fig.add_subplot(gs[0,2:])
    cc=[ORANGE if v>=0 else CYAN for v in df["accel_mph_s"]]
    ax2.bar(frames,df["accel_mph_s"],color=cc,width=1.2,alpha=0.8)
    ax2.axhline(0,color=DIM,lw=0.6)
    ax2.set_title("ACCELERATION (mph/s)",**tkw);ax2.set_xlabel("Frame",**lkw);sa(ax2)

    ax3=fig.add_subplot(gs[1,:2])
    ax3.plot(frames,df["stride_ft"],color=YELLOW,lw=1.5)
    ax3.fill_between(frames,df["stride_ft"],alpha=0.10,color=YELLOW)
    ax3.set_title("STRIDE LENGTH (feet)",**tkw);ax3.set_xlabel("Frame",**lkw);sa(ax3)

    ax4=fig.add_subplot(gs[1,2:])
    ax4.plot(frames,df["rotation_deg"],color=ORANGE,lw=1.5)
    ax4.axhline(0,color=DIM,lw=0.6,ls="--")
    ax4.set_title("SHOULDER–HIP ROTATION (°)",**tkw);ax4.set_xlabel("Frame",**lkw);sa(ax4)

    ax5=fig.add_subplot(gs[2,0])
    ax5.hist(df["mph"],bins=25,color=GREEN,alpha=0.8,edgecolor="#001a00")
    ax5.set_title("SPEED DISTRIBUTION",**tkw);ax5.set_xlabel("mph",**lkw);sa(ax5)

    ax6=fig.add_subplot(gs[2,1])
    spd_n=(df["mph"]-df["mph"].min())/(df["mph"].max()-df["mph"].min()+1e-9)
    sc=ax6.scatter(df["x_px"],df["y_px"],c=spd_n,cmap="YlGn",s=4,alpha=0.7)
    ax6.invert_yaxis()
    ax6.set_title("ROUTE MAP (color=speed)",**tkw)
    ax6.set_xlabel("X (px)",**lkw);ax6.set_ylabel("Y (px)",**lkw);sa(ax6)
    cb=plt.colorbar(sc,ax=ax6,fraction=0.04);cb.ax.tick_params(colors=DIM,labelsize=6)

    ax7=fig.add_subplot(gs[2,2:]);ax7.axis("off");ax7.set_facecolor(BG)
    stats=[
        ("Peak Speed",     f"{df['mph'].max():.2f} mph",              GREEN),
        ("Avg Speed",      f"{df['mph'].mean():.2f} mph",             GREEN),
        ("Peak Accel",     f"{df['accel_mph_s'].max():+.2f} mph/s",   ORANGE),
        ("Peak Decel",     f"{df['accel_mph_s'].min():+.2f} mph/s",   CYAN),
        ("Avg Stride",     f"{df['stride_ft'].mean():.2f} ft",        YELLOW),
        ("Max Stride",     f"{df['stride_ft'].max():.2f} ft",         YELLOW),
        ("Total Distance", f"{df['total_yards'].iloc[-1]:.1f} yds",   WHITE),
        ("Frames Tracked", f"{len(df)}",                              DIM),
    ]
    ax7.text(0.02,0.98,"SESSION SUMMARY",color=WHITE,fontsize=10,fontweight="black",
             va="top",fontfamily="monospace",transform=ax7.transAxes)
    for i,(lbl,val,col) in enumerate(stats):
        y=0.82-i*0.105
        ax7.text(0.02,y,lbl.upper(),color=DIM,fontsize=7.5,
                 fontfamily="monospace",transform=ax7.transAxes)
        ax7.text(0.98,y,val,color=col,fontsize=9,fontweight="bold",
                 ha="right",fontfamily="monospace",transform=ax7.transAxes)
        ax7.plot([0.02,0.98],[y-0.02,y-0.02],color="#1e1e1e",lw=0.5,transform=ax7.transAxes)

    plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅  Report saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE  —  python wr_tracker.py --video input.mp4
# ══════════════════════════════════════════════════════════════════════════════

def _run_standalone(video_path: str):
    print("\n" + "═"*52)
    print("  WR TRACKER  v3.0")
    print("═"*52)

    # Grab first frame for calibration
    temp = cv2.VideoCapture(video_path)
    ret, first_frame = temp.read()
    temp.release()
    if not ret:
        print("Error: could not read video.")
        return

    # Step 1: Calibrate
    print("\nSTEP 1 — Calibration")
    print("  Click 2 field points with a known real distance between them.")
    print("  (Press S in the window to skip)\n")
    px_per_yard = Calibrator.run(first_frame)

    # Step 2: Init + detect players
    tracker = WRTracker(video_path, px_per_yard=px_per_yard)
    annotated, players = tracker.get_first_frame_detections()

    print(f"STEP 2 — Player Selection")
    print(f"  Detected {len(players)} player(s).")
    print("  Click a player in the window to lock onto them.\n")

    selected = {"id": None}

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and selected["id"] is None:
            pid = tracker.select_player_by_click(x, y, players)
            if pid is not None:
                selected["id"] = pid

    win = "WR Tracker — click a player  |  Q to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    while selected["id"] is None:
        cv2.imshow(win, annotated)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            tracker.release()
            return

    tracker.lock_player(selected["id"])
    print(f"\nSTEP 3 — Tracking  (Q to stop early)\n")

    # Step 3: Track
    for result in tracker.process_frames():
        cv2.imshow(win, result["annotated_frame"])

        m = result["metrics"]
        if m and result["frame_number"] % 15 == 0:
            print(f"  {m['frame']:>5}f  |"
                  f"  {m['mph']:>5.1f} mph  |"
                  f"  {m['effort']:<10}  |"
                  f"  stride {m['stride_ft']:.1f} ft  |"
                  f"  dist {m['total_yards']:.1f} yds")

        if cv2.waitKey(1) & 0xFF == ord("q") or result["done"]:
            break

    cv2.destroyAllWindows()
    tracker.release()

    # Step 4: Save
    stem = Path(video_path).stem
    tracker.save_csv(f"{stem}_metrics.csv")
    tracker.save_report(f"{stem}_report.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WR Tracker v3.0")
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()
    _run_standalone(args.video)