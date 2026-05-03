"""
detection.py
────────────
YOLO-based detector with automatic model selection.

Priority:
  1. best.pt   — custom-trained football model (players + ball)
  2. yolov8n.pt — generic COCO fallback

Returns structured Detection objects per frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import: ultralytics is only required at runtime
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics not installed. Run: pip install ultralytics"
    ) from e


# ── Constants ───────────────────────────────────────────────────────────────
# COCO class indices used when falling back to yolov8n.pt
COCO_PERSON_CLS  = 0
COCO_SPORTS_BALL = 32

# Custom model class names (assumed when best.pt is used).
# Adjust to match the actual label names in your best.pt training config.
CUSTOM_PLAYER_LABELS = {"player", "person", "footballer"}
CUSTOM_BALL_LABELS   = {"ball", "sports ball", "football"}

# Minimum confidence to accept a detection
DEFAULT_CONF_THRESHOLD = 0.35


@dataclass
class Detection:
    """Single object detection for one frame."""
    track_id:   Optional[int]      # assigned by tracker; None before tracking
    class_id:   int
    class_name: str
    confidence: float
    bbox:       np.ndarray         # [x1, y1, x2, y2]  — pixel coords

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area(self) -> float:
        w = float(self.bbox[2] - self.bbox[0])
        h = float(self.bbox[3] - self.bbox[1])
        return w * h

    def is_player(self) -> bool:
        return self.class_name.lower() in CUSTOM_PLAYER_LABELS or self.class_id == COCO_PERSON_CLS

    def is_ball(self) -> bool:
        return self.class_name.lower() in CUSTOM_BALL_LABELS or self.class_id == COCO_SPORTS_BALL


@dataclass
class FrameDetections:
    """All detections for a single video frame."""
    frame_idx:  int
    players:    list[Detection] = field(default_factory=list)
    ball:       Optional[Detection] = None

    def player_count(self) -> int:
        return len(self.players)


# ── Detector class ───────────────────────────────────────────────────────────
class FootballDetector:
    """
    Wraps a YOLO model and exposes a clean per-frame detection API.

    Usage
    -----
    detector = FootballDetector.load(model_dir=".")
    for frame_idx, frame in enumerate(frames):
        fd = detector.detect(frame, frame_idx)
    """

    def __init__(self, model: "YOLO", is_custom: bool, conf: float = DEFAULT_CONF_THRESHOLD):
        self.model     = model
        self.is_custom = is_custom
        self.conf      = conf

    # ── Factory ─────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        model_dir: str = ".",
        conf: float = DEFAULT_CONF_THRESHOLD,
    ) -> "FootballDetector":
        """
        Attempt to load best.pt from model_dir; fall back to yolov8n.pt.
        yolov8n.pt is downloaded automatically by ultralytics on first use.
        """
        best_path = Path(model_dir) / "best.pt"

        if best_path.exists():
            print(f"[Detector] Loading custom model: {best_path}")
            model     = YOLO(str(best_path))
            is_custom = True
        else:
            print("[Detector] best.pt not found — falling back to yolov8n.pt")
            model     = YOLO("yolov8n.pt")
            is_custom = False

        return cls(model, is_custom, conf)

    # ── Core inference ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> FrameDetections:
        """
        Run YOLO on a single BGR frame (as returned by cv2.VideoCapture).

        Parameters
        ----------
        frame     : np.ndarray  H×W×3 BGR
        frame_idx : int         position in the source video

        Returns
        -------
        FrameDetections
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        fd      = FrameDetections(frame_idx=frame_idx)

        if results.boxes is None:
            return fd

        names = results.names  # {id: "name"}

        for box in results.boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            conf_val = float(box.conf[0].item())
            xyxy     = box.xyxy[0].cpu().numpy()

            det = Detection(
                track_id=None,
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf_val,
                bbox=xyxy,
            )

            if det.is_player():
                fd.players.append(det)
            elif det.is_ball() and fd.ball is None:
                # keep only highest-confidence ball detection
                fd.ball = det

        return fd

    # ── Batch helper ────────────────────────────────────────────────────────
    def detect_video_frames(
        self,
        frames: list[np.ndarray],
        start_idx: int = 0,
    ) -> list[FrameDetections]:
        """Run detection on a list of frames; returns one FrameDetections per frame."""
        return [
            self.detect(frame, frame_idx=start_idx + i)
            for i, frame in enumerate(frames)
        ]
