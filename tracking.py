"""
tracking.py
───────────
DeepSORT-based player tracker with Ball Memory (Ghost Tracking).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from detection import Detection, FrameDetections

# ── Try to import DeepSORT ───────────────────────────────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _DEEPSORT_AVAILABLE = True
except ImportError:
    _DEEPSORT_AVAILABLE = False
    print(
        "[Tracker] deep_sort_realtime not found — using IoU fallback tracker.\n"
        "          Install with: pip install deep-sort-realtime"
    )


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class TrackedPlayer:
    track_id:    int
    bbox:        np.ndarray          
    confidence:  float
    frame_idx:   int                 
    history:     List[np.ndarray] = field(default_factory=list)   

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class TrackingResult:
    frame_idx:      int
    tracked_players: List[TrackedPlayer] = field(default_factory=list)
    ball_bbox:      Optional[np.ndarray] = None

    def get_player(self, track_id: int) -> Optional[TrackedPlayer]:
        for p in self.tracked_players:
            if p.track_id == track_id:
                return p
        return None


# ── DeepSORT wrapper ─────────────────────────────────────────────────────────
class DeepSORTTracker:

    def __init__(
        self,
        max_age: int = 30,
        n_init: int  = 3,
        max_iou_distance: float = 0.7,
    ):
        self.max_age          = max_age
        self.n_init           = n_init
        self.max_iou_distance = max_iou_distance

        # ── NEW: Ball Memory (Ghost Tracking) ──
        self._last_ball_bbox = None
        self._ball_missing_frames = 0
        self.max_ball_missing_frames = 7  # Hold the ball for ~7 frames if it blurs

        if _DEEPSORT_AVAILABLE:
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                embedder="mobilenet",
                half=True,
                bgr=True,
            )
            self._use_deepsort = True
        else:
            self._iou_tracker  = _IoUFallbackTracker(max_age=max_age)
            self._use_deepsort = False

    def update(self, fd: FrameDetections, frame: np.ndarray) -> TrackingResult:
        
        # ── NEW: Ball Memory Logic ──
        if fd.ball is not None:
            self._last_ball_bbox = fd.ball.bbox.copy()
            self._ball_missing_frames = 0
        else:
            if self._last_ball_bbox is not None and self._ball_missing_frames < self.max_ball_missing_frames:
                self._ball_missing_frames += 1
            else:
                self._last_ball_bbox = None

        result = TrackingResult(
            frame_idx=fd.frame_idx,
            ball_bbox=self._last_ball_bbox, # Inject the ghost ball
        )

        if not fd.players:
            return result

        if self._use_deepsort:
            tracked = self._update_deepsort(fd, frame)
        else:
            tracked = self._iou_tracker.update(fd)

        result.tracked_players = tracked
        return result

    def _update_deepsort(
        self,
        fd: FrameDetections,
        frame: np.ndarray,
    ) -> List[TrackedPlayer]:
        raw_detections = []
        for det in fd.players:
            x1, y1, x2, y2 = det.bbox
            w, h = float(x2 - x1), float(y2 - y1)
            raw_detections.append(([float(x1), float(y1), w, h], det.confidence, det.class_id))

        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        players: List[TrackedPlayer] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()                        
            bbox = np.array(ltrb, dtype=np.float32)
            players.append(TrackedPlayer(
                track_id=tid,
                bbox=bbox,
                confidence=1.0,
                frame_idx=fd.frame_idx,
            ))

        return players

    def reset(self) -> None:
        self._last_ball_bbox = None
        self._ball_missing_frames = 0
        
        if self._use_deepsort:
            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                embedder="mobilenet",
                half=True,
                bgr=True,
            )
        else:
            self._iou_tracker.reset()


# ── IoU fallback tracker ─────────────────────────────────────────────────────
class _IoUFallbackTracker:
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age       = max_age
        self.iou_threshold = iou_threshold
        self._next_id: int                     = 1
        self._tracks: Dict[int, TrackedPlayer] = {}
        self._ages:   Dict[int, int]           = {}

    def update(self, fd: FrameDetections) -> List[TrackedPlayer]:
        detections = [d.bbox for d in fd.players]
        if not detections:
            self._age_out()
            return []

        matched, unmatched_dets = self._match(detections)

        for track_id, det_idx in matched:
            bbox = detections[det_idx]
            tp   = self._tracks[track_id]
            tp.bbox      = bbox
            tp.frame_idx = fd.frame_idx
            tp.history.append(bbox.copy())
            self._ages[track_id] = 0

        for det_idx in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            bbox = detections[det_idx]
            self._tracks[tid] = TrackedPlayer(
                track_id=tid,
                bbox=bbox,
                confidence=fd.players[det_idx].confidence,
                frame_idx=fd.frame_idx,
                history=[bbox.copy()],
            )
            self._ages[tid] = 0

        self._age_out()
        return list(self._tracks.values())

    def _match(
        self,
        detections: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        if not self._tracks:
            return [], list(range(len(detections)))

        track_ids  = list(self._tracks.keys())
        track_bbs  = [self._tracks[t].bbox for t in track_ids]

        matched:        List[Tuple[int, int]] = []
        unmatched_dets: List[int]             = list(range(len(detections)))

        for i, det_bbox in enumerate(detections):
            best_iou = self.iou_threshold
            best_tid = None
            for j, tid in enumerate(track_ids):
                iou = _iou(det_bbox, track_bbs[j])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                matched.append((best_tid, i))
                unmatched_dets.remove(i)

        return matched, unmatched_dets

    def _age_out(self) -> None:
        to_del = []
        for tid in list(self._ages):
            self._ages[tid] += 1
            if self._ages[tid] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            self._tracks.pop(tid, None)
            self._ages.pop(tid, None)

    def reset(self) -> None:
        self._tracks.clear()
        self._ages.clear()
        self._next_id = 1

def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0