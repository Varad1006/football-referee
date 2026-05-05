"""
interaction.py
──────────────
Detects player-player interaction events from tracking results,
and uses YOLO bounding boxes to determine if the ball was played.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking import TrackingResult, TrackedPlayer

# ── Configuration ─────────────────────────────────────────────────────────────
# INCREASED to 250px so the interaction "starts" recording before the slide tackle connects
DEFAULT_DISTANCE_THRESHOLD = 250.0   
DEFAULT_MIN_DURATION_FRAMES = 2      
DEFAULT_COOLDOWN_FRAMES = 15         

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class InteractionEvent:
    player_a_id:  int
    player_b_id:  int
    start_frame:  int
    end_frame:    int
    peak_frame:   int
    min_distance: float
    ball_won_cleanly: bool = False  # Driven by YOLO bounding boxes

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def center_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2

@dataclass
class _ActiveInteraction:
    player_a_id:  int
    player_b_id:  int
    start_frame:  int
    last_frame:   int
    distances:    List[float] = field(default_factory=list)
    ball_touches: List[bool]  = field(default_factory=list)

    def update(self, frame_idx: int, dist: float, pa_box: np.ndarray, pb_box: np.ndarray, ball_box: Optional[np.ndarray]) -> None:
        self.last_frame = frame_idx
        self.distances.append(dist)
        
        # True if either player's bounding box overlaps with the ball's bounding box
        pa_touch = _bboxes_intersect(pa_box, ball_box)
        pb_touch = _bboxes_intersect(pb_box, ball_box)
        self.ball_touches.append(pa_touch or pb_touch)

    def to_event(self) -> InteractionEvent:
        min_dist  = float(np.min(self.distances))
        peak_off  = int(np.argmin(self.distances))
        peak_frm  = self.start_frame + peak_off
        
        # Did a player intersect with the ball BEFORE or AT the peak collision?
        touches_before_peak = self.ball_touches[:peak_off + 1]
        ball_won_cleanly = any(touches_before_peak)

        return InteractionEvent(
            player_a_id=self.player_a_id,
            player_b_id=self.player_b_id,
            start_frame=self.start_frame,
            end_frame=self.last_frame,
            peak_frame=peak_frm,
            min_distance=min_dist,
            ball_won_cleanly=ball_won_cleanly,
        )

# ── Detector ──────────────────────────────────────────────────────────────────
class InteractionDetector:

    def __init__(
        self,
        distance_threshold:   float = DEFAULT_DISTANCE_THRESHOLD,
        min_duration_frames:  int   = DEFAULT_MIN_DURATION_FRAMES,
        cooldown_frames:      int   = DEFAULT_COOLDOWN_FRAMES,
    ):
        self.distance_threshold  = distance_threshold
        self.min_duration_frames = min_duration_frames
        self.cooldown_frames     = cooldown_frames

        self._active:    Dict[Tuple[int, int], _ActiveInteraction] = {}
        self._cooldowns: Dict[Tuple[int, int], int]                = {}
        self._confirmed: List[InteractionEvent]                    = []

    def update(self, tr: TrackingResult) -> List[InteractionEvent]:
        players    = tr.tracked_players
        ball_bbox  = tr.ball_bbox
        frame_idx  = tr.frame_idx
        newly_conf: List[InteractionEvent] = []

        seen_pairs: set[Tuple[int, int]] = set()

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pa = players[i]
                pb = players[j]
                pair = _canonical_pair(pa.track_id, pb.track_id)
                seen_pairs.add(pair)

                dist = _euclidean(pa, pb)

                if dist <= self.distance_threshold:
                    if _in_cooldown(pair, frame_idx, self._cooldowns, self.cooldown_frames):
                        continue

                    if pair not in self._active:
                        self._active[pair] = _ActiveInteraction(
                            player_a_id=pair[0],
                            player_b_id=pair[1],
                            start_frame=frame_idx,
                            last_frame=frame_idx,
                        )
                    # Pass the bounding boxes down to check for ball intersection
                    self._active[pair].update(frame_idx, dist, pa.bbox, pb.bbox, ball_bbox)

                    ai = self._active[pair]
                    if len(ai.distances) == self.min_duration_frames:
                        event = ai.to_event()
                        self._confirmed.append(event)
                        newly_conf.append(event)

                else:
                    if pair in self._active:
                        ai = self._active.pop(pair)
                        if len(ai.distances) >= self.min_duration_frames:
                            self._cooldowns[pair] = frame_idx

        stale = [p for p in self._active if p not in seen_pairs]
        for pair in stale:
            ai = self._active.pop(pair)
            if len(ai.distances) >= self.min_duration_frames:
                self._cooldowns[pair] = frame_idx

        return newly_conf

    def finalize(self) -> List[InteractionEvent]:
        for pair, ai in self._active.items():
            if len(ai.distances) >= self.min_duration_frames:
                event = ai.to_event()
                if event not in self._confirmed:
                    self._confirmed.append(event)
        self._active.clear()
        return list(self._confirmed)

    def reset(self) -> None:
        self._active.clear()
        self._cooldowns.clear()
        self._confirmed.clear()

    @property
    def confirmed_events(self) -> List[InteractionEvent]:
        return list(self._confirmed)

# ── Utilities ─────────────────────────────────────────────────────────────────
def _canonical_pair(a: int, b: int) -> Tuple[int, int]:
    return (min(a, b), max(a, b))

def _euclidean(pa: TrackedPlayer, pb: TrackedPlayer) -> float:
    return float(np.sqrt((pa.cx - pb.cx) ** 2 + (pa.cy - pb.cy) ** 2))

# INCREASED MARGIN to 100.0 to account for massive slide tackle extensions
def _bboxes_intersect(b1: Optional[np.ndarray], b2: Optional[np.ndarray], margin: float = 100.0) -> bool:
    """Returns True if two bounding boxes overlap, factoring in a reach margin."""
    if b1 is None or b2 is None:
        return False
        
    # Inflate the player bounding box (b1) by 'margin' pixels to account for outstretched legs
    p_x1 = b1[0] - margin
    p_y1 = b1[1] - margin
    p_x2 = b1[2] + margin
    p_y2 = b1[3] + margin
    
    # Ball bounding box (b2)
    b_x1, b_y1, b_x2, b_y2 = b2[0], b2[1], b2[2], b2[3]

    # Check for intersection with the inflated player box
    if p_x2 < b_x1 or p_x1 > b_x2 or p_y2 < b_y1 or p_y1 > b_y2:
        return False
        
    return True

def _in_cooldown(
    pair: Tuple[int, int],
    frame_idx: int,
    cooldowns: Dict[Tuple[int, int], int],
    cooldown_frames: int,
) -> bool:
    if pair not in cooldowns:
        return False
    return frame_idx - cooldowns[pair] < cooldown_frames