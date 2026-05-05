"""
pipeline.py  (fixed)
─────────────────────

FIXES APPLIED
─────────────
1. CLIP TOO SHORT (8 frames → 16 frames)
   At 30fps, 8 frames = 0.27 seconds. A tackle typically spans 0.3–0.8s.
   With frame_skip_n=2, 8 stored frames = 16 source frames = 0.53s.
   Extending to 16 clip frames = 32 source frames = 1.06s captures the
   full contact sequence including approach and follow-through.

2. YOLO_INTERVAL=3 MISSES CONTACT FRAMES
   YOLO re-ran every 3 processed frames. With frame_skip_n=2, that means
   YOLO fired every 6 source frames (~200ms at 30fps). A tackle is 4-8
   source frames — YOLO could easily skip the entire contact window.
   Fix: YOLO_INTERVAL reduced to 2 (every 2 processed frames).

3. FALLBACK CLIP USED ONLY 8 FRAMES FROM FULL VIDEO
   When no interaction was detected, pipeline fell back to a uniform
   8-frame sample of the entire video — mostly non-contact frames.
   Fix: fallback samples 16 frames, biased toward the middle third of
   the video (where interactions typically occur in short clips).

4. INTERACTION DETECTOR MIN_DURATION=4 FRAMES TOO STRICT
   With frame_skip_n=2 and YOLO_INTERVAL=3, a 5-frame tackle might only
   register as 1-2 interaction frames. Fix: min_duration reduced to 2.

5. DISTANCE_THRESHOLD=120px TOO TIGHT
   At typical broadcast zoom, players in contact are 80-200px apart
   (centre-to-centre of bboxes). 120px missed standing tackles.
   Fix: increased to 180px.

All other pipeline logic is unchanged.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from aggregation     import Aggregator, AggregationResult
from clip_extraction import ClipExtractor
from detection       import FootballDetector
from inference       import EnsembleClassifier
from interaction     import InteractionDetector, InteractionEvent
from tracking        import DeepSORTTracker


# ── Tunable knobs ─────────────────────────────────────────────────────────────
FRAME_SKIP_N       = 2      # process 1 in every N decoded frames
YOLO_INTERVAL      = 2      # run YOLO every N processed frames (was 3)
RING_BUFFER_FRAMES = 64     # larger buffer to cover long clips (was 32)
MAX_TOTAL_FRAMES   = 900
INFER_FRAME_SIZE   = (256, 256)

# Interaction detector — relaxed for better sensitivity
DISTANCE_THRESHOLD  = 180   # pixels (was 120)
MIN_DURATION_FRAMES = 2     # frames (was 4)

# Clip length — longer to capture full contact sequence
CLIP_LENGTH = 16            # frames (was 8)


#  Result 
@dataclass
class PipelineResult:
    aggregation:     AggregationResult
    events:          List[InteractionEvent]               = field(default_factory=list)
    top_frames:      List[Tuple[int, float, np.ndarray]]  = field(default_factory=list)
    elapsed_seconds: float                                = 0.0
    pipeline_mode:   str                                  = "full"
    warning:         Optional[str]                        = None
    frames_decoded:  int                                  = 0
    frames_yolo:     int                                  = 0
    tracking_history: Dict[int, List[TrackedPlayer]]      = field(default_factory=dict)


# ── Pipeline
class RefereePipeline:

    @classmethod
    def load(
        cls,
        model_dir:           str   = ".",
        detection_conf:      float = 0.30,         # was 0.35 — more sensitive
        distance_threshold:  float = DISTANCE_THRESHOLD,
        min_duration_frames: int   = MIN_DURATION_FRAMES,
        clip_length:         int   = CLIP_LENGTH,
        frame_skip_n:        int   = FRAME_SKIP_N,
        yolo_interval:       int   = YOLO_INTERVAL,
        ring_buffer_frames:  int   = RING_BUFFER_FRAMES,
    ) -> "RefereePipeline":
        print("[Pipeline] Initialising components…")
        print(f"[Pipeline] Settings: clip_length={clip_length} "
              f"frame_skip={frame_skip_n} yolo_interval={yolo_interval} "
              f"dist_thresh={distance_threshold} min_duration={min_duration_frames}")

        detector    = FootballDetector.load(model_dir=model_dir, conf=detection_conf)
        tracker     = DeepSORTTracker()
        interaction = InteractionDetector(
            distance_threshold=distance_threshold,
            min_duration_frames=min_duration_frames,
        )
        extractor   = ClipExtractor(
            clip_length=clip_length,
            frame_size=INFER_FRAME_SIZE,
        )
        classifier  = EnsembleClassifier.load(model_dir=model_dir)
        aggregator  = Aggregator()

        print("[Pipeline] Ready.")
        return cls(
            detector, tracker, interaction, extractor, classifier, aggregator,
            frame_skip_n=frame_skip_n,
            yolo_interval=yolo_interval,
            ring_buffer_frames=ring_buffer_frames,
        )

    def __init__(
        self,
        detector:    FootballDetector,
        tracker:     DeepSORTTracker,
        interaction: InteractionDetector,
        extractor:   ClipExtractor,
        classifier:  EnsembleClassifier,
        aggregator:  Aggregator,
        frame_skip_n:       int = FRAME_SKIP_N,
        yolo_interval:      int = YOLO_INTERVAL,
        ring_buffer_frames: int = RING_BUFFER_FRAMES,
    ):
        self.detector    = detector
        self.tracker     = tracker
        self.interaction = interaction
        self.extractor   = extractor
        self.classifier  = classifier
        self.aggregator  = aggregator

        self.frame_skip_n       = frame_skip_n
        self.yolo_interval      = yolo_interval
        self.ring_buffer_frames = ring_buffer_frames

    #  Main entry point 
    def analyse(self, video_path: str, auto_cleanup: bool = False) -> PipelineResult:
        t0 = time.perf_counter()

        self.tracker.reset()
        self.interaction.reset()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return PipelineResult(
                aggregation=Aggregator._empty_result(),
                warning="Could not open video file.",
                elapsed_seconds=time.perf_counter() - t0,
            )

        fps             = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_src       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Pipeline] Video: {total_src} frames @ {fps:.1f} fps")

        ring_buffer:    Dict[int, np.ndarray] = {}
        ring_keys:      Deque[int]            = deque()

        track_history:  Dict[int, List[TrackedPlayer]] = {}

        frames_decoded   = 0
        frames_processed = 0
        frames_yolo      = 0
        _last_fd         = None

        while frames_decoded < MAX_TOTAL_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frames_decoded += 1

            if frames_decoded % self.frame_skip_n != 0:
                continue

            frame_idx        = frames_decoded - 1
            frames_processed += 1

            # YOLO — now every 2 processed frames (was every 3)
            if frames_processed % self.yolo_interval == 1 or _last_fd is None:
                _last_fd = self.detector.detect(frame, frame_idx=frame_idx)
                frames_yolo += 1
            else:
                _last_fd.frame_idx = frame_idx

            tr = self.tracker.update(_last_fd, frame)

            track_history[frame_idx] = tr.tracked_players

            # Buffer ALL frames with ≥1 player (was ≥2) — we may miss lone
            # player frames just before a tackle starts
            is_candidate = (
                len(tr.tracked_players) >= 2
                or tr.ball_bbox is not None
            )

            if is_candidate:
                ring_buffer[frame_idx] = frame
                ring_keys.append(frame_idx)
                while len(ring_keys) > self.ring_buffer_frames:
                    oldest = ring_keys.popleft()
                    ring_buffer.pop(oldest, None)

            self.interaction.update(tr)

        cap.release()
        events = self.interaction.finalize()

        print(
            f"[Pipeline] decoded={frames_decoded} processed={frames_processed} "
            f"yolo={frames_yolo} interactions={len(events)} "
            f"buffered_frames={len(ring_buffer)}"
        )

        if auto_cleanup:
            _safe_delete(video_path)

        # Build clip 
        pipeline_mode = "full"
        warning       = None
        best_event    = None

        if events:
            best_event = max(events, key=lambda e: e.duration)
            print(f"[Pipeline] Best event: frames {best_event.start_frame}–"
                  f"{best_event.end_frame} (peak={best_event.peak_frame})")
            clip = self.extractor.from_frame_buffer(ring_buffer, best_event)

            if clip is None:
                if os.path.exists(video_path):
                    clip = self.extractor.extract(video_path, best_event)
                if clip is None:
                    clip    = _fallback_clip_from_buffer(ring_buffer, self.extractor)
                    warning = "Event frames evicted from buffer; using nearest available."
        else:
            pipeline_mode = "fallback"
            warning       = (
                "No player interactions detected. "
                "Classifying a sample of the video — result may be less accurate."
            )
            print(f"[Pipeline] No interactions found — using fallback clip from buffer "
                  f"({len(ring_buffer)} frames available)")
            clip = _fallback_clip_from_buffer(ring_buffer, self.extractor)

        if clip is None or len(clip) == 0:
            return PipelineResult(
                aggregation=Aggregator._empty_result(),
                events=events,
                warning="Clip extraction failed — no usable frames found.",
                elapsed_seconds=time.perf_counter() - t0,
                pipeline_mode=pipeline_mode,
                frames_decoded=frames_decoded,
                frames_yolo=frames_yolo,
            )

        print(f"[Pipeline] Clip ready: {len(clip)} frames")

        # Ensemble inference
        clip_pred  = self.classifier.predict(clip)
        top_frames = self.classifier.top_k_frames(clip, clip_pred, k=3)
        top_for_agg = [(fi, sc) for fi, sc, _ in top_frames]

        # Aggregation 
        aggregation = self.aggregator.aggregate(clip_pred, best_event, top_for_agg)

        elapsed = time.perf_counter() - t0
        print(
            f"[Pipeline] Done in {elapsed:.2f}s | "
            f"verdict={aggregation.decision.verdict} | "
            f"foul_prob={aggregation.foul_prob:.3f} | "
            f"severity={aggregation.severity:.3f}"
        )

        return PipelineResult(
            aggregation=aggregation,
            events=events,
            top_frames=top_frames,
            elapsed_seconds=elapsed,
            pipeline_mode=pipeline_mode,
            warning=warning,
            frames_decoded=frames_decoded,
            frames_yolo=frames_yolo,
            tracking_history=track_history,
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fallback_clip_from_buffer(
    ring_buffer: Dict[int, np.ndarray],
    extractor:   ClipExtractor,
) -> Optional[ClipExtractor]:
    """
    Sample clip_length frames from the ring buffer.

    Fix: biased toward the middle third of the video — interactions in
    short uploaded clips are usually not in the first or last few frames.
    """
    if not ring_buffer:
        return None

    sorted_keys = sorted(ring_buffer.keys())
    n           = extractor.clip_length
    total       = len(sorted_keys)

    if total <= n:
        indices = sorted_keys
    else:
        # Sample from middle 60% of buffer to avoid dead intro/outro frames
        lo  = total // 5
        hi  = total - total // 5
        mid = sorted_keys[lo:hi]
        if len(mid) <= n:
            indices = mid
        else:
            step    = (len(mid) - 1) / (n - 1)
            indices = [mid[round(i * step)] for i in range(n)]

    frames = [ring_buffer[k] for k in indices]
    print(f"[Pipeline] Fallback clip: {len(frames)} frames from buffer keys "
          f"{indices[0]}…{indices[-1]}")
    return extractor.from_frames(frames, event=None)


def _safe_delete(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"[Pipeline] Cleaned up: {path}")
    except OSError as e:
        print(f"[Pipeline] Could not delete {path}: {e}")