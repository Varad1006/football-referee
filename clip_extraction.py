"""
clip_extraction.py  (shape-safe, optimised)
────────────────────────────────────────────

Shape contract
──────────────
  raw_frames   : uint8   (128, 128, 3)  BGR  — display only, never fed to a model
  rgb_frames   : float32 (128, 128, 3)  [0,1]  — THUMBNAIL for ring-buffer storage
  gray_frames  : float32 (128, 128, 1)  [0,1]  — THUMBNAIL for ring-buffer storage

  ┌─ WHY thumbnails? ────────────────────────────────────────────────────────┐
  │  Storing 128-px frames keeps RAM low while the pipeline streams.         │
  │  Models require 256×256 — upscaling is done in inference.py, right       │
  │  before model.predict(), so no unnecessary large arrays live here.       │
  └──────────────────────────────────────────────────────────────────────────┘

All other public APIs are unchanged from the previous optimised version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from interaction import InteractionEvent


# ── Sizes ─────────────────────────────────────────────────────────────────────
# Storage size — small, kept in ring buffer during streaming
STORAGE_FRAME_SIZE: Tuple[int, int] = (128, 128)   # (W, H)
DEFAULT_CLIP_LENGTH: int = 8


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class Clip:
    """
    Preprocessed temporal clip.

    Stored at 128×128 to keep ring-buffer RAM minimal.
    inference.py upscales to 256×256 right before model.predict().

    Memory (8 frames @ 128×128):
      rgb_frames  : ~1.6 MB
      gray_frames : ~0.5 MB
      raw_frames  : ~0.4 MB
      Total       : ~2.5 MB  (vs ~40 MB at 256 × 16 frames)
    """
    rgb_frames:  List[np.ndarray]   # float32 (128, 128, 3)  [0, 1]
    gray_frames: List[np.ndarray]   # float32 (128, 128, 1)  [0, 1]
    raw_frames:  List[np.ndarray]   # uint8   (128, 128, 3)  BGR
    start_frame: int
    end_frame:   int
    event:       Optional[InteractionEvent] = None

    def __len__(self) -> int:
        return len(self.rgb_frames)


# ── Extractor ─────────────────────────────────────────────────────────────────
class ClipExtractor:
    """
    Builds Clip objects from:
      (a) the pipeline's in-memory ring-buffer  ← no disk I/O
      (b) a seek into the source video file     ← fallback
    """

    def __init__(
        self,
        clip_length: int             = DEFAULT_CLIP_LENGTH,
        frame_size:  Tuple[int, int] = STORAGE_FRAME_SIZE,
    ):
        self.clip_length = clip_length
        self.frame_size  = frame_size   # (W, H) — storage size only

    # ── Primary: from ring buffer ─────────────────────────────────────────────
    def from_frame_buffer(
        self,
        buffer: Dict[int, np.ndarray],
        event:  Optional[InteractionEvent] = None,
    ) -> Optional["Clip"]:
        if not buffer:
            return None

        sorted_keys = sorted(buffer.keys())

        centre = event.peak_frame if event is not None else sorted_keys[len(sorted_keys) // 2]
        half   = self.clip_length // 2
        want   = list(range(centre - half, centre - half + self.clip_length))

        selected: List[np.ndarray] = []
        for idx in want:
            if idx in buffer:
                selected.append(buffer[idx])
            elif sorted_keys:
                nearest = min(sorted_keys, key=lambda k: abs(k - idx))
                selected.append(buffer[nearest])

        if not selected:
            return None

        return self._build_clip(selected, want[0], want[-1], event)

    # ── Fallback: disk seek ───────────────────────────────────────────────────
    def extract(
        self,
        video_path: str,
        event:      InteractionEvent,
    ) -> Optional["Clip"]:
        centre    = event.peak_frame
        half      = self.clip_length // 2
        start_idx = max(0, centre - half)
        end_idx   = start_idx + self.clip_length - 1
        return self._extract_range(video_path, start_idx, end_idx, event)

    # ── From a raw frame list (fallback / test path) ──────────────────────────
    def from_frames(
        self,
        frames: List[np.ndarray],
        event:  Optional[InteractionEvent] = None,
    ) -> "Clip":
        if len(frames) > self.clip_length:
            indices = np.linspace(0, len(frames) - 1, self.clip_length, dtype=int)
            frames  = [frames[i] for i in indices]
        return self._build_clip(frames, 0, max(0, len(frames) - 1), event)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _extract_range(
        self,
        video_path:  str,
        start_frame: int,
        end_frame:   int,
        event:       Optional[InteractionEvent],
    ) -> Optional["Clip"]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

        raw_bgr: List[np.ndarray] = []
        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            raw_bgr.append(frame)
        cap.release()

        if not raw_bgr:
            return None

        return self._build_clip(raw_bgr, start_frame, end_frame, event)

    def _build_clip(
        self,
        raw_bgr_frames: List[np.ndarray],
        start_frame:    int,
        end_frame:      int,
        event:          Optional[InteractionEvent],
    ) -> "Clip":
        rgb_frames:  List[np.ndarray] = []
        gray_frames: List[np.ndarray] = []
        raw_out:     List[np.ndarray] = []

        for frame in raw_bgr_frames:
            rgb, gray, raw = self._preprocess(frame)
            rgb_frames.append(rgb)
            gray_frames.append(gray)
            raw_out.append(raw)

        # Pad short clips by repeating the last frame
        while len(rgb_frames) < self.clip_length:
            rgb_frames.append(rgb_frames[-1].copy())
            gray_frames.append(gray_frames[-1].copy())
            raw_out.append(raw_out[-1].copy())

        return Clip(
            rgb_frames=rgb_frames,
            gray_frames=gray_frames,
            raw_frames=raw_out,
            start_frame=start_frame,
            end_frame=end_frame,
            event=event,
        )

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize to STORAGE_FRAME_SIZE (128×128) and normalise.
        These are STORAGE thumbnails — inference.py upscales to 256×256.
        """
        W, H    = self.frame_size
        resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        raw     = resized.copy()   # uint8 BGR — kept for display

        rgb_norm  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_norm *= (1.0 / 255.0)

        gray_1ch  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_1ch *= (1.0 / 255.0)
        gray_norm = gray_1ch[:, :, np.newaxis]   # (H, W, 1)

        return rgb_norm, gray_norm, raw