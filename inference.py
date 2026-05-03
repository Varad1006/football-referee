from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from tensorflow import keras
except ImportError as e:
    raise ImportError("TensorFlow not installed. Run: pip install tensorflow") from e

from clip_extraction import Clip

# ── Model input spec ──────────────────────────────────────────────────────────
MODEL_INPUT_H   = 256
MODEL_INPUT_W   = 256
MODEL_INPUT_RGB  = (MODEL_INPUT_H, MODEL_INPUT_W, 3)
MODEL_INPUT_GRAY = (MODEL_INPUT_H, MODEL_INPUT_W, 1)

DEBUG_SHAPES = True

# ── Ensemble weights ─────────────────────────────────────────────────────────
RGB_WEIGHT  = 0.70
GRAY_WEIGHT = 0.30

# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class FramePrediction:
    frame_idx:     int
    rgb_pred:      float
    gray_pred:     float
    ensemble_pred: float

    @classmethod
    def combine(cls, frame_idx: int, rgb: float, gray: float) -> "FramePrediction":
        ensemble = RGB_WEIGHT * rgb + GRAY_WEIGHT * gray
        return cls(frame_idx=frame_idx, rgb_pred=rgb, gray_pred=gray, ensemble_pred=ensemble)

@dataclass
class ClipPrediction:
    frame_predictions: List[FramePrediction] = field(default_factory=list)

    @property
    def logs(self) -> List[float]:
        return [fp.ensemble_pred for fp in self.frame_predictions]

    @property
    def rgb_logs(self) -> List[float]:
        return [fp.rgb_pred for fp in self.frame_predictions]

    @property
    def gray_logs(self) -> List[float]:
        return [fp.gray_pred for fp in self.frame_predictions]

# ── Classifier ────────────────────────────────────────────────────────────────
class EnsembleClassifier:

    def __init__(self, model_rgb: keras.Model, model_gray: keras.Model):
        self.model_rgb  = model_rgb
        self.model_gray = model_gray

        self._rgb_hw  = _model_input_hw(model_rgb,  default=(MODEL_INPUT_H, MODEL_INPUT_W))
        self._gray_hw = _model_input_hw(model_gray, default=(MODEL_INPUT_H, MODEL_INPUT_W))

        print(
            f"[Inference] model_1 input: {self._rgb_hw[0]}×{self._rgb_hw[1]}×3  "
            f"| model_2 input: {self._gray_hw[0]}×{self._gray_hw[1]}×1"
        )

    @classmethod
    def load(cls, model_dir: str = ".") -> "EnsembleClassifier":
        base      = Path(model_dir)
        rgb_path  = base / "model_1.h5"
        gray_path = base / "model_2.h5"

        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB model not found: {rgb_path}")
        if not gray_path.exists():
            raise FileNotFoundError(f"Grayscale model not found: {gray_path}")

        print(f"[Inference] Loading RGB model:       {rgb_path}")
        model_rgb  = keras.models.load_model(str(rgb_path),  compile=False)

        print(f"[Inference] Loading grayscale model: {gray_path}")
        model_gray = keras.models.load_model(str(gray_path), compile=False)

        return cls(model_rgb, model_gray)

    def predict(self, clip: Clip) -> ClipPrediction:
        n_frames = len(clip.rgb_frames)
        if n_frames == 0:
            return ClipPrediction()

        # ── 1. Stack storage-size frames ─────────────────────────────────────
        rgb_small  = np.stack(clip.rgb_frames,  axis=0)   
        gray_small = np.stack(clip.gray_frames, axis=0)   

        # ── 2. Upscale to model input size ────────────────────────────────────
        rgb_batch  = _upscale_batch(rgb_small,  target_hw=self._rgb_hw,  channels=3)
        gray_batch = _upscale_batch(gray_small, target_hw=self._gray_hw, channels=1)

        # ── 3. Shape assertions + debug print ────────────────────────────────
        if DEBUG_SHAPES:
            expected_rgb  = (n_frames,) + MODEL_INPUT_RGB
            expected_gray = (n_frames,) + MODEL_INPUT_GRAY

            print(f"[Inference] rgb_batch  shape: {rgb_batch.shape}  | expected: {expected_rgb}")
            print(f"[Inference] gray_batch shape: {gray_batch.shape} | expected: {expected_gray}")

            assert rgb_batch.shape[1:]  == MODEL_INPUT_RGB,  (
                f"RGB batch shape mismatch: got {rgb_batch.shape[1:]}, "
                f"expected {MODEL_INPUT_RGB}"
            )
            assert gray_batch.shape[1:] == MODEL_INPUT_GRAY, (
                f"Gray batch shape mismatch: got {gray_batch.shape[1:]}, "
                f"expected {MODEL_INPUT_GRAY}"
            )

        # ── 4. Model inference ────────────────────────────────────────────────
        rgb_preds  = self._predict_batch(self.model_rgb,  rgb_batch)
        gray_preds = self._predict_batch(self.model_gray, gray_batch)

        # ── 5. Per-frame ensemble ─────────────────────────────────────────────
        result = ClipPrediction()
        for i in range(n_frames):
            fp = FramePrediction.combine(
                frame_idx=clip.start_frame + i,
                rgb=float(rgb_preds[i]),
                gray=float(gray_preds[i]),
            )
            result.frame_predictions.append(fp)

        return result

    @staticmethod
    def _predict_batch(model: keras.Model, batch: np.ndarray) -> np.ndarray:
        raw = model.predict(batch, verbose=0)
        raw = np.array(raw, dtype=np.float32)

        if raw.ndim == 1:
            return raw                          
        if raw.shape[-1] == 1:
            return raw.reshape(-1)              
        if raw.shape[-1] == 2:
            return raw[:, 1]                    
        return raw.max(axis=-1)                 

    def top_k_frames(
        self,
        clip:      Clip,
        clip_pred: ClipPrediction,
        k:         int = 3,
    ) -> List[Tuple[int, float, np.ndarray]]:
        if not clip_pred.frame_predictions:
            return []

        scored = sorted(
            zip(range(len(clip)), clip_pred.logs, clip.raw_frames),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            (clip.start_frame + local_idx, score, frame)
            for local_idx, score, frame in scored[:k]
        ]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _upscale_batch(
    batch:     np.ndarray,
    target_hw: Tuple[int, int],
    channels:  int,
) -> np.ndarray:
    N            = batch.shape[0]
    H_out, W_out = target_hw

    out = np.empty((N, H_out, W_out, channels), dtype=np.float32)

    for i in range(N):
        frame = batch[i]   

        if channels == 1:
            frame_2d   = frame[:, :, 0]                                    
            resized_2d = cv2.resize(frame_2d, (W_out, H_out),
                                    interpolation=cv2.INTER_LINEAR)        
            out[i]     = resized_2d[:, :, np.newaxis]                      
        else:
            out[i] = cv2.resize(frame, (W_out, H_out),
                                interpolation=cv2.INTER_LINEAR)            

    return out

def _model_input_hw(
    model:   keras.Model,
    default: Tuple[int, int] = (MODEL_INPUT_H, MODEL_INPUT_W),
) -> Tuple[int, int]:
    try:
        shape = model.input_shape   
        h, w  = shape[1], shape[2]
        if h is not None and w is not None:
            return int(h), int(w)
    except Exception:
        pass
    return default