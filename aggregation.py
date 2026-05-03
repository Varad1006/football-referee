"""
aggregation.py
──────────────
Pure model-driven aggregation. DL model handles severity, 
YOLO tracking handles the ball touch logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from inference import ClipPrediction
from interaction import InteractionEvent

# ── Calibration ───────────────────────────────────────────────────────────────
CALIBRATION_FACTOR = 1.25    

# ── Hybrid aggregation weights ────────────────────────────────────────────────
PEAK_WEIGHT = 0.60
MEAN_WEIGHT = 0.40

# ── Decision thresholds ───────────────────────────────────────────────────────
THRESH_RED    = 0.62
THRESH_YELLOW = 0.45

SEVERITY_PERCENTILE = 95

# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class AnalysisMetric:
    label:    str
    sublabel: str
    value:    float   

@dataclass
class RefereeDecision:
    verdict:     str
    badge_class: str
    explanation: str
    icon:        str

@dataclass
class ExplainabilityOutput:
    top_frames: List[Tuple[int, float]]

@dataclass
class AggregationResult:
    foul_prob:      float
    severity:       float
    consistency:    float
    peak_frame_idx: int
    ball_won:       bool

    metrics:        List[AnalysisMetric]       = field(default_factory=list)
    decision:       Optional[RefereeDecision]  = None
    explainability: Optional[ExplainabilityOutput] = None
    logs:           List[float]                = field(default_factory=list)

    def metrics_as_tuples(self) -> List[Tuple[str, str, float]]:
        return [(m.label, m.sublabel, round(m.value, 1)) for m in self.metrics]


# ── Aggregator ────────────────────────────────────────────────────────────────
class Aggregator:

    def aggregate(
        self,
        clip_pred:    ClipPrediction,
        event:        Optional[InteractionEvent] = None,
        top_k_frames: Optional[List[Tuple[int, float]]] = None,
    ) -> AggregationResult:

        logs = clip_pred.logs
        if not logs:
            return self._empty_result()

        arr = np.array(logs, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)

        # ── 1. Hybrid aggregation (peak-sensitive) ────────────────────────────
        raw_max  = float(arr.max())
        raw_mean = float(arr.mean())
        foul_prob_raw = PEAK_WEIGHT * raw_max + MEAN_WEIGHT * raw_mean
        foul_prob = min(foul_prob_raw * 1.35, 1.0)

        # ── 2. Severity ───────────────────────────────────────────────────────
        severity_raw = float(np.percentile(arr, SEVERITY_PERCENTILE))
        severity     = min(severity_raw * CALIBRATION_FACTOR, 1.0)

        # ── 3. Consistency & Peak Frame ───────────────────────────────────────
        std       = float(arr.std())
        norm_std  = min(std / 0.5, 1.0)
        consistency = 1.0 - norm_std

        peak_local = int(arr.argmax())
        peak_idx   = (clip_pred.frame_predictions[peak_local].frame_idx
                      if clip_pred.frame_predictions else 0)

        # ── 4. YOLO Object Detection Override ─────────────────────────────────
        ball_won_cleanly = event.ball_won_cleanly if event else False
        
        if ball_won_cleanly:
            print("[Aggregation] YOLO detected ball touch before peak collision. Overriding foul probability.")
            # Hardcap the foul probability so it results in "No Card" / "No Foul"
            foul_prob = min(foul_prob, 0.35) 
            severity = min(severity, 0.40) # Mute the severity UI slightly as it was a clean tackle

        # ── 5. UI metrics (Strictly Model-Driven) ─────────────────────────────
        metrics = self._build_metrics(foul_prob, severity, ball_won_cleanly)

        # ── 6. Decision ───────────────────────────────────────────────────────
        decision = _make_decision(foul_prob, ball_won_cleanly)

        expl = ExplainabilityOutput(top_frames=top_k_frames or [])

        return AggregationResult(
            foul_prob=round(foul_prob, 4),
            severity=round(severity, 4),
            consistency=round(consistency, 4),
            peak_frame_idx=peak_idx,
            ball_won=ball_won_cleanly,
            metrics=metrics,
            decision=decision,
            explainability=expl,
            logs=logs,
        )

    # ── Metric builder (Fake heuristics removed) ──────────────────────────────
    @staticmethod
    def _build_metrics(
        foul_prob: float,
        severity:  float,
        ball_won:  bool,
    ) -> List[AnalysisMetric]:
        fp_pct  = foul_prob * 100.0
        sev_pct = severity  * 100.0

        action_label = "Clean Tackle" if ball_won else ("Foul Challenge" if fp_pct > 45 else "Fair Contest")
        ball_label   = "Yes" if ball_won else "No / Missed"
        ball_val     = 100.0 if ball_won else 0.0

        return [
            AnalysisMetric("Action Class", action_label, round(fp_pct, 1)),
            AnalysisMetric("Collision Severity", "Physical Impact", round(sev_pct, 1)),
            AnalysisMetric("Ball Touched First", ball_label, round(ball_val, 1)),
        ]

    @staticmethod
    def _empty_result() -> AggregationResult:
        return AggregationResult(
            foul_prob=0.0, severity=0.0, consistency=1.0,
            peak_frame_idx=0, ball_won=False, metrics=[],
            decision=RefereeDecision(
                verdict="No Card", badge_class="badge-green",
                explanation="No interaction detected in this clip.", icon="👍",
            ),
            logs=[],
        )

# ── Decision logic ─────────────────────────────────────────────────────────────
def _make_decision(foul_prob: float, ball_won_cleanly: bool) -> RefereeDecision:
    if ball_won_cleanly:
        return RefereeDecision(
            verdict="No Foul",
            badge_class="badge-green",
            explanation=(
                "YOLO object tracking verified that the ball was cleanly played "
                "before the peak of the collision. No disciplinary action required."
            ),
            icon="✅",
        )
    elif foul_prob > THRESH_RED:
        return RefereeDecision(
            verdict="Red Card",
            badge_class="badge-red",
            explanation=(
                "Serious foul play detected with high confidence and no clean play "
                "on the ball. The challenge warrants a red card under FIFA Law 12."
            ),
            icon="🟥",
        )
    elif foul_prob > THRESH_YELLOW:
        return RefereeDecision(
            verdict="Yellow Card",
            badge_class="badge-yellow",
            explanation=(
                "Unsporting behaviour or a reckless challenge detected. The contact "
                "was careless without winning the ball, warranting a caution."
            ),
            icon="🟨",
        )
    else:
        return RefereeDecision(
            verdict="No Card",
            badge_class="badge-green",
            explanation=(
                "Interaction detected, but the action lacked significant "
                "aggression or recklessness to warrant a foul."
            ),
            icon="👍",
        )