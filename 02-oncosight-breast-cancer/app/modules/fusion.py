from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionResult:
    score: float | None
    category: str | None
    effective_weights: dict[str, float]


def redistribute_weights(scores: dict[str, float | None], weights: dict[str, float]) -> dict[str, float]:
    available = {k: v for k, v in weights.items() if scores.get(k) is not None}
    total = sum(available.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in available.items()}


def categorize(score: float) -> str:
    if score < 0.3:
        return "Low"
    if score <= 0.6:
        return "Moderate"
    return "High"


def weighted_late_fusion(
    mammography_prob: float | None,
    seer_risk: float | None,
    pcam_prob: float | None,
    weights: dict[str, float] | None = None,
) -> FusionResult:
    weights = weights or {"mammography": 0.50, "seer": 0.30, "pcam": 0.20}
    scores = {"mammography": mammography_prob, "seer": seer_risk, "pcam": pcam_prob}
    effective = redistribute_weights(scores, weights)
    if not effective:
        return FusionResult(score=None, category=None, effective_weights={})
    score = sum(float(scores[k]) * w for k, w in effective.items() if scores[k] is not None)
    return FusionResult(score=score, category=categorize(score), effective_weights=effective)
