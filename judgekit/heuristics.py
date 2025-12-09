from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .api import Criteria, JudgeResult


_CONNECTIVES = re.compile(r"\b(therefore|thus|hence|because|however|moreover|furthermore|although)\b", re.I)
_URL_OR_CITATION = re.compile(r"(https?://|\[\d+\]|\(\d{4}\))")
_COUNTERPOINT = re.compile(r"\b(on the other hand|alternatively|counter|oppos|critics?|skeptic)\b", re.I)
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class HeuristicJudge:
    """
    Lightweight, deterministic heuristic evaluation aligned with core dimensions.
    Signals are simple and fast; this is a fallback and for dry runs.
    """

    def evaluate(self, candidate: str, criteria: Optional[Criteria] = None, reference: str = "") -> JudgeResult:
        text = candidate.strip()
        length = len(text)
        if length == 0:
            return JudgeResult(rationale="Empty text.", score=0.0, passed=False, details={"heuristic": True})

        # Dimension scores in [0,1]
        reasoning = _clip(min(1.0, 0.2 + 0.1 * len(_CONNECTIVES.findall(text))))
        rigor = _clip(0.2 * len(_URL_OR_CITATION.findall(text)) + 0.1 * ("uncertain" in text.lower()))
        balance = _clip(0.25 * len(_COUNTERPOINT.findall(text)))
        specificity = _clip(0.1 * len(_NUMBER.findall(text)) + 0.05 * len(_PROPER_NOUN.findall(text)))
        # Clarity via sentence length distribution
        sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]
        avg_len = sum(len(s) for s in sentences) / max(1, len(sentences))
        clarity = _clip(1.0 - abs(avg_len - 120.0) / 200.0)

        # Length sufficiency penalty
        length_factor = _clip(math.log10(max(10, length)) / 4.0)  # ~0.5 at ~100 chars, ~0.75 at 1000

        # Aggregate
        dims: Dict[str, float] = {
            "reasoning": reasoning,
            "rigor": rigor,
            "balance": balance,
            "specificity": specificity,
            "clarity": clarity,
        }
        weights = None
        if criteria is not None and criteria.weights:
            weights = criteria.weights
        else:
            weights = {k: 1.0 / len(dims) for k in dims}
        weighted = sum(dims.get(k, 0.0) * weights.get(k, 0.0) for k in dims)
        score = _clip(0.15 + 0.7 * weighted + 0.15 * length_factor)

        rationale = "Heuristic score based on simple signals for reasoning, rigor, balance, specificity, and clarity."
        subscores = {k: round(float(v), 3) for k, v in dims.items()}
        passed = score >= (criteria.threshold if hasattr(criteria, "threshold") else 0.6)  # type: ignore[attr-defined]

        return JudgeResult(
            rationale=rationale,
            score=round(float(score), 3),
            passed=bool(passed),
            subscores=subscores,
            details={"heuristic": True},
        )


