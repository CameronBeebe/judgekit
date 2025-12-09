from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import dspy  # type: ignore

from .api import Judge, JudgeResult

try:
    from lmnr import observe  # type: ignore
except Exception:  # pragma: no cover
    observe = None  # type: ignore[assignment]

ValueResolver = Callable[[Any, Dict[str, Any]], Optional[str]]
KeyResolver = Union[str, ValueResolver, None]

_DEFAULT_DESCRIPTION = "Evaluate output quality"


class JudgeModule(dspy.Module):
    """
    Wrap a Judge in a DSPy module so verdicts become structured predictions.
    Integrates with Laminar via `lmnr.observe` when available.
    """

    def __init__(
        self,
        judge: Judge,
        *,
        description: Union[str, ValueResolver] = _DEFAULT_DESCRIPTION,
        context_key: KeyResolver = "context",
        reference_key: KeyResolver = None,
        candidate_key: KeyResolver = "candidate",
        default_llm_config: Optional[Dict[str, Any]] = None,
        observe_name: Optional[str] = "judgekit.judge_module",
    ) -> None:
        super().__init__()
        self._judge = judge
        self._description_resolver = description
        self._context_resolver = context_key
        self._reference_resolver = reference_key
        self._candidate_resolver = candidate_key
        self._llm_config = dict(default_llm_config or {})
        self._observe_name = observe_name

    def forward(  # type: ignore[override]
        self,
        candidate: Optional[str] = None,
        *,
        example: Any = None,
        description: Optional[str] = None,
        context: Optional[str] = None,
        reference: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        **fields: Any,
    ) -> dspy.Prediction:
        payload = dict(fields)
        candidate_text = self._resolve_candidate(candidate, example, payload)
        if not candidate_text:
            raise ValueError(
                "JudgeModule requires a candidate string; pass `candidate=` when calling "
                "or configure `candidate_key` / callable resolver."
            )

        resolved_description = self._resolve_description(example, payload, description)
        context_text = (
            context if context is not None else self._resolve_from_resolver(self._context_resolver, example, payload)
        )
        reference_text = (
            reference
            if reference is not None
            else self._resolve_from_resolver(self._reference_resolver, example, payload)
        )

        merged_llm_config = self._merge_llm_config(llm_config)

        def _run() -> JudgeResult:
            return self._judge.evaluate(
                candidate=candidate_text,
                description=resolved_description,
                context=context_text,
                reference=reference_text,
                llm_config=merged_llm_config,
            )

        result = self._call_with_observer(_run)

        return dspy.Prediction(
            score=result.score,
            passed=result.passed,
            rationale=result.rationale,
            subscores=result.subscores,
            details=result.details,
            judge_result=result,
        )

    def as_metric(self) -> Callable[[Any, Any], float]:
        """
        Expose the module as a DSPy metric callable: metric(example, pred) -> float.
        """

        def metric(example: Any, pred: Any) -> float:
            candidate_text = self._stringify_prediction(pred)
            verdict = self(example=example, candidate=candidate_text)
            return float(verdict.score)

        return metric

    def _resolve_description(
        self,
        example: Any,
        payload: Dict[str, Any],
        override: Optional[str],
    ) -> str:
        if override is not None:
            return str(override)
        resolver = self._description_resolver
        if isinstance(resolver, str):
            return resolver
        if callable(resolver):
            try:
                value = resolver(example, payload)
                if value is not None:
                    return str(value)
            except Exception:
                return _DEFAULT_DESCRIPTION
        return _DEFAULT_DESCRIPTION

    def _resolve_candidate(self, explicit: Optional[str], example: Any, payload: Dict[str, Any]) -> str:
        if explicit is not None:
            return str(explicit)
        value = self._resolve_from_resolver(self._candidate_resolver, example, payload)
        return value

    def _resolve_from_resolver(self, resolver: KeyResolver, example: Any, payload: Dict[str, Any]) -> str:
        if resolver is None:
            return ""
        if callable(resolver):
            try:
                value = resolver(example, payload)
                return "" if value is None else str(value)
            except Exception:
                return ""
        key = str(resolver)
        value = self._lookup_in_source(payload, key)
        if value is not None:
            return str(value)
        value = self._lookup_in_source(example, key)
        if value is not None:
            return str(value)
        return ""

    def _lookup_in_source(self, source: Any, key: str) -> Any:
        if source is None:
            return None
        if isinstance(source, dict) and key in source:
            return source[key]
        if hasattr(source, key):
            attr = getattr(source, key)
            if callable(attr):
                return None
            return attr
        getter = getattr(source, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                try:
                    return getter(key, None)
                except Exception:
                    return None
        return None

    def _merge_llm_config(self, override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self._llm_config and not override:
            return None
        merged: Dict[str, Any] = {}
        if self._llm_config:
            merged.update(self._llm_config)
        if override:
            merged.update(override)
        return merged

    def _call_with_observer(self, func: Callable[[], JudgeResult]) -> JudgeResult:
        if observe and self._observe_name:
            observed = observe(name=self._observe_name)(func)  # type: ignore[misc]
            return observed()
        return func()

    @staticmethod
    def _stringify_prediction(pred: Any) -> str:
        if isinstance(pred, str):
            return pred
        if isinstance(pred, dict):
            for key in ("text", "prediction", "output", "answer", "candidate"):
                value = pred.get(key)
                if value:
                    return str(value)
        for key in ("text", "prediction", "output", "answer", "candidate"):
            if hasattr(pred, key):
                value = getattr(pred, key)
                if value:
                    return str(value)
        return str(pred)


__all__ = ["JudgeModule", "KeyResolver", "ValueResolver"]

