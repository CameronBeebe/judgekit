from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

from .api import Judge
from .dspy_module import JudgeModule, KeyResolver, ValueResolver


def make_dspy_metric(
    judge: Judge,
    *,
    description: Union[str, ValueResolver] = "Evaluate output quality",
    context_key: KeyResolver = "context",
    reference_key: KeyResolver = None,
    candidate_key: KeyResolver = "candidate",
    observe_name: Optional[str] = "judgekit.metric",
) -> Callable[[Any, Any], float]:
    """
    Wrap a Judge in a DSPy metric callable using JudgeModule under the hood.
    """
    module = JudgeModule(
        judge,
        description=description,
        context_key=context_key,
        reference_key=reference_key,
        candidate_key=candidate_key,
        observe_name=observe_name,
    )
    return module.as_metric()


def attach_metric_to_program(
    program: Any,
    metric: Callable[[Dict[str, Any], Any], float],
    *,
    raise_on_error: bool = False,
) -> bool:
    """
    Attach the provided metric to a DSPy program via `set_metric`.
    Returns True on success; optionally raises if the attachment fails.
    """
    setter = getattr(program, "set_metric", None)
    if not callable(setter):
        if raise_on_error:
            raise AttributeError("Program does not expose set_metric()")
        return False
    try:
        setter(metric)
        return True
    except Exception as exc:
        if raise_on_error:
            raise exc
        return False


