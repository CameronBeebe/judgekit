from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union, List

from .api import Criteria, Judge
from .heuristics import HeuristicJudge
from .adapters import OpenAICompatAdapter
from .registry import default_registry, CriteriaRegistry
from .utils import console_hooks, init_laminar_from_env
from .xai_helpers import call_xai_with_retries


def generate_criteria_from_phrase(phrase: str, *, verbose: bool = False) -> Criteria:
    """
    Use xAI to transform a short phrase into a Criteria with 3-6 dimensions and a concise rubric template.
    Returns a Criteria with equal weights across generated dimensions.
    """
    system = (
        "You are an expert evaluator designer. Return STRICT JSON only. "
        "The resulting criteria must explicitly penalize any candidate that references the rubric, scoring, "
        "or meta instructions (e.g., phrases like 'Below is my improved output')."
    )
    user = (
        "Design a concise evaluation rubric for the following target, with 3-6 dimension names and a short template text.\n"
        f"Target: {phrase}\n\n"
        "Return JSON with keys: dimensions (list[str]), template (str)."
    )
    if verbose:
        print(f"[builders] Generating criteria from phrase: {phrase!r}")
    out = call_xai_with_retries(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=300,
        verbose=verbose,
    )
    text = out.get("text", "{}")
    try:
        obj = json.loads(text)
        dims = obj.get("dimensions") or []
        template = obj.get("template") or f"Evaluate with respect to: {phrase}. Score in [0,1]."
        if not isinstance(dims, list) or not dims:
            dims = ["task_success", "clarity", "specificity"]
        dims = [str(d) for d in dims][:6]
        weights = {d: 1.0 / len(dims) for d in dims}
        name = phrase.strip().replace(" ", "")
        crit = Criteria(name=name, dimensions=dims, weights=weights, template=template, version="1")
        if verbose:
            print(f"[builders] Generated dimensions={dims} weights={weights}")
        return crit
    except Exception:
        dims = ["task_success", "clarity", "specificity"]
        weights = {d: 1.0 / len(dims) for d in dims}
        template = f"Evaluate with respect to: {phrase}. Score in [0,1]."
        name = phrase.strip().replace(" ", "")
        crit = Criteria(name=name, dimensions=dims, weights=weights, template=template, version="1")
        if verbose:
            print("[builders] Falling back to default dimensions for generated criteria.")
        return crit


def create_judge(
    criteria: Union[Criteria, str],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    threshold: float = 0.6,
    verbose: bool = True,
    registry: Optional[CriteriaRegistry] = None,
) -> Judge:
    """
    Create a Judge from a Criteria object or a name/phrase.
    - If `criteria` is a string: lookup preset in registry, else generate via LLM from the phrase.
    - Uses xAI adapter with XAI_API_KEY and base URL (default https://api.x.ai/v1).
    """
    reg = registry or default_registry()
    crit: Criteria
    if isinstance(criteria, Criteria):
        crit = criteria
    else:
        found = reg.get(criteria)
        if found is not None:
            crit = found
            if verbose:
                print(f"[builders] Using preset criteria: {criteria!r}")
        else:
            crit = generate_criteria_from_phrase(criteria, verbose=verbose)

    # Initialize Laminar if available
    init_laminar_from_env()
    adapter = OpenAICompatAdapter(model=model or None or "grok-3-mini", api_key=None, base_url=base_url)
    hooks = console_hooks(verbose=verbose)
    return Judge(llm_callable=adapter, default_criteria=crit, threshold=threshold, hooks=hooks, heuristic_fallback=HeuristicJudge())

def refine_criteria(
    criteria: Criteria,
    feedback: str,
    *,
    phrase: Optional[str] = None,
    verbose: bool = False,
) -> Criteria:
    """
    Refine an existing criteria using feedback (e.g., from a meta-judge).
    """
    label = phrase or criteria.name
    if verbose:
        print(f"[builders] Refining criteria for {label!r}")
    system = (
        "You are an expert evaluation designer. Return STRICT JSON only. "
        "Ensure the refined criteria explicitly penalize any candidate referencing the rubric, scoring, or meta commentary."
    )
    user = (
        "Improve the following evaluation rubric based on the feedback.\n\n"
        f"Current dimensions: {criteria.dimensions}\n"
        f"Current template:\n{criteria.template}\n\n"
        f"Feedback:\n{feedback or 'Make it more precise and balanced.'}\n\n"
        "Return JSON with keys: dimensions (list[str]), template (str). Limit to 6 concise dimensions."
    )
    out = call_xai_with_retries(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=400,
        verbose=verbose,
    )
    text = out.get("text", "{}")
    try:
        obj = json.loads(text)
        dims = obj.get("dimensions") or criteria.dimensions
        template = obj.get("template") or criteria.template
        dims = [str(d) for d in dims][:6]
        weights = {d: 1.0 / len(dims) for d in dims}
        version = f"{criteria.version}-refined"
        refined = Criteria(name=criteria.name, dimensions=dims, weights=weights, template=template, version=version)
        if verbose:
            print(f"[builders] Refined dimensions={dims}")
        return refined
    except Exception:
        if verbose:
            print("[builders] Refine failed; returning original criteria.")
        return criteria
