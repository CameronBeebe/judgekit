from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .api import Criteria, Judge
from .heuristics import HeuristicJudge
from .adapters import OpenAICompatAdapter

try:
    from lmnr import Laminar  # type: ignore
except Exception:
    Laminar = None  # type: ignore[assignment]


def console_hooks(verbose: bool = True) -> Dict[str, Any]:
    def on_prompt_built(prompt, meta):
        if not verbose:
            return
        print(f"[hook] prompt_built criteria={meta.get('criteria')} v={meta.get('criteria_version')}")
        dims = meta.get("dimensions")
        weights = meta.get("weights")
        if dims and weights:
            print(f"[hook] dims={dims}")
            print(f"[hook] weights={weights}")
        if meta.get("threshold") is not None:
            print(f"[hook] threshold={meta.get('threshold')}")

    def on_prompt_rendered(rendered_text: str, meta: dict) -> None:
        if not verbose:
            return
        print("[hook] prompt_rendered (truncated to 800 chars):")
        preview = rendered_text if len(rendered_text) <= 800 else rendered_text[:800] + " ... [truncated]"
        print(preview)

    def on_llm_start(prompt, meta):
        if not verbose:
            return
        print(f"[hook] llm_start criteria={prompt.criteria_name}")

    def on_llm_finish(prompt, raw_output, timings_ms, meta):
        if not verbose:
            return
        model_id = (raw_output or {}).get("model_id") if isinstance(raw_output, dict) else None
        print(f"[hook] llm_finish model={model_id} llm_ms={timings_ms}")

    def on_heuristic_finish(candidate, result, timings_ms, meta):
        if not verbose:
            return
        print(f"[hook] heuristic_finish score={result.score:.3f} ms={timings_ms}")

    def on_item_finish(item_id, result, meta):
        if not verbose:
            return
        print(f"[hook] item_finish score={result.score:.3f} passed={result.passed}")

    return {
        "on_prompt_built": on_prompt_built,
        "on_prompt_rendered": on_prompt_rendered,
        "on_llm_start": on_llm_start,
        "on_llm_finish": on_llm_finish,
        "on_heuristic_finish": on_heuristic_finish,
        "on_item_finish": on_item_finish,
    }


def init_laminar_from_env() -> bool:
    key = os.getenv("LMNR_API_KEY") or os.getenv("LMNR_PROJECT_API_KEY")
    if Laminar and key:
        try:
            Laminar.initialize(project_api_key=key)  # type: ignore[misc]
            return True
        except Exception:
            return False
    return False


def make_default_judge(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
) -> Judge:
    # xAI defaults
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")
    base = base_url or os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1"
    mdl = model or os.getenv("JUDGEKIT_MODEL") or "grok-3-mini"

    # Initialize Laminar if available
    lmnr_init = init_laminar_from_env()

    # Criteria
    dims = ["reasoning", "rigor", "balance", "specificity", "clarity"]
    weights = {d: 1.0 / len(dims) for d in dims}
    template = (
        "Evaluate the candidate on: reasoning/logic, epistemic rigor, balance/synthesis, "
        "specificity/informativeness, and clarity/organization. Score in [0,1]."
    )
    criteria = Criteria(name="BalancedEssay", dimensions=dims, weights=weights, template=template, version="1")

    # Adapter and judge
    adapter = OpenAICompatAdapter(model=mdl, api_key=api_key, base_url=base)
    hooks = console_hooks(verbose=verbose)
    judge = Judge(llm_callable=adapter, default_criteria=criteria, threshold=0.6, hooks=hooks, heuristic_fallback=HeuristicJudge())

    if verbose:
        print(f"[config] Verbose mode ON (toggle in script)")
        print(f"[env] Model={mdl} BaseURL={base}")
        print(f"[criteria] name={criteria.name} v={criteria.version}")
        print(f"[criteria] dims={criteria.dimensions}")
        print(f"[criteria] weights={criteria.weights}")
        print(f"[integrations] lmnr={'yes, initialized' if lmnr_init else 'no or not initialized'}")
    return judge


