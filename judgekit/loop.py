from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from .api import Criteria, Judge
from .builders import create_judge, refine_criteria
from .xai_helpers import call_xai_with_retries


def _generate_candidate(
    agent_prompt: str,
    context: str,
    feedback: str,
    *,
    verbose: bool,
    temperature: float = 0.6,
    max_tokens: int = 400,
) -> str:
    system = agent_prompt or "You are an expert agent. Produce the best possible response."
    user = (
        "Context:\n"
        f"{context or 'N/A'}\n\n"
        "Feedback to address (improve upon this):\n"
        f"{feedback or 'Start from scratch and provide a strong candidate output.'}\n\n"
        "Produce an improved candidate output that satisfies the context and addresses all feedback.\n"
        "- DO NOT mention the evaluation rubric, scoring process, or that you are improving a prior answer.\n"
        "- DO NOT include meta commentary such as “Below is my improved...” or references to instructions.\n"
        "- Output only the final user-facing content (pitch/plan/etc.)."
    )
    out = call_xai_with_retries(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )
    return out.get("text", "").strip()


def _run_meta_judge(
    judge: Judge,
    *,
    meta_phrase: str,
    model: Optional[str],
    base_url: Optional[str],
    verbose: bool,
    label: str,
):
    try:
        meta_judge = create_judge(meta_phrase, model=model, base_url=base_url, verbose=verbose)
        res = meta_judge.evaluate(
            candidate=judge.default_criteria.template,
            description="Evaluate rubric quality.",
            context="",
        )
        print(f"[loop] Meta[{label}] Score={res.score:.3f} Passed={res.passed}")
        print(f"[loop] Meta[{label}] Rationale: {res.rationale}")
        return res
    except Exception as err:  # pragma: no cover
        print(f"[loop] Meta[{label}] failed: {err}")
        return None


def _print_iteration_header():
    print(f"{'Iter':>4} | {'Score':>6} | {'Result':>6} | Candidate Preview")
    print("-" * 80)


def _print_iteration_row(iteration: int, score: float, passed: bool, candidate: str):
    preview = candidate.replace("\n", " ")
    if len(preview) > 60:
        preview = preview[:57] + "..."
    result = "PASS" if passed else "FAIL"
    print(f"{iteration:4d} | {score:6.3f} | {result:>6} | {preview}")


def run_loop(
    *,
    criteria: str | Criteria,
    agent_prompt: str,
    context: str,
    max_iterations: int,
    recursion_limit: int,
    pass_threshold: float,
    model: Optional[str],
    base_url: Optional[str],
    verbose: bool,
    log_jsonl: Optional[str] = None,
) -> None:
    judge = create_judge(criteria, model=model, base_url=base_url, verbose=verbose)
    feedback = ""
    passes_completed = 0
    meta_phrase = f"Rubric quality for: {judge.default_criteria.name}"

    if log_jsonl:
        with open(log_jsonl, "w", encoding="utf-8") as sink:
            sink.write("")

    _print_iteration_header()
    for iteration in range(1, max_iterations + 1):
        candidate = _generate_candidate(agent_prompt, context, feedback, verbose=verbose)
        result = judge.evaluate(candidate=candidate, description="Evaluate the candidate quality.", context=context)
        passed = result.passed or (result.score >= pass_threshold)
        _print_iteration_row(iteration, result.score, passed, candidate)
        if log_jsonl:
            with open(log_jsonl, "a", encoding="utf-8") as sink:
                sink.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "candidate": candidate,
                            "score": result.score,
                            "passed": passed,
                            "rationale": result.rationale,
                            "criteria_version": judge.default_criteria.version,
                        }
                    )
                    + "\n"
                )

        if passed:
            passes_completed += 1
            print(f"[loop] Candidate passed ({passes_completed}/{recursion_limit}).")
            if passes_completed >= recursion_limit:
                print("[loop] Success: recursion limit satisfied.")
                break
            meta_res = _run_meta_judge(
                judge,
                meta_phrase=meta_phrase,
                model=model,
                base_url=base_url,
                verbose=verbose,
                label=f"{passes_completed}",
            )
            feedback = meta_res.rationale if meta_res else "Tighten the rubric and require higher quality."
            judge = create_judge(
                refine_criteria(judge.default_criteria, feedback, phrase=judge.default_criteria.name, verbose=verbose),
                model=model,
                base_url=base_url,
                verbose=verbose,
            )
        else:
            feedback = result.rationale
    else:
        print("[loop] Reached max iterations without satisfying recursion limit.")


