import argparse
import sys
import time
from typing import Optional

from .builders import create_judge, refine_criteria
from .loop import run_loop


def _read_stdin() -> str:
    try:
        # Avoid blocking when running in an interactive terminal with no piped input
        if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
            return ""
        data = sys.stdin.read()
        return data.strip()
    except Exception:
        return ""


def _run_meta_judge(criteria_text: str, meta_phrase: str, args, iteration_label: str):
    if args.verbose:
        print(f"[cli] Creating meta-judge for {meta_phrase!r} ({iteration_label}) ...")
    try:
        meta_judge = create_judge(meta_phrase, model=args.model, base_url=args.base_url, verbose=args.verbose)
        meta_res = meta_judge.evaluate(
            candidate=criteria_text,
            description="Evaluate rubric quality.",
            context="",
        )
        print(f"[cli] Meta[{iteration_label}] Score={meta_res.score:.3f} Passed={meta_res.passed}")
        print(f"[cli] Meta[{iteration_label}] Rationale: {meta_res.rationale}")
        return meta_res
    except Exception as err:  # pragma: no cover (network dependent)
        print(f"[cli] Meta[{iteration_label}] failed: {err}")
        return None


def factory_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="JudgeKit factory CLI.")
    parser.add_argument("--criteria", "-c", type=str, required=True, help="Preset name or phrase to create a judge.")
    parser.add_argument("--candidate", "-x", type=str, default="", help="Candidate text to evaluate (or read from stdin).")
    parser.add_argument("--context", "-k", type=str, default="", help="Optional context string.")
    parser.add_argument("--description", "-d", type=str, default="Evaluate the candidate quality.", help="Description.")
    parser.add_argument("--recursive", "-r", type=int, default=1, help="Number of meta-judge iterations (>=0).")
    parser.add_argument("--refine", "-R", type=int, default=0, help="Number of rubric refinement steps (>=0).")
    parser.add_argument("--meta-criteria", type=str, default=None, help="Phrase used to build the meta-judge.")
    parser.add_argument("--model", type=str, default=None, help="Override model used for the LLM judge.")
    parser.add_argument("--base-url", type=str, default=None, help="Override base URL for the LLM judge.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose console hooks.")
    args = parser.parse_args(argv)
    args.recursive = max(0, args.recursive)
    args.refine = max(0, args.refine)

    if args.verbose:
        print(f"[cli] Creating judge for criteria={args.criteria!r} ...")
    judge = create_judge(args.criteria, model=args.model, base_url=args.base_url, verbose=args.verbose)
    if args.verbose:
        print("[cli] Judge ready.")

    candidate = args.candidate or _read_stdin()
    if not candidate:
        print("No candidate provided. Use --candidate or pipe input via stdin.", file=sys.stderr)
        return 2

    print(f"[cli] Evaluating with judge={args.criteria!r} ...")
    t0 = time.time()
    res = judge.evaluate(candidate=candidate, description=args.description, context=args.context)
    print(f"[cli] Score={res.score:.3f} Passed={res.passed} Took={(time.time()-t0)*1000:.0f}ms")
    print(f"[cli] Rationale: {res.rationale}")

    latest_meta = None
    meta_phrase = args.meta_criteria or f"Rubric quality for: {args.criteria}"
    if args.recursive > 0:
        for idx in range(1, args.recursive + 1):
            latest_meta = _run_meta_judge(
                judge.default_criteria.template,
                meta_phrase,
                args,
                f"{idx}/{args.recursive}",
            )

    if args.refine > 0:
        feedback = latest_meta.rationale if latest_meta else ""
        for ridx in range(1, args.refine + 1):
            refined_criteria = refine_criteria(
                judge.default_criteria,
                feedback,
                phrase=args.criteria,
                verbose=args.verbose,
            )
            judge = create_judge(refined_criteria, model=args.model, base_url=args.base_url, verbose=args.verbose)
            rres = judge.evaluate(candidate=candidate, description=args.description, context=args.context)
            print(f"[cli] Refine[{ridx}/{args.refine}] Score={rres.score:.3f} Passed={rres.passed}")
            print(f"[cli] Refine[{ridx}/{args.refine}] Rationale: {rres.rationale}")
            latest_meta = _run_meta_judge(
                judge.default_criteria.template,
                meta_phrase,
                args,
                f"ref-{ridx}",
            )
            feedback = latest_meta.rationale if latest_meta else feedback

    return 0


def eval_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="JudgeKit eval CLI (quick eval using default BalancedEssay).")
    parser.add_argument("--candidate", "-x", type=str, default="", help="Candidate text to evaluate (or read from stdin).")
    parser.add_argument("--context", "-k", type=str, default="", help="Optional context string.")
    parser.add_argument("--description", "-d", type=str, default="Evaluate the candidate quality.", help="Description.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose console hooks.")
    args = parser.parse_args(argv)

    judge = create_judge("BalancedEssay", verbose=args.verbose)
    candidate = args.candidate or _read_stdin()
    if not candidate:
        print("No candidate provided. Use --candidate or pipe input via stdin.", file=sys.stderr)
        return 2
    res = judge.evaluate(candidate=candidate, description=args.description, context=args.context)
    print(f"{res.score:.3f}\t{res.passed}\t{res.rationale}")
    return 0


def loop_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="JudgeKit loop CLI (auto agent + judge iterations).")
    parser.add_argument("--criteria", "-c", type=str, required=True, help="Preset name or phrase for initial judge.")
    parser.add_argument("--agent-prompt", "-p", type=str, default="You are a persuasive agent. Produce the best response.", help="System prompt for the agent generating candidates.")
    parser.add_argument("--context", "-k", type=str, default="", help="Context passed to both agent and judge.")
    parser.add_argument("--max-iters", "-n", type=int, default=10, help="Maximum candidate iterations.")
    parser.add_argument("--recursion-limit", "-r", type=int, default=1, help="Number of passes required (judge refinements).")
    parser.add_argument("--threshold", "-t", type=float, default=0.6, help="Score threshold treated as pass.")
    parser.add_argument("--log-jsonl", type=str, default=None, help="Optional path to write iteration logs (JSONL).")
    parser.add_argument("--model", type=str, default=None, help="Override model for judge creation.")
    parser.add_argument("--base-url", type=str, default=None, help="Override base URL for judge creation.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    run_loop(
        criteria=args.criteria,
        agent_prompt=args.agent_prompt,
        context=args.context,
        max_iterations=max(1, args.max_iters),
        recursion_limit=max(1, args.recursion_limit),
        pass_threshold=args.threshold,
        model=args.model,
        base_url=args.base_url,
        verbose=args.verbose,
        log_jsonl=args.log_jsonl,
    )
    return 0


if __name__ == "__main__":
    sys.exit(factory_main())


