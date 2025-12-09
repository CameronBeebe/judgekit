import os
import time
from typing import Dict, Any

try:
    from lmnr import observe  # type: ignore
except Exception:
    observe = None  # type: ignore[assignment]

from judgekit.utils import make_default_judge
from judgekit.llm_chat import xai_chat
from judgekit import JudgeModule


def _read_block(prompt: str) -> str:
    print(prompt)
    print("(Finish with an empty line)")
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def ask_clarifying_questions(context: str) -> str:
    system = "You are a helpful assistant that asks concise clarifying questions to improve a build plan."
    user = f"Given this context, ask 2-3 short clarifying questions to resolve ambiguities:\n\n{context}"
    res = xai_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=300,
    )
    return res["text"].strip()


def synthesize_candidate(context: str, clarifications: str) -> str:
    system = "You are a senior engineer. Propose a concrete plan focused on next actions."
    user = f"Context:\n{context}\n\nClarifications:\n{clarifications}\n\nDraft a short plan (3-6 bullets) to build the requested thing. Be specific."
    res = xai_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=400,
    )
    return res["text"].strip()


def improve_with_feedback(candidate: str, rationale: str) -> str:
    system = "You are a constructive reviewer who turns critiques into improvements."
    user = f"Candidate:\n{candidate}\n\nCritique to address:\n{rationale}\n\nRewrite an improved candidate addressing the critique."
    res = xai_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=500,
    )
    return res["text"].strip()


def main() -> None:
    judge = make_default_judge(verbose=True)

    # Interactive or env-provided context
    is_interactive = os.getenv("JUDGEKIT_NONINTERACTIVE") not in ("1", "true", "True")
    context = os.getenv("JUDGEKIT_CONTEXT", "").strip()
    if not context and is_interactive:
        context = _read_block("Enter initial context describing what to build:")
    if not context:
        # Fallback seed if nothing provided
        context = (
            "We want a small microservice to summarize issues from GitHub and post weekly updates to Slack. "
            "The service should be easy to deploy and cheap to run."
        )
    description = "Evaluate if the plan addresses the context and is specific, balanced, rigorous, and clear."

    max_iters = 3
    best: Dict[str, Any] = {"score": 0.0, "candidate": "", "rationale": ""}

    for i in range(1, max_iters + 1):
        print(f"\n=== Iteration {i} ===")
        clarifications = ask_clarifying_questions(context)
        print(f"[agent] Clarifying questions:\n{clarifications}\n")

        enriched_context = context
        if is_interactive:
            answers = _read_block("Answer clarifying questions (optional):")
            if answers:
                enriched_context = f"{context}\n\nUser clarifications:\n{answers}"

        candidate = synthesize_candidate(enriched_context, clarifications)
        print(f"[agent] Candidate plan:\n{candidate}\n")

        print("[judge] Evaluating candidate...")
        t0 = time.time()
        if observe:
            @observe(name="judge_iterate")  # type: ignore[misc]
            def run_eval():
                return judge.evaluate(candidate=candidate, description=description, context=context)
            result = run_eval()
        else:
            result = judge.evaluate(candidate=candidate, description=description, context=context)
        print(f"[judge] Score={result.score:.3f} Passed={result.passed} Took={(time.time()-t0)*1000:.0f}ms")
        print(f"[judge] Rationale: {result.rationale}\n")

        if result.score > best["score"]:
            best = {"score": result.score, "candidate": candidate, "rationale": result.rationale}
        if result.passed:
            print("[loop] Passed threshold, stopping.")
            break

        # Improve for next iteration using judge rationale as feedback
        if is_interactive:
            hints = _read_block("Optional improvement hints (press Enter to skip):")
            fb = f"{result.rationale}\n\nUser hints:\n{hints}" if hints else result.rationale
        else:
            fb = result.rationale
        candidate = improve_with_feedback(candidate, fb)
        # Context can be updated as needed; we keep enriched_context modifications persistent
        context = enriched_context

    print("\n=== Best Candidate ===")
    print(best["candidate"])
    print(f"Best score: {best['score']:.3f}")

    dspy_module = JudgeModule(
        judge,
        description=description,
        context_key="context",
        observe_name="judgekit.loop.metric",
    )
    metric = dspy_module.as_metric()
    sample_candidate = best["candidate"] or candidate
    sample_score = metric({"context": context}, sample_candidate)
    print(f"[loop] DSPy metric ready; sample score for best candidate: {sample_score:.3f}")


if __name__ == "__main__":
    main()


