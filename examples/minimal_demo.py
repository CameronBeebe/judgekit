import time

try:
    from lmnr import observe  # type: ignore
except Exception:
    observe = None  # type: ignore[assignment]

from judgekit.utils import make_default_judge
from judgekit import JudgeModule


def main() -> None:
    # Build a default, verbose Judge with xAI adapter, criteria, heuristic, console hooks, and Laminar init
    judge = make_default_judge(verbose=True)

    candidate = (
        "The policy could reduce emissions significantly; however, costs may burden small businesses. "
        "A balanced approach should include subsidies and phased targets. Therefore, adopting the plan with "
        "adjustments seems reasonable."
    )
    description = "Assess the quality of this short policy analysis per the rubric."
    context = "Topic: Emissions policy; Stakeholders: small businesses, consumers, regulators."

    print("Running JudgeKit LLM evaluation...")
    t0 = time.time()
    if observe:
        @observe(name="judge_evaluate")  # type: ignore[misc]
        def run_eval():
            return judge.evaluate(candidate=candidate, description=description, context=context)
        result = run_eval()
    else:
        result = judge.evaluate(candidate=candidate, description=description, context=context)
    elapsed = (time.time() - t0) * 1000
    print(f"Score: {result.score:.3f} | Passed: {result.passed} | Rationale: {result.rationale}")
    print(f"Elapsed: {elapsed:.0f} ms")

    print("\nConstructing DSPy judge module...")
    dspy_judge = JudgeModule(
        judge,
        description="Evaluate model output quality",
        context_key="context",
        observe_name="judgekit.demo",
    )
    module_result = dspy_judge(candidate=candidate, context=context, description=description)
    print(f"DSPy module score: {module_result.score:.3f} | Passed: {module_result.passed}")
    metric = dspy_judge.as_metric()
    print("DSPy metric ready; returns JudgeKit score for predictions.")


if __name__ == "__main__":
    main()


