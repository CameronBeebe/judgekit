import os

try:
    import dspy  # type: ignore
    from dspy.optimizers import GEPO  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("This example requires dspy-ai with GEPO support installed.") from exc

from judgekit.utils import make_default_judge
from judgekit import JudgeModule


def main() -> None:
    # Configure DSPy with your LLM (use environment variables or dspy config as needed)
    lm_model = os.getenv("JUDGEKIT_DSPY_MODEL", "gpt-4o-mini")
    dspy.configure(lm=dspy.OpenAI(model=lm_model))  # type: ignore[attr-defined]

    # Example training set (context + target keywords)
    trainset = [
        dspy.Example(context="Families comparing hybrid mattresses and needing cooling.").with_inputs("context"),
        dspy.Example(context="Athletes needing lumbar support after training.").with_inputs("context"),
        dspy.Example(context="Remote workers seeking budget foam mattresses for studio apartments.").with_inputs("context"),
    ]

    judge = make_default_judge(verbose=False)
    judge_module = JudgeModule(
        judge,
        description="Evaluate mattress sales pitch quality",
        context_key="context",
        observe_name="judgekit.gepo.metric",
    )
    metric = judge_module.as_metric()

    class MattressPitcher(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pitch = dspy.Prompt("context -> pitch", template="Context: {{context}}\nPitch:")

        def forward(self, context: str):
            prediction = self.pitch(context=context)
            pitch_text = prediction.text if hasattr(prediction, "text") else str(prediction)
            return dspy.Prediction(pitch=pitch_text)

    program = MattressPitcher()
    optimizer = GEPO(metric=metric, trainset=trainset, max_iters=3)
    optimized_program = optimizer(program)

    example_context = "Busy parents wanting quick delivery and easy setup."
    outcome = optimized_program(context=example_context)
    print("Optimized pitch:\n", outcome.pitch)
    verdict = judge_module(candidate=outcome.pitch, context=example_context)
    print(f"Judge score: {verdict.score:.3f} | Passed: {verdict.passed}")


if __name__ == "__main__":
    main()


