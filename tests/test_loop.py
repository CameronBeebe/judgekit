from __future__ import annotations

from pathlib import Path
from typing import List

from judgekit import loop as loop_mod
from judgekit.api import JudgeResult


class DummyCriteria:
    def __init__(self, name: str = "Crit", template: str = "t", version: str = "1") -> None:
        self.name = name
        self.template = template
        self.version = version


class DummyJudge:
    """
    Simple stand-in for a Judge used to exercise the recursive loop behavior.
    Returns scores from a preconfigured sequence and records evaluated candidates.
    """

    def __init__(self, scores: List[float]) -> None:
        self.default_criteria = DummyCriteria()
        self._scores = list(scores)
        self.evaluated_candidates: List[str] = []

    def evaluate(self, candidate: str, description: str, context: str) -> JudgeResult:  # type: ignore[override]
        self.evaluated_candidates.append(candidate)
        score = self._scores.pop(0) if self._scores else 0.0
        passed = score >= 0.5
        return JudgeResult(rationale=f"score={score}", score=score, passed=passed)


class DummyMetaResult:
    def __init__(self, rationale: str) -> None:
        self.rationale = rationale


def test_run_loop_uses_meta_feedback_and_refinement(monkeypatch, tmp_path: Path):
    """
    The recursive loop should:
    - generate an initial candidate with empty feedback
    - when a pass occurs, call the meta-judge and use its rationale as feedback
    - call refine_criteria with that feedback
    - stop once recursion_limit passes have been achieved.
    """

    refine_calls = []
    generated_feedbacks = []
    judge_instances: List[DummyJudge] = []

    def fake_refine_criteria(criteria, feedback: str, phrase=None, verbose: bool = False):
        refine_calls.append((criteria, feedback))
        # Return a new dummy criteria object to simulate refinement
        return DummyCriteria(name=criteria.name if hasattr(criteria, "name") else "Crit", template="refined", version="2")

    def fake_meta_judge(judge, **kwargs):
        # Always return the same feedback string
        return DummyMetaResult(rationale="meta-feedback")

    def fake_generate_candidate(
        agent_prompt: str,
        context: str,
        feedback: str,
        *,
        verbose: bool,
        temperature: float = 0.6,
        max_tokens: int = 400,
    ) -> str:
        generated_feedbacks.append(feedback)
        return f"candidate-{len(generated_feedbacks)}"

    def fake_create_judge(criteria, model=None, base_url=None, threshold: float = 0.6, verbose: bool = True, registry=None):
        # First judge handles up to 2 iterations, refined judge could handle further ones if needed.
        if not judge_instances:
            j = DummyJudge(scores=[0.8, 0.8])
        else:
            j = DummyJudge(scores=[0.9])
        judge_instances.append(j)
        return j

    monkeypatch.setattr(loop_mod, "create_judge", fake_create_judge)
    monkeypatch.setattr(loop_mod, "refine_criteria", fake_refine_criteria)
    monkeypatch.setattr(loop_mod, "_run_meta_judge", fake_meta_judge)
    monkeypatch.setattr(loop_mod, "_generate_candidate", fake_generate_candidate)

    log_path = tmp_path / "loop.jsonl"

    loop_mod.run_loop(
        criteria="BalancedEssay",
        agent_prompt="You are an agent.",
        context="ctx",
        max_iterations=5,
        recursion_limit=2,
        pass_threshold=0.7,
        model=None,
        base_url=None,
        verbose=False,
        log_jsonl=str(log_path),
    )

    # Two iterations should have run (until recursion_limit=2 is satisfied)
    assert judge_instances
    # First judge handles the initial iteration, refined judge handles the next
    total_evals = sum(len(j.evaluated_candidates) for j in judge_instances)
    assert total_evals == 2

    # First candidate is generated with empty feedback, second with meta-feedback
    assert generated_feedbacks[0] == ""
    assert generated_feedbacks[1] == "meta-feedback"

    # Refinement should have been invoked once with feedback from the meta-judge
    assert len(refine_calls) == 1
    _, fb = refine_calls[0]
    assert fb == "meta-feedback"

    # JSONL log should have one line per iteration
    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2


