import pytest

from judgekit.api import Criteria
from judgekit.dspy_module import JudgeModule
from judgekit.testing import make_mock_judge


def _build_judge(score: float = 0.42):
    dims = ["reasoning", "rigor", "balance", "specificity", "clarity"]
    weights = {d: 1.0 / len(dims) for d in dims}
    crit = Criteria(name="BalancedEssay", dimensions=dims, weights=weights, template="t")
    return make_mock_judge(crit, score=score)


def test_judge_module_forward_returns_prediction():
    pytest.importorskip("dspy")
    judge = _build_judge(score=0.55)
    module = JudgeModule(judge, description="Check quality")
    verdict = module(candidate="example output", context="ctx")
    assert pytest.approx(verdict.score, rel=1e-6) == 0.55
    assert verdict.passed is False  # threshold defaults to 0.6
    assert verdict.judge_result.score == verdict.score


def test_candidate_and_context_resolvers_are_used():
    pytest.importorskip("dspy")
    judge = _build_judge(score=0.73)
    module = JudgeModule(judge, description="Resolver demo", context_key="context")
    verdict = module(example={"candidate": "auto text", "context": "ctx block"})
    assert pytest.approx(verdict.score, rel=1e-6) == 0.73


def test_metric_wrapper_reuses_module():
    pytest.importorskip("dspy")
    judge = _build_judge(score=0.33)
    module = JudgeModule(judge, description="Metric demo", context_key="context")
    metric = module.as_metric()
    score = metric({"context": "ctx"}, "prediction text")
    assert 0.0 <= score <= 1.0

