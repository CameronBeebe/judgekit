from judgekit.api import Criteria
from judgekit.testing import make_mock_judge, MockLLM


def test_judge_parses_structured_response():
    dims = ["reasoning", "rigor", "balance", "specificity", "clarity"]
    weights = {d: 1.0 / len(dims) for d in dims}
    crit = Criteria(name="BalancedEssay", dimensions=dims, weights=weights, template="t")
    judge = make_mock_judge(crit, score=0.75)
    res = judge.evaluate(candidate="abc", description="desc")
    assert 0.0 <= res.score <= 1.0
    assert isinstance(res.passed, bool)
    assert isinstance(res.rationale, str)


def test_judge_retry_and_fallback():
    dims = ["reasoning", "rigor", "balance", "specificity", "clarity"]
    weights = {d: 1.0 / len(dims) for d in dims}
    crit = Criteria(name="BalancedEssay", dimensions=dims, weights=weights, template="t")
    # LLM returns non-JSON text; forces fallback to heuristic
    noisy = MockLLM(response={"text": "not json"}, as_text=True)
    from judgekit.api import Judge
    judge = Judge(llm_callable=noisy, default_criteria=crit, threshold=0.6, heuristic_fallback=None)
    res = judge.evaluate(candidate="short", description="desc")
    # No heuristic fallback configured; expect failure object with error marker
    assert res.score == 0.0
    assert res.passed is False
    assert res.details.get("error") == "llm_parse_failure"


