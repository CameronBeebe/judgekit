from judgekit.heuristics import HeuristicJudge
from judgekit.api import Criteria


def test_heuristic_determinism():
    dims = ["reasoning", "rigor", "balance", "specificity", "clarity"]
    weights = {d: 1.0 / len(dims) for d in dims}
    crit = Criteria(name="BalancedEssay", dimensions=dims, weights=weights, template="t")
    hj = HeuristicJudge()
    candidate = "This is a clear, specific, balanced argument; however, evidence is limited."
    r1 = hj.evaluate(candidate, crit)
    r2 = hj.evaluate(candidate, crit)
    assert r1.score == r2.score
    assert r1.passed == r2.passed


