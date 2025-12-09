import asyncio

import pytest

from judgekit.api import Criteria, Judge, JudgePrompt, _parse_llm_json
from judgekit.heuristics import HeuristicJudge


def _make_criteria() -> Criteria:
    dims = ["reasoning", "rigor", "balance", "specificity", "clarity"]
    # Equal weights; Criteria will normalize if needed
    weights = {d: 1.0 for d in dims}
    return Criteria(name="BalancedEssay", dimensions=dims, weights=weights, template="t")


def test_parse_llm_json_extracts_object_from_wrapped_text():
    text = "prefix\n  {\"rationale\": \"ok\", \"score\": 0.9}\ntrailing text"
    obj = _parse_llm_json(text)
    assert obj["rationale"] == "ok"
    assert obj["score"] == 0.9


def test_parse_llm_json_raises_when_no_braces():
    with pytest.raises(ValueError):
        _parse_llm_json("there is definitely no json object here")


def test_evaluate_async_with_async_callable_parses_success():
    async def async_llm(prompt: JudgePrompt, config=None):
        return {"rationale": "async ok", "score": 0.8}

    crit = _make_criteria()
    judge = Judge(llm_callable=async_llm, default_criteria=crit, threshold=0.6)

    res = asyncio.run(judge.evaluate_async(candidate="text", description="desc"))
    assert 0.0 <= res.score <= 1.0
    assert isinstance(res.rationale, str)


def test_evaluate_async_heuristic_fallback_on_parse_failure():
    async def async_noisy_llm(prompt: JudgePrompt, config=None):
        # Return non-JSON text to force parse failure path
        return {"text": "not valid json at all"}

    crit = _make_criteria()
    judge = Judge(
        llm_callable=async_noisy_llm,
        default_criteria=crit,
        threshold=0.6,
        heuristic_fallback=HeuristicJudge(),
    )

    res = asyncio.run(judge.evaluate_async(candidate="text with some content", description="desc"))
    # Heuristic fallback should still produce a bounded score and mark error metadata
    assert 0.0 <= res.score <= 1.0
    assert res.details.get("error") == "llm_parse_failure"
    # Heuristic path should mark that it ran
    assert res.details.get("heuristic") is True



