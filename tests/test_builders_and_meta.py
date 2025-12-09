from judgekit.builders import create_judge, generate_criteria_from_phrase
from judgekit.registry import default_registry
from judgekit import builders


def test_registry_presets_listable():
    reg = default_registry()
    names = set(reg.list())
    assert "BalancedEssay" in names
    assert "CodeReview" in names


def test_create_judge_from_preset():
    j = create_judge("BalancedEssay", verbose=False)
    res = j.evaluate(candidate="A short argument with clarity and some evidence.", description="desc")
    assert 0.0 <= res.score <= 1.0


def test_generate_criteria_from_phrase_mockable(monkeypatch):
    # Monkeypatch call_xai_with_retries (as imported inside builders) to return deterministic rubric JSON
    def fake_call_xai_with_retries(*, retries=3, backoff=1.5, verbose=False, **kwargs):
        return {
            "text": '{"dimensions":["accuracy","clarity","specificity"], '
            '"template":"Evaluate accuracy, clarity, specificity."}'
        }

    monkeypatch.setattr(builders, "call_xai_with_retries", fake_call_xai_with_retries)
    crit = generate_criteria_from_phrase("Test rubric", verbose=False)
    assert crit.name == "Testrubric"
    assert crit.dimensions == ["accuracy", "clarity", "specificity"]


def test_meta_judge_on_rubric(monkeypatch):
    # Deterministic rubric generation for create_judge
    def fake_call_xai_with_retries(*, retries=3, backoff=1.5, verbose=False, **kwargs):
        return {
            "text": '{"dimensions":["soundness","clarity"], '
            '"template":"Evaluate soundness and clarity."}'
        }

    monkeypatch.setattr(builders, "call_xai_with_retries", fake_call_xai_with_retries)
    j = create_judge("My Custom Rubric", verbose=False)

    # Now meta-judge the rubric text with a preset judge
    reg = default_registry()
    assert reg.get("BalancedEssay") is not None
    rubric_text = j.default_criteria.template
    meta = create_judge("BalancedEssay", verbose=False)
    res = meta.evaluate(candidate=rubric_text, description="Evaluate rubric quality.", context="")
    assert isinstance(res.rationale, str)
    assert 0.0 <= res.score <= 1.0


