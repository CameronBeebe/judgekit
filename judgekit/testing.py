from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .api import Criteria, Judge, JudgePrompt
from .heuristics import HeuristicJudge


@dataclass
class MockLLM:
    response: Dict[str, Any]
    as_text: bool = False

    def __call__(self, prompt: JudgePrompt, config: Optional[dict] = None) -> Dict[str, Any] | str:
        if self.as_text:
            return (
                '{"rationale":"ok","score":0.9}'
                if not self.response
                else self.response.get("text", '{"rationale":"ok","score":0.9}')
            )
        return self.response or {"rationale": "ok", "score": 0.9}


def make_mock_judge(criteria: Criteria, score: float = 0.8) -> Judge:
    mock = MockLLM(response={"rationale": "mocked", "score": score})
    return Judge(llm_callable=mock, default_criteria=criteria, threshold=0.6, heuristic_fallback=HeuristicJudge())


