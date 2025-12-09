from .api import Criteria, JudgePrompt, JudgeResult, BatchConfig, Judge
from .heuristics import HeuristicJudge
from .registry import CriteriaRegistry, default_registry
from .builders import create_judge, generate_criteria_from_phrase, refine_criteria
from .dspy_module import JudgeModule
from .dspy_integration import make_dspy_metric, attach_metric_to_program
from .utils import make_default_judge

__version__ = "0.1.0"

__all__ = [
    "Criteria",
    "JudgePrompt",
    "JudgeResult",
    "BatchConfig",
    "Judge",
    "HeuristicJudge",
    "CriteriaRegistry",
    "default_registry",
    "create_judge",
    "generate_criteria_from_phrase",
    "refine_criteria",
    "make_dspy_metric",
    "JudgeModule",
    "attach_metric_to_program",
    "make_default_judge",
    "__version__",
]


