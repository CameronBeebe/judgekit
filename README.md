# judgekit

Status: an experimental LLM-as-judge library. I'm no longer actively developing it.

judgekit scores text against a rubric with an LLM judge. It ships five preset rubrics, can generate or refine a rubric from a short phrase, wraps a judge as a DSPy metric, and includes a mock judge for offline tests. It has rough edges: the DSPy metric uses an old two-argument signature, the GEPA example targets an API that no longer exists, the LLM path is hardwired to xAI, and the heuristic fallback only makes sense for essay-style rubrics.

## Where to look instead

- [DSPy](https://dspy.ai) supports LLM judges as metrics directly, and [GEPA](https://dspy.ai/current/api/optimizers/GEPA/overview/) takes a score plus text feedback.
- Judge alignment against human labels: [MLflow's GEPA judge aligner](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/gepa/), [LangSmith](https://docs.langchain.com/langsmith/improve-judge-evaluator-feedback), [Ragas](https://docs.ragas.io/en/stable/howtos/applications/align-llm-as-judge/).
- Papers:
  - [Who Validates the Validators? (EvalGen)](https://arxiv.org/abs/2404.12272)
  - [Debate Training Reduces Reward Hacking in RLAIF](https://arxiv.org/abs/2608.17776)
  - [LLM-as-a-Judge Is Not an Oracle (PROCTOR)](https://arxiv.org/abs/2609.02246)

My ongoing DSPy work on dialectical and critic-based evaluation is in [diaspy](https://github.com/CameronBeebe/diaspy).
