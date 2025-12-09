## JudgeKit Architecture & Core Judge Summary

### High-Level Architecture

- **Core types and judge** (`api.py`):
  - `Criteria`, `JudgePrompt`, `JudgeResult`, and `Judge` define the main evaluation abstraction.
  - `Judge` is provider-agnostic: it accepts any callable matching `LLMCallable` / `AsyncLLMCallable` that turns a `JudgePrompt` into a JSON-like dict or JSON string.
  - `Criteria` objects carry rubric dimensions, weights, a free-text template, and a simple `version` string. Weights are auto-normalized and default to equal weights if unspecified.
- **Criteria registry and builders** (`registry.py`, `builders.py`):
  - `CriteriaRegistry` stores named `Criteria` and provides a default library of presets (`BalancedEssay`, `AnalyticalSummary`, `CodeReview`, `SafetyRedTeam`, `DataExtraction`).
  - `generate_criteria_from_phrase` and `refine_criteria` use xAI (via `call_xai_with_retries`) to synthesize or refine rubrics from natural language phrases/feedback.
  - `create_judge` resolves a preset or phrase to `Criteria`, initializes an xAI-backed adapter, attaches console hooks and a `HeuristicJudge` fallback, and returns a configured `Judge`.
- **Heuristic fallback** (`heuristics.py`):
  - `HeuristicJudge` implements a fast, deterministic scoring path using text features (connectives, citations, counterpoints, numbers, sentence lengths).
  - It outputs a `JudgeResult` with scalar `score` and per-dimension `subscores`, and marks results with `details={"heuristic": True}`.
- **Adapters and provider integration** (`adapters/openai_compat.py`, `llm_chat.py`, `xai_helpers.py`, `utils.py`):
  - `OpenAICompatAdapter` wraps OpenAI-compatible chat APIs, targeting xAI (using `XAI_API_KEY` and `XAI_BASE_URL`).
  - `xai_chat` and `call_xai_with_retries` serve as helpers for calling xAI directly for rubric generation and candidate generation.
  - `console_hooks` and `init_laminar_from_env` support logging/tracing and optional Laminar integration.
- **DSPy integration** (`dspy_module.py`, `dspy_integration.py`):
  - `JudgeModule` wraps a `Judge` as a `dspy.Module` and exposes structured `dspy.Prediction` outputs and a `metric(example, pred) -> float` interface via `as_metric()`.
  - `make_dspy_metric` and `attach_metric_to_program` offer convenience helpers to connect JudgeKit to DSPy programs.
- **CLIs and optimization loop** (`cli.py`, `loop.py`, `examples/`):
  - `factory_main`, `eval_main`, and `loop_main` provide CLIs for one-off evaluation, meta-judging/refinement, and iterative agent–judge loops.
  - `run_loop` in `loop.py` implements a simple agent → judge → meta-judge → rubric refinement loop, with optional JSONL logging.

### Judge Implementation Summary

- **Prompt building and safety**:
  - `Judge.build_prompt` creates a `JudgePrompt` from candidate text, description, criteria, and optional context/reference.
  - The actual rendered prompt is assembled either by `_render_prompt_text` (for hooks/demos) or by `OpenAICompatAdapter._render_messages` (for OpenAI-compatible APIs).
  - Both renderers:
    - Wrap `context`, `reference`, and `candidate` in quoted fenced blocks.
    - Explicitly instruct the LLM to ignore instructions in these blocks and to output *JSON only*.
- **Evaluation flow (sync and async)**:
  - `evaluate`:
    - Builds a `JudgePrompt`, runs `on_llm_start` / `on_prompt_rendered` hooks, calls the LLM, times the call in ms, and triggers `on_llm_finish`.
    - Tries to interpret the returned object directly or via `"text"` as JSON, using `_parse_llm_json` and `_result_from_llm_dict`.
    - If parsing fails, it tries again on the `"text"` field (if any), then falls back to `HeuristicJudge` if configured, or returns a failure `JudgeResult` with `error="llm_parse_failure"`.
    - It invokes `on_item_finish` for successful LLM results and for heuristic fallbacks.
  - `evaluate_async`:
    - Mirrors the sync logic but attempts to handle both sync and async LLM callables by inspecting `__await__`.
    - On parse failure, it follows the same fallback strategy as `evaluate_async`.
  - `BatchConfig` is defined (concurrency, retries, jitter, timeouts, preserve_order) but is **not** yet wired into any batch API.
- **Heuristic fallback path**:
  - When the LLM output cannot be parsed into a valid JSON object with `rationale` and `score`, the judge can call `HeuristicJudge.evaluate` to get a deterministic score.
  - Fallback results store error metadata (at least `error="llm_parse_failure"` and `retries=1`) and are surfaced to hooks as completed items.

---

## Suggested Updates & Fixes

The items below are grouped by concern. Each item includes:
- **What to change** (goal and scope).
- **Where** (specific files/functions).
- **Why** (rationale).
- **How** (high-level implementation steps suitable for junior engineers).

### 1. Implement Batch Evaluation and Use `BatchConfig`

- **1.1 Implement `evaluate_batch_async` and `evaluate_batch`**
  - **What**: Add first-class batch evaluation APIs that accept a list of work items and return a list of `JudgeResult` objects, respecting concurrency, retries, and timeouts.
  - **Where**:
    - `judgekit/api.py`: add `evaluate_batch_async` and `evaluate_batch` functions near `BatchConfig`.
    - Optionally expose them in `judgekit/__init__.py`.
  - **Why**:
    - The PRD explicitly calls for batch evaluation with concurrency limits, per-item retries, and heuristic fallback.
    - Current code defines `BatchConfig` but does not use it; this is a noticeable feature gap for users wanting to score datasets.
  - **How**:
    1. **Define item schema**:
       - Decide on a dict shape for batch items, e.g.:
         - `{"item_id": Optional[Hashable], "candidate": str, "description": str, "criteria": Criteria | None, "context": str, "reference": str, "llm_config": dict | None}`.
       - Document this in the function docstrings and type hints.
    2. **Implement `evaluate_batch_async`**:
       - Signature (approximate):
         - `async def evaluate_batch_async(items: list[dict], judge: Judge, fallback_heuristic: HeuristicJudge | None = None, batch_config: BatchConfig | None = None) -> list[JudgeResult]:`
       - Inside:
         - Default `batch_config` if `None`.
         - Create an `asyncio.Semaphore(batch_config.max_concurrency)`.
         - For each item, create an async worker coroutine that:
           - Extracts item fields and calls `judge.evaluate_async` or `judge.evaluate` (wrapped with `asyncio.to_thread` if needed).
           - Implements retry logic:
             - Try up to `batch_config.retries + 1` times.
             - On transient exceptions or parse failures (use a broad `except Exception` at first), apply jitter `asyncio.sleep(batch_config.jitter_ms / 1000.0)` between attempts.
             - If `batch_config.timeout_ms` is set, wrap each attempt in `asyncio.wait_for`.
           - On final failure, use `fallback_heuristic` (if provided) to produce a `JudgeResult`; otherwise, construct a `JudgeResult` with `score=0.0`, `passed=False`, and a clear `details["error"]` code (e.g. `"llm_batch_failure"`).
         - Use `asyncio.gather` to run all worker coroutines, with `return_exceptions=False`.
         - If `batch_config.preserve_order` is `True`, maintain a mapping from index to result and return results in the original order. Otherwise, you may return in completion order (document your choice).
    3. **Implement `evaluate_batch`**:
       - Sync wrapper:
         - `def evaluate_batch(...):` should call the async function using `asyncio.run` when no event loop is running.
         - If you detect an already running event loop (e.g., `asyncio.get_event_loop().is_running()`), fall back to `asyncio.get_event_loop().create_task(...)` + `loop.run_until_complete(...)` or document that `evaluate_batch` must be used only in non-async contexts.
    4. **Hooks & metadata**:
       - Call `on_item_finish` from within each worker (already handled by `Judge` internally, but you may want a separate `on_batch_finish` hook; see item 4.3).
       - For each `JudgeResult`, enrich `details` with:
         - `item_id`, `retries`, any timeout flags, `timings_ms`.
    5. **Testing**:
       - Add unit tests in `tests/`:
         - Use a `MockLLM` that can simulate success, failure, and parse errors.
         - Ensure `BatchConfig.max_concurrency` limits concurrent tasks (can assert via counters).
         - Validate `preserve_order` behavior.

### 2. Align Retry Behavior with the PRD (Second Prompt Attempt)

- **2.1 Add a real second LLM attempt with stricter JSON instructions**
  - **What**: When JSON parsing fails, re-call the LLM once with a stricter reminder and explicit schema before resorting to heuristics.
  - **Where**:
    - `judgekit/api.py`: inside `Judge.evaluate` and `Judge.evaluate_async`.
    - Potential helper: `_build_strict_llm_config` or `_decorate_description_for_retry`.
  - **Why**:
    - The PRD says: “If parse fails, retry once with stricter reminder and echo of the expected JSON schema; then fallback to heuristic.”
    - Currently we only re-parse the same text; we never issue a second LLM call. This under-utilizes the retry budget and may reduce robustness.
  - **How**:
    1. **Introduce a retry flag / config**:
       - Extend `llm_config` to carry an internal `{"_retry": True}` marker or similar (not exposed to users), or create a local config copy for the second call.
    2. **Build stricter instructions**:
       - On parse failure:
         - Modify the `JudgePrompt` or synthesize a modified prompt for the second call that:
           - Explicitly includes the JSON schema (e.g., in `output_schema_hint` or a new field).
           - Adds a firm instruction such as:
             - “You previously produced output that was not valid JSON. Now output ONLY valid JSON matching this schema: ... No prose before or after.”
         - Option A (simpler): include this text in `prompt_description`.
         - Option B (cleaner): extend `JudgePrompt` with a `retry_hint` field and update renderers to include it when present.
    3. **Call LLM again**:
       - After the first call and parse failure:
         - Perform a second `self._llm_callable(prompt, llm_config)` invocation with the stricter text.
         - Parse this second result using `_parse_llm_json` and `_result_from_llm_dict`.
         - If it still fails, then go to the heuristic fallback path.
    4. **Record retries and reasons**:
       - For successful second-try parses, set `result.details["retries"] = 1`.
       - For heuristic fallback cases, set `result.details["retries"] = 1` (or more if you add network retries later) and `result.details["error"] = "llm_parse_failure"`.
    5. **Tests**:
       - In `tests/test_judge_with_mock.py`, extend tests to cover:
         - A `MockLLM` that returns invalid text the first time and valid JSON the second time.
         - Assertions that:
           - `Judge.evaluate` calls the LLM twice.
           - Returned `JudgeResult.details["retries"] == 1`.

### 3. Improve Async Story and Avoid Event Loop Pitfalls

- **3.1 Refactor to a shared async core for evaluation**
  - **What**: Centralize evaluation logic in a single async helper and make `evaluate` a thin sync wrapper to avoid duplicated logic and dangerous `run_until_complete` behavior.
  - **Where**:
    - `judgekit/api.py`: inside `Judge`.
  - **Why**:
    - Today `evaluate` and `evaluate_async` have overlapping logic and slightly different code paths.
    - `evaluate` attempts to call `asyncio.get_event_loop().run_until_complete(...)` on a potentially already-running loop, which can break in notebooks and async frameworks.
  - **How**:
    1. **Create an internal async method**:
       - Example: `async def _evaluate_inner(self, prompt: JudgePrompt, criteria: Criteria, llm_config: Optional[dict]) -> JudgeResult:`
       - Move the common logic (hooks, LLM call, parsing, retries, heuristic fallback, details) into this method.
    2. **Implement `evaluate_async` via the core**:
       - `evaluate_async` would:
         - Build `prompt`.
         - Call `return await self._evaluate_inner(prompt, criteria, llm_config)`.
    3. **Implement `evaluate` as a wrapper**:
       - In `evaluate`, build `prompt` and then:
         - If no event loop is running:
           - Use `asyncio.run(self._evaluate_inner(prompt, criteria, llm_config))`.
         - If an event loop is already running:
           - Either:
             - Document that `evaluate` is not safe when an event loop is already running (and advise using `evaluate_async`), or
             - Use `nest_asyncio`-like behavior (not recommended as a hidden dependency).
       - For a first pass, you can:
         - Detect running loop (`loop = asyncio.get_event_loop(); loop.is_running()`).
         - If running, call `loop.run_until_complete(self._evaluate_inner(...))` only if you are sure you created the loop yourself. Otherwise, raise a clear error suggesting `evaluate_async`.
    4. **Keep `LLMCallable` / `AsyncLLMCallable` support**:
       - In `_evaluate_inner`, handle both sync and async LLM callables by:
         - Calling `maybe = self._llm_callable(prompt, llm_config)`.
         - If `inspect.isawaitable(maybe)`, `raw = await maybe`.
         - Else, `raw = maybe`.
    5. **Tests**:
       - Re-run existing tests; add a small test that ensures a sync callable still works when used via `evaluate_async` (should run synchronously but not error).

### 4. Observability, Hooks, and `JudgeResult.details`

- **4.1 Standardize `details` schema on `JudgeResult`**
  - **What**: Define and consistently populate a minimal schema in `JudgeResult.details`.
  - **Where**:
    - `judgekit/api.py`: `_result_from_llm_dict`, `evaluate`, `evaluate_async`.
    - `heuristics.py`: `HeuristicJudge.evaluate`.
    - Possibly in new batch APIs.
  - **Why**:
    - The PRD describes a rich `details` dict (model IDs, timings, error codes, retries, item IDs).
    - Right now, different code paths only set partial information (e.g., criteria name/version, error codes) and do not attach consistent timing or model metadata.
  - **How**:
    1. **Define a target schema** (document in comments):
       - Suggested keys:
         - `"criteria"` (str), `"criteria_version"` (str)
         - `"model_id"` (str | None)
         - `"timings_ms"` (dict with keys: `"llm"`, `"heuristic"`, `"total"`; optionally `"build"`)
         - `"error"` (str | None)
         - `"retries"` (int)
         - `"heuristic"` (bool)
         - `"item_id"` (optional, for batch)
    2. **Populate from LLM path**:
       - When `Judge` gets `raw` from the LLM:
         - Extract `model_id` from `raw.get("model_id")` if it’s a dict, or set it to `None`.
         - Measure timings:
           - Already have `llm_ms` in `evaluate`/`evaluate_async`.
           - Compute `total_ms` (e.g., start timing before prompt build and end after result construction).
         - Set `details` accordingly when constructing `JudgeResult`.
         - Preserve or merge existing details, especially if `_result_from_llm_dict` already sets criteria-related details.
    3. **Populate from heuristic path**:
       - In `HeuristicJudge.evaluate`, add:
         - `timings_ms` measurement for heuristic evaluation.
         - `heuristic=True`, and optional `error`/`retries` if calling in fallback contexts.
    4. **Integrate with hooks**:
       - Continue to use `console_hooks` for printing, but make sure all the structured values you rely on are present in `details`, not just logs.

- **4.2 Add `on_batch_finish` hook support**
  - **What**: Implement the `on_batch_finish(results, meta)` hook described in the PRD.
  - **Where**:
    - `judgekit/utils.py`: extend `console_hooks` to include `"on_batch_finish"`.
    - New batch evaluation functions in `api.py`.
  - **Why**:
    - Batch-level observability is important for tracing and for summarizing runs in logs/metrics.
  - **How**:
    1. In `console_hooks`, add:
       - A function `on_batch_finish(results, meta)` that prints batch size, average score, and maybe total time.
       - Add this function to the returned dict.
    2. In `evaluate_batch_async`, After gathering results:
       - If `"on_batch_finish"` is present in `self._hooks`, call it with the results list and a meta dict (e.g., `{"num_items": len(results)}`).

### 5. Provider Abstraction and xAI Coupling

- **5.1 Decouple rubric generation from xAI-specific helpers**
  - **What**: Introduce a provider-agnostic interface for rubric generation/refinement and make xAI one implementation behind that interface.
  - **Where**:
    - `judgekit/builders.py`
    - Possibly a new module `judgekit/providers.py` or similar.
  - **Why**:
    - The PRD states “Not tied to any model/provider; must work with various LLMs via adapters or user-supplied callables.”
    - Today, `generate_criteria_from_phrase` and `refine_criteria` depend directly on `call_xai_with_retries` and xAI-specific environment variables.
  - **How**:
    1. **Define an abstraction**:
       - For example, a simple protocol:
         - `class RubricGenerator(Protocol): def generate(self, phrase: str) -> dict: ...`
         - `class RubricRefiner(Protocol): def refine(self, criteria: Criteria, feedback: str, phrase: Optional[str] = None) -> dict: ...`
    2. **Create xAI-backed implementations**:
       - Implement `XaiRubricGenerator` and `XaiRubricRefiner` that use `call_xai_with_retries`.
       - Keep the same JSON contract (dimensions, template).
    3. **Refactor `generate_criteria_from_phrase` / `refine_criteria`**:
       - Accept an optional provider object (defaulting to the xAI implementation).
       - Use the provider to get raw JSON, then map it to `Criteria` (with fallbacks as before).
    4. **Future-proofing**:
       - This will allow adding other providers (e.g., OpenAI, Anthropic) via extras without changing core logic.

- **5.2 Make `create_judge` friendlier for non-xAI callables**
  - **What**: Provide a convenient way to create a `Judge` around an arbitrary user-supplied callable without requiring `XAI_API_KEY` or the `openai` package.
  - **Where**:
    - `judgekit/builders.py`
  - **Why**:
    - Not all users will use xAI; some may want to pass in a local model or another provider.
  - **How**:
    1. Add a new helper:
       - Example: `def create_judge_from_callable(criteria: Criteria | str, llm_callable: LLMCallable | AsyncLLMCallable, threshold: float = 0.6, verbose: bool = True, registry: CriteriaRegistry | None = None) -> Judge:`
       - This should:
         - Resolve `Criteria` from the registry or phrase (optionally via `generate_criteria_from_phrase` or a user-supplied `Criteria`).
         - Skip OpenAI/xAI adapter instantiation.
         - Attach `console_hooks(verbose)` and `HeuristicJudge()` as before.
    2. Optionally relax `create_judge`:
       - Add a parameter like `use_xai_adapter: bool = True`. When `False`, require a user-provided callable instead of building `OpenAICompatAdapter`.
       - For clarity, keep the new helper separate and keep `create_judge`’s current behavior as the xAI-specific convenience path.

### 6. Heuristic Judge Fixes and Enhancements

- **6.1 Fix heuristic threshold usage**
  - **What**: Stop trying to read `criteria.threshold` (which does not exist) and instead take a clear threshold value.
  - **Where**:
    - `judgekit/heuristics.py`: `HeuristicJudge.evaluate`.
  - **Why**:
    - The current line:
      - `passed = score >= (criteria.threshold if hasattr(criteria, "threshold") else 0.6)`
      - relies on a non-existent `threshold` attribute on `Criteria` and quietly falls back to `0.6`.
    - This is confusing and fragile for future changes.
  - **How**:
    1. Change the function signature to:
       - `def evaluate(self, candidate: str, criteria: Optional[Criteria] = None, reference: str = "", threshold: float = 0.6) -> JudgeResult:`
    2. Compute `passed` as:
       - `passed = score >= threshold`
    3. Update all callsites:
       - When used as a fallback from `Judge`, pass the same threshold as the `Judge` instance (e.g., `self._threshold`).
       - In tests, rely on the default `0.6` unless you want to exercise different thresholds.

- **6.2 - Optional: incorporate reference-awareness**
  - **What**: Use the `reference` argument to adjust heuristic scoring when a reference text is provided.
  - **Where**:
    - `heuristics.py`: inside `HeuristicJudge.evaluate`.
  - **Why**:
    - The PRD notes that reference-based judging should be supported, even in heuristic mode (e.g., simple similarity or overlap checks).
  - **How** (minimal version):
    1. If `reference` is non-empty:
       - Compute a simple similarity score, e.g., Jaccard overlap over word sets or character-level overlap.
       - Blend this into the final score with a modest weight (e.g., 10–20% of the final heuristic score).
    2. Ensure this behavior is documented and deterministic.

### 7. Prompt Rendering Consolidation and Redaction

- **7.1 Consolidate prompt rendering logic**
  - **What**: De-duplicate the logic used in `_render_prompt_text` and `OpenAICompatAdapter._render_messages` so they share a single “source of truth.”
  - **Where**:
    - `judgekit/api.py`: `_render_prompt_text`.
    - `judgekit/adapters/openai_compat.py`: `_render_messages`.
  - **Why**:
    - Today we maintain two slightly different prompt builders; any future change (e.g., updating instructions) must be applied in two places.
  - **How**:
    1. Extract a shared helper, e.g.:
       - `def render_prompt_segments(prompt: JudgePrompt) -> list[str]:` or similar, which returns the key text blocks.
    2. Use this helper:
       - In `_render_prompt_text`, join with `"\n\n"` for a human-readable string.
       - In `_render_messages`, convert segments into the `{"type": "text", "text": ...}` form required by the OpenAI client.
    3. Make sure tests still pass and that hooks receive the same (or intentionally improved) prompt content.

- **7.2 Add a simple built-in redactor implementation**
  - **What**: Provide a default redactor function that removes obvious PII/secrets from `JudgePrompt` before sending it to external LLMs.
  - **Where**:
    - New module: `judgekit/redaction.py` (or inside `utils.py`).
    - Wire it through `Judge` usage (optional).
  - **Why**:
    - The PRD mentions a redaction API; `Judge` already accepts a `redactor` callable, but the package does not ship any built-in redaction logic.
  - **How**:
    1. Implement a function like:
       - `def regex_redactor(prompt: JudgePrompt) -> JudgePrompt:`
       - Use simple regexes to replace patterns like emails, URLs, obvious API keys (`[A-Za-z0-9]{32,}`) with placeholders (`"[REDACTED_EMAIL]"`, etc.).
    2. Ensure the function:
       - Returns a *new* `JudgePrompt` (do not mutate in place).
       - Does not log unredacted content.
    3. Provide a helper in `utils.py`:
       - `def make_default_redactor() -> Callable[[JudgePrompt], JudgePrompt]: ...`
    4. Optionally integrate it into `create_judge` / `make_default_judge` as an opt-in parameter (e.g., `enable_redaction: bool = False`).

### 8. Export Utilities and JSONL Schema

- **8.1 Add a reusable JSONL/CSV exporter**
  - **What**: Provide helper functions to export `JudgeResult` objects to JSONL/CSV following the schema described in the PRD.
  - **Where**:
    - New module: `judgekit/export.py` (or similar).
    - Integrate in `loop.py` and future batch APIs.
  - **Why**:
    - `loop.py` currently writes a minimal JSONL (iteration, candidate, score, etc.) but not the full schema (candidate hash, criteria name/version, timings, etc.).
    - Having a reusable exporter avoids duplicate, ad-hoc logging across scripts.
  - **How**:
    1. Implement:
       - `def result_to_record(item_id: Any, candidate: str, judge: Judge, result: JudgeResult) -> dict:`
         - Include at least:
           - `schema_version`, `item_id`, `candidate_hash` (using `_hash_text` from `api.py`), `candidate_preview` (truncated),
           - `criteria_name`, `criteria_version`, `weights`, `threshold`,
           - `score`, `passed`, `rationale`, `subscores`,
           - `model_id`, `timings_ms`, `timestamp`, `error`.
       - `def write_jsonl(path: str, records: list[dict]) -> None`.
    2. Update `loop.py`:
       - Replace ad-hoc `json.dumps` + manual fields with calls to `result_to_record` / `write_jsonl`.
    3. Optionally provide a CSV writer highlighting frequently used fields.

### 9. Criteria Versioning and Configuration Models

- **9.1 Introduce explicit versioning helpers for criteria**
  - **What**: Provide helper functions or a small registry for managing `Criteria.version` evolution.
  - **Where**:
    - `judgekit/registry.py` or a new `versioning` module.
  - **Why**:
    - Today, versions are hardcoded (`"1"` or `"1-refined"`) and must be manually updated when criteria change.
    - This makes reproducibility harder to track over time.
  - **How**:
    1. Decide on a simple convention:
       - e.g., `"1"`, `"1.1"`, `"1.2"`, or `"1-refined"`, `"1-refined2"`.
    2. Provide helpers such as:
       - `def bump_major(version: str) -> str`, `def bump_minor(version: str) -> str`.
    3. Use these helpers:
       - In `refine_criteria`, call `bump_minor(criteria.version)` instead of appending `"-refined"`.
       - In `default_registry`, keep version lookup consolidated in one place (so changing a criteria’s text can deliberately bump its version).

- **9.2 Config dataclasses for Judge / LLM / Batch settings (optional but helpful)**
  - **What**: Encapsulate configuration options (criteria, threshold, hooks, redactor, model params, batch params) in typed dataclasses.
  - **Where**:
    - New module: `judgekit/config.py` (or similar).
  - **Why**:
    - The PRD mentions `JudgeConfig`, `LLMConfig`, `BatchConfig` as conceptual objects; only `BatchConfig` is currently implemented.
    - Having explicit config objects can make client code and tests clearer.

### 10. Tests and Documentation Sync

- **10.1 Bring tests in sync with current implementations**
  - **What**: Update tests that reference outdated APIs or behaviors (e.g., `xai_chat` in `builders`).
  - **Where**:
    - `tests/test_builders_and_meta.py` and any other tests that refer to `xai_chat`.
  - **Why**:
    - Tests that monkeypatch non-existent functions or rely on outdated call paths will either fail or silently misrepresent behavior.
  - **How**:
    1. Replace references to `builders.xai_chat` with `xai_helpers.call_xai_with_retries` or the new abstraction introduced in item 5.1.
    2. Ensure tests simulate realistic xAI responses (`{"text": "{...json...}"}`) matching current builder expectations.
    3. Add new tests for:
       - Batch evaluation.
       - Retry behavior with two LLM calls.
       - New export functions.
       - Redactor behavior (ensuring PII is removed).

- **10.2 Update documentation to match implemented and planned features**
  - **What**: Keep `judgekit_prd.md` and any README in sync with actual behavior and planned roadmap.
  - **Where**:
    - `judgekit_prd.md`
    - Any user-facing docs/README files.
  - **Why**:
    - The PRD includes features (batch eval, detailed export schema, etc.) that are not yet implemented; being explicit about current version vs. future roadmap will reduce user confusion.
  - **How**:
    1. Add a short “Implementation status” section summarizing which PRD items are complete, partial, or pending.
    2. Cross-reference this improvement plan file so engineers and users know what’s coming next.

---

This file is intended as a living implementation plan. As items are completed, consider updating it (or a separate CHANGELOG) to reflect the current state and to keep future work focused on the highest-impact gaps.


