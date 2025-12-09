from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def xai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """
    Minimal helper for xAI chat completion calls using the OpenAI-compatible SDK.
    - messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Returns: {"text": str, "model_id": str}
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please `pip install openai`.")
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY not set")
    base = base_url or os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1"
    mdl = model or os.getenv("JUDGEKIT_MODEL") or "grok-3-mini"
    client = OpenAI(api_key=key, base_url=base)
    resp = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return {"text": content, "model_id": getattr(resp, "model", mdl)}


