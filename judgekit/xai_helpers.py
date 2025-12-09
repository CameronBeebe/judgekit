from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .llm_chat import xai_chat


def call_xai_with_retries(
    *,
    retries: int = 3,
    backoff: float = 1.5,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return xai_chat(**kwargs)
        except Exception as err:  # pragma: no cover (network dependent)
            last_err = err
            if verbose:
                print(f"[xai] attempt {attempt}/{retries} failed: {err}")
            if attempt == retries:
                raise
            time.sleep(backoff ** (attempt - 1))
    if last_err:
        raise last_err
    raise RuntimeError("xai_chat failed unexpectedly")


