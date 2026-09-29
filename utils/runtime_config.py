"""Resolve MLLM credentials without storing secrets in project files."""

import os
import re
from typing import Mapping, Optional


def resolve_api_key(mllm: Mapping[str, object]) -> str:
    """Read the environment variable named by ``mllm.api_key_env``.

    For an unauthenticated local server, explicitly set that variable to EMPTY.
    Literal credentials in YAML are intentionally unsupported.
    """
    env_name = mllm.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(env_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("mllm.api_key_env must be an environment variable name, not a credential.")
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(f"Set {env_name} before running inference (use EMPTY for an unauthenticated local server).")
    return value


def resolve_base_url(mllm: Mapping[str, object]) -> Optional[str]:
    return os.environ.get("MLLM_BASE_URL") or mllm.get("base_url") or None


def resolve_model_name(mllm: Mapping[str, object]) -> str:
    value = os.environ.get("MLLM_MODEL_NAME") or mllm.get("model_name")
    if not value:
        raise ValueError("Set mllm.model_name or MLLM_MODEL_NAME to your vision model's served name.")
    return str(value)
