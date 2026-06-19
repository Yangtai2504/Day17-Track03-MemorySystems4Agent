from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Support DASHSCOPE env vars (OpenAI-compatible) or generic CUSTOM_* vars
    api_key = (
        os.getenv("CUSTOM_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("CUSTOM_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
    )
    model_name = os.getenv("MODEL_NAME", "qwen-max")

    # Determine provider from env, defaulting to "custom" when a base_url is present
    raw_provider = os.getenv("LLM_PROVIDER", "")
    if not raw_provider:
        if base_url:
            raw_provider = "custom"
        elif os.getenv("ANTHROPIC_API_KEY"):
            raw_provider = "anthropic"
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        elif os.getenv("GEMINI_API_KEY"):
            raw_provider = "gemini"
            api_key = api_key or os.getenv("GEMINI_API_KEY")
        else:
            raw_provider = "openai"

    main_model = ProviderConfig(
        provider=raw_provider,
        model_name=model_name,
        temperature=0.7,
        api_key=api_key,
        base_url=base_url,
    )

    judge_model_name = os.getenv("JUDGE_MODEL_NAME", model_name)
    judge_model = ProviderConfig(
        provider=raw_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=main_model,
        judge_model=judge_model,
    )
