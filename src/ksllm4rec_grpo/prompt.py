"""Authoritative prompt rendering and tokenization for GRPO V3.1."""

from __future__ import annotations

from typing import Any

from ksllm4rec_sft.data import SourceRecord, render_qwen3_nothink


def render_prompt_text(system: str, prompt: str) -> str:
    """Reuse the exact single-turn qwen3_nothink SFT source layout."""

    source, _ = render_qwen3_nothink(SourceRecord(system, prompt, ""))
    return source


def encode_prompt(
    tokenizer: Any,
    system: str,
    prompt: str,
    *,
    cutoff_len: int,
) -> list[int]:
    """Tokenize without truncation and fail if the approved cutoff is exceeded."""

    token_ids = [
        int(value)
        for value in tokenizer.encode(
            render_prompt_text(system, prompt), add_special_tokens=False
        )
    ]
    if not token_ids:
        raise ValueError("Rendered prompt produced no tokens.")
    if len(token_ids) > cutoff_len:
        raise ValueError(
            f"Prompt length {len(token_ids)} exceeds cutoff_len={cutoff_len}; "
            "silent truncation is forbidden."
        )
    return token_ids
