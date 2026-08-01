"""Fixed-shape no-cache scoring shared by sampling verification and training."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F

from .rloo_probability import legal_log_probs
from .rloo_scoring import (
    legal_logits_from_hidden,
    unwrap_causal_lm,
)

from . import dapo_contract as contract


_FORCE_DENSE_SCORING: ContextVar[bool] = ContextVar(
    "rloo_dapo_force_dense_scoring", default=False
)


@dataclass(frozen=True)
class DecisionDistribution:
    position: int
    allowed_ids: tuple[int, ...]
    log_probs: torch.Tensor

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("Decision position must be non-negative.")
        if len(self.allowed_ids) <= 1:
            raise ValueError("A decision distribution needs at least two legal IDs.")
        if self.log_probs.shape != (len(self.allowed_ids),):
            raise ValueError("Decision log_probs must match allowed_ids.")


@dataclass(frozen=True)
class DenseCompletionScores:
    log_probs: torch.Tensor
    valid_mask: torch.Tensor
    decision_mask: torch.Tensor
    decisions: tuple[tuple[DecisionDistribution, ...], ...]

    def __post_init__(self) -> None:
        if self.log_probs.ndim != 2:
            raise ValueError("log_probs must have shape (C,T).")
        if self.valid_mask.shape != self.log_probs.shape:
            raise ValueError("valid_mask must match log_probs.")
        if self.decision_mask.shape != self.log_probs.shape:
            raise ValueError("decision_mask must match log_probs.")
        if len(self.decisions) != self.log_probs.shape[0]:
            raise ValueError("decisions must contain one row per completion.")


@dataclass(frozen=True)
class AlignedPromptCache:
    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]
    prompt_prediction: torch.Tensor
    prompt_length: int

    def __post_init__(self) -> None:
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive.")
        if not self.layer_keys or len(self.layer_keys) != len(self.layer_values):
            raise ValueError("Aligned prompt K/V layers are incomplete.")
        if self.prompt_prediction.ndim != 3 or self.prompt_prediction.shape[:2] != (1, 1):
            raise ValueError("prompt_prediction must have shape (1,1,H).")

    @property
    def storage_bytes(self) -> int:
        return sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in zip(self.layer_keys, self.layer_values, strict=True)
        )


def grammar_completion_width(
    grammar: Any, *, safety_limit: int = contract.MAX_COMPLETION_LENGTH
) -> int:
    prefixes = getattr(grammar, "prefix_tokens", None)
    suffix = getattr(grammar, "suffix_tokens", None)
    if not isinstance(prefixes, dict) or not prefixes or not isinstance(suffix, tuple):
        return int(safety_limit)
    width = max(len(tuple(tokens)) + 3 + len(suffix) for tokens in prefixes.values())
    if width <= 0 or width > int(safety_limit):
        raise RuntimeError(
            f"Grammar completion width {width} exceeds safety limit {safety_limit}."
        )
    return width


def _validate_fixed_shape(
    prompt_ids: Sequence[int],
    *,
    batch_rows: int,
    completion_width: int,
) -> list[int]:
    prompt = [int(value) for value in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty.")
    if isinstance(batch_rows, bool) or not 1 <= int(batch_rows) <= contract.LOSS_CHUNK_SIZE:
        raise ValueError("batch_rows must be in 1..8.")
    if isinstance(completion_width, bool) or int(completion_width) <= 0:
        raise ValueError("completion_width must be positive.")
    return prompt


def _fixed_input_rows(
    prompt: Sequence[int],
    completion_rows: Sequence[Sequence[int]],
    *,
    eos_id: int,
    completion_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    for values in completion_rows:
        completion = [int(value) for value in values]
        if len(completion) > completion_width:
            raise ValueError("Completion exceeds the fixed completion width.")
        rows.append(
            list(prompt)
            + completion
            + [int(eos_id)] * (completion_width - len(completion))
        )
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    # Future filler tokens are deliberately visible as query positions.  The
    # causal mask prevents them from affecting any scored predecessor while
    # keeping every scoring call on exactly the same kernel shape.
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def _forward_hidden_no_cache(
    causal_lm: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    backbone = getattr(causal_lm, "model", None)
    if backbone is None:
        backbone = getattr(causal_lm, "transformer", None)
    if backbone is None:
        raise TypeError("Causal LM has no supported transformer backbone.")
    output = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if hidden is None:
        raise RuntimeError("Transformer backbone returned no hidden state.")
    return hidden


def _supports_shared_prefix_forward(causal_lm: torch.nn.Module) -> bool:
    """Return whether the production Qwen3 backbone can share a prompt per layer."""

    if _FORCE_DENSE_SCORING.get() or not torch.cuda.is_available():
        return False
    backbone = getattr(causal_lm, "model", None)
    if backbone is None:
        return False
    return (
        type(backbone).__name__ == "Qwen3Model"
        and type(backbone).__module__.endswith(".modeling_qwen3")
        and hasattr(backbone, "embed_tokens")
        and hasattr(backbone, "rotary_emb")
        and hasattr(backbone, "layers")
        and hasattr(backbone, "norm")
    )


@contextmanager
def dense_scoring_mode() -> Iterator[None]:
    """Force the fixed dense scorer for a reproducible gate-only baseline."""

    token = _FORCE_DENSE_SCORING.set(True)
    try:
        yield
    finally:
        _FORCE_DENSE_SCORING.reset(token)


def _qwen3_project(
    attention: torch.nn.Module,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    shape = (*hidden.shape[:2], -1, int(attention.head_dim))
    query = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
    value = attention.v_proj(hidden).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def _qwen3_window_size(attention: torch.nn.Module) -> tuple[int, int]:
    return (
        (-1, -1)
        if attention.sliding_window is None
        else (int(attention.sliding_window) - 1, 0)
    )


def build_aligned_prompt_cache(
    causal_lm: torch.nn.Module,
    *,
    prompt_input_ids: torch.Tensor,
) -> AlignedPromptCache:
    """Evaluate one prompt and retain its per-layer BF16 K/V representation."""

    from flash_attn import flash_attn_func

    if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("prompt_input_ids must have shape (1,P).")
    if not _supports_shared_prefix_forward(causal_lm):
        raise TypeError("Aligned prompt caching requires the production Qwen3 backbone.")
    backbone = causal_lm.model
    prompt_length = prompt_input_ids.shape[1]
    prompt_hidden = backbone.embed_tokens(prompt_input_ids)
    prompt_positions = torch.arange(
        prompt_length, dtype=torch.long, device=prompt_input_ids.device
    ).unsqueeze(0)
    prompt_rope = backbone.rotary_emb(prompt_hidden, prompt_positions)
    layer_keys: list[torch.Tensor] = []
    layer_values: list[torch.Tensor] = []
    use_checkpoint = bool(
        getattr(backbone, "gradient_checkpointing", False)
        and backbone.training
        and torch.is_grad_enabled()
    )
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint

    for layer in backbone.layers:
        def prompt_layer_forward(
            hidden: torch.Tensor,
            *,
            current_layer: torch.nn.Module = layer,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            residual = hidden
            normalized = current_layer.input_layernorm(hidden)
            attention = current_layer.self_attn
            if float(attention.attention_dropout) != 0.0:
                raise RuntimeError(
                    "Shared-prefix scoring requires attention dropout=0."
                )
            query, key, value = _qwen3_project(
                attention, normalized, prompt_rope
            )
            attention_output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=0.0,
                softmax_scale=float(attention.scaling),
                causal=True,
                window_size=_qwen3_window_size(attention),
                deterministic=True,
            )
            hidden = residual + attention.o_proj(
                attention_output.reshape(1, prompt_length, -1)
            )
            residual = hidden
            hidden = residual + current_layer.mlp(
                current_layer.post_attention_layernorm(hidden)
            )
            return hidden, key, value

        if use_checkpoint:
            prompt_hidden, prompt_key, prompt_value = checkpoint(
                prompt_layer_forward,
                prompt_hidden,
                use_reentrant=False,
            )
        else:
            prompt_hidden, prompt_key, prompt_value = prompt_layer_forward(
                prompt_hidden
            )
        layer_keys.append(prompt_key)
        layer_values.append(prompt_value)

    prompt_hidden = backbone.norm(prompt_hidden)
    return AlignedPromptCache(
        layer_keys=tuple(layer_keys),
        layer_values=tuple(layer_values),
        prompt_prediction=prompt_hidden[:, -1:],
        prompt_length=int(prompt_length),
    )


def aligned_cached_prediction_hidden(
    causal_lm: torch.nn.Module,
    prompt_cache: AlignedPromptCache,
    *,
    completion_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Score fixed completion rows from an aligned prompt cache."""

    from flash_attn import flash_attn_func

    if completion_input_ids.ndim != 2 or completion_input_ids.shape[0] <= 0:
        raise ValueError("completion_input_ids must have shape (C,T).")
    backbone = causal_lm.model
    completion_rows, completion_width = completion_input_ids.shape
    completion_hidden = backbone.embed_tokens(completion_input_ids)
    completion_positions = torch.arange(
        prompt_cache.prompt_length,
        prompt_cache.prompt_length + completion_width,
        dtype=torch.long,
        device=completion_input_ids.device,
    ).unsqueeze(0).expand(completion_rows, -1)
    completion_rope = backbone.rotary_emb(
        completion_hidden, completion_positions
    )
    use_checkpoint = bool(
        getattr(backbone, "gradient_checkpointing", False)
        and backbone.training
        and torch.is_grad_enabled()
    )
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint

    for layer_index, layer in enumerate(backbone.layers):
        prompt_key = prompt_cache.layer_keys[layer_index]
        prompt_value = prompt_cache.layer_values[layer_index]

        def completion_layer_forward(
            hidden: torch.Tensor,
            cached_key: torch.Tensor,
            cached_value: torch.Tensor,
            *,
            current_layer: torch.nn.Module = layer,
        ) -> torch.Tensor:
            residual = hidden
            normalized = current_layer.input_layernorm(hidden)
            attention = current_layer.self_attn
            query, key, value = _qwen3_project(
                attention, normalized, completion_rope
            )
            branch_keys = torch.cat(
                [cached_key.expand(completion_rows, -1, -1, -1), key], dim=1
            )
            branch_values = torch.cat(
                [cached_value.expand(completion_rows, -1, -1, -1), value], dim=1
            )
            attention_output = flash_attn_func(
                query,
                branch_keys,
                branch_values,
                dropout_p=0.0,
                softmax_scale=float(attention.scaling),
                causal=True,
                window_size=_qwen3_window_size(attention),
                deterministic=True,
            )
            hidden = residual + attention.o_proj(
                attention_output.reshape(
                    completion_rows, completion_width, -1
                )
            )
            residual = hidden
            return residual + current_layer.mlp(
                current_layer.post_attention_layernorm(hidden)
            )

        if use_checkpoint:
            completion_hidden = checkpoint(
                completion_layer_forward,
                completion_hidden,
                prompt_key,
                prompt_value,
                use_reentrant=False,
            )
        else:
            completion_hidden = completion_layer_forward(
                completion_hidden, prompt_key, prompt_value
            )

    completion_hidden = backbone.norm(completion_hidden)
    return torch.cat(
        [
            prompt_cache.prompt_prediction.expand(completion_rows, -1, -1),
            completion_hidden[:, :-1],
        ],
        dim=1,
    )


def _shared_prefix_prediction_hidden(
    causal_lm: torch.nn.Module,
    *,
    prompt_input_ids: torch.Tensor,
    completion_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the differentiable no-persistent-cache scorer used by replay."""

    prompt_cache = build_aligned_prompt_cache(
        causal_lm, prompt_input_ids=prompt_input_ids
    )
    return aligned_cached_prediction_hidden(
        causal_lm,
        prompt_cache,
        completion_input_ids=completion_input_ids,
    )


def _prediction_hidden_no_cache(
    causal_lm: torch.nn.Module,
    prompt: Sequence[int],
    completion_rows: Sequence[Sequence[int]],
    *,
    eos_id: int,
    completion_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return shape (C,T,H), where column t predicts completion token t."""

    input_ids, attention_mask = _fixed_input_rows(
        prompt,
        completion_rows,
        eos_id=eos_id,
        completion_width=completion_width,
        device=device,
    )
    if not _supports_shared_prefix_forward(causal_lm):
        hidden = _forward_hidden_no_cache(
            causal_lm,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        start = len(prompt) - 1
        return hidden[:, start : start + int(completion_width)]

    prompt_input_ids = torch.tensor(
        [list(prompt)], dtype=torch.long, device=device
    )
    completion_input_ids = input_ids[:, len(prompt) :]
    return _shared_prefix_prediction_hidden(
        causal_lm,
        prompt_input_ids=prompt_input_ids,
        completion_input_ids=completion_input_ids,
    )


def _validate_completion(grammar: Any, completion: Sequence[int]) -> None:
    prefix: list[int] = []
    eos_id = int(grammar.eos_token_id)
    for position, raw_token in enumerate(completion):
        token = int(raw_token)
        allowed = [int(value) for value in grammar.allowed_next(prefix)]
        if token not in allowed:
            raise ValueError(f"Completion token {position} is illegal.")
        prefix.append(token)
        if token == eos_id and position != len(completion) - 1:
            raise ValueError("Completion contains tokens after grammar EOS.")
    if not prefix or prefix[-1] != eos_id:
        raise ValueError("Completion must terminate with grammar EOS.")
    grammar.parse(prefix)


def score_completions_dense_fixed(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[Sequence[int]],
    grammar: Any,
    *,
    device: str | torch.device,
    temperature: float = contract.TEMPERATURE,
    batch_rows: int = contract.LOSS_CHUNK_SIZE,
    completion_width: int = contract.MAX_COMPLETION_LENGTH,
) -> DenseCompletionScores:
    """Score every completion decision in one fixed-shape no-cache forward."""

    prompt = _validate_fixed_shape(
        prompt_ids, batch_rows=batch_rows, completion_width=completion_width
    )
    completions = [tuple(int(value) for value in row) for row in completion_ids]
    if len(completions) != int(batch_rows):
        raise ValueError(f"Exactly {int(batch_rows)} completion rows are required.")
    if any(not row for row in completions):
        raise ValueError("Every completion row must be non-empty.")
    for completion in completions:
        _validate_completion(grammar, completion)

    torch_device = torch.device(device)
    causal_lm = unwrap_causal_lm(model)
    hidden = _prediction_hidden_no_cache(
        causal_lm,
        prompt,
        completions,
        eos_id=int(grammar.eos_token_id),
        completion_width=int(completion_width),
        device=torch_device,
    )

    score_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    decision_rows: list[torch.Tensor] = []
    all_decisions: list[tuple[DecisionDistribution, ...]] = []
    for row_index, completion in enumerate(completions):
        prefix: list[int] = []
        values: list[torch.Tensor] = []
        decisions: list[bool] = []
        distributions: list[DecisionDistribution] = []
        for position, token_id in enumerate(completion):
            allowed = tuple(int(value) for value in grammar.allowed_next(prefix))
            if len(allowed) == 1:
                value = torch.zeros((), dtype=torch.float32, device=torch_device)
                decision = False
            else:
                logits = legal_logits_from_hidden(
                    causal_lm, hidden[row_index, position], allowed
                )
                log_probs = legal_log_probs(
                    logits,
                    list(range(len(allowed))),
                    temperature=float(temperature),
                )
                local_index = allowed.index(int(token_id))
                value = log_probs[local_index]
                decision = True
                distributions.append(
                    DecisionDistribution(
                        position=position,
                        allowed_ids=allowed,
                        log_probs=log_probs,
                    )
                )
            values.append(value)
            decisions.append(decision)
            prefix.append(int(token_id))

        padding = int(completion_width) - len(values)
        score_rows.append(F.pad(torch.stack(values), (0, padding), value=0.0))
        valid_rows.append(
            torch.tensor(
                [True] * len(values) + [False] * padding,
                dtype=torch.bool,
                device=torch_device,
            )
        )
        decision_rows.append(
            torch.tensor(
                decisions + [False] * padding,
                dtype=torch.bool,
                device=torch_device,
            )
        )
        all_decisions.append(tuple(distributions))

    return DenseCompletionScores(
        log_probs=torch.stack(score_rows),
        valid_mask=torch.stack(valid_rows),
        decision_mask=torch.stack(decision_rows),
        decisions=tuple(all_decisions),
    )


def score_prefix_distribution_fixed(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
    grammar: Any,
    *,
    device: str | torch.device,
    temperature: float = contract.TEMPERATURE,
    batch_rows: int = contract.LOSS_CHUNK_SIZE,
    completion_width: int = contract.MAX_COMPLETION_LENGTH,
) -> DecisionDistribution:
    """Return the target distribution at one prefix using the training shape."""

    return score_prefix_distributions_fixed(
        model,
        prompt_ids,
        [prefix_ids],
        grammar,
        device=device,
        temperature=temperature,
        batch_rows=batch_rows,
        completion_width=completion_width,
    )[0]


def score_prefix_distributions_fixed(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    prefixes_ids: Sequence[Sequence[int]],
    grammar: Any,
    *,
    device: str | torch.device,
    temperature: float = contract.TEMPERATURE,
    batch_rows: int = contract.LOSS_CHUNK_SIZE,
    completion_width: int = contract.MAX_COMPLETION_LENGTH,
) -> tuple[DecisionDistribution, ...]:
    """Score up to eight decision prefixes in one fixed eight-row forward."""

    prompt = _validate_fixed_shape(
        prompt_ids, batch_rows=batch_rows, completion_width=completion_width
    )
    prefixes = [[int(value) for value in row] for row in prefixes_ids]
    if not 1 <= len(prefixes) <= int(batch_rows):
        raise ValueError("prefixes_ids must contain one to batch_rows prefixes.")
    allowed_rows: list[tuple[int, ...]] = []
    for prefix_index, prefix in enumerate(prefixes):
        if len(prefix) >= int(completion_width):
            raise ValueError("Prefix leaves no room for another completion token.")
        checked: list[int] = []
        for position, token in enumerate(prefix):
            allowed = [int(value) for value in grammar.allowed_next(checked)]
            if token not in allowed:
                raise ValueError(
                    f"Prefix {prefix_index} token {position} is illegal."
                )
            if token == int(grammar.eos_token_id):
                raise ValueError("Cannot score a prefix that already contains EOS.")
            checked.append(token)
        allowed_ids = tuple(int(value) for value in grammar.allowed_next(prefix))
        if len(allowed_ids) <= 1:
            raise ValueError("Every prefix must end at a decision node.")
        allowed_rows.append(allowed_ids)

    torch_device = torch.device(device)
    causal_lm = unwrap_causal_lm(model)
    padded_prefixes = prefixes + [prefixes[0]] * (int(batch_rows) - len(prefixes))
    hidden = _prediction_hidden_no_cache(
        causal_lm,
        prompt,
        padded_prefixes,
        eos_id=int(grammar.eos_token_id),
        completion_width=int(completion_width),
        device=torch_device,
    )
    result: list[DecisionDistribution] = []
    for row_index, (prefix, allowed_ids) in enumerate(
        zip(prefixes, allowed_rows, strict=True)
    ):
        logits = legal_logits_from_hidden(
            causal_lm, hidden[row_index, len(prefix)], allowed_ids
        )
        result.append(
            DecisionDistribution(
                position=len(prefix),
                allowed_ids=allowed_ids,
                log_probs=legal_log_probs(
                    logits,
                    list(range(len(allowed_ids))),
                    temperature=float(temperature),
                ),
            )
        )
    return tuple(result)


__all__ = [
    "DecisionDistribution",
    "DenseCompletionScores",
    "grammar_completion_width",
    "score_completions_dense_fixed",
    "score_prefix_distribution_fixed",
    "score_prefix_distributions_fixed",
]
