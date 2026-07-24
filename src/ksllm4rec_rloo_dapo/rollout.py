"""Multi-prompt constrained rollout with prompt KV-cache reuse."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterator, Sequence

import torch

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.probability import legal_log_probs
from ksllm4rec_rloo.rollout import rollout_group, rollout_seed
from ksllm4rec_rloo.scoring import (
    legal_logits_from_hidden,
    unwrap_causal_lm,
)

from . import contract
from .kv_cache import (
    KVCacheGeometry,
    estimate_kv_cache_bytes,
    fork_dynamic_cache,
    resolve_active_sequences,
)
from .scoring import (
    aligned_cached_prediction_hidden,
    build_aligned_prompt_cache,
    DecisionDistribution,
    grammar_completion_width,
    _supports_shared_prefix_forward,
    score_completions_dense_fixed,
    score_prefix_distributions_fixed,
)
from .speculative import speculative_accept_or_residual


@dataclass(frozen=True)
class PromptRequest:
    group_id: str
    prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class ProposalDecision:
    position: int
    allowed_ids: tuple[int, ...]
    log_probs: tuple[float, ...]


@dataclass(frozen=True)
class SampledCandidate:
    candidate_index: int
    sid: Sid
    token_ids: tuple[int, ...]
    sample_log_probs: tuple[float, ...]
    decision_mask: tuple[bool, ...]
    proposal_decisions: tuple[ProposalDecision, ...] = ()


@dataclass(frozen=True)
class CanonicalCandidate:
    candidate_index: int
    sid: Sid
    token_ids: tuple[int, ...]
    sample_log_probs: tuple[float, ...]
    old_log_probs: tuple[float, ...]
    decision_mask: tuple[bool, ...]


@dataclass(frozen=True)
class PromptRollout:
    request: PromptRequest
    candidates: tuple[SampledCandidate, ...]
    policy_aligned: bool = False


@dataclass(frozen=True)
class CanonicalPromptRollout:
    request: PromptRequest
    candidates: tuple[CanonicalCandidate, ...]
    max_sample_canonical_logp_difference: float
    max_proposal_canonical_logp_difference: float
    corrected_candidate_count: int
    target_forward_calls: int


@dataclass(frozen=True)
class CacheRolloutStats:
    prompt_count: int
    rollout_count: int
    prefill_calls: int
    decode_calls: int
    cache_forks: int
    requested_active_sequences: int
    resolved_active_sequences: int
    estimated_peak_cache_bytes: int
    fallback_prompt_count: int


@dataclass(frozen=True)
class ForcedSpan:
    token_ids: tuple[int, ...]
    finished: bool


@dataclass
class _CandidateState:
    prompt_index: int
    group_id: str
    prompt_length: int
    candidate_index: int
    generator: torch.Generator
    tokens: list[int]
    logps: list[float]
    decisions: list[bool]
    proposal_decisions: list[ProposalDecision]


@dataclass
class _TargetSuffixState:
    candidate_index: int
    generator: torch.Generator
    tokens: list[int]
    logps: list[float]
    decisions: list[bool]


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def candidate_wave_pairs(
    prompt_count: int,
    *,
    candidates_per_prompt: int = contract.GROUP_SIZE,
    candidates_per_wave: int = contract.CANDIDATES_PER_PROMPT_WAVE,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive.")
    if candidates_per_prompt <= 0 or candidates_per_wave <= 0:
        raise ValueError("candidate counts must be positive.")
    if candidates_per_prompt % candidates_per_wave:
        raise ValueError("candidates_per_prompt must divide by candidates_per_wave.")
    return tuple(
        tuple(
            (prompt_index, candidate_index)
            for prompt_index in range(prompt_count)
            for candidate_index in range(start, start + candidates_per_wave)
        )
        for start in range(0, candidates_per_prompt, candidates_per_wave)
    )


def collect_forced_span(
    grammar: Any,
    prefix_ids: Sequence[int],
    *,
    maximum_tokens: int,
) -> ForcedSpan:
    """Collect consecutive singleton grammar tokens up to a branch or EOS."""

    if isinstance(maximum_tokens, bool) or int(maximum_tokens) < 0:
        raise ValueError("maximum_tokens must be a non-negative integer.")
    prefix = [int(value) for value in prefix_ids]
    span: list[int] = []
    eos_id = int(grammar.eos_token_id)
    while True:
        allowed = [int(value) for value in grammar.allowed_next(prefix)]
        if not allowed:
            raise RuntimeError("Grammar ended without emitting EOS.")
        if len(allowed) != 1:
            return ForcedSpan(tuple(span), False)
        if len(span) >= int(maximum_tokens):
            raise RuntimeError("Forced grammar span exceeds max_completion_length.")
        token_id = allowed[0]
        prefix.append(token_id)
        span.append(token_id)
        if token_id == eos_id:
            grammar.parse(prefix)
            return ForcedSpan(tuple(span), True)


def _backbone(causal_lm: torch.nn.Module) -> torch.nn.Module:
    value = getattr(causal_lm, "model", None)
    if value is None:
        value = getattr(causal_lm, "transformer", None)
    if value is None:
        raise TypeError("Causal LM has no supported transformer backbone.")
    return value


def _cached_forward(
    backbone: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Any | None,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    output = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    cache = getattr(output, "past_key_values", None)
    if hidden is None or cache is None:
        raise RuntimeError("Cached backbone forward returned no hidden state or cache.")
    return hidden, cache


@contextmanager
def cached_rollout_mode(model: torch.nn.Module) -> Iterator[None]:
    was_training = bool(model.training)
    causal_lm = unwrap_causal_lm(model)
    previous_use_cache = getattr(causal_lm.config, "use_cache", None)
    model.eval()
    causal_lm.config.use_cache = True
    try:
        with torch.inference_mode():
            yield
    finally:
        if previous_use_cache is not None:
            causal_lm.config.use_cache = previous_use_cache
        model.train(was_training)


@contextmanager
def canonical_scoring_mode(model: torch.nn.Module) -> Iterator[None]:
    was_training = bool(model.training)
    causal_lm = unwrap_causal_lm(model)
    previous_use_cache = getattr(causal_lm.config, "use_cache", None)
    model.eval()
    causal_lm.config.use_cache = False
    try:
        with torch.inference_mode():
            yield
    finally:
        if previous_use_cache is not None:
            causal_lm.config.use_cache = previous_use_cache
        model.train(was_training)


def _left_padded_prompts(
    requests: Sequence[PromptRequest], *, eos_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(request.prompt_ids) for request in requests]
    if any(length <= 0 for length in lengths):
        raise ValueError("Every prompt must contain at least one token.")
    maximum = max(lengths)
    rows = [
        [eos_id] * (maximum - len(request.prompt_ids)) + list(request.prompt_ids)
        for request in requests
    ]
    masks = [
        [0] * (maximum - len(request.prompt_ids)) + [1] * len(request.prompt_ids)
        for request in requests
    ]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids, lengths


def _advance_candidate_state(
    causal_lm: torch.nn.Module,
    prediction_hidden: torch.Tensor,
    state: _CandidateState,
    grammar: Any,
    *,
    maximum_length: int,
    temperature: float,
) -> ForcedSpan:
    if len(state.tokens) >= int(maximum_length):
        raise RuntimeError(
            f"Cached rollout exceeded max_completion_length={maximum_length}."
        )
    allowed = tuple(int(value) for value in grammar.allowed_next(state.tokens))
    if not allowed:
        raise RuntimeError("Grammar ended without emitting EOS.")
    appended: list[int] = []
    finished = False
    if len(allowed) > 1:
        legal_logits = legal_logits_from_hidden(causal_lm, prediction_hidden, allowed)
        log_probs = legal_log_probs(
            legal_logits,
            list(range(len(allowed))),
            temperature=float(temperature),
        )
        local_index = int(
            torch.multinomial(
                log_probs.exp(), 1, generator=state.generator
            ).item()
        )
        token_id = allowed[local_index]
        position = len(state.tokens)
        state.proposal_decisions.append(
            ProposalDecision(
                position=position,
                allowed_ids=allowed,
                log_probs=tuple(
                    float(value) for value in log_probs.detach().float().cpu().tolist()
                ),
            )
        )
        state.tokens.append(token_id)
        state.logps.append(float(log_probs[local_index].detach().float().item()))
        state.decisions.append(True)
        appended.append(token_id)
        finished = token_id == int(grammar.eos_token_id)

    if not finished:
        remaining = int(maximum_length) - len(state.tokens)
        span = collect_forced_span(
            grammar, state.tokens, maximum_tokens=remaining
        )
        state.tokens.extend(span.token_ids)
        state.logps.extend([0.0] * len(span.token_ids))
        state.decisions.extend([False] * len(span.token_ids))
        appended.extend(span.token_ids)
        finished = span.finished
    if not appended:
        raise RuntimeError("Candidate advance produced no token.")
    if finished:
        grammar.parse(state.tokens)
    return ForcedSpan(tuple(appended), finished)


def _complete_cache_rows(
    causal_lm: torch.nn.Module,
    backbone: torch.nn.Module,
    cache: Any,
    prediction_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    states: list[_CandidateState],
    grammar: Any,
    *,
    maximum_length: int,
    temperature: float,
    device: torch.device,
) -> tuple[list[_CandidateState], int, int]:
    finished: list[_CandidateState] = []
    decode_calls = 0
    while states:
        active_rows: list[int] = []
        active_spans: list[tuple[int, ...]] = []
        for row_index, state in enumerate(states):
            span = _advance_candidate_state(
                causal_lm,
                prediction_hidden[row_index, -1],
                state,
                grammar,
                maximum_length=maximum_length,
                temperature=temperature,
            )
            if span.finished:
                finished.append(state)
            else:
                active_rows.append(row_index)
                active_spans.append(span.token_ids)
        if not active_rows:
            break

        row_selector = torch.tensor(active_rows, dtype=torch.long, device=device)
        cache.batch_select_indices(row_selector)
        states = [states[index] for index in active_rows]
        attention_mask = attention_mask.index_select(0, row_selector)
        maximum_advance = max(len(tokens) for tokens in active_spans)
        eos_id = int(grammar.eos_token_id)
        input_ids = torch.tensor(
            [
                list(tokens) + [eos_id] * (maximum_advance - len(tokens))
                for tokens in active_spans
            ],
            dtype=torch.long,
            device=device,
        )
        appended_mask = torch.tensor(
            [
                [1] * len(tokens) + [0] * (maximum_advance - len(tokens))
                for tokens in active_spans
            ],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.cat([attention_mask, appended_mask], dim=1)
        position_ids = torch.zeros(
            (len(states), maximum_advance), dtype=torch.long, device=device
        )
        for row_index, (state, tokens) in enumerate(zip(states, active_spans, strict=True)):
            start = state.prompt_length + len(state.tokens) - len(tokens)
            position_ids[row_index, : len(tokens)] = torch.arange(
                start, start + len(tokens), dtype=torch.long, device=device
            )
        cache_start = int(cache.get_seq_length())
        hidden, cache = _cached_forward(
            backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=torch.arange(
                cache_start,
                cache_start + maximum_advance,
                dtype=torch.long,
                device=device,
            ),
        )
        decode_calls += 1
        gather = torch.tensor(
            [len(tokens) - 1 for tokens in active_spans],
            dtype=torch.long,
            device=device,
        )
        prediction_hidden = hidden[
            torch.arange(len(states), dtype=torch.long, device=device), gather
        ].unsqueeze(1)
    return finished, decode_calls, 0


def _sample_cache_chunk_complete(
    causal_lm: torch.nn.Module,
    backbone: torch.nn.Module,
    base_cache: Any,
    base_hidden: torch.Tensor,
    base_attention_mask: torch.Tensor,
    prompt_lengths: Sequence[int],
    requests: Sequence[PromptRequest],
    pairs: Sequence[tuple[int, int]],
    grammar: Any,
    *,
    sampling_nonce: int,
    maximum_length: int,
    temperature: float,
    device: torch.device,
) -> tuple[list[_CandidateState], int, int]:
    cache_indices = [prompt_index for prompt_index, _ in pairs]
    cache = fork_dynamic_cache(base_cache, cache_indices)
    selector = torch.tensor(cache_indices, dtype=torch.long, device=device)
    prediction_hidden = base_hidden.index_select(0, selector)[:, -1:, :]
    attention_mask = base_attention_mask.index_select(0, selector)
    states = [
        _CandidateState(
            prompt_index=prompt_index,
            group_id=requests[prompt_index].group_id,
            prompt_length=int(prompt_lengths[prompt_index]),
            candidate_index=candidate_index,
            generator=_device_generator(
                device,
                rollout_seed(requests[prompt_index].group_id, sampling_nonce, candidate_index),
            ),
            tokens=[],
            logps=[],
            decisions=[],
            proposal_decisions=[],
        )
        for prompt_index, candidate_index in pairs
    ]
    finished_states, decode_calls, split_forks = _complete_cache_rows(
        causal_lm,
        backbone,
        cache,
        prediction_hidden,
        attention_mask,
        states,
        grammar,
        maximum_length=maximum_length,
        temperature=temperature,
        device=device,
    )
    if len(finished_states) != len(pairs):
        raise RuntimeError("Cached rollout did not finish every candidate.")
    return finished_states, decode_calls, 1 + split_forks


def _sample_aligned_prompt(
    causal_lm: torch.nn.Module,
    request: PromptRequest,
    grammar: Any,
    *,
    sampling_nonce: int,
    maximum_length: int,
    temperature: float,
    device: torch.device,
) -> tuple[PromptRollout, int, int]:
    """Sample canonical policy rows from one reusable aligned prompt cache."""

    scoring_width = grammar_completion_width(
        grammar, safety_limit=int(maximum_length)
    )
    prompt_tensor = torch.tensor(
        [list(request.prompt_ids)], dtype=torch.long, device=device
    )
    prompt_cache = build_aligned_prompt_cache(
        causal_lm, prompt_input_ids=prompt_tensor
    )
    cache_bytes = prompt_cache.storage_bytes
    states = [
        _CandidateState(
            prompt_index=0,
            group_id=request.group_id,
            prompt_length=len(request.prompt_ids),
            candidate_index=candidate_index,
            generator=_device_generator(
                device,
                rollout_seed(request.group_id, sampling_nonce, candidate_index),
            ),
            tokens=[],
            logps=[],
            decisions=[],
            proposal_decisions=[],
        )
        for candidate_index in range(contract.GROUP_SIZE)
    ]
    active = states
    finished: list[_CandidateState] = []
    completion_forwards = 0
    eos_id = int(grammar.eos_token_id)

    while active:
        decision_states: list[_CandidateState] = []
        for state in active:
            remaining = int(maximum_length) - len(state.tokens)
            span = collect_forced_span(
                grammar, state.tokens, maximum_tokens=remaining
            )
            state.tokens.extend(span.token_ids)
            state.logps.extend([0.0] * len(span.token_ids))
            state.decisions.extend([False] * len(span.token_ids))
            if span.finished:
                finished.append(state)
            else:
                if len(state.tokens) >= scoring_width:
                    raise RuntimeError(
                        "Aligned cached prefix leaves no decision-token position."
                    )
                decision_states.append(state)
        if not decision_states:
            break

        next_active: list[_CandidateState] = []
        for start in range(0, len(decision_states), contract.LOSS_CHUNK_SIZE):
            chunk = decision_states[start : start + contract.LOSS_CHUNK_SIZE]
            padded = chunk + [chunk[0]] * (contract.LOSS_CHUNK_SIZE - len(chunk))
            completion_ids = torch.tensor(
                [
                    state.tokens
                    + [eos_id] * (scoring_width - len(state.tokens))
                    for state in padded
                ],
                dtype=torch.long,
                device=device,
            )
            prediction_hidden = aligned_cached_prediction_hidden(
                causal_lm,
                prompt_cache,
                completion_input_ids=completion_ids,
            )
            completion_forwards += 1
            for row_index, state in enumerate(chunk):
                position = len(state.tokens)
                allowed = tuple(
                    int(value) for value in grammar.allowed_next(state.tokens)
                )
                if len(allowed) <= 1:
                    raise RuntimeError(
                        "Aligned cached sampler expected a decision node."
                    )
                logits = legal_logits_from_hidden(
                    causal_lm, prediction_hidden[row_index, position], allowed
                )
                distribution = legal_log_probs(
                    logits,
                    list(range(len(allowed))),
                    temperature=float(temperature),
                )
                local_index = int(
                    torch.multinomial(
                        distribution.exp(), 1, generator=state.generator
                    ).item()
                )
                token_id = allowed[local_index]
                state.proposal_decisions.append(
                    ProposalDecision(
                        position=position,
                        allowed_ids=allowed,
                        log_probs=tuple(
                            float(value)
                            for value in distribution.detach().float().cpu().tolist()
                        ),
                    )
                )
                state.tokens.append(token_id)
                state.logps.append(
                    float(distribution[local_index].detach().float().item())
                )
                state.decisions.append(True)
                if token_id == eos_id:
                    grammar.parse(state.tokens)
                    finished.append(state)
                else:
                    next_active.append(state)
        active = next_active

    finished.sort(key=lambda state: state.candidate_index)
    if [state.candidate_index for state in finished] != list(range(contract.GROUP_SIZE)):
        raise RuntimeError("Aligned cached sampling lost or duplicated a candidate.")
    candidates = tuple(
        SampledCandidate(
            candidate_index=state.candidate_index,
            sid=grammar.parse(state.tokens),
            token_ids=tuple(state.tokens),
            sample_log_probs=tuple(state.logps),
            decision_mask=tuple(state.decisions),
            proposal_decisions=tuple(state.proposal_decisions),
        )
        for state in finished
    )
    return (
        PromptRollout(
            request=request,
            candidates=candidates,
            policy_aligned=True,
        ),
        completion_forwards,
        cache_bytes,
    )


def _rollout_aligned_prompt_batch(
    model: torch.nn.Module,
    requests: Sequence[PromptRequest],
    grammar: Any,
    *,
    sampling_nonce: int,
    maximum_length: int,
    temperature: float,
    requested_active: int,
    resolved_active: int,
    device: torch.device,
) -> tuple[tuple[PromptRollout, ...], CacheRolloutStats]:
    causal_lm = unwrap_causal_lm(model)
    outputs: list[PromptRollout] = []
    completion_forwards = 0
    peak_cache_bytes = 0
    with cached_rollout_mode(model):
        for request in requests:
            output, calls, cache_bytes = _sample_aligned_prompt(
                causal_lm,
                request,
                grammar,
                sampling_nonce=sampling_nonce,
                maximum_length=maximum_length,
                temperature=temperature,
                device=device,
            )
            outputs.append(output)
            completion_forwards += calls
            peak_cache_bytes = max(peak_cache_bytes, cache_bytes)
    return tuple(outputs), CacheRolloutStats(
        prompt_count=len(requests),
        rollout_count=len(requests) * contract.GROUP_SIZE,
        prefill_calls=len(requests),
        decode_calls=completion_forwards,
        cache_forks=0,
        requested_active_sequences=requested_active,
        resolved_active_sequences=resolved_active,
        estimated_peak_cache_bytes=peak_cache_bytes,
        fallback_prompt_count=0,
    )


def rollout_prompt_batch(
    model: torch.nn.Module,
    requests: Sequence[PromptRequest],
    grammar: Any,
    *,
    sampling_nonce: int,
    config: dict[str, Any],
    device: str | torch.device,
    aligned_cache: bool = True,
) -> tuple[tuple[PromptRollout, ...], CacheRolloutStats]:
    prompt_requests = tuple(requests)
    if not 1 <= len(prompt_requests) <= contract.PROMPT_BATCH_SIZE:
        raise ValueError("A prompt batch must contain one to eight prompts.")
    if len({request.group_id for request in prompt_requests}) != len(prompt_requests):
        raise ValueError("A prompt batch cannot contain duplicate group IDs.")
    torch_device = torch.device(device)
    causal_lm = unwrap_causal_lm(model)
    backbone = _backbone(causal_lm)
    eos_id = int(grammar.eos_token_id)
    input_ids, attention_mask, position_ids, prompt_lengths = _left_padded_prompts(
        prompt_requests, eos_id=eos_id, device=torch_device
    )
    sequence_length = max(prompt_lengths) + int(config["rollout"]["max_completion_length"])
    geometry = KVCacheGeometry(
        int(config["memory"]["model_num_layers"]),
        int(config["memory"]["model_num_kv_heads"]),
        int(config["memory"]["model_head_dim"]),
        int(config["memory"]["cache_dtype_bytes"]),
    )
    requested_active = int(config["rollout"]["max_active_sequences"])
    active_limit = resolve_active_sequences(
        geometry,
        base_rows=len(prompt_requests),
        requested_active_rows=requested_active,
        sequence_length=sequence_length,
        budget_gib=float(config["rollout"]["kv_cache_budget_gib"]),
    )
    estimated = estimate_kv_cache_bytes(
        geometry,
        base_rows=len(prompt_requests),
        active_rows=active_limit,
        sequence_length=sequence_length,
    )

    if (
        bool(aligned_cache)
        and
        active_limit >= contract.LOSS_CHUNK_SIZE
        and _supports_shared_prefix_forward(causal_lm)
    ):
        return _rollout_aligned_prompt_batch(
            model,
            prompt_requests,
            grammar,
            sampling_nonce=sampling_nonce,
            maximum_length=int(config["rollout"]["max_completion_length"]),
            temperature=float(config["rollout"]["temperature"]),
            requested_active=requested_active,
            resolved_active=active_limit,
            device=torch_device,
        )

    if active_limit == 0:
        outputs: list[PromptRollout] = []
        for request in prompt_requests:
            candidates = rollout_group(
                model,
                request.prompt_ids,
                grammar,
                group_id=request.group_id,
                epoch_index=sampling_nonce,
                chunk_size=contract.LOSS_CHUNK_SIZE,
                num_candidates=contract.GROUP_SIZE,
                max_completion_length=int(config["rollout"]["max_completion_length"]),
                device=torch_device,
                temperature=float(config["rollout"]["temperature"]),
            )
            outputs.append(
                PromptRollout(
                    request=request,
                    candidates=tuple(
                        SampledCandidate(
                            candidate_index=item.candidate_index,
                            sid=item.sid,
                            token_ids=item.token_ids,
                            sample_log_probs=item.old_log_probs,
                            decision_mask=item.decision_mask,
                            proposal_decisions=(),
                        )
                        for item in candidates
                    ),
                )
            )
        return tuple(outputs), CacheRolloutStats(
            prompt_count=len(prompt_requests),
            rollout_count=len(prompt_requests) * contract.GROUP_SIZE,
            prefill_calls=0,
            decode_calls=0,
            cache_forks=0,
            requested_active_sequences=requested_active,
            resolved_active_sequences=0,
            estimated_peak_cache_bytes=estimated,
            fallback_prompt_count=len(prompt_requests),
        )

    by_prompt: list[list[SampledCandidate]] = [[] for _ in prompt_requests]
    decode_calls = 0
    cache_forks = 0
    with cached_rollout_mode(model):
        base_hidden, base_cache = _cached_forward(
            backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=torch.arange(input_ids.shape[1], device=torch_device),
        )
        for wave in candidate_wave_pairs(len(prompt_requests)):
            for start in range(0, len(wave), active_limit):
                pairs = wave[start : start + active_limit]
                states, calls, forks = _sample_cache_chunk_complete(
                    causal_lm,
                    backbone,
                    base_cache,
                    base_hidden,
                    attention_mask,
                    prompt_lengths,
                    prompt_requests,
                    pairs,
                    grammar,
                    sampling_nonce=sampling_nonce,
                    maximum_length=int(config["rollout"]["max_completion_length"]),
                    temperature=float(config["rollout"]["temperature"]),
                    device=torch_device,
                )
                cache_forks += forks
                decode_calls += calls
                for state in states:
                    by_prompt[state.prompt_index].append(
                        SampledCandidate(
                            candidate_index=state.candidate_index,
                            sid=grammar.parse(state.tokens),
                            token_ids=tuple(state.tokens),
                            sample_log_probs=tuple(state.logps),
                            decision_mask=tuple(state.decisions),
                            proposal_decisions=tuple(state.proposal_decisions),
                        )
                    )
        del base_cache

    outputs = []
    for request, candidates in zip(prompt_requests, by_prompt, strict=True):
        candidates.sort(key=lambda item: item.candidate_index)
        if [item.candidate_index for item in candidates] != list(range(contract.GROUP_SIZE)):
            raise RuntimeError("Cached rollout did not return candidate indices 0..15.")
        outputs.append(PromptRollout(request=request, candidates=tuple(candidates)))
    return tuple(outputs), CacheRolloutStats(
        prompt_count=len(prompt_requests),
        rollout_count=len(prompt_requests) * contract.GROUP_SIZE,
        prefill_calls=1,
        decode_calls=decode_calls,
        cache_forks=cache_forks,
        requested_active_sequences=requested_active,
        resolved_active_sequences=active_limit,
        estimated_peak_cache_bytes=estimated,
        fallback_prompt_count=0,
    )


def correction_seed(group_id: str, sampling_nonce: int, candidate_index: int) -> int:
    if not group_id:
        raise ValueError("group_id must be non-empty.")
    if int(sampling_nonce) < 0 or not 0 <= int(candidate_index) < contract.GROUP_SIZE:
        raise ValueError("Invalid correction seed coordinates.")
    payload = (
        f"speculative-correction|42|{int(sampling_nonce)}|"
        f"{group_id}|{int(candidate_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _uniform(generator: torch.Generator, device: torch.device) -> float:
    return float(torch.rand((), generator=generator, device=device).item())


def _append_proposal_singletons(
    state: _TargetSuffixState,
    proposal_tokens: Sequence[int],
    stop: int,
    grammar: Any,
) -> None:
    while len(state.tokens) < int(stop):
        position = len(state.tokens)
        token_id = int(proposal_tokens[position])
        allowed = [int(value) for value in grammar.allowed_next(state.tokens)]
        if allowed != [token_id]:
            raise RuntimeError(
                "Proposal/canonical decision positions disagree at a singleton node."
            )
        state.tokens.append(token_id)
        state.logps.append(0.0)
        state.decisions.append(False)


def _complete_target_suffixes(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    states: Sequence[_TargetSuffixState],
    grammar: Any,
    *,
    temperature: float,
    maximum_length: int,
    device: torch.device,
) -> tuple[list[_TargetSuffixState], int]:
    active = list(states)
    finished: list[_TargetSuffixState] = []
    target_forward_calls = 0
    eos_id = int(grammar.eos_token_id)
    scoring_width = grammar_completion_width(
        grammar, safety_limit=int(maximum_length)
    )
    while active:
        decisions: list[_TargetSuffixState] = []
        for state in active:
            remaining = int(maximum_length) - len(state.tokens)
            span = collect_forced_span(
                grammar, state.tokens, maximum_tokens=remaining
            )
            state.tokens.extend(span.token_ids)
            state.logps.extend([0.0] * len(span.token_ids))
            state.decisions.extend([False] * len(span.token_ids))
            if span.finished:
                finished.append(state)
            else:
                decisions.append(state)
        if not decisions:
            break

        next_active: list[_TargetSuffixState] = []
        for start in range(0, len(decisions), contract.LOSS_CHUNK_SIZE):
            chunk = decisions[start : start + contract.LOSS_CHUNK_SIZE]
            distributions = score_prefix_distributions_fixed(
                model,
                prompt_ids,
                [state.tokens for state in chunk],
                grammar,
                device=device,
                temperature=float(temperature),
                batch_rows=contract.LOSS_CHUNK_SIZE,
                completion_width=scoring_width,
            )
            target_forward_calls += 1
            for state, distribution in zip(chunk, distributions, strict=True):
                local_index = int(
                    torch.multinomial(
                        distribution.log_probs.detach().exp(),
                        1,
                        generator=state.generator,
                    ).item()
                )
                token_id = distribution.allowed_ids[local_index]
                state.tokens.append(token_id)
                state.logps.append(
                    float(distribution.log_probs[local_index].detach().float().item())
                )
                state.decisions.append(True)
                if token_id == eos_id:
                    grammar.parse(state.tokens)
                    finished.append(state)
                else:
                    next_active.append(state)
        active = next_active

    if len(finished) != len(states):
        raise RuntimeError("Target suffix sampler did not finish every candidate.")
    return finished, target_forward_calls


def _canonicalize_aligned_rollout(
    rollout: PromptRollout,
    grammar: Any,
) -> CanonicalPromptRollout:
    """Validate and promote samples produced by the canonical aligned cache."""

    canonical: list[CanonicalCandidate] = []
    for candidate in rollout.candidates:
        if not (
            len(candidate.token_ids)
            == len(candidate.sample_log_probs)
            == len(candidate.decision_mask)
        ):
            raise RuntimeError("Aligned sample tensors have inconsistent lengths.")
        prefix: list[int] = []
        decision_index = 0
        for position, (token_id, value, is_decision) in enumerate(
            zip(
                candidate.token_ids,
                candidate.sample_log_probs,
                candidate.decision_mask,
                strict=True,
            )
        ):
            allowed = tuple(int(item) for item in grammar.allowed_next(prefix))
            if int(token_id) not in allowed:
                raise RuntimeError("Aligned sample contains an illegal token.")
            expected_decision = len(allowed) > 1
            if bool(is_decision) != expected_decision or not math.isfinite(float(value)):
                raise RuntimeError("Aligned sample decision metadata is invalid.")
            if expected_decision:
                if decision_index >= len(candidate.proposal_decisions):
                    raise RuntimeError("Aligned sample is missing a decision distribution.")
                distribution = candidate.proposal_decisions[decision_index]
                if distribution.position != position or distribution.allowed_ids != allowed:
                    raise RuntimeError("Aligned decision distribution has the wrong prefix.")
                selected = allowed.index(int(token_id))
                if float(value) != float(distribution.log_probs[selected]):
                    raise RuntimeError("Aligned chosen-token logp is inconsistent.")
                decision_index += 1
            elif float(value) != 0.0:
                raise RuntimeError("Forced aligned tokens must have zero logp.")
            prefix.append(int(token_id))
        if decision_index != len(candidate.proposal_decisions):
            raise RuntimeError("Aligned sample contains unused decision distributions.")
        sid = grammar.parse(prefix)
        if sid != candidate.sid:
            raise RuntimeError("Aligned sample SID differs from its token sequence.")
        values = tuple(float(item) for item in candidate.sample_log_probs)
        canonical.append(
            CanonicalCandidate(
                candidate_index=candidate.candidate_index,
                sid=sid,
                token_ids=tuple(candidate.token_ids),
                sample_log_probs=values,
                old_log_probs=values,
                decision_mask=tuple(candidate.decision_mask),
            )
        )
    canonical.sort(key=lambda item: item.candidate_index)
    if [item.candidate_index for item in canonical] != list(range(contract.GROUP_SIZE)):
        raise RuntimeError("Aligned canonicalization lost or duplicated a candidate.")
    return CanonicalPromptRollout(
        request=rollout.request,
        candidates=tuple(canonical),
        max_sample_canonical_logp_difference=0.0,
        max_proposal_canonical_logp_difference=0.0,
        corrected_candidate_count=0,
        target_forward_calls=0,
    )


def _canonicalize_prompt_rollout_impl(
    model: torch.nn.Module,
    rollout: PromptRollout,
    grammar: Any,
    *,
    sampling_nonce: int,
    temperature: float,
    max_difference: float,
    device: str | torch.device,
) -> CanonicalPromptRollout:
    """Correct cached proposals into exact fixed-shape no-cache policy samples."""

    if len(rollout.candidates) != contract.GROUP_SIZE:
        raise ValueError("Canonical replay requires exactly 16 candidates.")
    if float(max_difference) < 0.0:
        raise ValueError("max_difference must be non-negative.")
    if rollout.policy_aligned:
        return _canonicalize_aligned_rollout(rollout, grammar)
    torch_device = torch.device(device)
    scoring_width = grammar_completion_width(grammar)
    target_rows: dict[int, tuple[DecisionDistribution, ...]] = {}
    target_forward_calls = 0
    cache_proposals = all(candidate.proposal_decisions for candidate in rollout.candidates)
    if cache_proposals:
        for start in (0, contract.LOSS_CHUNK_SIZE):
            chunk = rollout.candidates[start : start + contract.LOSS_CHUNK_SIZE]
            scores = score_completions_dense_fixed(
                model,
                rollout.request.prompt_ids,
                [candidate.token_ids for candidate in chunk],
                grammar,
                device=torch_device,
                temperature=float(temperature),
                batch_rows=contract.LOSS_CHUNK_SIZE,
                completion_width=scoring_width,
            )
            target_forward_calls += 1
            for row, candidate in enumerate(chunk):
                target_rows[candidate.candidate_index] = scores.decisions[row]

    finished_states: list[_TargetSuffixState] = []
    suffix_states: list[_TargetSuffixState] = []
    maximum_proposal_delta = 0.0
    corrected_candidates = 0
    for candidate in rollout.candidates:
        state = _TargetSuffixState(
            candidate_index=candidate.candidate_index,
            generator=_device_generator(
                torch_device,
                correction_seed(
                    rollout.request.group_id,
                    sampling_nonce,
                    candidate.candidate_index,
                ),
            ),
            tokens=[],
            logps=[],
            decisions=[],
        )
        if not cache_proposals:
            corrected_candidates += 1
            suffix_states.append(state)
            continue

        proposal_decisions = candidate.proposal_decisions
        target_decisions = target_rows[candidate.candidate_index]
        if len(proposal_decisions) != len(target_decisions):
            raise RuntimeError("Proposal and target decision counts differ.")
        rejected = False
        for proposal, target in zip(
            proposal_decisions, target_decisions, strict=True
        ):
            if (
                proposal.position != target.position
                or proposal.allowed_ids != target.allowed_ids
            ):
                raise RuntimeError("Proposal and target decision definitions differ.")
            _append_proposal_singletons(
                state, candidate.token_ids, proposal.position, grammar
            )
            proposal_token = int(candidate.token_ids[proposal.position])
            proposal_index = proposal.allowed_ids.index(proposal_token)
            proposal_log_probs = torch.tensor(
                proposal.log_probs,
                dtype=torch.float32,
                device=target.log_probs.device,
            )
            proposal_delta = (
                target.log_probs[proposal_index].detach().float()
                - proposal_log_probs[proposal_index]
            ).abs()
            maximum_proposal_delta = max(
                maximum_proposal_delta, float(proposal_delta.item())
            )
            selection = speculative_accept_or_residual(
                proposal_log_probs,
                target.log_probs,
                proposal_index=proposal_index,
                acceptance_uniform=_uniform(state.generator, torch_device),
                residual_uniform=_uniform(state.generator, torch_device),
            )
            selected_token = target.allowed_ids[selection.selected_index]
            state.tokens.append(selected_token)
            state.logps.append(
                float(
                    target.log_probs[selection.selected_index]
                    .detach()
                    .float()
                    .item()
                )
            )
            state.decisions.append(True)
            if not selection.accepted:
                corrected_candidates += 1
                suffix_states.append(state)
                rejected = True
                break
        if rejected:
            continue
        _append_proposal_singletons(
            state, candidate.token_ids, len(candidate.token_ids), grammar
        )
        grammar.parse(state.tokens)
        finished_states.append(state)

    if suffix_states:
        completed, calls = _complete_target_suffixes(
            model,
            rollout.request.prompt_ids,
            suffix_states,
            grammar,
            temperature=float(temperature),
            maximum_length=contract.MAX_COMPLETION_LENGTH,
            device=torch_device,
        )
        finished_states.extend(completed)
        target_forward_calls += calls

    canonical: list[CanonicalCandidate] = []
    for state in finished_states:
        if not (
            len(state.tokens) == len(state.logps) == len(state.decisions)
            and state.tokens
            and state.tokens[-1] == int(grammar.eos_token_id)
        ):
            raise RuntimeError("Corrected candidate tensors are inconsistent.")
        values = tuple(float(value) for value in state.logps)
        canonical.append(
            CanonicalCandidate(
                candidate_index=state.candidate_index,
                sid=grammar.parse(state.tokens),
                token_ids=tuple(state.tokens),
                sample_log_probs=values,
                old_log_probs=values,
                decision_mask=tuple(state.decisions),
            )
        )
    canonical.sort(key=lambda item: item.candidate_index)
    if [item.candidate_index for item in canonical] != list(range(contract.GROUP_SIZE)):
        raise RuntimeError("Canonical correction lost or duplicated a candidate.")

    # The corrected sampler's behavior distribution is the target distribution
    # itself.  The independent GPU parity gate replays final candidates through
    # the same dense scorer and verifies the configured numerical threshold.
    maximum_sample_delta = 0.0
    if maximum_sample_delta > float(max_difference):  # pragma: no cover
        raise RuntimeError("Corrected sampling/canonical probability gate failed.")
    return CanonicalPromptRollout(
        request=rollout.request,
        candidates=tuple(canonical),
        max_sample_canonical_logp_difference=maximum_sample_delta,
        max_proposal_canonical_logp_difference=maximum_proposal_delta,
        corrected_candidate_count=corrected_candidates,
        target_forward_calls=target_forward_calls,
    )


def canonicalize_prompt_rollout(
    model: torch.nn.Module,
    rollout: PromptRollout,
    grammar: Any,
    *,
    sampling_nonce: int,
    temperature: float,
    max_difference: float,
    device: str | torch.device,
) -> CanonicalPromptRollout:
    with canonical_scoring_mode(model):
        return _canonicalize_prompt_rollout_impl(
            model,
            rollout,
            grammar,
            sampling_nonce=sampling_nonce,
            temperature=temperature,
            max_difference=max_difference,
            device=device,
        )


__all__ = [
    "CacheRolloutStats",
    "CanonicalCandidate",
    "CanonicalPromptRollout",
    "PromptRequest",
    "PromptRollout",
    "ProposalDecision",
    "SampledCandidate",
    "cached_rollout_mode",
    "candidate_wave_pairs",
    "canonicalize_prompt_rollout",
    "collect_forced_span",
    "rollout_prompt_batch",
]
