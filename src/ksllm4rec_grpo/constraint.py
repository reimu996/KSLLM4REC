"""One finite-state grammar shared by rollout, scoring, and probe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from transformers import LogitsProcessor

from ksllm4rec_orpo.data import Sid

from .contract import DOMAINS, EMPTY_THINK, MAX_SID_COMPONENT, RESPONSE_PREFIX
from .probability import legal_log_probs
from .trie import SidPrefixTrie


@dataclass
class _PrefixNode:
    children: dict[int, int] = field(default_factory=dict)
    domain: str | None = None


class RecommendationGrammar:
    """Exact completion grammar ending in one baseline SID and EOS."""

    def __init__(
        self,
        tokenizer,
        trie: SidPrefixTrie,
        *,
        mode: str = "train_recommendation",
    ) -> None:
        self.tokenizer = tokenizer
        self.trie = trie
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.domain_token_ids = {
            domain: self._single_token(f"<|{domain}_begin|>") for domain in DOMAINS
        }
        self.a_offset = self._component_offset("a")
        self.b_offset = self._component_offset("b")
        self.c_offset = self._component_offset("c")
        self.empty_think_ids = tuple(self._encode(EMPTY_THINK))
        if mode == "train_recommendation":
            prefix_text = RESPONSE_PREFIX
            common_prefix: tuple[int, ...] = self.empty_think_ids
            self.suffix_tokens = (self.eos_token_id,)
        elif mode == "probe_text_to_sid":
            prefix_text = {domain: "" for domain in DOMAINS}
            common_prefix = self.empty_think_ids
            self.suffix_tokens = (self.eos_token_id,)
        elif mode == "probe_recommendation":
            prefix_text = {domain: "" for domain in DOMAINS}
            common_prefix = (*self.empty_think_ids, *self._encode('["'))
            self.suffix_tokens = (*self._encode('"]'), self.eos_token_id)
        else:
            raise ValueError(f"Unsupported grammar mode: {mode!r}.")
        self.mode = mode
        self.nodes = [_PrefixNode()]
        self.prefix_tokens: dict[str, tuple[int, ...]] = {}
        for domain in DOMAINS:
            if self.trie.allowed_a(domain).size == 0:
                continue
            tokens = tuple(
                [
                    *common_prefix,
                    *self._encode(prefix_text[domain]),
                    self.domain_token_ids[domain],
                ]
            )
            self.prefix_tokens[domain] = tokens
            node_index = 0
            for token_id in tokens:
                node = self.nodes[node_index]
                node_index = node.children.setdefault(token_id, len(self.nodes))
                if node_index == len(self.nodes):
                    self.nodes.append(_PrefixNode())
            if self.nodes[node_index].domain is not None:
                raise ValueError(
                    "Canonical completion prefixes collide after tokenization."
                )
            self.nodes[node_index].domain = domain

    def _encode(self, text: str) -> list[int]:
        if text == "":
            return []
        return [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=False)
        ]

    def _single_token(self, text: str) -> int:
        ids = self._encode(text)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for {text!r}, got {ids}.")
        return ids[0]

    def _component_offset(self, level: str) -> int:
        zero = self._single_token(f"<s_{level}_0>")
        last = self._single_token(f"<s_{level}_{MAX_SID_COMPONENT}>")
        if last - zero != MAX_SID_COMPONENT:
            raise ValueError(f"SID {level} token IDs are not contiguous.")
        return zero

    @staticmethod
    def _component(token_id: int, offset: int) -> int | None:
        value = int(token_id) - offset
        return value if 0 <= value <= MAX_SID_COMPONENT else None

    def _prefix_state(self, generated: Sequence[int]) -> tuple[int, str | None, int]:
        node_index = 0
        for index, token_id in enumerate(generated):
            node = self.nodes[node_index]
            if node.domain is not None:
                return node_index, node.domain, index
            next_index = node.children.get(int(token_id))
            if next_index is None:
                raise ValueError(
                    f"Completion left the canonical prefix at token {index}."
                )
            node_index = next_index
        node = self.nodes[node_index]
        return node_index, node.domain, len(generated)

    def allowed_next(self, generated: Sequence[int]) -> list[int]:
        node_index, domain, consumed = self._prefix_state(generated)
        if domain is None:
            return sorted(self.nodes[node_index].children)
        tail = [int(value) for value in generated[consumed:]]
        if not tail:
            return (self.trie.allowed_a(domain).astype(int) + self.a_offset).tolist()
        a = self._component(tail[0], self.a_offset)
        if a is None:
            raise ValueError("Completion has an invalid s_a token.")
        if len(tail) == 1:
            return (self.trie.allowed_b(domain, a).astype(int) + self.b_offset).tolist()
        b = self._component(tail[1], self.b_offset)
        if b is None:
            raise ValueError("Completion has an invalid s_b token.")
        if len(tail) == 2:
            return (
                self.trie.allowed_c(domain, a, b).astype(int) + self.c_offset
            ).tolist()
        c = self._component(tail[2], self.c_offset)
        if c is None or not self.trie.contains(Sid(domain, a, b, c)):
            raise ValueError("Completion has a non-baseline s_c token.")
        suffix = tail[3:]
        if len(suffix) < len(self.suffix_tokens):
            if tuple(suffix) != self.suffix_tokens[: len(suffix)]:
                raise ValueError("Completion has an invalid canonical suffix.")
            return [self.suffix_tokens[len(suffix)]]
        if tuple(suffix) == self.suffix_tokens:
            return []
        raise ValueError("Completion contains tokens after the canonical suffix.")

    def parse(self, generated: Sequence[int]) -> Sid:
        _, domain, consumed = self._prefix_state(generated)
        if domain is None:
            raise ValueError("Completion ended before selecting a domain.")
        tail = [int(value) for value in generated[consumed:]]
        if (
            len(tail) != 3 + len(self.suffix_tokens)
            or tuple(tail[3:]) != self.suffix_tokens
        ):
            raise ValueError("Completion must end with exactly one SID and EOS.")
        values = (
            self._component(tail[0], self.a_offset),
            self._component(tail[1], self.b_offset),
            self._component(tail[2], self.c_offset),
        )
        if any(value is None for value in values):
            raise ValueError("Completion contains malformed SID components.")
        sid = Sid(domain, int(values[0]), int(values[1]), int(values[2]))
        if not self.trie.contains(sid):
            raise ValueError("Completion SID is absent from the baseline trie.")
        return sid

    def encode_sid(self, sid: Sid) -> list[int]:
        if not self.trie.contains(sid):
            raise ValueError(f"Cannot encode non-baseline SID: {sid.render()}")
        return [
            *self.prefix_tokens[sid.domain],
            self.a_offset + sid.a,
            self.b_offset + sid.b,
            self.c_offset + sid.c,
            *self.suffix_tokens,
        ]

    def decision_mask(self, generated: Sequence[int]) -> list[bool]:
        prefix: list[int] = []
        result: list[bool] = []
        for token_id in generated:
            allowed = self.allowed_next(prefix)
            if int(token_id) not in allowed:
                raise ValueError("Target token is not allowed by the grammar.")
            result.append(len(allowed) > 1)
            prefix.append(int(token_id))
        self.parse(prefix)
        return result


class RecommendationLogitsProcessor(LogitsProcessor):
    """Mask every token outside the grammar's current legal set."""

    def __init__(self, grammar: RecommendationGrammar, prompt_length: int) -> None:
        self.grammar = grammar
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        masked = torch.full(
            scores.shape, -torch.inf, dtype=torch.float32, device=scores.device
        )
        for row in range(input_ids.size(0)):
            generated = input_ids[row, self.prompt_length :].tolist()
            allowed = self.grammar.allowed_next(generated)
            if not allowed:
                allowed = [self.grammar.eos_token_id]
            ids = torch.tensor(allowed, dtype=torch.long, device=scores.device)
            masked[row, ids] = legal_log_probs(scores[row], allowed)
        return masked
