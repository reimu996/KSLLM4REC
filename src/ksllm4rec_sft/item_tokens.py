"""Competition item-mask vocabulary rules."""

from __future__ import annotations


EXPECTED_ITEM_TOKEN_COUNT = 24610


def build_item_token_ids(tokenizer) -> list[int]:
    """Match the platform rule: every tokenizer added-vocabulary id is an item id."""

    item_ids = set(tokenizer.get_added_vocab().values())
    if len(item_ids) != EXPECTED_ITEM_TOKEN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ITEM_TOKEN_COUNT} item token ids, found {len(item_ids)}. "
            "Refusing to train with an ambiguous item mask."
        )
    expected_ids = set(range(tokenizer.vocab_size, len(tokenizer)))
    if item_ids != expected_ids:
        raise ValueError(
            "Added-vocabulary ids are not the expected contiguous tokenizer suffix."
        )
    return sorted(item_ids)
