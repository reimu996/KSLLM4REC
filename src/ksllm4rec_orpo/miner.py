"""Resumable base-model hard-negative mining and final pair audit."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ksllm4rec_sft.data import SourceRecord, render_qwen3_nothink, sha256_file
from ksllm4rec_sft.manifest import atomic_write_json, now_iso

from .data import (
    EMPTY_THINK,
    EXPECTED_RECORDS,
    EXPECTED_TASK_COUNTS,
    PairTask,
    final_answer_suffix,
    find_sids,
    iter_pair_rows,
    sha256_text,
)
from .pairs import CANDIDATE_SCHEMA_VERSION, _atomic_jsonl


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def _longest_common_prefix(sequences: list[list[int]]) -> int:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Candidate token sequences must not be empty.")
    limit = min(map(len, sequences))
    for index in range(limit):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            return index
    raise ValueError("Rejected candidates have no token-level divergence.")


class BaseFirstDivergenceMiner:
    def __init__(
        self,
        model_path: Path,
        *,
        min_free_gib: float = 20.5,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for base hard-negative mining.")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        if free_gib < min_free_gib:
            raise RuntimeError(
                f"Mining GPU gate failed: {free_gib:.3f} GiB free, "
                f"need {min_free_gib:.3f} GiB."
            )
        self.torch = torch
        self.model_path = model_path.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda:0")
        self.model.eval()
        self.gpu_free_gib_before = free_gib
        self.gpu_total_gib = total_bytes / 1024**3
        self.forward_calls = 0

    def _token_ids(self, system: str, instruction: str, response: str) -> list[int]:
        source, target = render_qwen3_nothink(
            SourceRecord(system=system, prompt=instruction, response=response)
        )
        return self.tokenizer.encode(
            source + target,
            add_special_tokens=False,
        )

    def select(
        self,
        *,
        system: str,
        instruction: str,
        candidates: list[str],
    ) -> tuple[int, dict[str, Any]]:
        if len(candidates) < 2:
            return 0, {
                "method": "single_candidate",
                "common_prefix_tokens": None,
                "candidate_next_token_ids": [],
                "candidate_logits": [],
            }
        sequences = [
            self._token_ids(system, instruction, response)
            for response in candidates
        ]
        prefix_length = _longest_common_prefix(sequences)
        if prefix_length == 0:
            raise RuntimeError("Candidate conversations diverge at the first token.")
        next_token_ids = [sequence[prefix_length] for sequence in sequences]
        input_ids = self.torch.tensor(
            [sequences[0][:prefix_length]],
            dtype=self.torch.long,
            device="cuda:0",
        )
        with self.torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            ).logits[0, -1].float()
        self.forward_calls += 1
        scores = [float(logits[token_id].item()) for token_id in next_token_ids]
        selected = max(range(len(candidates)), key=lambda index: (scores[index], -index))
        return selected, {
            "method": "base_first_divergence_logit",
            "common_prefix_tokens": prefix_length,
            "candidate_next_token_ids": next_token_ids,
            "candidate_logits": scores,
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_weights_sha256": sha256_file(
                self.model_path / "model.safetensors"
            ),
            "gpu_free_gib_before": self.gpu_free_gib_before,
            "gpu_total_gib": self.gpu_total_gib,
            "forward_calls": self.forward_calls,
            "peak_memory_allocated_gib": self.torch.cuda.max_memory_allocated()
            / 1024**3,
            "peak_memory_reserved_gib": self.torch.cuda.max_memory_reserved()
            / 1024**3,
        }


def _completed_part_rows(part_path: Path) -> int:
    if not part_path.exists():
        return 0
    count = 0
    with part_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Incomplete mining part at line {line_number}; remove only after audit."
                ) from exc
            if not isinstance(value, dict) or "rejected" not in value:
                raise RuntimeError(f"Invalid mining part row {line_number}.")
            count += 1
    return count


def _finalize_audit_row(
    candidate: dict[str, Any], selected_index: int, mining: dict[str, Any]
) -> dict[str, Any]:
    rejected_candidates = candidate["rejected_candidates"]
    rejected = rejected_candidates[selected_index]
    pair_payload = json.dumps(
        [
            candidate["system"],
            candidate["instruction"],
            candidate["chosen"],
            rejected,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    result = dict(candidate)
    result.pop("rejected_candidates")
    result.update(
        pair_id=sha256_text(pair_payload),
        rejected=rejected,
        selected_candidate_index=selected_index,
        selected_candidate_count=len(rejected_candidates),
        mining=mining,
    )
    return result


def _assert_audit_row(row: dict[str, Any]) -> None:
    chosen = row["chosen"]
    rejected = row["rejected"]
    if chosen == rejected:
        raise RuntimeError(f"Pair {row['pair_id']} has identical responses.")
    if not chosen.startswith(EMPTY_THINK) or not rejected.startswith(EMPTY_THINK):
        raise RuntimeError(f"Pair {row['pair_id']} is not direct no-think.")
    if sha256_text(final_answer_suffix(chosen)) != row["original_final_suffix_sha256"]:
        raise RuntimeError(f"Pair {row['pair_id']} changed the chosen final suffix.")
    task = PairTask(row["task"])
    if task in (PairTask.RECOMMEND, PairTask.TEXT_TO_SID):
        rejected_sids = find_sids(final_answer_suffix(rejected))
        if len(rejected_sids) != 1:
            raise RuntimeError(f"Pair {row['pair_id']} rejected SID is malformed.")
        if rejected_sids[0].render() in set(row["positive_set"]):
            raise RuntimeError(f"Pair {row['pair_id']} uses a known positive as rejected.")
    elif task == PairTask.USER_INTEREST:
        json.loads(final_answer_suffix(rejected).strip())
        prompt_sids = {sid.render() for sid in find_sids(row["instruction"])}
        rejected_sids = {sid.render() for sid in find_sids(rejected)}
        if not rejected_sids or not rejected_sids.isdisjoint(prompt_sids):
            raise RuntimeError(
                f"Pair {row['pair_id']} user rejected is not SID-disjoint."
            )
    elif task == PairTask.CEVAL:
        if len(find_sids(rejected)) != 0:
            raise RuntimeError(f"Pair {row['pair_id']} CEval rejected contains SID.")


def audit_final_pairs(audit_path: Path, train_path: Path) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    pair_ids: set[str] = set()
    source_lines: set[int] = set()
    audit_count = 0
    for row in _iter_jsonl(audit_path):
        _assert_audit_row(row)
        audit_count += 1
        task_counts[row["task"]] += 1
        tier_counts[
            f"{row['task']}|tier{row['negative_tier']}|{row['negative_reason']}"
        ] += 1
        if row["pair_id"] in pair_ids:
            raise RuntimeError(f"Duplicate pair_id: {row['pair_id']}")
        if int(row["source_line"]) in source_lines:
            raise RuntimeError(f"Duplicate source line: {row['source_line']}")
        pair_ids.add(row["pair_id"])
        source_lines.add(int(row["source_line"]))

    train_count = sum(1 for _ in iter_pair_rows(train_path))
    expected = {task.value: count for task, count in EXPECTED_TASK_COUNTS.items()}
    if audit_count != EXPECTED_RECORDS or train_count != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} final pairs, audit={audit_count}, train={train_count}."
        )
    if dict(task_counts) != expected:
        raise RuntimeError(f"Final task counts mismatch: {dict(task_counts)}")
    if source_lines != set(range(1, EXPECTED_RECORDS + 1)):
        raise RuntimeError("Final pair source lines are not exactly 1..32705.")
    return {
        "status": "passed",
        "records": audit_count,
        "task_counts": dict(sorted(task_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "unique_pair_ids": len(pair_ids),
        "audit_path": str(audit_path.resolve()),
        "audit_size": audit_path.stat().st_size,
        "audit_sha256": sha256_file(audit_path),
        "train_path": str(train_path.resolve()),
        "train_size": train_path.stat().st_size,
        "train_sha256": sha256_file(train_path),
    }


def mine_pair_candidates(
    candidates_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    min_free_gib: float = 20.5,
    progress_every: int = 100,
) -> dict[str, Any]:
    candidates_path = candidates_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_sha = sha256_file(candidates_path)
    state_path = output_dir / "mining_state.json"
    part_path = output_dir / "audit.jsonl.part"
    audit_path = output_dir / "audit.jsonl"
    train_path = output_dir / "train.jsonl"

    if audit_path.exists() or train_path.exists():
        raise RuntimeError(
            "Final mining outputs already exist; use a new output directory."
        )
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("candidates_sha256") != candidate_sha:
            raise RuntimeError("Mining candidate file changed since the partial run.")
        if Path(state.get("model_path", "")).resolve() != model_path.resolve():
            raise RuntimeError("Mining model path changed since the partial run.")
    else:
        state = {
            "schema_version": 1,
            "status": "running",
            "started_at": now_iso(),
            "candidates_path": str(candidates_path),
            "candidates_sha256": candidate_sha,
            "model_path": str(model_path.resolve()),
            "completed_rows": 0,
        }
        atomic_write_json(state_path, state)

    completed = _completed_part_rows(part_path)
    if completed > EXPECTED_RECORDS:
        raise RuntimeError(f"Mining part contains too many rows: {completed}")
    miner = BaseFirstDivergenceMiner(model_path, min_free_gib=min_free_gib)
    mode = "a" if completed else "w"
    with part_path.open(mode, encoding="utf-8") as target:
        for index, candidate in enumerate(_iter_jsonl(candidates_path), start=1):
            if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
                raise RuntimeError(f"Candidate schema mismatch at line {index}.")
            if index <= completed:
                continue
            candidates = candidate["rejected_candidates"]
            if candidate["requires_model_mining"]:
                selected, mining = miner.select(
                    system=candidate["system"],
                    instruction=candidate["instruction"],
                    candidates=candidates,
                )
            else:
                selected = 0
                mining = {
                    "method": "deterministic_first_candidate",
                    "common_prefix_tokens": None,
                    "candidate_next_token_ids": [],
                    "candidate_logits": [],
                }
            row = _finalize_audit_row(candidate, selected, mining)
            _assert_audit_row(row)
            target.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            if index % progress_every == 0:
                target.flush()
                os.fsync(target.fileno())
                state.update(
                    status="running",
                    updated_at=now_iso(),
                    completed_rows=index,
                    miner=miner.metrics(),
                )
                atomic_write_json(state_path, state)
                print(f"mined_rows={index}/{EXPECTED_RECORDS}", flush=True)
        target.flush()
        os.fsync(target.fileno())

    completed = _completed_part_rows(part_path)
    if completed != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Mining ended with {completed} rows, expected {EXPECTED_RECORDS}."
        )
    os.replace(part_path, audit_path)

    def train_rows() -> Iterator[dict[str, str]]:
        for row in _iter_jsonl(audit_path):
            yield {
                "instruction": row["instruction"],
                "input": "",
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "system": row["system"],
            }

    _atomic_jsonl(train_path, train_rows())
    dataset_info = {
        "orpo_all_tasks_32705": {
            "file_name": train_path.name,
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system",
            },
        },
        "orpo_memory_gate_16384": {
            "file_name": "memory_gate.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
                "system": "system",
            },
        },
    }
    atomic_write_json(output_dir / "dataset_info.json", dataset_info)
    audit = audit_final_pairs(audit_path, train_path)
    report = {
        "schema_version": 1,
        "status": "passed",
        "finished_at": now_iso(),
        "candidates_path": str(candidates_path),
        "candidates_sha256": candidate_sha,
        "model_path": str(model_path.resolve()),
        "miner": miner.metrics(),
        "audit": audit,
        "dataset_info_path": str((output_dir / "dataset_info.json").resolve()),
        "dataset_info_sha256": sha256_file(output_dir / "dataset_info.json"),
    }
    atomic_write_json(output_dir / "pair_report.json", report)
    deterministic_manifest = {
        "schema_version": 1,
        "source_kind": "base_first_divergence_orpo_pairs",
        "candidates_sha256": candidate_sha,
        "base_model_weights_sha256": report["miner"]["model_weights_sha256"],
        "base_model_forward_calls": report["miner"]["forward_calls"],
        "records": audit["records"],
        "task_counts": audit["task_counts"],
        "tier_counts": audit["tier_counts"],
        "unique_pair_ids": audit["unique_pair_ids"],
        "audit_size": audit["audit_size"],
        "audit_sha256": audit["audit_sha256"],
        "train_size": audit["train_size"],
        "train_sha256": audit["train_sha256"],
        "dataset_info_sha256": report["dataset_info_sha256"],
    }
    atomic_write_json(output_dir / "pair_manifest.json", deterministic_manifest)
    state.update(
        status="passed",
        updated_at=now_iso(),
        completed_rows=EXPECTED_RECORDS,
        final_report=str((output_dir / "pair_report.json").resolve()),
        miner=miner.metrics(),
    )
    atomic_write_json(state_path, state)
    return report
