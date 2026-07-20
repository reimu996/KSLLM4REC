"""Reproducible command-line entry points for GRPO Spec V3.1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ksllm4rec_orpo.data import Sid

from .catalog import build_baseline_trie
from .config import load_config
from .constraint import RecommendationGrammar
from .contract import (
    DOMAINS,
    EXPECTED_TOKEN_IDS,
    expected_trie_leaf_count,
    profile_for_config,
)
from .data import build_recommendation_groups, iter_groups, write_groups
from .integrity import sha256_file
from .prompt import encode_prompt
from .trie import SidPrefixTrie


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(
            f"Input SHA256 mismatch for {path}: expected={expected_sha256}, "
            f"actual={actual}"
        )
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": actual,
    }


def _optional_record(
    path: Path | None, expected_sha256: str | None = None
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": str(path) if path is not None else None, "present": False}
    record = _record(path, expected_sha256)
    record["present"] = True
    return record


def config_check(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    profile = profile_for_config(config)
    base = Path(config["model"]["base_model"])
    adapter = Path(config["model"]["sft_adapter"])
    source = Path(config["data"]["source"])
    provenance_value = config["data"].get("provenance")
    provenance = Path(provenance_value) if provenance_value else profile.provenance
    probe = Path(config["evaluation"]["fixed_probe"])
    inputs = {
        "base_weights": _record(
            base / "model.safetensors",
            "28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90",
        ),
        "sft_adapter": _record(
            adapter / "adapter_model.safetensors", profile.adapter_sha256
        ),
        "sft_adapter_config": _record(
            adapter / "adapter_config.json", profile.adapter_config_sha256
        ),
        "source": _record(source, profile.source_sha256),
    }
    if profile.provenance is None:
        # Preserve the exact historical V3.1 report schema.
        inputs["fixed_probe"] = _record(probe, profile.fixed_probe_sha256)
        return {"spec_version": config["spec_version"], "inputs": inputs}
    inputs["provenance"] = _optional_record(provenance, profile.provenance_sha256)
    inputs["fixed_probe"] = _optional_record(probe, profile.fixed_probe_sha256)
    return {
        "spec_version": config["spec_version"],
        "profile": profile.name,
        "inputs": inputs,
    }


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    profile = profile_for_config(config)
    source = Path(config["data"]["source"])
    source_record = _record(source, profile.source_sha256)
    if profile.provenance is not None:
        provenance = Path(config["data"].get("provenance", profile.provenance))
        _record(provenance, profile.provenance_sha256)
        from .frontier_data import build_frontier_groups

        if args.output_dir.exists():
            manifest_path = args.output_dir / "data_manifest.json"
            groups_path = args.output_dir / "groups.jsonl"
            if not manifest_path.is_file() or not groups_path.is_file():
                raise RuntimeError(
                    f"Existing data directory is incomplete: {args.output_dir}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source_lock", {}).get("source", {}).get("sha256")
                != source_record["sha256"]
                or manifest.get("groups", {}).get("rows") != profile.groups
                or sha256_file(groups_path)
                != manifest.get("groups", {}).get("sha256")
            ):
                raise RuntimeError("Existing Frontier grouped data failed manifest check.")
            return manifest
        return build_frontier_groups(source, provenance, args.output_dir)
    if args.output_dir.exists():
        manifest_path = args.output_dir / "data_manifest.json"
        groups_path = args.output_dir / "groups.jsonl"
        if not manifest_path.is_file() or not groups_path.is_file():
            raise RuntimeError(
                f"Existing data directory is incomplete: {args.output_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("groups", {})
        if (
            expected.get("rows") != int(config["data"]["groups"])
            or expected.get("sha256") != profile.groups_sha256
            or (
                profile.groups_sha256 is not None
                and sha256_file(groups_path) != profile.groups_sha256
            )
            or manifest.get("source") != source_record
        ):
            raise RuntimeError(
                "Existing grouped data failed its frozen manifest check."
            )
        return manifest
    groups = build_recommendation_groups(source)
    group_record = write_groups(groups, args.output_dir / "groups.jsonl")
    if profile.groups_sha256 is not None and group_record["sha256"] != profile.groups_sha256:
        raise RuntimeError("Rebuilt grouped data differs from the frozen artifact.")
    manifest = {
        "schema_version": 3,
        "spec_version": "3.1",
        "kind": "recommendation_prompt_groups_without_completions",
        "source": source_record,
        "groups": group_record,
        "positive_rows": sum(len(item.positive_sids) for item in groups),
        "unique_positive_sids": len(
            {sid for item in groups for sid in item.positive_sids}
        ),
        "positive_set_size_distribution": dict(
            sorted(Counter(len(item.positive_sids) for item in groups).items())
        ),
        "completion_fields": [],
    }
    if manifest["positive_rows"] != profile.positives:
        raise RuntimeError("Prepared group positives differ from the frozen contract.")
    _write_json(args.output_dir / "data_manifest.json", manifest)
    return manifest


def build_trie(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    profile = profile_for_config(config)
    source = Path(config["data"]["source"])
    _record(source, profile.source_sha256)
    if profile.provenance is not None:
        provenance = Path(config["data"].get("provenance", profile.provenance))
        _record(provenance, profile.provenance_sha256)
        from .frontier_data import build_frontier_trie

        if args.output_dir.exists():
            manifest_path = args.output_dir / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Existing trie directory is incomplete: {args.output_dir}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("counts", {}).get("leaves") != profile.unique_sids
                or (
                    profile.trie_manifest_sha256 is not None
                    and sha256_file(manifest_path) != profile.trie_manifest_sha256
                )
            ):
                raise RuntimeError("Existing Frontier trie failed its manifest check.")
            return manifest
        return build_frontier_trie(source, provenance, args.output_dir)
    if args.output_dir.exists():
        trie = SidPrefixTrie.load(
            args.output_dir, expected_leaf_count=expected_trie_leaf_count(config)
        )
        manifest = json.loads(
            (args.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        metadata = manifest.get("metadata", {})
        if (
            metadata.get("source_sha256") != profile.source_sha256
            or metadata.get("strategy") != "baseline_all_system_prompt_response_sids"
            or manifest.get("counts") != trie.counts
            or sha256_file(args.output_dir / "manifest.json")
            != profile.trie_manifest_sha256
        ):
            raise RuntimeError("Existing trie failed its frozen manifest check.")
        return manifest
    manifest = build_baseline_trie(source, args.output_dir)
    if manifest["counts"]["leaves"] != profile.unique_sids:
        raise RuntimeError("Saved trie has an unexpected leaf count.")
    if sha256_file(args.output_dir / "manifest.json") != profile.trie_manifest_sha256:
        raise RuntimeError("Rebuilt trie differs from the frozen artifact.")
    return manifest


def test_tokenizer(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer"],
        local_files_only=True,
        trust_remote_code=True,
    )
    trie = SidPrefixTrie.load(
        args.trie_dir, expected_leaf_count=expected_trie_leaf_count(config)
    )
    grammar = RecommendationGrammar(tokenizer, trie)
    actual_token_ids = {
        "eos": int(tokenizer.eos_token_id),
        "think_open": tokenizer.encode("<think>", add_special_tokens=False)[0],
        "think_close": tokenizer.encode("</think>", add_special_tokens=False)[0],
        "a_zero": grammar.a_offset,
        "a_last": grammar.a_offset + 8191,
        "b_zero": grammar.b_offset,
        "b_last": grammar.b_offset + 8191,
        "c_zero": grammar.c_offset,
        "c_last": grammar.c_offset + 8191,
        **grammar.domain_token_ids,
    }
    if actual_token_ids != EXPECTED_TOKEN_IDS:
        raise RuntimeError(
            f"Tokenizer ID contract mismatch: expected={EXPECTED_TOKEN_IDS}, "
            f"actual={actual_token_ids}"
        )
    if list(grammar.empty_think_ids) != [151667, 198, 151668, 198]:
        raise RuntimeError(f"Unexpected empty-think IDs: {grammar.empty_think_ids}")

    examples: dict[str, Any] = {}
    for domain in DOMAINS:
        a = int(trie.allowed_a(domain)[0])
        b = int(trie.allowed_b(domain, a)[0])
        c = int(trie.allowed_c(domain, a, b)[0])
        sid = Sid(domain, a, b, c)
        encoded = grammar.encode_sid(sid)
        examples[domain] = {
            "sid": sid.render(),
            "completion_tokens": len(encoded),
            "round_trip": grammar.parse(encoded).render(),
        }

    groups = list(iter_groups(args.groups))
    cutoff_len = int(config["data"]["cutoff_len"])
    lengths = [
        len(
            encode_prompt(
                tokenizer,
                group.system,
                group.prompt,
                cutoff_len=cutoff_len,
            )
        )
        for group in groups
    ]
    return {
        "token_ids": actual_token_ids,
        "empty_think_ids": list(grammar.empty_think_ids),
        "completion_examples": examples,
        "prompt_groups": len(groups),
        "prompt_min_tokens": min(lengths),
        "prompt_max_tokens": max(lengths),
        "cutoff_len": cutoff_len,
    }


def memory_gate(args: argparse.Namespace) -> dict[str, Any]:
    from .gates import run_memory_gate

    config = load_config(args.config)
    return run_memory_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        device=args.device,
    )


def signal_gate(args: argparse.Namespace) -> dict[str, Any]:
    from .gates import load_memory_gate, run_signal_gate

    config = load_config(args.config)
    rollout_chunk, loss_chunk = load_memory_gate(
        args.memory_report,
        config=config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
    )
    return run_signal_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        output_dir=args.output_dir,
        rollout_chunk=rollout_chunk,
        loss_chunk=loss_chunk,
        device=args.device,
    )


def timing_gate(args: argparse.Namespace) -> dict[str, Any]:
    from .gates import load_memory_gate, run_timing_gate

    config = load_config(args.config)
    rollout_chunk, loss_chunk = load_memory_gate(
        args.memory_report,
        config=config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
    )
    return run_timing_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        output_dir=args.output_dir,
        rollout_chunk=rollout_chunk,
        loss_chunk=loss_chunk,
        device=args.device,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    from .gates import load_training_gates
    from .trainer import run_training

    config = load_config(args.config)
    rollout_chunk, loss_chunk = load_training_gates(
        config=config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        memory_path=args.memory_report,
        signal_path=args.signal_report,
        timing_path=args.timing_report,
    )
    return run_training(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        output_dir=args.output_dir,
        rollout_chunk=rollout_chunk,
        loss_chunk=loss_chunk,
        resume=not args.no_resume,
        save_recovery=True,
        save_epochs=True,
        device=args.device,
    )


def probe(args: argparse.Namespace) -> dict[str, Any]:
    from .probe import run_fixed_probe

    config = load_config(args.config)
    return run_fixed_probe(
        config,
        policy_adapter_path=args.policy_adapter,
        trie_dir=args.trie_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    from .verify import verify_run

    config = load_config(args.config)
    return verify_run(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        run_dir=args.run_dir,
        probe_root=args.probe_root,
        gate_root=args.gate_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grpo/onereason_lora_grpo.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("config-check")
    check_parser.set_defaults(handler=config_check)

    data_parser = subparsers.add_parser("prepare-data")
    data_parser.add_argument("--output-dir", type=Path, required=True)
    data_parser.set_defaults(handler=prepare_data)

    trie_parser = subparsers.add_parser("build-trie")
    trie_parser.add_argument("--output-dir", type=Path, required=True)
    trie_parser.set_defaults(handler=build_trie)

    tokenizer_parser = subparsers.add_parser("test-tokenizer")
    tokenizer_parser.add_argument("--groups", type=Path, required=True)
    tokenizer_parser.add_argument("--trie-dir", type=Path, required=True)
    tokenizer_parser.set_defaults(handler=test_tokenizer)

    def add_gpu_inputs(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--groups", type=Path, required=True)
        command_parser.add_argument("--trie-dir", type=Path, required=True)
        command_parser.add_argument("--device", default="cuda:0")

    memory_parser = subparsers.add_parser("memory-gate")
    add_gpu_inputs(memory_parser)
    memory_parser.set_defaults(handler=memory_gate)

    signal_parser = subparsers.add_parser("signal-gate")
    add_gpu_inputs(signal_parser)
    signal_parser.add_argument("--memory-report", type=Path, required=True)
    signal_parser.add_argument("--output-dir", type=Path, required=True)
    signal_parser.set_defaults(handler=signal_gate)

    timing_parser = subparsers.add_parser("timing-gate")
    add_gpu_inputs(timing_parser)
    timing_parser.add_argument("--memory-report", type=Path, required=True)
    timing_parser.add_argument("--output-dir", type=Path, required=True)
    timing_parser.set_defaults(handler=timing_gate)

    train_parser = subparsers.add_parser("train")
    add_gpu_inputs(train_parser)
    train_parser.add_argument("--memory-report", type=Path, required=True)
    train_parser.add_argument("--signal-report", type=Path, required=True)
    train_parser.add_argument("--timing-report", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--no-resume", action="store_true")
    train_parser.set_defaults(handler=train)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--policy-adapter", type=Path, required=True)
    probe_parser.add_argument("--trie-dir", type=Path, required=True)
    probe_parser.add_argument("--output-dir", type=Path, required=True)
    probe_parser.add_argument("--device", default="cuda:0")
    probe_parser.set_defaults(handler=probe)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--groups", type=Path, required=True)
    verify_parser.add_argument("--trie-dir", type=Path, required=True)
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    verify_parser.add_argument("--probe-root", type=Path, required=True)
    verify_parser.add_argument("--gate-root", type=Path, required=True)
    verify_parser.set_defaults(handler=verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
