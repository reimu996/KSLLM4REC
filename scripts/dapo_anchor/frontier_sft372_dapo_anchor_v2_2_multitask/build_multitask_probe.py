#!/usr/bin/env python3
"""Build the Frontier Multitask Fixed Probe v1 dataset.

Spec: FR-001..FR-007 (approved). Deterministic, read-only, self-checking.
Outputs:
  artifacts/probe/frontier_multitask_probe_v1/fixed_probe_1024.jsonl
  artifacts/probe/frontier_multitask_probe_v1/probe_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TRAIN = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"
)
PROVENANCE = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
)
OUT_DIR = Path(
    "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/probe/frontier_multitask_probe_v1"
)
OUT_FILE = OUT_DIR / "fixed_probe_1024.jsonl"
MANIFEST = OUT_DIR / "probe_manifest.json"

COUNTS = {"recommend": 640, "text_to_sid": 384}
TOTAL = sum(COUNTS.values())

# SID: <|video_begin|> | <|ad_begin|> | <|prod_begin|> | <|living_begin|> + 3 x <s_x_N>
SID_RE = re.compile(
    r"<\|(?:video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
TASK_MAP = {
    "recommendation": "recommend",
    "item_text_to_sid": "text_to_sid",
}

# 原始输入锁定（Spec AC-005 参考值）
LOCKED_TRAIN_SHA256 = "9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a"
LOCKED_PROVENANCE_SHA256 = (
    "e56dddff865780273573e06e5a44d4c1dd9a27013c78c166aa6565f7b9ad8908"
)

SCHEMA_VERSION = 1
KIND = "frontier_multitask_fixed_probe"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def load_locked_sha256(path: Path, expected: str) -> None:
    actual = sha256_bytes(path.read_bytes())
    if actual != expected:
        raise SystemExit(
            f"Input drift: {path} SHA256={actual}, expected {expected}"
        )


def read_aligned_rows() -> list[tuple[str, dict, dict]]:
    """Return (raw_line, train_record, provenance_record) per line, aligned 1:1."""
    with open(TRAIN, encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]
    train_rows = [json.loads(line)[0] for line in raw_lines]
    with open(PROVENANCE, encoding="utf-8") as f:
        prov_rows = [json.loads(line) for line in f]
    if not (len(raw_lines) == len(train_rows) == len(prov_rows)):
        raise SystemExit("Row-count mismatch between train/provenance.")
    return list(zip(raw_lines, train_rows, prov_rows))


def selection_key(raw_line: str, record_sha256: str) -> str:
    """Spec FR-002: sha256(f'{line}:{record_sha256}') ascending."""
    return sha256_text(f"{raw_line}:{record_sha256}")


def extract_unique_sid(response: str) -> str | None:
    found = SID_RE.findall(response)
    if len(found) != 1:
        return None
    return found[0]


def build_candidates(
    rows: list[tuple[str, dict, dict]]
) -> list[tuple[str, str, str, str, dict, dict, int]]:
    """Filter + key + sort + dedup per task.

    Returns list of (task_probe, key, prompt, sid, train, prov, line_no).
    """
    per_task: dict[str, list] = {name: [] for name in TASK_MAP.values()}
    for line_no, (raw_line, train, prov) in enumerate(rows, start=1):
        task = prov.get("task")
        if task not in TASK_MAP:
            continue
        sid = extract_unique_sid(train.get("response", ""))
        if sid is None:
            continue  # FR-003: exactly one SID
        key = selection_key(raw_line, prov["record_sha256"])
        probe_task = TASK_MAP[task]
        per_task[probe_task].append(
            (probe_task, key, train.get("prompt", ""), sid, train, prov, line_no)
        )
    result: list = []
    for name in TASK_MAP.values():
        entries = sorted(per_task[name], key=lambda e: e[1])  # key ascending
        seen_prompts: set[str] = set()
        deduped = []
        for entry in entries:
            prompt = entry[2]
            if prompt in seen_prompts:
                continue  # FR-004: one row per group (prompt)
            seen_prompts.add(prompt)
            deduped.append(entry)
        if len(deduped) < COUNTS[name]:
            raise SystemExit(
                f"Not enough candidates for {name}: {len(deduped)} < {COUNTS[name]}"
            )
        result.extend(deduped[: COUNTS[name]])
    return result


def emit_record(entry: tuple) -> dict:
    task, _key, _prompt, sid, train, prov, line_no = entry
    return {
        "task": task,
        "system": train.get("system", ""),
        "instruction": train.get("prompt", ""),
        "target": train.get("response", ""),
        "target_sid": sid,
        "source_file": str(TRAIN),
        "source_line": line_no,
        "source_row_sha256": prov["record_sha256"],
    }


def self_check(records: list[dict]) -> None:
    """Script-internal checks (spec FR-007)."""
    counts = {name: 0 for name in COUNTS}
    for rec in records:
        assert set(rec.keys()) == {
            "task", "system", "instruction", "target", "target_sid",
            "source_file", "source_line", "source_row_sha256",
        }, f"field set mismatch: {rec['source_line']}"
        assert isinstance(rec["source_line"], int), rec["source_line"]
        counts[rec["task"]] += 1
        assert SID_RE.fullmatch(rec["target_sid"]), rec["target_sid"]
    assert counts == COUNTS, f"task_counts={counts}"
    prompts = [rec["instruction"] for rec in records]
    assert len(set(prompts)) == len(prompts), "duplicate prompts in output"


def write_outputs(records: list[dict], *, force: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not force:
        if OUT_FILE.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {OUT_FILE}")
        if MANIFEST.exists():
            raise SystemExit(f"Refusing to overwrite existing manifest: {MANIFEST}")

    lines = [json.dumps(rec, ensure_ascii=False) + "\n" for rec in records]
    data = "".join(lines).encode("utf-8")
    OUT_FILE.write_bytes(data)
    out_sha256 = sha256_bytes(data)
    out_size = len(data)

    manifest = {
        "kind": KIND,
        "path": str(OUT_FILE),
        "provenance": {
            "path": str(PROVENANCE),
            "sha256": sha256_bytes(PROVENANCE.read_bytes()),
            "size": PROVENANCE.stat().st_size,
        },
        "source": {
            "path": str(TRAIN),
            "sha256": sha256_bytes(TRAIN.read_bytes()),
            "size": TRAIN.stat().st_size,
        },
        "records": len(records),
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "algorithm": "sha256(f'{candidate_line}:{raw_record_sha256}') ascending; "
            "per-task; single-SID response filter; prompt-level dedup; "
            "first COUNTS[task] kept",
            "count_each": COUNTS,
        },
        "task_counts": {
            name: sum(1 for r in records if r["task"] == name) for name in COUNTS
        },
        "sha256": out_sha256,
        "size": out_size,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite outputs")
    args = parser.parse_args()
    if not args.force:
        if OUT_FILE.exists() or MANIFEST.exists():
            raise SystemExit("Outputs exist; use --force to overwrite.")

    load_locked_sha256(TRAIN, LOCKED_TRAIN_SHA256)  # AC-005 guard
    load_locked_sha256(PROVENANCE, LOCKED_PROVENANCE_SHA256)
    rows = read_aligned_rows()
    candidates = build_candidates(rows)
    records = [emit_record(entry) for entry in candidates]
    self_check(records)
    manifest = write_outputs(records, force=args.force)
    print(
        f"OK: {len(records)} rows -> {OUT_FILE}\n"
        f"task_counts={manifest['task_counts']} sha256={manifest['sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
