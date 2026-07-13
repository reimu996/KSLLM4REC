"""Atomic run manifests for auditable SFT operations."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)
    return {
        "args": args,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class RunManifest:
    def __init__(self, path: Path, initial: dict[str, Any]) -> None:
        self.path = path
        self.value = dict(initial)
        self.value.setdefault("created_at", now_iso())
        self.value.setdefault("updated_at", self.value["created_at"])
        self.flush()

    def update(self, **values: Any) -> None:
        self.value.update(values)
        self.value["updated_at"] = now_iso()
        self.flush()

    def flush(self) -> None:
        atomic_write_json(self.path, self.value)
