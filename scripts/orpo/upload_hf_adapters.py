#!/usr/bin/env python3
"""Upload the two approved ORPO LoRA adapters to new public HF repos.

The script uploads exactly three approved files, permits only HF's generated
`.gitattributes` beside them, and refuses existing repos. It never loads a
model or performs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/orpo/hf_upload.json"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/orpo/hf_upload"
LOG_ROOT = PROJECT_ROOT / "operation_logs/orpo"
UPLOADED_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
)
HUB_MANAGED_FILES = (".gitattributes",)
REMOTE_FILES = HUB_MANAGED_FILES + UPLOADED_FILES
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TOKEN_RE = re.compile(r"\b(?:hf|api|oauth)_[A-Za-z0-9_-]{12,}\b", re.IGNORECASE)


class UploadError(RuntimeError):
    """A contract violation or an upload verification failure."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id() -> str:
    return datetime.now().strftime("hf_upload_%Y%m%d_%H%M%S_%f")


def redact(value: object) -> str:
    return TOKEN_RE.sub("[REDACTED]", str(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UploadError(f"cannot read upload config: {redact(exc)}") from exc

    if config.get("schema_version") != 1:
        raise UploadError("upload config schema_version must be 1")
    if config.get("visibility") != "public":
        raise UploadError("only public repositories are approved")
    if tuple(config.get("required_uploaded_files", ())) != UPLOADED_FILES:
        raise UploadError(
            f"required_uploaded_files must be exactly {list(UPLOADED_FILES)}"
        )
    if tuple(config.get("allowed_hub_managed_files", ())) != HUB_MANAGED_FILES:
        raise UploadError(
            "allowed_hub_managed_files must contain only .gitattributes"
        )
    if tuple(config.get("required_remote_files", ())) != REMOTE_FILES:
        raise UploadError(
            f"required_remote_files must be exactly {list(REMOTE_FILES)}"
        )

    namespace = config.get("namespace")
    repositories = config.get("repositories")
    if not isinstance(namespace, str) or not namespace:
        raise UploadError("namespace must be a non-empty string")
    if not isinstance(repositories, list) or len(repositories) != 2:
        raise UploadError("repositories must contain exactly two entries")

    seen_repo_ids: set[str] = set()
    seen_epochs: set[int] = set()
    for item in repositories:
        if not isinstance(item, dict):
            raise UploadError("each repository entry must be an object")
        repo_id = item.get("repo_id")
        epoch = item.get("epoch")
        if not isinstance(repo_id, str) or not repo_id.startswith(f"{namespace}/"):
            raise UploadError(f"repo_id must be owned by {namespace}: {repo_id}")
        if repo_id in seen_repo_ids:
            raise UploadError(f"duplicate repo_id: {repo_id}")
        if not isinstance(epoch, int) or epoch not in (1, 2):
            raise UploadError(f"epoch must be 1 or 2: {epoch}")
        if epoch in seen_epochs:
            raise UploadError(f"duplicate epoch: {epoch}")
        source_dir = item.get("source_dir")
        if not isinstance(source_dir, str) or Path(source_dir).is_absolute():
            raise UploadError(f"source_dir must be project-relative: {source_dir}")
        for key in ("adapter_model", "adapter_config"):
            expected = item.get(key)
            if not isinstance(expected, dict):
                raise UploadError(f"missing expected file metadata: {key}")
            if not isinstance(expected.get("size"), int) or expected["size"] <= 0:
                raise UploadError(f"invalid expected size for {repo_id}/{key}")
            digest = expected.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise UploadError(f"invalid expected sha256 for {repo_id}/{key}")
        seen_repo_ids.add(repo_id)
        seen_epochs.add(epoch)

    if seen_epochs != {1, 2}:
        raise UploadError("repository epochs must be exactly {1, 2}")
    return config


def resolve_source_dir(relative_path: str) -> Path:
    source_dir = (PROJECT_ROOT / relative_path).resolve()
    try:
        source_dir.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise UploadError(f"source_dir escapes project root: {relative_path}") from exc
    if not source_dir.is_dir():
        raise UploadError(f"source checkpoint directory does not exist: {source_dir}")
    return source_dir


def assert_expected(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise UploadError(f"missing required file: {path}")
    actual = file_record(path)
    if actual != expected:
        raise UploadError(f"{label} mismatch: expected={expected}, actual={actual}")
    return actual


def render_readme(repo: dict[str, Any], training: dict[str, Any]) -> str:
    modules = ", ".join(training["target_modules"])
    return (
        f"# OneReason-0.8B ORPO LoRA - Epoch {repo['epoch']}\n\n"
        "This repository contains one LoRA adapter checkpoint for the "
        "Kuaishou LLM4Rec competition.\n\n"
        "## Training\n\n"
        f"- Start: {training['base_description']}\n"
        f"- Method: {training['method']} from the first optimizer step\n"
        f"- Preference pairs: {training['pairs']:,}\n"
        f"- Completed epoch: {repo['epoch']}\n"
        f"- LoRA rank / alpha / dropout: {training['lora_rank']} / "
        f"{training['lora_alpha']} / {training['lora_dropout']}\n"
        f"- Target modules: {modules}\n\n"
        "## Files\n\n"
        "- `adapter_model.safetensors`: LoRA weights\n"
        "- `adapter_config.json`: the original training checkpoint config, "
        "preserved byte-for-byte\n\n"
        "## Evaluation status\n\n"
        "The official constrained-beam competition evaluation has not been run "
        "for this adapter. This repository makes no performance claim.\n"
    )


def assert_allowlist(directory: Path) -> list[str]:
    actual = sorted(
        path.name for path in directory.iterdir() if path.is_file()
    )
    expected = sorted(UPLOADED_FILES)
    if actual != expected:
        raise UploadError(f"file allowlist mismatch in {directory}: {actual}")
    if any(path.is_dir() for path in directory.iterdir()):
        raise UploadError(f"subdirectories are not allowed in staging: {directory}")
    return actual


def stage_repository(
    repo: dict[str, Any], training: dict[str, Any], staging_root: Path
) -> dict[str, Any]:
    source_dir = resolve_source_dir(repo["source_dir"])
    source_model = source_dir / "adapter_model.safetensors"
    source_config = source_dir / "adapter_config.json"
    assert_expected(source_model, repo["adapter_model"], "source adapter_model")
    assert_expected(source_config, repo["adapter_config"], "source adapter_config")

    stage_dir = staging_root / f"epoch_{repo['epoch']}"
    stage_dir.mkdir(parents=False, exist_ok=False)
    shutil.copyfile(source_model, stage_dir / "adapter_model.safetensors")
    shutil.copyfile(source_config, stage_dir / "adapter_config.json")
    (stage_dir / "README.md").write_text(
        render_readme(repo, training), encoding="utf-8", newline="\n"
    )

    files = assert_allowlist(stage_dir)
    records = {name: file_record(stage_dir / name) for name in files}
    if records["adapter_model.safetensors"] != repo["adapter_model"]:
        raise UploadError(f"staged adapter_model changed for {repo['repo_id']}")
    if records["adapter_config.json"] != repo["adapter_config"]:
        raise UploadError(f"staged adapter_config changed for {repo['repo_id']}")
    if (stage_dir / "adapter_config.json").read_bytes() != source_config.read_bytes():
        raise UploadError(f"adapter_config is not byte-identical for {repo['repo_id']}")
    return {
        "repo_id": repo["repo_id"],
        "epoch": repo["epoch"],
        "source_dir": str(source_dir),
        "stage_dir": str(stage_dir),
        "files": records,
        "remote_status": "not_started",
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_absent(api: Any, repo_id: str) -> None:
    from huggingface_hub.utils import HfHubHTTPError

    try:
        api.model_info(repo_id=repo_id, token=None)
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return
        raise UploadError(f"cannot check target repository {repo_id}: HTTP {status}") from exc
    raise UploadError(f"target repository already exists; refusing overwrite: {repo_id}")


def upload_and_verify(
    config: dict[str, Any],
    states: list[dict[str, Any]],
    download_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    identity = api.whoami(token=None)
    username = identity.get("name")
    if username != config["namespace"]:
        raise UploadError(
            f"HF identity mismatch: expected={config['namespace']}, actual={username}"
        )
    manifest["hf_identity"] = username

    for state in states:
        require_absent(api, state["repo_id"])
    manifest["remote_preflight"] = "all_targets_absent"
    write_manifest(manifest_path, manifest)

    for state in states:
        repo_id = state["repo_id"]
        stage_dir = Path(state["stage_dir"])
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=False,
            token=None,
        )
        state["remote_status"] = "created_empty"
        write_manifest(manifest_path, manifest)

        operations = [
            CommitOperationAdd(
                path_in_repo=name,
                path_or_fileobj=str(stage_dir / name),
            )
            for name in UPLOADED_FILES
        ]
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=config["commit_message"],
            token=None,
            num_threads=1,
        )
        state["remote_status"] = "uploaded"
        state["hf_commit_sha"] = commit.oid
        state["url"] = f"https://huggingface.co/{repo_id}"
        write_manifest(manifest_path, manifest)

    download_root.mkdir(parents=False, exist_ok=False)
    for state in states:
        repo_id = state["repo_id"]
        revision = state["hf_commit_sha"]
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            token=False,
            files_metadata=True,
        )
        remote_files = sorted(sibling.rfilename for sibling in info.siblings)
        if info.private:
            raise UploadError(f"repository is not public: {repo_id}")
        if remote_files != sorted(REMOTE_FILES):
            raise UploadError(
                f"remote file allowlist mismatch for {repo_id}: {remote_files}"
            )
        if info.sha != revision:
            raise UploadError(
                f"remote revision mismatch for {repo_id}: expected={revision}, actual={info.sha}"
            )

        repo_download_dir = download_root / f"epoch_{state['epoch']}"
        repo_download_dir.mkdir(parents=False, exist_ok=False)
        downloaded: dict[str, Any] = {}
        for name in UPLOADED_FILES:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    repo_type="model",
                    revision=revision,
                    local_dir=repo_download_dir,
                    token=False,
                    force_download=True,
                )
            )
            downloaded[name] = file_record(downloaded_path)
            if downloaded[name] != state["files"][name]:
                raise UploadError(
                    f"anonymous download differs from staging: {repo_id}/{name}"
                )

        state["remote_files"] = remote_files
        state["anonymous_download_files"] = downloaded
        state["remote_status"] = "verified"
        write_manifest(manifest_path, manifest)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create repositories and upload. Without this flag, only stage and verify locally.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise UploadError("run-id may contain only letters, digits, dot, dash, underscore")

    config_path = args.config.resolve()
    config = load_config(config_path)
    artifact_run_dir = ARTIFACT_ROOT / args.run_id
    staging_root = artifact_run_dir / "staging"
    download_root = artifact_run_dir / "anonymous_download"
    log_dir = LOG_ROOT / args.run_id
    manifest_path = log_dir / "manifest.json"
    if artifact_run_dir.exists() or log_dir.exists():
        raise UploadError(f"run-id already exists; refusing reuse: {args.run_id}")
    staging_root.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "preparing",
        "execute": args.execute,
        "started_at_utc": now_utc(),
        "project_root": str(PROJECT_ROOT),
        "git_head_before_upload": git_head(),
        "config_path": str(config_path),
        "config": file_record(config_path),
        "script": file_record(Path(__file__).resolve()),
        "required_uploaded_files": list(UPLOADED_FILES),
        "allowed_hub_managed_files": list(HUB_MANAGED_FILES),
        "required_remote_files": list(REMOTE_FILES),
        "repositories": [],
    }
    write_manifest(manifest_path, manifest)

    try:
        states = [
            stage_repository(repo, config["training"], staging_root)
            for repo in config["repositories"]
        ]
        manifest["repositories"] = states
        manifest["status"] = "prepared"
        write_manifest(manifest_path, manifest)
        if args.execute:
            upload_and_verify(
                config, states, download_root, manifest, manifest_path
            )
            manifest["status"] = "completed"
        else:
            manifest["status"] = "prepared_only"
        manifest["completed_at_utc"] = now_utc()
        write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = now_utc()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = redact(exc)
        write_manifest(manifest_path, manifest)
        raise
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest_path = run(args)
    except Exception as exc:
        print(f"ERROR: {redact(exc)}", file=sys.stderr)
        return 1
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
