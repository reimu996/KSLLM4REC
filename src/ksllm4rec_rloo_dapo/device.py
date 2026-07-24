"""Stable CUDA device identity for binding measured gates to formal training."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from uuid import UUID

import torch


_GPU_IDENTITY_FIELDS = {
    "device",
    "name",
    "uuid",
    "total_memory",
    "compute_capability",
}


def validate_gpu_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one canonical, JSON-safe GPU identity."""

    if not isinstance(value, Mapping) or set(value) != _GPU_IDENTITY_FIELDS:
        raise ValueError("GPU identity has an invalid schema.")
    device = value["device"]
    name = value["name"]
    raw_uuid = value["uuid"]
    total_memory = value["total_memory"]
    capability = value["compute_capability"]
    if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("GPU identity device must be a resolved CUDA index.")
    if not isinstance(name, str) or not name:
        raise ValueError("GPU identity name must be non-empty.")
    if not isinstance(raw_uuid, str):
        raise ValueError("GPU identity UUID must be a string.")
    try:
        canonical_uuid = str(UUID(raw_uuid.removeprefix("GPU-")))
    except ValueError as exc:
        raise ValueError("GPU identity UUID is invalid.") from exc
    if type(total_memory) is not int or total_memory <= 0:
        raise ValueError("GPU identity total_memory must be a positive byte count.")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(part) is not int or part < 0 for part in capability)
    ):
        raise ValueError("GPU identity compute_capability must contain two integers.")
    return {
        "device": device,
        "name": name,
        "uuid": canonical_uuid,
        "total_memory": total_memory,
        "compute_capability": list(capability),
    }


def gpu_identity(device: str | torch.device = "cuda:0") -> dict[str, Any]:
    """Read the physical CUDA identity used by one gate or training invocation."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU identity cannot be established.")
    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("GPU identity requires a CUDA device.")
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(properties, "uuid", None)
    if raw_uuid is None:
        raise RuntimeError("CUDA device properties do not expose a UUID.")
    if isinstance(raw_uuid, bytes):
        raw_uuid = str(UUID(bytes=raw_uuid))
    return validate_gpu_identity(
        {
            "device": f"cuda:{index}",
            "name": str(properties.name),
            "uuid": str(raw_uuid),
            "total_memory": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    )


def require_gpu_identity(
    expected: Mapping[str, Any],
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    """Reject formal training when its current GPU differs from measured gates."""

    required = validate_gpu_identity(expected)
    actual = gpu_identity(device)
    if actual != required:
        raise RuntimeError("Current GPU identity differs from the gate reports.")
    return actual


__all__ = ["gpu_identity", "require_gpu_identity", "validate_gpu_identity"]
