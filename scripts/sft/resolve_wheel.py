#!/usr/bin/env python3
"""Resolve an exact Linux x86_64 wheel URL and its publisher SHA256."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def read_url(url: str, attempts: int = 20) -> bytes:
    """Read a small package-index response with bounded retry delays."""

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Could not read package index after {attempts} attempts: {url}"
    ) from last_error


def resolve_pypi(package: str, version: str) -> tuple[str, str, str, int]:
    endpoint = f"https://pypi.org/pypi/{urllib.parse.quote(package)}/{urllib.parse.quote(version)}/json"
    payload = json.loads(read_url(endpoint))
    candidates = []
    for item in payload["urls"]:
        filename = item["filename"]
        if item["packagetype"] != "bdist_wheel" or "x86_64" not in filename:
            continue
        if (
            not (filename.startswith("triton-") and "cp311-cp311" in filename)
            and "py3-none" not in filename
        ):
            continue
        candidates.append(
            (item["url"], filename, item["digests"]["sha256"], int(item["size"]))
        )
    if len(candidates) == 1:
        return candidates[0]

    # Some NVIDIA releases publish both a legacy manylinux2014 filename and an
    # equivalent dual-tag filename. Prefer the dual-tag artifact deterministically.
    dual_tag = [item for item in candidates if ".manylinux_2_17_x86_64.whl" in item[1]]
    if len(dual_tag) == 1:
        return dual_tag[0]

    names = ", ".join(item[1] for item in candidates)
    raise RuntimeError(
        f"Expected one compatible wheel for {package}=={version}, found {len(candidates)}: {names}"
    )


def _content_length(url: str, attempts: int = 20) -> int:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=120) as response:
                value = response.headers.get("Content-Length")
                if value is None:
                    raise RuntimeError(f"No Content-Length returned for {url}")
                return int(value)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Could not read wheel size after {attempts} attempts: {url}"
    ) from last_error


def resolve_simple(
    index_url: str, expected_filename: str, expected_size: int | None
) -> tuple[str, str, str, int]:
    html = read_url(index_url).decode("utf-8")
    parser = LinkParser()
    parser.feed(html)
    for href in parser.links:
        absolute = urllib.parse.urljoin(index_url, href)
        parsed = urllib.parse.urlparse(absolute)
        filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        if filename != expected_filename:
            continue
        fragments = urllib.parse.parse_qs(parsed.fragment)
        hashes = fragments.get("sha256", [])
        if len(hashes) != 1:
            raise RuntimeError(f"Missing publisher SHA256 for {expected_filename}")
        size = expected_size if expected_size is not None else _content_length(absolute)
        return absolute, filename, hashes[0], size
    raise RuntimeError(f"Wheel {expected_filename} was not found at {index_url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pypi-package")
    group.add_argument("--simple-index")
    parser.add_argument("--version")
    parser.add_argument("--filename")
    parser.add_argument("--size", type=int)
    args = parser.parse_args()
    if args.pypi_package:
        if not args.version:
            parser.error("--version is required with --pypi-package")
        result = resolve_pypi(args.pypi_package, args.version)
    else:
        if not args.filename:
            parser.error("--filename is required with --simple-index")
        result = resolve_simple(args.simple_index, args.filename, args.size)
    sys.stdout.write("\t".join(str(value) for value in result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
