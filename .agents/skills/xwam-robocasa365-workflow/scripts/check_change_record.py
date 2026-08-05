#!/usr/bin/env python3
"""Require progress documentation for implementation and workflow changes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REQUIRED_DOCS = {"docs/CHANGELOG.md", "docs/PROGRESS.md"}
MATERIAL_PREFIXES = (
    ".agents/",
    "configs/",
    "data/",
    "evaluation/",
    "modules/",
    "runners/",
    "scripts/",
    "utils/",
)
MATERIAL_FILES = {
    ".gitignore",
    ".gitmodules",
    "AGENTS.md",
    "requirements.txt",
}


def git_lines(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def changed_paths(base: str) -> set[str]:
    paths: set[str] = set()
    paths.update(git_lines("diff", "--name-only", f"{base}...HEAD"))
    paths.update(git_lines("diff", "--name-only"))
    paths.update(git_lines("diff", "--cached", "--name-only"))
    paths.update(git_lines("ls-files", "--others", "--exclude-standard"))
    return paths


def is_material(path: str) -> bool:
    return path in MATERIAL_FILES or path.startswith(MATERIAL_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main", help="Base Git ref used for the branch diff (default: main).")
    args = parser.parse_args()

    if not Path(".git").exists():
        print("error: run this check from the repository root", file=sys.stderr)
        return 2

    try:
        paths = changed_paths(args.base)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or str(exc)
        print(f"error: unable to inspect Git diff: {message}", file=sys.stderr)
        return 2

    material = sorted(path for path in paths if is_material(path))
    if not material:
        print("change-record check: no implementation or workflow changes detected")
        return 0

    missing = sorted(REQUIRED_DOCS - paths)
    if missing:
        print("error: implementation/workflow changes require documentation updates", file=sys.stderr)
        print(f"material changes: {', '.join(material)}", file=sys.stderr)
        print(f"missing updates: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"change-record check: passed ({len(material)} material path(s), required docs updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
