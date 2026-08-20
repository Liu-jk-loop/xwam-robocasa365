#!/usr/bin/env python3
"""Check explicit outer-shell to ``srun ... env ... bash -lc`` contracts.

This catches a class of failures that ``bash -n`` cannot detect: a variable is
defined by the submitting shell and read by a single-quoted node shell, but is
not forwarded through the explicit ``env`` boundary.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


OUTER_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:(?:export|readonly)\s+)?([A-Z][A-Z0-9_]*)=", re.MULTILINE
)
INNER_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:(?:export|readonly|local)\s+)?([A-Z][A-Z0-9_]*)=", re.MULTILINE
)
VARIABLE_REFERENCE_RE = re.compile(
    r"(?<!\\)\$(?:\{([A-Z][A-Z0-9_]*)(?:[^}]*)\}|([A-Z][A-Z0-9_]*))"
)
ENV_ASSIGNMENT_RE = re.compile(r"(?:^|[\s\\])([A-Z][A-Z0-9_]*)=")
SRUN_START_RE = re.compile(r"^\s*(?:if\s+)?srun\b")
BASH_LC_SINGLE_RE = re.compile(r"\bbash\s+-lc\s+'\s*$")
SINGLE_QUOTE_END_RE = re.compile(r"^\s*'\s*(?:(?:[0-9]*>|[|&;]).*)?$")
QUOTED_HEREDOC_RE = re.compile(r"<<-?\s*(['\"])([A-Za-z_][A-Za-z0-9_]*)\1")


@dataclass(frozen=True)
class ShellBlock:
    command_start_line: int
    start_line: int
    end_line: int
    command: str
    body: str


@dataclass(frozen=True)
class ContractError:
    path: Path
    line: int
    missing: tuple[str, ...]


def find_shell_blocks(text: str) -> list[ShellBlock]:
    lines = text.splitlines(keepends=True)
    blocks: list[ShellBlock] = []
    index = 0
    while index < len(lines):
        if not BASH_LC_SINGLE_RE.search(lines[index].rstrip("\n")):
            index += 1
            continue

        command_start = index
        while command_start >= 0 and not SRUN_START_RE.match(lines[command_start]):
            command_start -= 1
        if command_start < 0:
            index += 1
            continue

        end = index + 1
        while end < len(lines) and not SINGLE_QUOTE_END_RE.match(lines[end]):
            end += 1
        if end >= len(lines):
            raise ValueError(f"unterminated single-quoted bash -lc block at line {index + 1}")

        blocks.append(
            ShellBlock(
                command_start_line=command_start + 1,
                start_line=index + 1,
                end_line=end + 1,
                command="".join(lines[command_start : index + 1]),
                body="".join(lines[index + 1 : end]),
            )
        )
        index = end + 1
    return blocks


def mask_blocks(text: str, blocks: list[ShellBlock]) -> str:
    lines = text.splitlines(keepends=True)
    for block in blocks:
        for line_index in range(block.command_start_line - 1, block.end_line):
            lines[line_index] = "\n" if lines[line_index].endswith("\n") else ""
    return "".join(lines)


def mask_quoted_heredocs(body: str) -> str:
    """Remove quoted heredoc contents because the shell does not expand them."""

    output: list[str] = []
    delimiter: str | None = None
    for line in body.splitlines(keepends=True):
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            output.append("\n" if line.endswith("\n") else "")
            continue
        match = QUOTED_HEREDOC_RE.search(line)
        output.append(line)
        if match:
            delimiter = match.group(2)
    return "".join(output)


def variable_references(body: str) -> set[str]:
    references: set[str] = set()
    for match in VARIABLE_REFERENCE_RE.finditer(mask_quoted_heredocs(body)):
        references.add(match.group(1) or match.group(2))
    return references


def passed_environment(command: str) -> set[str]:
    env_match = re.search(r"\benv\b", command)
    if env_match is None:
        return set()
    return set(ENV_ASSIGNMENT_RE.findall(command[env_match.end() :]))


def audit_path(path: Path) -> list[ContractError]:
    text = path.read_text(encoding="utf-8")
    blocks = find_shell_blocks(text)
    outer_assignments = set(OUTER_ASSIGNMENT_RE.findall(mask_blocks(text, blocks)))
    errors: list[ContractError] = []

    for block in blocks:
        if re.search(r"\benv\b", block.command) is None:
            continue
        references = variable_references(block.body)
        inner_assignments = set(INNER_ASSIGNMENT_RE.findall(block.body))
        passed = passed_environment(block.command)
        missing = tuple(sorted((references & outer_assignments) - inner_assignments - passed))
        if missing:
            errors.append(ContractError(path=path, line=block.start_line, missing=missing))
    return errors


def resolve_paths(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return sorted(Path("deployment/clariden").glob("*.sbatch"))

    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.sbatch")))
        else:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Slurm files or directories (default: deployment/clariden/*.sbatch)",
    )
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    if not paths:
        print("error: no Slurm files found", file=sys.stderr)
        return 2

    errors: list[ContractError] = []
    try:
        for path in paths:
            errors.extend(audit_path(path))
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: unable to audit Slurm environment contracts: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("error: incomplete explicit Slurm environment contract", file=sys.stderr)
        for error in errors:
            print(
                f"{error.path}:{error.line}: node shell reads outer variable(s) not "
                f"forwarded by srun env: {', '.join(error.missing)}",
                file=sys.stderr,
            )
        print(
            "hint: add NAME=\"$NAME\" after the srun env boundary, or define NAME "
            "inside the node shell",
            file=sys.stderr,
        )
        return 1

    print(f"slurm-env-contract check: passed ({len(paths)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
