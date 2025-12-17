#!/usr/bin/env python3
"""
Immutability doctrine enforcement.

Fails fast if any immutability bypass patterns are present in code paths.
See documents/engineering/immutability_doctrine.md for policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

FORBIDDEN_SUBSTRINGS: dict[str, str] = {
    "object.__setattr__": "Immutability bypass using object.__setattr__",
    "__dict__[": "Immutability bypass via __dict__ mutation",
}

VARS_MUTATION_MARKERS: tuple[str, ...] = ("[", ".update", ".pop", ".setdefault")

CODE_ROOTS: tuple[Path, ...] = (
    Path("src"),
    Path("tests"),
    Path("examples"),
)


@dataclass(frozen=True)
class Violation:
    """Immutability violation instance."""

    path: Path
    line: int
    code: str
    message: str


def _iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield Python source files under provided roots."""
    for base in paths:
        yield from base.rglob("*.py")


def _scan_file(path: Path) -> list[Violation]:
    """Scan a single file for forbidden immutability bypass patterns."""
    violations: list[Violation] = []
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        # Skip lines with immutability exception marker
        if "# immutability-exception:" in line:
            continue

        for pattern, message in FORBIDDEN_SUBSTRINGS.items():
            if pattern in line:
                violations.append(Violation(path=path, line=idx, code=pattern, message=message))
        if "vars(" in line and any(marker in line for marker in VARS_MUTATION_MARKERS):
            violations.append(
                Violation(
                    path=path,
                    line=idx,
                    code="vars(",
                    message="Immutability bypass via vars() mutation",
                )
            )
    return violations


def main() -> int:
    """Run immutability audit; return non-zero if violations detected."""
    repo_root = Path(__file__).parent.parent
    roots = [repo_root / root for root in CODE_ROOTS if (repo_root / root).exists()]

    violations: list[Violation] = []
    for file_path in _iter_python_files(roots):
        violations.extend(_scan_file(file_path))

    if violations:
        print("❌ Immutability doctrine violations detected:")
        for violation in violations:
            rel_path = violation.path.relative_to(repo_root)
            print(f"   {rel_path}:{violation.line}: {violation.message} " f"[{violation.code}]")
        print("\nRefer to documents/engineering/immutability_doctrine.md")
        return 1

    print("✅ Immutability doctrine audit passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
