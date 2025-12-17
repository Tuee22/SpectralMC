"""
Fail if direct Pydantic model construction is used in src/.

Allowed:
- validate_model(...) / build_* helpers returning Result
- Direct BaseModel instantiation in tests/ and examples/ (optionally permitted)

Disallowed:
- <BaseModelSubclass>(...) in src/ without going through a builder
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _iter_py_files(root: Path, allow_tests: bool) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if not allow_tests and ("tests" in rel.parts or "examples" in rel.parts):
            continue
        files.append(path)
    return files


def _collect_base_model_classes(files: list[Path]) -> set[str]:
    """Collect class names that subclass BaseModel in the scanned files."""
    names: set[str] = set()
    for file in files:
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseModel":
                        names.add(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                        names.add(node.name)
    return names


def _get_caller_name(node: ast.Call) -> str | None:
    """
    Extract callable name from AST Call node.

    Handles both direct names (Model()) and attribute access (module.Model()).

    Args:
        node: AST Call node

    Returns:
        Callable name if extractable, None otherwise
    """
    match node.func:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=attr):
            return attr
        case _:
            return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect direct Pydantic model construction.")
    parser.add_argument("--root", type=Path, default=Path("src"), help="Root directory to scan")
    parser.add_argument(
        "--allow-tests",
        action="store_true",
        help="Allow direct construction in tests/examples",
    )
    args = parser.parse_args()

    files = _iter_py_files(args.root, args.allow_tests)
    base_model_classes = _collect_base_model_classes(files)
    allowed_callers = {"validate_model", "build_"}

    offending: list[tuple[Path, int, str]] = []
    for file in files:
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if not name:
                continue

            if name not in base_model_classes:
                continue

            # Skip builder/helper invocations (improved to handle attribute access)
            caller_name = _get_caller_name(node)
            if caller_name is not None and any(
                caller_name.startswith(prefix) for prefix in allowed_callers
            ):
                continue
            offending.append((file, node.lineno, name))

    if offending:
        print("Disallowed direct Pydantic constructions found:", file=sys.stderr)
        for path, lineno, name in offending:
            print(f"  {path}:{lineno}: {name}(...)", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
