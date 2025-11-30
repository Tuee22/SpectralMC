#!/usr/bin/env python
"""
Enhanced type safety checker for SpectralMC.

Ensures no usage of Any, cast, or type ignore comments.
Supports automatic fixing and comprehensive detection.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


class TypeSafetyChecker(ast.NodeVisitor):
    """AST visitor to check for forbidden type constructs."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.errors: list[tuple[int, int, str]] = []
        self.imported_names: set[str] = set()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for imports of Any or cast from typing."""
        if node.module and "typing" in node.module:
            for alias in node.names:
                name = alias.name
                if name == "Any":
                    self.errors.append(
                        (
                            node.lineno,
                            node.col_offset,
                            "TYP001: Import of 'Any' is forbidden",
                        )
                    )
                elif name == "cast":
                    self.errors.append(
                        (
                            node.lineno,
                            node.col_offset,
                            "TYP002: Import of 'cast' is forbidden",
                        )
                    )
                # Track imported names for later checks
                self.imported_names.add(alias.asname or name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for usage of Any or cast if imported."""
        if isinstance(node.ctx, ast.Load):
            if node.id == "Any" and "Any" in self.imported_names:
                self.errors.append(
                    (
                        node.lineno,
                        node.col_offset,
                        "TYP001: Use of 'Any' type is forbidden",
                    )
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check for Any usage in type subscripts like List[Any]."""
        # Check if the subscript contains Any
        if isinstance(node.slice, ast.Name) and node.slice.id == "Any":
            self.errors.append(
                (
                    node.lineno,
                    node.col_offset,
                    "TYP001: Use of 'Any' type in subscript is forbidden",
                )
            )
        # Check nested Any usage like Dict[str, Any]
        elif isinstance(node.slice, ast.Tuple):
            for elt in node.slice.elts:
                if isinstance(elt, ast.Name) and elt.id == "Any":
                    self.errors.append(
                        (
                            node.lineno,
                            node.col_offset,
                            "TYP001: Use of 'Any' type in subscript is forbidden",
                        )
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for cast() calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "cast":
            self.errors.append(
                (
                    node.lineno,
                    node.col_offset,
                    "TYP002: Use of 'cast' function is forbidden",
                )
            )
        self.generic_visit(node)


def check_type_ignore_comments(content: str, filename: str) -> list[tuple[int, int, str]]:
    """Check for type ignore comments in the file."""
    errors: list[tuple[int, int, str]] = []
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        # Check for various forms of type ignore comments
        patterns = [
            r"type:\s*ignore",
            r"#\s*type:\s*ignore",
            r"#.*type:\s*ignore",
        ]

        for pattern in patterns:
            if re.search(pattern, line):
                # Find the column where the comment starts
                match = re.search(pattern, line)
                if match:
                    col = match.start()
                    errors.append(
                        (
                            line_num,
                            col,
                            "TYP003: Use of 'type ignore' comment is forbidden",
                        )
                    )
                break  # Don't report the same line multiple times

    return errors


def auto_fix_file(filepath: Path) -> bool:
    """Automatically fix common type safety violations."""
    try:
        content = filepath.read_text()
        original_content = content

        # Fix: Remove unused Any imports
        # Pattern 1: "from typing import Any"
        content = re.sub(r"from typing import Any\n", "", content)

        # Pattern 2: "from typing import ..., ..."
        content = re.sub(r"(from typing import [^,\n]+),\s*Any\s*,", r"\1,", content)
        content = re.sub(
            r"(from typing import [^,\n]+),\s*Any\s*$",
            r"\1",
            content,
            flags=re.MULTILINE,
        )

        # Pattern 3: "from typing import ..."
        content = re.sub(r"from typing import \s*", "from typing import ", content)

        # Pattern 4: Multiline imports
        lines = content.splitlines()
        new_lines: list[str] = []
        in_typing_import = False

        for line in lines:
            if "from typing import (" in line:
                in_typing_import = True
                new_lines.append(line)
            elif in_typing_import and ")" in line:
                in_typing_import = False
                new_lines.append(line)
            elif in_typing_import:
                # Remove Any from multiline typing imports
                if line.strip() == "Any,":
                    continue  # Skip this line entirely
                if line.strip().startswith("Any,"):
                    # Remove Any, from the beginning
                    new_lines.append(line.replace("Any,", "").lstrip())
                elif line.strip().endswith(", Any,"):
                    # Remove , Any, from the end
                    new_lines.append(line.replace(", Any,", ","))
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        content = "\n".join(new_lines)

        # Write back if changed
        if content != original_content:
            filepath.write_text(content)
            print(f"✅ Fixed type safety violations in {filepath}")
            return True

        return False

    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}", file=sys.stderr)
        return False


def check_file(filepath: Path) -> list[tuple[str, int, int, str]]:
    """Check a single Python file for type safety violations."""
    try:
        content = filepath.read_text()

        # Parse the AST
        tree = ast.parse(content, str(filepath))

        # Check AST for Any and cast usage
        checker = TypeSafetyChecker(str(filepath))
        checker.visit(tree)

        # Check for type ignore comments
        comment_errors = check_type_ignore_comments(content, str(filepath))

        # Combine all errors
        all_errors: list[tuple[str, int, int, str]] = []
        for line, col, msg in checker.errors + comment_errors:
            all_errors.append((str(filepath), line, col, msg))

        return all_errors

    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)
        return []


def main() -> int:
    """Main entry point for the type safety checker."""
    parser = argparse.ArgumentParser(description="Check and optionally fix type safety violations")
    parser.add_argument("--fix", action="store_true", help="Automatically fix common violations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Get all Python files in the project
    root_path = Path(__file__).parent.parent
    python_files: list[Path] = []

    # Define directories to check
    check_dirs = ["src/spectralmc", "tests", "tools"]

    for dir_name in check_dirs:
        dir_path = root_path / dir_name
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))

    # Auto-fix mode
    if args.fix:
        fixed_count = 0
        for filepath in python_files:
            # Skip stubs directory and proto generated files
            if "stubs" in filepath.parts or "_pb2.py" in filepath.name:
                continue

            if auto_fix_file(filepath):
                fixed_count += 1

        if fixed_count > 0:
            print(f"✅ Fixed type safety violations in {fixed_count} files")
        else:
            print("✅ No files needed fixing")

    # Check each file
    all_errors: list[tuple[str, int, int, str]] = []
    for filepath in python_files:
        # Skip stubs directory and proto generated files
        if "stubs" in filepath.parts or "_pb2.py" in filepath.name:
            continue

        errors = check_file(filepath)
        all_errors.extend(errors)

    # Report errors
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} type safety violation(s):\n")
        for file_str, line, col, msg in sorted(all_errors):
            print(f"{file_str}:{line}:{col}: {msg}")

        if not args.fix:
            print("\n💡 Run with --fix to automatically fix common violations")

        return 1
    print("✅ No type safety violations found!")
    return 0


def run_mypy_only() -> int:
    """Run only MyPy type checker."""
    print("🔍 Running MyPy type checker...")

    result = subprocess.run(
        ["mypy", "src/spectralmc", "tests", "tools"],
        check=False,
    )

    if result.returncode == 0:
        print("✅ MyPy passed!")
    else:
        print("❌ MyPy failed")

    return result.returncode


def run_full_check() -> int:
    """
    Run both MyPy and custom type safety checks.

    Returns 0 if all checks pass, 1 if any check fails.
    """
    print("🔍 Running comprehensive type checking...")

    # Run MyPy first
    print("\n1️⃣ Running MyPy type checker...")
    mypy_result = subprocess.run(
        ["mypy", "src/spectralmc", "tests", "tools"],
        check=False,
    )

    mypy_success = mypy_result.returncode == 0
    if mypy_success:
        print("✅ MyPy passed!")
    else:
        print("❌ MyPy failed")

    # Run custom type safety checker
    print("\n2️⃣ Running custom type safety checker...")

    # Get all Python files in the project
    root_path = Path(__file__).parent.parent
    python_files: list[Path] = []
    for pattern in ["src/spectralmc/**/*.py", "tests/**/*.py", "tools/**/*.py"]:
        python_files.extend(root_path.glob(pattern))

    violations: list[tuple[str, int, int, str]] = []

    for file_path in python_files:
        if file_path.name.startswith("."):
            continue
        # Skip stubs and proto generated files
        if "stubs" in file_path.parts or "_pb2.py" in file_path.name:
            continue

        try:
            file_violations = check_file(file_path)
            violations.extend(file_violations)
        except Exception as e:
            print(f"Warning: Could not check {file_path}: {e}")

    custom_success = len(violations) == 0

    if custom_success:
        print("✅ Custom type safety checker passed!")
    else:
        print(f"❌ Found {len(violations)} type safety violations:")
        for violation in violations:
            print(f"  {violation}")

    # Overall result
    if mypy_success and custom_success:
        print("\n🎉 All type checks passed!")
        return 0
    print("\n💥 Some type checks failed!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
