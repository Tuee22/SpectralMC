# File: tools/purity/rules.py
"""AST-based purity rule checker.

Implements 5 purity detection rules:
- PUR001: No for loops in business logic
- PUR002: No while loops in business logic
- PUR003: No if statements in business logic
- PUR004: No raise for expected errors
- PUR005: No side effects (print, logger)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tools.purity.classifier import FileTier


class ParentTracker(ast.NodeVisitor):
    """Track parent relationships for AST nodes."""

    def __init__(self) -> None:
        """Initialize parent tracker with empty stack."""
        self.parent_stack: list[ast.AST] = []
        self.parent_map: dict[ast.AST, ast.AST | None] = {}

    def visit(self, node: ast.AST) -> None:
        """Record parent relationship while visiting children.

        Args:
            node: AST node to visit
        """
        parent = self.parent_stack[-1] if self.parent_stack else None
        self.parent_map[node] = parent

        self.parent_stack.append(node)
        self.generic_visit(node)
        self.parent_stack.pop()


@dataclass(frozen=True)
class PurityViolation:
    """Represents a purity rule violation."""

    rule_code: str  # PUR001, PUR002, etc.
    filepath: Path
    line_number: int
    column: int
    context: str  # Code snippet causing violation
    whitelisted: bool = False
    whitelist_reason: str = ""


class PurityChecker(ast.NodeVisitor):
    """AST visitor for purity rule checking.

    Only checks Tier 2 (business logic) files.
    Tier 1 (infrastructure) and Tier 3 (effects) are exempt.
    """

    def __init__(
        self,
        filepath: Path,
        tier: FileTier,
        source: str,
        parent_map: dict[ast.AST, ast.AST | None],
        whitelist: dict[int, str] | None = None,
    ) -> None:
        """Initialize purity checker.

        Args:
            filepath: Path to file being checked
            tier: File tier classification
            source: Source code content
            parent_map: Mapping of AST node -> parent for traversal helpers
            whitelist: Dict of line_number -> justification for whitelisted violations
        """
        self.filepath = filepath
        self.tier = tier
        self.source = source
        self.source_lines = source.splitlines()
        self.parent_map = parent_map
        self.whitelist = whitelist or {}
        self.violations: list[PurityViolation] = []

    def check(self, tree: ast.AST) -> list[PurityViolation]:
        """Check AST for purity violations.

        Args:
            tree: AST to check

        Returns:
            List of purity violations
        """
        # Only check Tier 2 files
        if self.tier != FileTier.TIER2_BUSINESS_LOGIC:
            return []

        self.visit(tree)
        return self.violations

    def _parent_of(self, node: ast.AST) -> ast.AST | None:
        """Return parent of node recorded by ParentTracker."""
        return self.parent_map.get(node)

    def _get_code_context(self, node: ast.AST, max_length: int = 50) -> str:
        """Extract code context from AST node.

        Args:
            node: AST node
            max_length: Maximum context string length

        Returns:
            Code snippet (truncated if needed)
        """
        if not hasattr(node, "lineno"):
            return ""

        line_idx = node.lineno - 1
        if line_idx >= len(self.source_lines):
            return ""

        line: str = self.source_lines[line_idx].strip()
        if len(line) > max_length:
            return line[:max_length] + "..."
        return line

    def _is_whitelisted(self, line_number: int) -> tuple[bool, str]:
        """Check if line is whitelisted.

        Args:
            line_number: Line number to check

        Returns:
            (is_whitelisted, reason)
        """
        reason = self.whitelist.get(line_number, "")
        return (bool(reason), reason)

    def _add_violation(
        self,
        rule_code: str,
        node: ast.AST,
    ) -> None:
        """Add a purity violation.

        Args:
            rule_code: Rule code (PUR001, etc.)
            node: AST node with violation
        """
        line_number = getattr(node, "lineno", 0)
        column = getattr(node, "col_offset", 0)
        context = self._get_code_context(node)
        is_whitelisted, reason = self._is_whitelisted(line_number)

        violation = PurityViolation(
            rule_code=rule_code,
            filepath=self.filepath,
            line_number=line_number,
            column=column,
            context=context,
            whitelisted=is_whitelisted,
            whitelist_reason=reason,
        )
        self.violations.append(violation)

    # Pattern detection helpers

    def _is_guard_clause(self, node: ast.If) -> bool:
        """Check if if statement is a guard clause returning Failure(...).

        Guard clauses are acceptable per purity doctrine (lines 180-186).
        Pattern: Single-statement if body with return Failure(...).

        Args:
            node: If statement node

        Returns:
            True if node is a guard clause, False otherwise

        Example:
            if x < 0:
                return Failure(ValueError("negative"))
        """
        # Guard clause: single-statement body with return
        if len(node.body) != 1:
            return False

        stmt = node.body[0]
        if not isinstance(stmt, ast.Return):
            return False

        # Check if returns Failure(...)
        if stmt.value is None:
            return False

        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Name) and func.id == "Failure":
                return True

        return False

    def _is_cuda_jit_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is @cuda.jit.

        Recognizes both forms:
        - @cuda.jit
        - @cuda.jit(...)

        Args:
            decorator: Decorator expression

        Returns:
            True if decorator is @cuda.jit, False otherwise
        """
        # Simple case: @cuda.jit
        if isinstance(decorator, ast.Attribute):
            if isinstance(decorator.value, ast.Name):
                if decorator.value.id == "cuda" and decorator.attr == "jit":
                    return True

        # Call case: @cuda.jit(...)
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id == "cuda" and func.attr == "jit":
                        return True

        return False

    def _is_cuda_kernel(self, node: ast.AST) -> bool:
        """Check if node is inside a CUDA kernel (@cuda.jit decorated function).

        CUDA kernels are Tier 3 GPU boundaries and exempt from purity rules.
        Imperative patterns (if, for, while) are necessary for GPU efficiency.

        Args:
            node: AST node to check

        Returns:
            True if node is inside CUDA kernel, False otherwise
        """
        # Walk up AST tree to find containing function
        parent = self._parent_of(node)
        while parent:
            if isinstance(parent, ast.FunctionDef):
                # Check decorators for @cuda.jit
                for decorator in parent.decorator_list:
                    if self._is_cuda_jit_decorator(decorator):
                        return True
                break
            parent = self._parent_of(parent)

        return False

    def _has_noreturn_annotation(self, returns: ast.expr) -> bool:
        """Check if return annotation is NoReturn.

        Args:
            returns: Return type annotation expression

        Returns:
            True if annotation is NoReturn, False otherwise
        """
        # Direct name: -> NoReturn
        if isinstance(returns, ast.Name) and returns.id == "NoReturn":
            return True

        # Constant (Python 3.8+): -> typing.NoReturn
        if isinstance(returns, ast.Attribute):
            if returns.attr == "NoReturn":
                return True

        return False

    def _is_boundary_unwrap(self, node: ast.Raise) -> bool:
        """Check if raise is in a boundary unwrap function.

        Boundary unwrap functions convert Result → Exception at system boundaries.
        Pattern: Method named 'unwrap' with return type NoReturn.

        Args:
            node: Raise statement node

        Returns:
            True if raise is in boundary unwrap function, False otherwise

        Example:
            def unwrap(self) -> NoReturn:
                raise RuntimeError(f"Called unwrap() on Failure")
        """
        # Walk up AST tree to find containing function
        parent = self._parent_of(node)
        while parent:
            if isinstance(parent, ast.FunctionDef):
                # Check if function is named 'unwrap'
                if parent.name == "unwrap":
                    # Check return type annotation for NoReturn
                    if parent.returns:
                        return self._has_noreturn_annotation(parent.returns)
                break
            parent = self._parent_of(parent)

        return False

    # PUR001: No for loops
    def visit_For(self, node: ast.For) -> None:
        """Detect for loops in business logic (PUR001).

        Exempt: CUDA kernels (@cuda.jit decorated functions).
        """
        # Skip CUDA kernels (Tier 3 GPU boundary)
        if not self._is_cuda_kernel(node):
            self._add_violation("PUR001", node)
        self.generic_visit(node)

    # PUR002: No while loops
    def visit_While(self, node: ast.While) -> None:
        """Detect while loops in business logic (PUR002).

        Exempt: CUDA kernels (@cuda.jit decorated functions).
        """
        # Skip CUDA kernels (Tier 3 GPU boundary)
        if not self._is_cuda_kernel(node):
            self._add_violation("PUR002", node)
        self.generic_visit(node)

    # PUR003: No if statements
    def visit_If(self, node: ast.If) -> None:
        """Detect if statements in business logic (PUR003).

        Allows:
        - Conditional expressions (handled by visit_IfExp)
        - Match/case (handled by visit_Match)
        - Guard clauses returning Failure(...) (automatic detection)
        - CUDA kernels (@cuda.jit decorated functions)
        """
        # Skip CUDA kernels (Tier 3 GPU boundary)
        if self._is_cuda_kernel(node):
            self.generic_visit(node)
            return

        # Skip guard clauses (acceptable pattern per purity doctrine)
        if not self._is_guard_clause(node):
            self._add_violation("PUR003", node)
        self.generic_visit(node)

    # PUR004: No raise for expected errors
    def visit_Raise(self, node: ast.Raise) -> None:
        """Detect raise statements in business logic (PUR004).

        Allows:
        - Programming error assertions (AssertionError, RuntimeError with "unreachable")
        - Inside assert_never() calls
        - Boundary unwrap functions (method named 'unwrap' with NoReturn)
        """
        # Check if this is inside assert_never() call
        if self._is_inside_assert_never(node):
            self.generic_visit(node)
            return

        # Check if this is a programming error assertion
        if self._is_programming_error(node):
            self.generic_visit(node)
            return

        # Check if this is a boundary unwrap function
        if self._is_boundary_unwrap(node):
            self.generic_visit(node)
            return

        self._add_violation("PUR004", node)
        self.generic_visit(node)

    def _is_inside_assert_never(self, node: ast.AST) -> bool:
        """Check if node is inside assert_never() call.

        This is a simplified check - a full implementation would walk the AST tree.
        For now, we check if the line contains 'assert_never'.
        """
        if not hasattr(node, "lineno"):
            return False

        line_idx = node.lineno - 1
        if line_idx >= len(self.source_lines):
            return False

        return "assert_never" in self.source_lines[line_idx]

    def _is_programming_error(self, node: ast.Raise) -> bool:
        """Check if raise is for programming error (not expected error).

        Programming errors:
        - AssertionError
        - RuntimeError with "unreachable" in message
        - NotImplementedError
        """
        if node.exc is None:
            return False

        # Check exception type
        if isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name):
                # AssertionError, NotImplementedError
                if func.id in ("AssertionError", "NotImplementedError"):
                    return True

                # RuntimeError with "unreachable"
                if func.id == "RuntimeError" and node.exc.args:
                    first_arg = node.exc.args[0]
                    if isinstance(first_arg, ast.Constant):
                        msg = str(first_arg.value).lower()
                        if "unreachable" in msg or "programming error" in msg:
                            return True

        return False

    # PUR005: No side effects (print, logger)
    def visit_Call(self, node: ast.Call) -> None:
        """Detect side effect calls in business logic (PUR005).

        Forbidden:
        - print()
        - logger.info/debug/warning/error()
        """
        # Check for print()
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self._add_violation("PUR005", node)

        # Check for logger.*(...)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == "logger":
                    # logger.info(), logger.debug(), etc.
                    if node.func.attr in (
                        "debug",
                        "info",
                        "warning",
                        "error",
                        "critical",
                    ):
                        self._add_violation("PUR005", node)

        self.generic_visit(node)


def check_file_purity(
    filepath: Path,
    tier: FileTier,
    whitelist: dict[int, str] | None = None,
) -> list[PurityViolation]:
    """Check a file for purity violations.

    Args:
        filepath: Path to Python file
        tier: File tier classification
        whitelist: Dict of line_number -> justification for whitelisted violations

    Returns:
        List of purity violations
    """
    # Only check Tier 2 files
    if tier != FileTier.TIER2_BUSINESS_LOGIC:
        return []

    # Read source code
    source = filepath.read_text(encoding="utf-8")

    # Parse AST
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        # Skip files with syntax errors (will be caught by MyPy/Black)
        return []

    # Add parent tracking for pattern detection
    tracker = ParentTracker()
    tracker.visit(tree)

    # Run checker
    checker = PurityChecker(
        filepath=filepath,
        tier=tier,
        source=source,
        parent_map=tracker.parent_map,
        whitelist=whitelist,
    )
    return checker.check(tree)
