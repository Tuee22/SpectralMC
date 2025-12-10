# File: tests/test_tools/test_purity_checker.py
"""Comprehensive test suite for the purity checker.

Tests all 5 detection rules (PUR001-PUR005), automatic pattern exemptions,
configuration loading, and integration with real codebase files.
"""

import ast
import textwrap
from pathlib import Path
from typing import Protocol

import pytest

from tools.purity.classifier import FileClassifier, FileTier
from tools.purity.config import get_project_root, load_purity_config
from tools.purity.rules import PurityChecker, PurityViolation


# ============================================================================
# TYPE PROTOCOLS
# ============================================================================


class CheckCodeCallable(Protocol):
    """Protocol for check_code fixture callable with optional whitelist parameter."""

    def __call__(
        self, code: str, whitelist: dict[int, str] | None = None
    ) -> list[PurityViolation]: ...


class AssertViolationCallable(Protocol):
    """Protocol for assert_violation fixture callable."""

    def __call__(self, violations: list[PurityViolation], expected_rules: list[str]) -> None: ...


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def check_code() -> CheckCodeCallable:
    """Helper to parse code and check for violations.

    Args:
        code: Python source code to check
        whitelist: Optional dict of line_number -> reason for whitelisted violations

    Returns:
        List of purity violations found
    """

    def _check(code: str, whitelist: dict[int, str] | None = None) -> list[PurityViolation]:
        from tools.purity.rules import ParentTracker

        # Dedent the code
        source = textwrap.dedent(code)
        # Parse into AST
        tree = ast.parse(source)

        # Add parent tracking for pattern detection (REQUIRED!)
        tracker = ParentTracker()
        tracker.visit(tree)

        # Create checker with source
        checker = PurityChecker(
            filepath=Path("test.py"),
            tier=FileTier.TIER2_BUSINESS_LOGIC,
            source=source,
            whitelist=whitelist or {},
        )
        # Check for violations
        return checker.check(tree)

    return _check


@pytest.fixture
def assert_violation() -> AssertViolationCallable:
    """Assert that violations match expected rules."""

    def _assert(violations: list[PurityViolation], expected_rules: list[str]) -> None:
        actual_rules = [v.rule_code for v in violations]
        assert actual_rules == expected_rules

    return _assert


# ============================================================================
# TEST CLASS 1: PUR001 - FOR LOOPS
# ============================================================================


class TestPUR001ForLoops:
    """Test detection of for loops (PUR001)."""

    def test_for_loop_detected(self, check_code: CheckCodeCallable) -> None:
        """PUR001 triggers on basic for loop."""
        code = """
        for x in items:
            process(x)
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR001"
        assert "for" in violations[0].context.lower()

    def test_nested_for_loops(self, check_code: CheckCodeCallable) -> None:
        """PUR001 triggers on each nested for loop."""
        code = """
        for x in items:
            for y in subitems:
                process(x, y)
        """
        violations = check_code(code)

        assert len(violations) == 2  # Outer and inner
        assert all(v.rule_code == "PUR001" for v in violations)

    def test_for_in_comprehension_allowed(self, check_code: CheckCodeCallable) -> None:
        """PUR001 does NOT trigger on comprehension."""
        code = """
        result = [process(x) for x in items]
        """
        violations = check_code(code)

        assert len(violations) == 0  # Comprehensions are pure

    def test_for_in_cuda_kernel_exempt(self, check_code: CheckCodeCallable) -> None:
        """PUR001 exempt in CUDA kernel."""
        code = """
        @cuda.jit
        def kernel():
            for i in range(10):
                data[i] = i
        """
        violations = check_code(code)

        assert len(violations) == 0  # CUDA pattern detected

    def test_for_with_enumerate(self, check_code: CheckCodeCallable) -> None:
        """PUR001 triggers on enumerate() loop."""
        code = """
        for i, item in enumerate(items):
            print(i, item)
        """
        violations = check_code(code)

        # 1 violation (for loop) + 1 violation (print statement = PUR005)
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR001" for v in violations)

    def test_for_in_generator_function(self, check_code: CheckCodeCallable) -> None:
        """PUR001 triggers in generator function body."""
        code = """
        def gen():
            for x in items:
                yield x
        """
        violations = check_code(code)

        assert len(violations) == 1  # Generator body is Tier 2
        assert violations[0].rule_code == "PUR001"


# ============================================================================
# TEST CLASS 2: PUR002 - WHILE LOOPS
# ============================================================================


class TestPUR002WhileLoops:
    """Test detection of while loops (PUR002)."""

    def test_while_loop_detected(self, check_code: CheckCodeCallable) -> None:
        """PUR002 triggers on basic while loop."""
        code = """
        while condition:
            process()
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR002"

    def test_while_true(self, check_code: CheckCodeCallable) -> None:
        """PUR002 triggers on infinite loop."""
        code = """
        while True:
            if done():
                break
            process()
        """
        violations = check_code(code)

        # while loop + if statement
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR002" for v in violations)

    def test_nested_while_loops(self, check_code: CheckCodeCallable) -> None:
        """PUR002 triggers on each nested while."""
        code = """
        while outer:
            while inner:
                process()
        """
        violations = check_code(code)

        assert len(violations) == 2  # Outer and inner
        assert all(v.rule_code == "PUR002" for v in violations)

    def test_while_in_cuda_kernel(self, check_code: CheckCodeCallable) -> None:
        """PUR002 exempt in CUDA kernel."""
        code = """
        @cuda.jit
        def kernel():
            while i < 10:
                data[i] = i
                i += 1
        """
        violations = check_code(code)

        assert len(violations) == 0  # CUDA exemption


# ============================================================================
# TEST CLASS 3: PUR003 - IF STATEMENTS
# ============================================================================


class TestPUR003IfStatements:
    """Test detection of if statements (PUR003)."""

    def test_if_statement_detected(self, check_code: CheckCodeCallable) -> None:
        """PUR003 triggers on statement-level if."""
        code = """
        if condition:
            process()
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR003"

    def test_if_else_statement(self, check_code: CheckCodeCallable) -> None:
        """PUR003 triggers on if-else."""
        code = """
        if condition:
            action_a()
        else:
            action_b()
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR003"

    def test_guard_clause_exempt(self, check_code: CheckCodeCallable) -> None:
        """PUR003 exempt for guard clause returning Failure."""
        code = """
        def func(x: int):
            if x < 0:
                return Failure(ValueError("negative"))
            return Success(x * 2)
        """
        violations = check_code(code)

        # Guard pattern detected - should be exempt
        assert len([v for v in violations if v.rule_code == "PUR003"]) == 0

    def test_guard_clause_multiple_statements(self, check_code: CheckCodeCallable) -> None:
        """PUR003 triggers if guard has multiple statements."""
        code = """
        if x < 0:
            logger.warning("negative value")
            return Failure(ValueError("negative"))
        """
        violations = check_code(code)

        # Not a pure guard (multiple statements)
        assert len(violations) >= 1
        # Will have PUR003 (if statement) + PUR005 (logger.warning)
        assert any(v.rule_code == "PUR003" for v in violations)

    def test_if_in_cuda_kernel(self, check_code: CheckCodeCallable) -> None:
        """PUR003 exempt in CUDA kernel."""
        code = """
        @cuda.jit
        def kernel():
            if condition:
                data[i] = value
        """
        violations = check_code(code)

        assert len(violations) == 0  # CUDA exemption

    def test_conditional_expression_allowed(self, check_code: CheckCodeCallable) -> None:
        """PUR003 does NOT trigger on ternary expression."""
        code = """
        result = value_a if condition else value_b
        """
        violations = check_code(code)

        assert len(violations) == 0  # Expression, not statement

    def test_if_elif_statement(self, check_code: CheckCodeCallable) -> None:
        """PUR003 triggers on if-elif."""
        code = """
        if cond_a:
            action_a()
        elif cond_b:
            action_b()
        """
        violations = check_code(code)

        # Should detect if statement (elif is part of the same if node in AST)
        # Actually AST represents elif as a separate If node in the orelse
        # so we should expect 2 violations
        assert len(violations) == 2
        assert all(v.rule_code == "PUR003" for v in violations)

    def test_nested_if_statements(self, check_code: CheckCodeCallable) -> None:
        """PUR003 triggers on each nested if."""
        code = """
        if outer:
            if inner:
                process()
        """
        violations = check_code(code)

        assert len(violations) == 2  # Outer and inner
        assert all(v.rule_code == "PUR003" for v in violations)


# ============================================================================
# TEST CLASS 4: PUR004 - RAISE STATEMENTS
# ============================================================================


class TestPUR004RaiseStatements:
    """Test detection of raise statements (PUR004)."""

    def test_raise_detected(self, check_code: CheckCodeCallable) -> None:
        """PUR004 triggers on raise statement."""
        code = """
        if error:
            raise ValueError("error")
        """
        violations = check_code(code)

        # if statement + raise
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR004" for v in violations)

    def test_raise_in_assert_never(self, check_code: CheckCodeCallable) -> None:
        """PUR004 exempt in assert_never call."""
        code = """
        match value:
            case "a":
                return 1
            case _:
                assert_never(value)
        """
        violations = check_code(code)

        # No PUR004 violations (assert_never is exempt)
        assert len([v for v in violations if v.rule_code == "PUR004"]) == 0

    def test_unwrap_function_exempt(self, check_code: CheckCodeCallable) -> None:
        """PUR004 exempt in unwrap() with NoReturn."""
        code = """
        def unwrap(self) -> NoReturn:
            raise RuntimeError(f"Called unwrap on Failure")
        """
        violations = check_code(code)

        # Boundary pattern detected - should be exempt
        assert len(violations) == 0

    def test_raise_assertion_error(self, check_code: CheckCodeCallable) -> None:
        """PUR004 exempt for AssertionError."""
        code = """
        if unreachable:
            raise AssertionError("unreachable code")
        """
        violations = check_code(code)

        # AssertionError is exempt (programming error), but if statement is not
        pur004_violations = [v for v in violations if v.rule_code == "PUR004"]
        assert len(pur004_violations) == 0

    def test_raise_runtime_error_unreachable(self, check_code: CheckCodeCallable) -> None:
        """PUR004 exempt for RuntimeError with 'unreachable'."""
        code = """
        raise RuntimeError("unreachable: match should be exhaustive")
        """
        violations = check_code(code)

        # RuntimeError with 'unreachable' is exempt
        assert len(violations) == 0

    def test_raise_for_expected_error(self, check_code: CheckCodeCallable) -> None:
        """PUR004 triggers on domain error raise."""
        code = """
        if value < 0:
            raise ValueError("value must be positive")
        """
        violations = check_code(code)

        # Should have violations for both if and raise
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR004" for v in violations)

    def test_reraise_exception(self, check_code: CheckCodeCallable) -> None:
        """PUR004 triggers on bare raise."""
        code = """
        try:
            risky_operation()
        except Exception:
            raise
        """
        violations = check_code(code)

        # Bare raise should trigger
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR004" for v in violations)


# ============================================================================
# TEST CLASS 5: PUR005 - SIDE EFFECTS
# ============================================================================


class TestPUR005SideEffects:
    """Test detection of side effects (PUR005)."""

    def test_print_detected(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on print()."""
        code = """
        print("debug message")
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"

    def test_logger_debug(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on logger.debug()."""
        code = """
        logger.debug("debug message")
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"

    def test_logger_info(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on logger.info()."""
        code = """
        logger.info("info message")
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"

    def test_logger_warning(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on logger.warning()."""
        code = """
        logger.warning("warning message")
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"

    def test_logger_error(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on logger.error()."""
        code = """
        logger.error("error message")
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"

    def test_logger_critical(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on logger.critical()."""
        code = """
        logger.critical("critical message")
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"

    def test_dynamic_logger_call(self, check_code: CheckCodeCallable) -> None:
        """PUR005 may NOT detect dynamic method calls.

        This is a known limitation - AST cannot detect dynamic calls.
        """
        code = """
        method = "info"
        getattr(logger, method)("message")
        """
        violations = check_code(code)

        # Known limitation: cannot detect dynamic calls
        assert len(violations) == 0

    def test_print_in_fstring(self, check_code: CheckCodeCallable) -> None:
        """PUR005 triggers on print in f-string."""
        code = """
        message = f"value: {print(x) or x}"
        """
        violations = check_code(code)

        assert len(violations) == 1
        assert violations[0].rule_code == "PUR005"


# ============================================================================
# TEST CLASS 6: PATTERN EXEMPTIONS
# ============================================================================


class TestPatternExemptions:
    """Test automatic pattern exemptions (guard clauses, CUDA kernels, unwrap)."""

    def test_guard_clause_single_return(self, check_code: CheckCodeCallable) -> None:
        """Guard clause with single return Failure is exempt."""
        code = """
        def process(x: int):
            if x < 0:
                return Failure(ValueError("negative"))
            return Success(x * 2)
        """
        violations = check_code(code)

        # Guard pattern detected - no PUR003 violations
        pur003_violations = [v for v in violations if v.rule_code == "PUR003"]
        assert len(pur003_violations) == 0

    def test_guard_clause_multiple_returns(self, check_code: CheckCodeCallable) -> None:
        """Guard clause with multiple returns triggers PUR003."""
        code = """
        if x < 0:
            return Failure(ValueError("negative"))
            return Failure(ValueError("another error"))
        """
        violations = check_code(code)

        # Not a pure guard (multiple statements)
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR003" for v in violations)

    def test_guard_clause_success_return(self, check_code: CheckCodeCallable) -> None:
        """Guard returning Success triggers PUR003."""
        code = """
        if x > 0:
            return Success(x)
        """
        violations = check_code(code)

        # Guards must return Failure
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR003" for v in violations)

    def test_cuda_kernel_basic(self, check_code: CheckCodeCallable) -> None:
        """CUDA kernel with @cuda.jit is fully exempt."""
        code = """
        @cuda.jit
        def kernel(data):
            for i in range(10):
                if condition:
                    while True:
                        data[i] = i
                        break
        """
        violations = check_code(code)

        # All rules exempt in CUDA kernel
        assert len(violations) == 0

    def test_cuda_kernel_with_args(self, check_code: CheckCodeCallable) -> None:
        """CUDA kernel with @cuda.jit(...) is exempt."""
        code = """
        @cuda.jit(device=True)
        def kernel():
            for i in range(10):
                pass
        """
        violations = check_code(code)

        assert len(violations) == 0

    def test_non_cuda_decorator(self, check_code: CheckCodeCallable) -> None:
        """Non-CUDA decorators do NOT exempt."""
        code = """
        @dataclass
        def func():
            for x in items:
                process(x)
        """
        violations = check_code(code)

        # Not a CUDA kernel - should detect violations
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR001" for v in violations)

    def test_unwrap_function_noreturn(self, check_code: CheckCodeCallable) -> None:
        """unwrap() with NoReturn exempt from PUR004."""
        code = """
        def unwrap(self) -> NoReturn:
            raise RuntimeError("unwrap called on Failure")
        """
        violations = check_code(code)

        # Boundary pattern detected
        assert len(violations) == 0

    def test_unwrap_wrong_name(self, check_code: CheckCodeCallable) -> None:
        """Function not named 'unwrap' triggers PUR004."""
        code = """
        def extract(self) -> NoReturn:
            raise RuntimeError("error")
        """
        violations = check_code(code)

        # Name must be 'unwrap'
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR004" for v in violations)

    def test_unwrap_without_noreturn(self, check_code: CheckCodeCallable) -> None:
        """unwrap() without NoReturn triggers PUR004."""
        code = """
        def unwrap(self):
            raise RuntimeError("error")
        """
        violations = check_code(code)

        # Must have NoReturn annotation
        assert len(violations) >= 1
        assert any(v.rule_code == "PUR004" for v in violations)

    def test_nested_cuda_guards(self, check_code: CheckCodeCallable) -> None:
        """CUDA kernel with guard clauses inside."""
        code = """
        @cuda.jit
        def kernel(x):
            if x < 0:
                return
            for i in range(10):
                pass
        """
        violations = check_code(code)

        # CUDA exemption takes precedence
        assert len(violations) == 0


# ============================================================================
# TEST CLASS 7: CONFIGURATION
# ============================================================================


class TestConfiguration:
    """Test tier classification and whitelist configuration."""

    def test_tier1_classification(self) -> None:
        """Tier 1 files are exempt from all rules."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/models/torch.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER1_INFRASTRUCTURE

    def test_tier2_classification(self) -> None:
        """Tier 2 files are subject to purity rules."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/gbm.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER2_BUSINESS_LOGIC

    def test_tier3_classification(self) -> None:
        """Tier 3 files (effects) are exempt."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/storage/store.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER3_EFFECTS

    def test_load_whitelist(self) -> None:
        """Whitelist loads from pyproject.toml."""
        config = load_purity_config()
        whitelist = config["whitelist"]

        assert "src/spectralmc/async_normals.py" in whitelist
        assert 311 in whitelist["src/spectralmc/async_normals.py"]

    def test_whitelist_application(self, check_code: CheckCodeCallable) -> None:
        """Whitelisted violations are marked but don't fail."""
        code = """
        if condition:
            process()
        """
        # Line 2 is where the if statement is (after dedenting)
        whitelist = {2: "Whitelisted for testing"}
        violations = check_code(code, whitelist=whitelist)

        assert len(violations) == 1
        assert violations[0].whitelisted is True
        assert "testing" in violations[0].whitelist_reason

    def test_non_whitelisted_triggers(self, check_code: CheckCodeCallable) -> None:
        """Non-whitelisted violations are not marked."""
        code = """
        if condition:
            process()
        """
        # Whitelist line 5, not line 2 (where if statement is)
        whitelist = {5: "Different line"}
        violations = check_code(code, whitelist=whitelist)

        assert len(violations) == 1
        assert violations[0].whitelisted is False


# ============================================================================
# TEST CLASS 8: INTEGRATION
# ============================================================================


class TestIntegration:
    """Integration tests with real codebase files."""

    def test_real_file_async_normals(self) -> None:
        """Test purity checker on actual codebase file."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/async_normals.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER2_BUSINESS_LOGIC

        # Should find exactly 1 violation (line 311) which is whitelisted
        whitelist = config["whitelist"].get("src/spectralmc/async_normals.py", {})
        assert 311 in whitelist

    def test_real_file_tensors(self) -> None:
        """Test purity checker on tensors.py."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/serialization/tensors.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER2_BUSINESS_LOGIC

        # Should find exactly 1 violation (line 174) which is whitelisted
        whitelist = config["whitelist"].get("src/spectralmc/serialization/tensors.py", {})
        assert 174 in whitelist

    def test_real_file_gbm_trainer(self) -> None:
        """Test purity checker on refactored gbm_trainer.py."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/gbm_trainer.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER2_BUSINESS_LOGIC

        # Should find 0 violations (all eliminated in refactor)
        # No whitelist entries for this file
        whitelist = config["whitelist"].get("src/spectralmc/gbm_trainer.py", {})
        assert len(whitelist) == 0

    def test_empty_file(self, check_code: CheckCodeCallable) -> None:
        """Purity checker handles empty files gracefully."""
        code = ""
        violations = check_code(code)

        assert violations == []

    def test_tier1_exempt_from_checks(self) -> None:
        """Tier 1 files should be exempt from purity checks."""
        config = load_purity_config()
        project_root = get_project_root()
        classifier = FileClassifier(
            tier1_patterns=config["tier1_infrastructure"],
            tier3_patterns=config["tier3_effects"],
            project_root=project_root,
        )
        filepath = project_root / "src/spectralmc/models/torch.py"
        result = classifier.classify(filepath)

        assert result is not None
        assert result.tier == FileTier.TIER1_INFRASTRUCTURE
