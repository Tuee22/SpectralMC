"""
Result type for explicit error handling.

This module provides a generic Result[T, E] ADT that makes error handling explicit
and eliminates the need for Optional types that hide failures or bare exception handling.

Type Safety:
    - All functions returning Result must explicitly specify both success and error types
    - Pattern matching ensures exhaustive handling of both cases
    - No implicit None returns or silent exception swallowing

Usage:
    >>> def divide(a: float, b: float) -> Result[float, str]:
    ...     if b == 0:
    ...         return Failure("Division by zero")
    ...     return Success(a / b)
    ...
    >>> result = divide(10, 2)
    >>> match result:
    ...     case Success(value):
    ...         print(f"Result: {value}")
    ...     case Failure(error):
    ...         print(f"Error: {error}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, NoReturn, TypeVar


T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")


@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful result containing a value of type T."""

    value: T

    def is_success(self) -> bool:
        """Check if this result is a success."""
        return True

    def is_failure(self) -> bool:
        """Check if this result is a failure."""
        return False

    def unwrap(self) -> T:
        """Unwrap the success value. Safe to call on Success."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return the success value, or default if failure."""
        return self.value

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Return the success value, or call f on error if failure."""
        return self.value

    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """Map the success value through function f."""
        return Success(f(self.value))

    def map_error(self, f: Callable[[E], F]) -> Result[T, F]:
        """Map the error value through function f. No-op on Success."""
        # Explicit type annotation eliminates need for cast
        result: Result[T, F] = Success(self.value)
        return result

    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Monadic bind: chain operations that return Result."""
        return f(self.value)

    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Alias for flat_map for more readable chaining."""
        return self.flat_map(f)


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed result containing an error of type E."""

    error: E

    def is_success(self) -> bool:
        """Check if this result is a success."""
        return False

    def is_failure(self) -> bool:
        """Check if this result is a failure."""
        return True

    def unwrap(self) -> NoReturn:
        """
        Unwrap the success value.

        Raises:
            RuntimeError: Always, since Failure has no value.
        """
        raise RuntimeError(f"Called unwrap() on Failure: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Return default value since this is a failure."""
        return default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Call f on the error value to produce a default."""
        return f(self.error)

    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """Map the success value through function f. No-op on Failure."""
        # Explicit type annotation eliminates need for cast
        result: Result[U, E] = Failure(self.error)
        return result

    def map_error(self, f: Callable[[E], F]) -> Result[T, F]:
        """Map the error value through function f."""
        return Failure(f(self.error))

    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Monadic bind. No-op on Failure."""
        # Explicit type annotation eliminates need for cast
        result: Result[U, E] = Failure(self.error)
        return result

    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Alias for flat_map. No-op on Failure."""
        return self.flat_map(f)


# Type alias for the union of Success and Failure
Result = Success[T] | Failure[E]


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """
    Collect a list of Results into a Result of list.

    If all results are Success, returns Success with list of values.
    If any result is Failure, returns the first Failure encountered.

    Args:
        results: List of Result values to collect

    Returns:
        Success(list of values) if all succeed, or first Failure
    """
    first_failure = next((result for result in results if isinstance(result, Failure)), None)
    return (
        first_failure
        if isinstance(first_failure, Failure)
        else Success([result.value for result in results if isinstance(result, Success)])
    )


def partition_results(
    results: list[Result[T, E]],
) -> tuple[list[T], list[E]]:
    """
    Partition a list of Results into successes and failures.

    Args:
        results: List of Result values to partition

    Returns:
        Tuple of (successes, failures)
    """
    successes: list[T] = [result.value for result in results if isinstance(result, Success)]
    failures: list[E] = [result.error for result in results if isinstance(result, Failure)]
    return (successes, failures)


def fold_results(
    items: list[T],
    f: Callable[[U, T], Result[U, E]],
    initial: U,
) -> Result[U, E]:
    """
    Functional fold with early exit on first failure.

    Equivalent to reduce() but stops on first Failure.

    Args:
        items: List of items to fold over
        f: Fold function taking (accumulator, item) and returning Result
        initial: Initial accumulator value

    Returns:
        Success(final_accumulator) if all steps succeed, else first Failure

    Example:
        >>> def add_if_positive(acc: int, x: int) -> Result[int, str]:
        ...     return Success(acc + x) if x > 0 else Failure("negative")
        >>> fold_results([1, 2, 3], add_if_positive, 0)
        Success(value=6)
        >>> fold_results([1, -2, 3], add_if_positive, 0)
        Failure(error="negative")
    """

    # Functional fold using recursion instead of for loop
    def fold_impl(remaining: list[T], current: U) -> Result[U, E]:
        match remaining:
            case []:
                return Success(current)
            case [head, *tail]:
                match f(current, head):
                    case Failure(err):
                        return Failure(err)
                    case Success(val):
                        return fold_impl(tail, val)
            case _:
                raise AssertionError("Unreachable: list pattern match exhaustive")

    return fold_impl(items, initial)
