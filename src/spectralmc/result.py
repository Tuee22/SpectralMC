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
from typing import Callable, Generic, TypeVar, cast, NoReturn

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
        return cast("Result[T, F]", self)

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
        return cast("Result[U, E]", self)

    def map_error(self, f: Callable[[E], F]) -> Result[T, F]:
        """Map the error value through function f."""
        return Failure(f(self.error))

    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Monadic bind. No-op on Failure."""
        return cast("Result[U, E]", self)

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
    values: list[T] = []
    for result in results:
        match result:
            case Success(value):
                values.append(value)
            case Failure(error):
                return Failure(error)
    return Success(values)


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
    successes: list[T] = []
    failures: list[E] = []
    for result in results:
        match result:
            case Success(value):
                successes.append(value)
            case Failure(error):
                failures.append(error)
    return (successes, failures)
