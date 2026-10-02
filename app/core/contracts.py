"""Explicit runtime assertion contracts for defensive programming."""

from typing import Any


class ContractViolation(RuntimeError):
    """Precondition or postcondition invariant violation."""


def require(condition: Any, message: str) -> None:
    """Precondition. Raises ContractViolation when condition is falsy."""
    if not condition:
        raise ContractViolation(f"precondition failed: {message}")


def ensure(condition: Any, message: str) -> None:
    """Postcondition. Raises ContractViolation when condition is falsy."""
    if not condition:
        raise ContractViolation(f"postcondition failed: {message}")
