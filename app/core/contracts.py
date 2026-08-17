"""
Explicit runtime contracts (NASA Power of 10, rule 5).

Rule 5 asks for assertion density on safety-critical paths. Python's `assert`
is the wrong tool for that here, for two reasons:

  1. `python -O` strips every assert statement. A safety check that vanishes
     under an optimisation flag is not a safety check — it is a comment that
     happens to run in development. Production containers are exactly where
     someone eventually adds -O.
  2. The rule's stated intent is that a violated assertion takes a defined
     recovery action, not that the process dies with an unhandled exception.

So: `require()` for preconditions, `ensure()` for postconditions, both raising
ContractViolation. They cannot be stripped, they carry a message naming the
component, and they are catchable at the boundary so the API can answer with
something honest rather than a stack trace.

Deliberately NOT applied at rule 5's literal "two per function" density. That
number was chosen for C flight software where a wild pointer write is
unrecoverable and silent. Python raises on the equivalent mistakes by itself,
so blanket assertions would be noise that trains reviewers to skim. These are
placed only where a violation would otherwise pass silently into a plate read:

  - boundaries between components (YOLO -> cropping -> OCR -> API)
  - anything derived from model output or a third-party response
  - anything that indexes, slices, or bounds a loop

See docs/NASA_RULES.md for the full rule-by-rule mapping.
"""

from __future__ import annotations

from typing import Any, NoReturn, Optional, Sequence


class ContractViolation(RuntimeError):
    """
    A precondition or postcondition failed.

    Subclasses RuntimeError rather than ANPRServiceError on purpose: this is an
    internal invariant break, not a domain error the caller did something to
    cause. It should surface as a 500 and be investigated, never be quietly
    mapped to a 400 and blamed on the client.
    """


def require(condition: Any, message: str) -> None:
    """
    Precondition. Raises ContractViolation when `condition` is falsy.

    Use at the entry of a function that is about to trust a value it did not
    create — model output, a parsed response, a caller-supplied box.
    """
    if not condition:
        raise ContractViolation(f"precondition failed: {message}")


def ensure(condition: Any, message: str) -> None:
    """
    Postcondition. Raises ContractViolation when `condition` is falsy.

    Use at the exit of a function whose output something downstream will trust
    without re-checking.
    """
    if not condition:
        raise ContractViolation(f"postcondition failed: {message}")


def unreachable(message: str) -> NoReturn:
    """For branches that should be impossible. Documents the assumption."""
    raise ContractViolation(f"unreachable: {message}")


def bounded(items: Optional[Sequence[Any]], limit: int, what: str) -> Sequence[Any]:
    """
    Enforce rule 2 (every loop has a fixed upper bound) at the data source.

    Returns at most `limit` items. Truncation is a warning-level event, not an
    error: the caller asked for work we are declining to do without bound, and
    that decision should be visible in the logs rather than silent.

    Bounding the sequence rather than counting inside the loop keeps the bound
    stated once, next to the data, instead of repeated in every loop body where
    it can drift.
    """
    require(limit > 0, f"{what} limit must be positive, got {limit}")

    if not items:
        return []
    if len(items) <= limit:
        return items

    # Imported here to keep this module importable by anything, including the
    # logging configuration itself.
    from app.core.logging import logger  # noqa: PLC0415

    logger.warning(
        f"[bounds] {what}: {len(items)} exceeds cap of {limit}; "
        f"processing first {limit} and discarding {len(items) - limit}."
    )
    return items[:limit]
