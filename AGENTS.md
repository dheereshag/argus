# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Python Quality & Verification Gates

**Always enforce the 3-step verification loop. All three must pass.**

For any Python modifications:
1. **Lint & Code Style**: Run `uv run ruff check --fix`. If manual errors remain, create an implementation plan and fix them cleanly without using `# noqa` suppressions.
2. **Type Checking**: Run `uv run ty check`. If type diagnostics are found, create an implementation plan and fix the underlying typing without using `# type: ignore`.
3. **Test Suite**: Run `uv run pytest` to confirm all unit and integration tests pass without regression.
4. **Iterative Verification**: Keep looping through `uv run ruff check --fix`, `uv run ty check`, and `uv run pytest` until all three yield **0 errors, 0 warnings, and all tests pass**.

**The 3 mandatory commands that must always work:**
- `uv run ruff check --fix` (0 errors, 0 warnings)
- `uv run ty check` (0 errors, 0 warnings)
- `uv run pytest` (all tests pass)

## 6. Update Documentation (`docs/` and `README.md`)

**Always keep documentation in `docs/` and `README.md` in sync with codebase changes.**

- Explicitly update `README.md` and relevant guides in `docs/` (such as `docs/ARCHITECTURE.md`) whenever adding, modifying, or removing features, API endpoints, environment variables, architecture patterns, dependencies, or workflows.
- Ensure all setup instructions, configuration options, usage examples, and architectural diagrams remain accurate and up-to-date.

## 7. Dependency Management via `uv`

**Always use `uv` commands to add or remove dependencies. Never directly edit `pyproject.toml` dependencies.**

- Use `uv add <package>` to install and record new dependencies.
- Use `uv remove <package>` to uninstall and remove dependencies.
- Never manually edit the `dependencies` or `dependency-groups` arrays in `pyproject.toml` directly; let `uv` manage dependency specification, lockfile synchronization (`uv.lock`), and virtual environment state.

## 8. No Backward-Compatibility Shims

**Never add or keep backward-compatibility code for obsolete tests or legacy implementations.**

- Do not compromise production code types or signatures to satisfy stale test mocks or old conventions.
- If a function returns a typed dataclass/model, never add `isinstance(x, dict)` checks, dict subscription shims (`__getitem__`, `get`), or fallback adapters.
- Update tests and callers to strictly match the current, clean production contracts. Never bend production code backward for tests.

## 9. NASA JPL Rule 4: Single Sheet of Paper Architecture (≤ 60 Lines)

**Every file and function must fit on a single printed sheet of paper.**

Derived from Gerard J. Holzmann's NASA JPL *Power of 10* safety-critical code rules (Rule 4):
- **Function Limit**: No function should exceed **60 lines of code** (single sheet of standard reference paper).
- **Module Limit**: Service files must be compact and single-purpose, targeting **≤ 60 lines per file** (excluding docstrings where reasonable, hard ceiling of 60 lines).
- **Folder / Subpackage Hierarchy**: Decompose multi-step or multi-concern domains into structured subpackages (e.g. `app/services/detector/`, `app/services/ocr/`, `app/services/plate_rules/`) rather than large monolithic files.
- **Single Cohesive Concern**: Each module must do one thing (e.g. geometry, occupancy policy, enhancement, spatial clustering).
- **Harmony with Surgical Changes (§3)**: Apply this standard to new or refactored components; do not arbitrarily refactor untouched adjacent code unless requested.

## 10. Python 3.14 Standards & Modern Idioms

**Write native Python 3.14 code. Leverage modern typing and runtime enhancements.**

- **PEP 649 Deferred Evaluation of Annotations**:
  - Never use string quotes for forward references (e.g. use `def add(node: Node) -> Node:`, not `'Node'`).
  - Do not use `from __future__ import annotations`.
- **PEP 695 Type Parameter Syntax**:
  - Prefer the `type` statement for type aliases (e.g. `type TokenMap = dict[str, str]`).
  - Use native generic parameter syntax for functions and classes (`class Pipeline[T]:`, `def process[T](item: T) -> T:`).
- **Modern Collections & Unions**:
  - Exclusively use built-in collection types (`list`, `dict`, `tuple`, `set`) and pipe unions (`str | None`, `int | float`).
  - Never import `List`, `Dict`, `Optional`, or `Union` from `typing`.

## 11. Code Modularization Protocol & Master Prompt

**Systematically decompose monolithic code into compact, cohesive modules.**

When breaking down large files, services, or functions:
1. **Target Single Concerns**: Map responsibilities into dedicated subpackages (e.g., `app/services/<domain>/`).
2. **Enforce NASA JPL Rule 4**: Every resulting file and function must be strictly ≤ 60 lines.
3. **FastAPI & Async Conventions**:
   - Use `Annotated[..., Depends(...)]` for all parameter and dependency declarations.
   - Attach route prefixes, tags, and shared dependencies directly to `APIRouter(...)`.
   - Keep route handlers thin; delegate all business logic to service layers.
   - Use regular `def` for synchronous/CPU-bound tasks, `async def` only for non-blocking I/O, and `asyncer` when bridging.
4. **Standard Modularization Master Prompt**:
   Use the following structured prompt for LLMs or subagents performing refactoring:

   ```markdown
   ### Role & Context
   Senior Python Architect enforcing Python 3.14, FastAPI conventions, and NASA JPL Rule 4.

   ### Objectives
   1. Decompose: Break the monolithic code into cohesive, single-responsibility files (≤ 60 lines each).
   2. Modernize: Apply Python 3.14 native deferred annotations (PEP 649), `type` aliases (PEP 695), and `|` unions.
   3. Decouple: Separate API endpoints, schemas, dependencies, and business logic.
   4. No Shims: Never add backwards-compatibility shims or dictionary wrappers.

   ### Verification Criteria
   Must pass `uv run ruff check --fix`, `uv run ty check`, and `uv run pytest` with 0 errors.
   ```

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

