# NASA Power of 10 — how they map to Argus

Reference: [Perforce summary](https://www.perforce.com/blog/kw/NASA-rules-for-developing-safety-critical-code),
originally Gerard Holzmann, JPL Laboratory for Reliable Software, *The Power of 10: Rules for
Developing Safety-Critical Code* (2006).

The rules were written for C flight software, where the failure modes are wild pointer writes,
stack overflow from recursion, and heap fragmentation over a multi-year mission. Argus is Python
running a web service. Several rules therefore describe hazards this codebase cannot have.

**This document records which rules were applied, which were translated, and which were
deliberately skipped.** A rule skipped with a reason is an engineering decision. A rule skipped
silently is an omission, and six months from now nobody can tell the two apart.

The guiding test for every rule below: *does applying this catch a defect that actually exists in
this code?* Where the answer was no, the rule was not forced.

---

## Summary

| # | Rule | Verdict | What it caught here |
|---|---|---|---|
| 1 | Simple control flow, no `goto`/recursion | **Already satisfied** | Nothing — enforced going forward by ruff `C901` |
| 2 | Fixed upper bound on every loop | **Applied** | Unbounded vehicle-box and OCR-line loops |
| 3 | No dynamic allocation after init | **Skipped, intent honoured** | Leaked image handles under a threadpool |
| 4 | Function fits on one page | **Applied, selectively** | Three functions past a page |
| 5 | ≥2 assertions per function | **Translated** | `assert` vanishes under `-O` |
| 6 | Smallest possible scope | **Applied** | Live race on both model singletons |
| 7 | Check returns, validate parameters | **Applied** | Unclamped boxes, blind `**item` splatting |
| 8 | Limit the preprocessor | **Not applicable** | Python has no preprocessor |
| 9 | Single pointer dereference | **Translated** | Four-deep unchecked subscript chain |
| 10 | All warnings on, all addressed | **Applied** | No linting or type checking existed |

---

## Rule 1 — Simple control flow

**Verdict: already satisfied. No changes.**

Python has no `goto`, `setjmp` or `longjmp`. A grep for recursion across `app/` finds none — the
waterfall is iterative, and the tier ladder is a flat sequence.

Rather than declare victory, `C901` (cyclomatic complexity, max 12) and `PLR0912` (max branches)
are now enforced in CI, so the property is maintained rather than merely observed once.

---

## Rule 2 — Fixed upper bound on every loop

**Verdict: applied. The highest-value rule for this codebase.**

`BasePlateRecognizer.recognize` iterated whatever box list it was handed. Cost per request:

```
len(vehicle_boxes) x 5 ROI tiers x 2 (plain + warped) x N providers
```

The tier count was fixed at 5, but `vehicle_boxes` was not bounded at all — YOLO will return a
dozen boxes for a yard with parked vehicles in frame. This is the mechanism behind the 27.4 s
maximum in `eval_report.json`.

Three loops now have stated ceilings:

| Loop | Bound | Setting |
|---|---|---|
| Vehicle boxes in the waterfall | 3 | `MAX_VEHICLE_BOXES` |
| OCR text lines fed to pairing | 40 | `MAX_OCR_LINES` |
| YOLO detections examined | 100 | `MAX_DETECTIONS` |

Bounds are applied to the *sequence* via `contracts.bounded()`, not counted inside the loop body,
so each bound is stated once next to its data instead of being repeated where it can drift.

**Truncation is safe only because the ordering is right.** Boxes are sorted by area descending
before the cap, and at a weighbridge the vehicle on the platform dominates the frame. Truncation
discards the least plausible candidates, and logs a warning when it does.

---

## Rule 3 — No dynamic memory allocation after initialisation

**Verdict: skipped as written. Intent honoured.**

Literally impossible in a garbage-collected language: every Python object is a heap allocation,
and there is no `malloc` to avoid. Implementing this rule would mean not writing Python.

The rule's *purpose* — bounded, predictable memory, no fragmentation or leak over a long run — is
real, and two things address it:

1. **Bounded input.** Issue #7 added the upload cap, the pixel budget, and downscaling to a
   1920px edge. Peak memory per request now has a stated ceiling.
2. **Prompt handle release.** `Image.open` is lazy and holds its source open until the pixels are
   read. The previous code left one handle per call for the collector to reclaim whenever it
   chose. Under a threadpool that is a slow leak. All decoding now routes through `load_rgb()`,
   which uses context managers and calls `load()` explicitly, and a test asserts no bare
   `Image.open` survives outside that helper.

---

## Rule 4 — No function longer than one page

**Verdict: applied to the three functions that exceeded it. Not enforced globally.**

| Function | Before | Change |
|---|---|---|
| `filter_vehicle_and_occupancy` | ~100 lines | Inference split into `_run_detection`; the outer function is now policy only |
| `recognize_plate` | ~90 lines | Provider orchestration split into `_run_waterfall` and `_validate_plate_results` |
| `save_debug_images` | ~75 lines | Repeated ROI blocks collapsed into `_save_bottom_rois` |

The split in `filter_vehicle_and_occupancy` earns its keep beyond line count: separating
inference from policy means the rejection branches can be tested without loading a model.

A hard line-count lint rule was **not** added. Line count is a proxy for complexity, and enforcing
the proxy directly rewards splitting a coherent function into two incoherent ones. `PLR0915`
(max statements) and `C901` are the honest versions and both are on.

---

## Rule 5 — Minimum two assertions per function

**Verdict: translated. Applied selectively, not at the stated density.**

Two problems with the literal form in Python:

1. **`python -O` strips every `assert`.** A safety check that disappears under an optimisation
   flag is not a safety check. Production containers are exactly where someone eventually adds
   `-O`. There is a test (`test_contracts_survive_optimised_mode`) that runs a subprocess under
   `-O` and asserts the check still fires.
2. **A raw `AssertionError` is not a defined recovery action.** The rule wants a violation
   handled; FastAPI renders an uncaught `AssertionError` as an opaque 500.

So `app/core/contracts.py` provides `require()` (precondition), `ensure()` (postcondition) and
`bounded()`, all raising `ContractViolation`, with a handler in `main.py` that logs the detail
and returns a 500 that does not leak internals.

**The "two per function" density was not applied.** That number was chosen for C, where a wild
pointer write is silent and unrecoverable. Python raises on the equivalent mistakes unaided, so
blanket assertions would be noise — and noise trains reviewers to skim, which costs more than it
buys. Contracts are placed only where a violation would otherwise pass *silently* into a plate
read:

- boundaries between components (YOLO → cropping → OCR → API)
- anything derived from model output or a third-party response
- anything that indexes, slices, or bounds a loop

---

## Rule 6 — Declare data at the smallest possible scope

**Verdict: applied. Fixed a live race.**

`_YOLO_MODEL` and `_PADDLE_OCR_INSTANCE` are module-level mutable globals with lazy
check-then-set initialisation and no synchronisation.

This was latent while the request handler was `async` and effectively single-threaded. **Issue #7
activated it**: making the handler a sync `def` moved it into FastAPI's threadpool, so two
concurrent first-requests can both observe `None` and both construct a model — doubling peak
memory during startup, on a Raspberry Pi. Our own fix created the exposure, which is a good
argument for the rule.

Fixed with double-checked locking (fast path stays lock-free once loaded) plus eager warming in
the `lifespan` handler, so in normal operation the race is never run at all. A source-level test
fails if either lock is removed.

The globals themselves stay module-scoped: a true dependency-injected registry is the cleaner
answer, but it touches every call site and this is a pilot.

---

## Rule 7 — Check return values; validate all parameters

**Verdict: applied at the component boundaries.**

Two concrete defects:

**Unclamped model output.** YOLO box coordinates flowed into `crop_image_roi` with only partial
min/max clamping, and `warp_perspective_crop` did none. **PIL does not raise on an out-of-range
crop — it pads with black.** So a box running off the frame produced a crop partly made of
invented pixels, OCR ran on it, and nothing anywhere reported a problem. Silent wrong input is
precisely what this rule exists to prevent. `clamp_box()` is now the single validated entry
point, returning `None` for anything degenerate rather than cropping with it.

**Blind splatting of provider output.** `[PlateResult(**item) for item in raw_results]` trusts a
strategy to return exactly the right keys. An unexpected key raises `TypeError`, a missing one
raises `ValidationError`, and either turns a request that had *successfully read a plate* into a
500. `_validate_plate_results` validates each item and drops bad ones with a log line, so one
malformed entry no longer discards the good ones.

`_run_waterfall` deliberately re-raises `ContractViolation` instead of failing over. Falling
through to the next provider would hide an internal defect behind a slower path.

---

## Rule 8 — Limit preprocessor use

**Verdict: not applicable. No equivalent exists.**

Python has no preprocessor. There is no token pasting, no conditional compilation, no macro
expansion — none of the hazards this rule addresses can occur.

The nearest analogues are import-time side effects and metaprogramming. Argus has one of the
former (`Image.MAX_IMAGE_PIXELS` set at import) which is deliberate and commented, and none of
the latter. Nothing to do.

---

## Rule 9 — Limit pointer dereferencing to one level

**Verdict: translated. Fixed a real crash path.**

No pointers in Python. The direct analogue is a chain of unchecked subscripts, and there was one
four levels deep:

```python
res_json['choices'][0]['message']['content']
```

Every link is a separate way to raise. A rate-limit body, an error object returned with HTTP 200,
a content filter, or an empty `choices` list each produce `KeyError`, `IndexError` or
`TypeError` — from inside a `try` that catches `Exception` and logs the provider as merely
"failed". A malformed response and a network outage were indistinguishable in the logs.

Replaced with `extract_message_content()`, which checks each level and returns `None` with a log
line naming what was actually received. Verified against ten malformed payload shapes.

Function pointers have no meaningful analogue worth restricting — the Strategy pattern here is
the codebase's best structural idea, not a hazard.

---

## Rule 10 — All warnings enabled; all warnings addressed

**Verdict: applied. Fails CI.**

There was no linting and no type checking. Added:

- **ruff** — pycodestyle, pyflakes, import sorting, bugbear, mccabe complexity, pylint subset
- **mypy** — `warn_unreachable`, `warn_return_any`, `no_implicit_optional`, `check_untyped_defs`
- **pytest** `filterwarnings = ["error"]` so a `DeprecationWarning` from a pinned dependency
  surfaces now rather than on upgrade day
- `.github/workflows/quality.yml`, which **fails** rather than annotates

Two deliberate limits:

`mypy` is **not** `strict = true`. Requiring full annotations on every existing function would
produce hundreds of findings nobody reads, and an ignored gate is worse than no gate — it looks
like coverage. The flags above catch real bugs without that. Tighten toward strict after the
pilot.

Untyped third-party packages are listed **individually** under `[[tool.mypy.overrides]]` rather
than with a blanket `ignore_missing_imports`, so adding a new untyped dependency is a visible
decision.

---

## What this did not do

Applying these rules is not the same as making the service safe. Still open:

- **No authentication** on `/recognize` (issue #12)
- **No ground truth**, so no measured accuracy (issues #2, #3)
- **The bottom-third ROI guess** remains the largest source of wrong reads (issues #5, #6)
- **YOLO weights are an unverified pickle** — `torch.load` on untrusted data is remote code
  execution, and no rule here addresses it (audit P2.7)

The Power of 10 makes code analysable and its failure modes explicit. It does not make a
guessing pipeline correct. Rules 2, 6, 7 and 9 fixed four real defects here; the rest of the
safety case is the rest of the plan.
