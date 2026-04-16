---
name: Feature / Enhancement
about: A single deliverable that rolls up to a theme epic.
title: "<scope>: <short description>"
labels: ["type:feature"]
---

## Problem / Request
<!-- What's needed? One or two sentences. -->

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Motivation
<!-- Why now; what it enables; what breaks if we don't have it. -->

## Proposed API
```python
# Signatures, types, docstring stubs.
```

<!--
OPTIONAL — Inline context ("Design Snapshot")
Copy here any API signatures, code snippets, configuration examples, or
excerpts from external / private design docs that the implementer needs to
work on this issue in isolation. Goal: another contributor (human or AI
agent) can implement this issue without opening other repos or chats.

RENAME the heading to fit the issue type. Examples:
  ## Design Snapshot            (typical library feature)
  ## Demo To Implement          (walkthrough / example / notebook issue)
  ## Demo Snippet To Include    (docs / API-reference issue)
  ## Config Snippet             (CI / infra issue)
  ## Reference Trace            (bug reproduction)

Lead with the exact snippet the implementer will reproduce — prose last.
Delete this section if not relevant.
-->

## Design Snapshot
<!-- Delete or rename. -->
```python
# Lead with the exact code/config the implementer will reproduce.
```

<!--
OPTIONAL — Inline math / numerical context ("Mathematical Notes")
For algorithmic / numerical issues: inline equations, sign conventions,
numerical-stability notes, edge cases. Keep everything the implementer
needs in one place.

STYLE — prefer unicode math in prose (σ², E₁, ∑, ⊗, ≈, Λ⁻¹, O(d³)) so
the issue reads in the GH UI and plain-text tools. Reach for a `text`
code fence for multi-line equation blocks so syntax-highlighting doesn't
try to parse pseudo-math:

    ```text
    s_next   = (1 - ρ) * s   + ρ * (s₀   - h)
    η_next   = (1 - ρ) * η   + ρ * (η₀ + g - h * m)
    ```

Rename the heading if the content type warrants (e.g. "Numerical Notes",
"Stability Notes", "Equations To Test"). Delete if not relevant.
-->

## Mathematical Notes
<!-- Delete or rename. -->
```text
<equations, conventions, numerical considerations>
```

## References & Existing Code
- Design doc / spec: `<path or URL>`
- Reference impl: `<path:line>`
- Related prior art: `<repo / paper / issue>`

## Implementation Steps
<!-- Concrete, file-level steps. Each should be checkable. -->
- [ ] Add `<symbol>` in `src/<package>/<module>.py`
- [ ] Wire <symbol> into `src/<package>/__init__.py` (if public)
- [ ] ...

## Definition of Done
- [ ] Code lands at the intended path
- [ ] Public API exported via `src/<package>/__init__.py` (if user-facing)
- [ ] Tests pass: `make test`
- [ ] Lint + typecheck pass: `make lint && make typecheck`
- [ ] Docstrings (Google-style) on all public symbols

## Testing
<!-- One checkbox per test, so progress is trackable. -->
- [ ] Unit test: `<what it asserts>`
- [ ] Property / round-trip test: `<what it asserts>` (if applicable)
- [ ] Regression test: `<what it asserts>` (if applicable)

## Documentation
- [ ] API reference page / section
- [ ] Notebook or recipe (if user-facing flow)
- [ ] Docstrings (covered by Definition of Done)

## Relationships
- Parent (theme epic): #
- Blocked by: #
- Blocks: #
- Related: #
