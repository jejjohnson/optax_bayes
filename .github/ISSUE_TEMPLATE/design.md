---
name: Design / ADR
about: Resolve an open design question for a new API.
title: "[Design] <question>"
labels: ["type:design"]
---

## Problem / Question
<!-- The design question being resolved. -->

## User Story
> As a <role>, I want <capability>, so that <outcome>.

## Motivation
<!-- Why this needs a design decision now. Upstream/downstream consumers. -->

## Proposed API
```python
# Option A
...
```
```python
# Option B
...
```

## Alternatives Considered
- **Option A** — pros/cons
- **Option B** — pros/cons
- **Option C (rejected)** — pros/cons

## Decision
<!-- Final decision (filled in when the issue is resolved). -->

## Consequences
- Ripple effects, downstream changes, backwards-compat implications.

## References & Existing Code
- Design doc: `jej_vc_snippets/design_docs/optax_bayes/decisions.md` §D?
- Related packages / prior art

## Implementation Steps (post-decision)
- [ ] Land reference implementation in `src/optax_bayes/...`
- [ ] Update `decisions.md` / `docs/` with the ADR record
- [ ] Open follow-up issues if surface changes

## Definition of Done
- [ ] Decision written into the issue body (Decision + Consequences)
- [ ] ADR entry added to `docs/decisions.md` (or equivalent)
- [ ] Follow-up feature issues opened and linked

## Testing
<!-- Tests that encode the decision (if reference impl lands in same PR). -->

## Documentation
<!-- ADR page; docstring notes referencing the decision. -->

## Relationships
- Parent (theme epic): #
- Blocked by: #
- Blocks: # (issues that can't start until this decision resolves)
- Related: #
