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

## References & Existing Code
- Design doc: `jej_vc_snippets/design_docs/optax_bayes/...`
- Reference impl: `research/base.py:L##`
- Related packages: gaussx `src/gaussx/...`, Khan & Rue (2023) §N

## Implementation Steps
- [ ] ...

## Definition of Done
- [ ] Code lands in `src/optax_bayes/...`
- [ ] Public API exported via `src/optax_bayes/__init__.py` (if user-facing)
- [ ] Tests pass: `make test`
- [ ] Lint + typecheck pass: `make lint && make typecheck`
- [ ] Docstrings (Google-style) on all public symbols

## Testing
<!-- Unit tests, property tests, numerical recovery tests, composition tests. -->

## Documentation
<!-- API reference page(s), notebook(s), recipe(s). -->

## Relationships
- Parent (theme epic): #
- Blocked by: #
- Blocks: #
- Related: #
