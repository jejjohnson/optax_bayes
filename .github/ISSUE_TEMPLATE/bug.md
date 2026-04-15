---
name: Bug report
about: Something isn't working correctly.
title: "bug: <short description>"
labels: ["type:bug"]
---

## Problem
<!-- What's broken? One or two sentences. -->

## Reproduction
```python
# Minimal reproducing example.
```

## Expected Behaviour
<!-- What should happen. -->

## Actual Behaviour
<!-- What happens instead. Include traceback. -->

## Environment
- `optax_bayes` version:
- `jax` / `optax` / `gaussx` versions:
- Python:
- Platform:

## References & Existing Code
- Related code: `src/optax_bayes/...`
- Related issue/PR: #

## Implementation Steps (fix)
- [ ] ...

## Definition of Done
- [ ] Regression test captures the bug
- [ ] Fix lands; regression test green
- [ ] `make test && make lint && make typecheck` green

## Testing
<!-- Regression test path + what it asserts. -->

## Documentation
<!-- If user-facing behaviour changed, update relevant pages. -->

## Relationships
- Parent (theme epic, if any): #
- Blocked by: #
- Blocks: #
