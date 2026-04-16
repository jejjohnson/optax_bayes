---
status: draft
version: 0.1.0
---

# optax-bayes Design Doc

**The Bayesian Learning Rule as optax GradientTransformations.**

*Formerly bayes_rule — renamed to optax-bayes.*

## Structure

```
optax_bayes/
├── README.md              # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Layer stack, optax integration, gaussx backend, dependencies
├── boundaries.md          # Ownership, ecosystem, scope, testing, roadmap
├── api/
│   ├── README.md          # Surface inventory, conventions, notation
│   ├── primitives.md      # Layer 0 — pure JAX update functions (BLR math)
│   ├── components.md      # Layer 1 — optax GradientTransformations, state types
│   └── models.md          # Layer 2 — loss wrappers, posterior extraction, schedule composition
├── examples/
│   ├── README.md          # Index and reading order
│   ├── primitives.md      # Layer 0 — BLR update step by step
│   ├── components.md      # Layer 1 — optax integration, composability
│   ├── models.md          # Layer 2 — drop-in training, posterior uncertainty
│   └── integration.md     # Layer 3 — gaussx, pyrox-gp, ekalmX, xtremax composition
├── decisions.md           # Design decisions with rationale
└── research/
    ├── README.md          # Index
    ├── overview.md        # Full mathematical treatment (original monolithic design doc)
    └── base.py            # Reference implementation
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the optax integration
3. **[boundaries.md](boundaries.md)** — understand the scope
4. **[api/README.md](api/README.md)** — scan the surface
5. **[api/primitives.md](api/primitives.md)** → **[components.md](api/components.md)** → **[models.md](api/models.md)** — drill into detail
6. **[examples/](examples/)** — see it in action
7. **[decisions.md](decisions.md)** — understand the tradeoffs
8. **[research/overview.md](research/overview.md)** — full mathematical treatment (BLR theory, model zoo, efax/gaussx backends)
