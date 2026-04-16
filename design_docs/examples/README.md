---
status: draft
version: 0.1.0
---

# optax-bayes — Examples

## Structure

```
examples/
├── README.md              # This file
├── primitives.md          # Layer 0 — BLR update mechanics
├── components.md          # Layer 1 — optax integration, composability
├── models.md              # Layer 2 — drop-in training, posterior extraction
└── integration.md         # Layer 3 — gaussx, pyrox-gp, ekalmX, xtremax
```

## Reading Order

1. **[primitives.md](primitives.md)** — L0: the BLR update step by step
2. **[components.md](components.md)** — L1: optax transforms in action
3. **[models.md](models.md)** — L2: drop-in training with uncertainty
4. **[integration.md](integration.md)** — L3: ecosystem composition
