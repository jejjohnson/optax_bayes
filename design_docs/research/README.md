---
status: draft
version: 0.1.0
---

# optax-bayes — Research

Full mathematical treatment and reference implementation.

## Structure

```
research/
├── README.md        # This file
├── overview.md      # Complete BLR design doc: math, model zoo, efax/gaussx backends, applications
└── base.py          # Reference implementation of BLR as optax transforms
```

## How These Relate to the Design Docs

| File | Informs | Key content |
|---|---|---|
| `overview.md` | All API and example files | §1: BLR math (variational objective, Bonnet-Price, diagonal/full-rank/low-rank). §2: numerical requirements. §3: model zoo (20+ algorithms as BLR special cases). §4: optax API design. §5: 8 application domains. §6: efax + gaussx backends. §7: references. |
| `base.py` | `api/components.md` | Working implementation of `blr_diagonal`, `blr_full_rank`, state types, posterior extraction |
