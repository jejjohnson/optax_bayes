---
status: draft
version: 0.1.0
---

# Layer 1 — Component Examples

optax GradientTransformations in action.

---

## Drop-in Adam Replacement

```python
import optax
from optax_bayes import blr_diagonal_for_loss, get_posterior_diagonal

opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-4)
state = opt.init(params)

for batch in dataloader:
    grads = jax.grad(loss_fn)(params, batch)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)

# Uncertainty at any time — no extra computation
mean, variance = get_posterior_diagonal(state)
```

---

## Chain with Gradient Clipping

```python
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    blr_diagonal_for_loss(learning_rate=1e-3),
)
```

---

## With Learning Rate Schedule

```python
from optax_bayes import blr_with_schedule

schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4, peak_value=1e-2,
    warmup_steps=1000, decay_steps=50000,
)
opt = blr_with_schedule(schedule, prior_precision=1e-3)
```
