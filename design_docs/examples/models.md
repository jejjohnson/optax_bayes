---
status: draft
version: 0.1.0
---

# Layer 2 — Model Examples

Drop-in training with posterior uncertainty.

---

## Equinox Neural Network Training

```python
import equinox as eqx
from optax_bayes import blr_diagonal_for_loss, get_posterior_diagonal

model = MyNetwork(key=jr.PRNGKey(0))
opt = blr_diagonal_for_loss(learning_rate=1e-3, prior_precision=1e-4)
state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, state, batch):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates, state = opt.update(grads, state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss

# After training: per-parameter uncertainty for free
mean, variance = get_posterior_diagonal(state)
```

---

## Continual Learning (EWC-style)

```python
# Task 1: train as usual
for batch in task1_data:
    ...
mean_1, var_1 = get_posterior_diagonal(state)

# Task 2: task-1 posterior becomes task-2 prior
opt2 = blr_diagonal_for_loss(
    learning_rate=1e-3,
    prior_precision=jax.tree.map(lambda v: 1.0 / v, var_1),  # per-parameter
    prior_mean=mean_1,
)
state2 = opt2.init(params)
for batch in task2_data:
    ...
```

---

## Predictive Uncertainty

```python
mean, variance = get_posterior_diagonal(state)

# Sample parameter vectors from posterior
def sample_predict(key):
    eps = jax.tree.map(lambda m, v: m + jr.normal(key, m.shape) * jnp.sqrt(v), mean, variance)
    return model_apply(eps, x_test)

keys = jr.split(jr.PRNGKey(0), 50)
preds = jax.vmap(sample_predict)(keys)
pred_mean, pred_std = preds.mean(0), preds.std(0)
```
