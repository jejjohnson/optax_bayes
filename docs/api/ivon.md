# IVON

Improved Variational Online Newton (Shen et al., 2024): a practical diagonal
BLR variant designed for deep learning, with separate EMA rates for the mean
and precision, explicit weight decay acting as the prior, and gradient
clipping in natural-parameter space. Accepts standard **loss** gradients
(same convention as `optax.adam`).

For proper Bayesian MC training, draw a posterior sample with
[`sample_ivon`][optax_bayes.sample_ivon] *before* computing gradients at each
step; for optimizer-only use, feed gradients at the current params directly.

## Optimizer

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ivon]

## Posterior sampling & extraction

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [sample_ivon, get_posterior_ivon]

## State

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [IVONState]
