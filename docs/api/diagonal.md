# Diagonal BLR

The workhorse variant: a diagonal Gaussian
$q(\theta) = \mathcal{N}(m, \operatorname{diag}(1/s))$ with Adam-like $O(d)$
cost per step. Works on arbitrary parameter pytrees, so it drops into any
optax training loop unchanged.

## Transforms

Most users want [`blr_diagonal_for_loss`][optax_bayes.blr_diagonal_for_loss]
(standard loss gradients, like `optax.adam`); `blr_diagonal` is the inner
log-likelihood-convention transform; `blr_with_schedule` runs the loss
wrapper under an optax learning-rate schedule via `optax.inject_hyperparams`.

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [blr_diagonal_for_loss, blr_diagonal, blr_with_schedule]

## Posterior extraction & sampling

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [get_posterior_diagonal, sample_posterior_diagonal]

## State

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [BLRDiagState]
