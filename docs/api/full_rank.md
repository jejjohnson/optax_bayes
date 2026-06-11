# Full-Rank BLR

The exact end of the spectrum: a dense precision matrix
$\Lambda \in \mathbb{R}^{d \times d}$ over flat parameter vectors. With an
exact-Hessian callable the fixed point is the exact Gaussian posterior of a
conjugate model, which makes this variant the reference implementation the
cheaper families are tested against. Solves go through `gaussx.solve`
(requires the optional `gaussx` extra).

## Transforms

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [blr_full_rank_for_loss, blr_full_rank]

## Newton's method

Classic damped Newton falls out of full-rank BLR at $\rho = 1$ with a
near-flat prior and an exact Hessian.

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [newton_for_loss, newton]

## Posterior extraction & sampling

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [get_posterior_full_rank, sample_posterior_full_rank]

## State

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [BLRFullRankState]
