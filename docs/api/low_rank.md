# Low-Rank BLR

The scalable middle ground: precision parameterised as
$\Lambda = \operatorname{diag}(D) + UU^\top$ with $U \in \mathbb{R}^{d \times r}$,
giving $O(dr)$ storage and $O(dr^2 + r^3)$ solves through the Woodbury
identity. Structured solves are dispatched via
[gaussx](https://github.com/jejjohnson/gaussx) `LowRankUpdate` operators
(requires the optional `gaussx` extra).

## Transforms

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [blr_low_rank_for_loss, blr_low_rank]

## Posterior extraction & sampling

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [get_posterior_low_rank, sample_posterior_low_rank]

## State

::: optax_bayes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [BLRLowRankState]
