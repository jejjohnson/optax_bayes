from __future__ import annotations

from optax_bayes._src.diagonal import blr_diagonal as blr_diagonal
from optax_bayes._src.full_rank import blr_full_rank as blr_full_rank
from optax_bayes._src.ivon import (
    IVONState as IVONState,
    get_posterior_ivon as get_posterior_ivon,
    ivon as ivon,
    sample_ivon as sample_ivon,
)
from optax_bayes._src.low_rank import blr_low_rank as blr_low_rank
from optax_bayes._src.posterior import (
    get_posterior_diagonal as get_posterior_diagonal,
    get_posterior_full_rank as get_posterior_full_rank,
    get_posterior_low_rank as get_posterior_low_rank,
)
from optax_bayes._src.presets import (
    newton as newton,
    newton_for_loss as newton_for_loss,
)
from optax_bayes._src.sampling import (
    sample_posterior_diagonal as sample_posterior_diagonal,
    sample_posterior_full_rank as sample_posterior_full_rank,
    sample_posterior_low_rank as sample_posterior_low_rank,
)
from optax_bayes._src.schedules import blr_with_schedule as blr_with_schedule
from optax_bayes._src.types import (
    BLRDiagState as BLRDiagState,
    BLRFullRankState as BLRFullRankState,
    BLRLowRankState as BLRLowRankState,
)
from optax_bayes._src.wrappers import (
    blr_diagonal_for_loss as blr_diagonal_for_loss,
    blr_full_rank_for_loss as blr_full_rank_for_loss,
    blr_low_rank_for_loss as blr_low_rank_for_loss,
)


__version__ = "0.1.0"
