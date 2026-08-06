"""randomSL baseline: random-scheduling Split Learning.

The pipeline is identical to the MSL mainline except that Stage 3 client
selection is uniform random instead of predicted-cluster-aware scheduling.
Rounds that do not cover every predicted cluster are tolerated and recorded as
empty binding rounds instead of raising, keeping the fusion structure and the
label-guided pseudo-binding protocol unchanged.
"""

from baseline.randomSL.scheduling import RandomScheduler
from baseline.randomSL.training import run_random_sl_stage3_split_training

__all__ = ["RandomScheduler", "run_random_sl_stage3_split_training"]
