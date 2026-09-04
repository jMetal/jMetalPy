"""Component that selects survivors from a parent and an offspring population.

No adapter is needed here: `jmetal.operator.replacement.Replacement` already matches
the `replace(current, offspring) -> list` contract the `EvolutionaryAlgorithm`
template expects, so this module simply re-exports it and its existing
implementations instead of duplicating them under a second hierarchy.
"""

from jmetal.operator.replacement import (
    RankingAndCrowdingDistanceReplacement,
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
    Replacement,
    SMSEMOAReplacement,
)

__all__ = [
    "RankingAndCrowdingDistanceReplacement",
    "RankingAndDensityEstimatorReplacement",
    "RemovalPolicyType",
    "Replacement",
    "SMSEMOAReplacement",
]
