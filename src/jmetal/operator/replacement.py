from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

import numpy as np

from jmetal.util.density_estimator import (
    CrowdingDistanceDensityEstimator,
    DensityEstimator,
    HypervolumeContributionDensityEstimator,
)
from jmetal.util.ranking import FastNonDominatedRanking, Ranking

S = TypeVar("S")


class Replacement(ABC, Generic[S]):
    """Base class for population replacement strategies.

    A replacement strategy decides which solutions from a parent population and an
    offspring population survive into the next generation. Concrete strategies
    (ranking-based, crowding-distance-based, hypervolume-based, ...) differ enough in
    their selection logic that this base class only fixes the shared contract, not any
    implementation.
    """

    @abstractmethod
    def replace(self, solution_list: list[S], offspring_list: list[S]) -> list[S]:
        """Combine a parent and an offspring population and select the survivors.

        Args:
            solution_list: The parent population.
            offspring_list: The offspring population.

        Returns:
            The population that survives into the next generation.
        """
        pass


class RemovalPolicyType(Enum):
    """Defines the policy for removing solutions in replacement strategies.

    Attributes:
        SEQUENTIAL: Remove solutions one by one, updating density estimates after each removal.
                   This is more computationally expensive but can lead to better diversity.
        ONE_SHOT: Remove all solutions at once based on initial density estimates.
                 This is faster but may be less accurate in maintaining diversity.
    """

    SEQUENTIAL = 1
    ONE_SHOT = 2


class RankingAndDensityEstimatorReplacement(Replacement[S]):
    """A replacement strategy that combines non-dominated ranking with density estimation.

    This replacement strategy is commonly used in multi-objective evolutionary algorithms
    to maintain a good balance between convergence and diversity in the population.
    It first ranks solutions using non-dominated sorting and then applies a density
    estimator to select solutions within each front.

    The replacement process works as follows:
    1. Combine parent and offspring populations
    2. Rank all solutions using non-dominated sorting
    3. Fill the new population with solutions from the best fronts
    4. When a front needs to be split, use the density estimator to select the most diverse solutions

    Args:
        ranking: The ranking strategy to use (e.g., FastNonDominatedRanking)
        density_estimator: The density estimator to use (e.g., CrowdingDistance)
        removal_policy: The policy for removing solutions (SEQUENTIAL or ONE_SHOT)

    Example:
        >>> from jmetal.operator import RankingAndDensityEstimatorReplacement
        >>> from jmetal.util.ranking import FastNonDominatedRanking
        >>> from jmetal.util.density_estimator import CrowdingDistance
        >>>
        >>> # Create a replacement operator with crowding distance
        >>> replacement = RankingAndDensityEstimatorReplacement(
        ...     ranking=FastNonDominatedRanking(),
        ...     density_estimator=CrowdingDistance(),
        ...     removal_policy=RemovalPolicyType.SEQUENTIAL
        ... )
        >>>
        >>> # Apply replacement to combine parent and offspring populations
        >>> new_population = replacement.replace(parents, offspring)
    """

    def __init__(
        self,
        ranking: Ranking,
        density_estimator: DensityEstimator,
        removal_policy: RemovalPolicyType = RemovalPolicyType.ONE_SHOT,
    ):
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.removal_policy = removal_policy

    def replace(self, solution_list: list[S], offspring_list: list[S]) -> list[S]:
        """Combine parent and offspring populations and select the best solutions.

        This method combines the parent and offspring populations, ranks all solutions
        using non-dominated sorting, and then applies the specified removal policy
        to select the best solutions.

        Args:
            solution_list: The parent population (list of solutions).
            offspring_list: The offspring population (list of solutions).

        Returns:
            A new population with the same size as solution_list containing the
            best solutions from the combined population.

        Note:
            The size of the returned population will be equal to the size of
            solution_list, not the combined size of both populations.
        """
        join_population = solution_list + offspring_list
        self.ranking.compute_ranking(join_population)

        if self.removal_policy is RemovalPolicyType.SEQUENTIAL:
            result_list: list[S] = self.sequential_truncation(0, len(solution_list))
        else:
            result_list = self.one_shot_truncation(0, len(solution_list))

        return result_list

    def sequential_truncation(self, ranking_id: int, size_of_the_result_list: int) -> list[S]:
        """Select solutions using sequential truncation based on density estimation.

        This method is called recursively to fill the new population with solutions
        from the best non-dominated fronts. When a front needs to be split, it uses
        the density estimator to select the most diverse solutions.

        Args:
            ranking_id: The current front index to process.
            size_of_the_result_list: Number of solutions still needed to fill the population.

        Returns:
            A list of selected solutions from the current and subsequent fronts.

        Note:
            This method is typically called internally by the replace() method and
            should not be called directly in most cases.
        """
        current_ranked_solutions = self.ranking.get_subfront(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)

        result_list: list[S] = []

        if len(current_ranked_solutions) < size_of_the_result_list:
            # If the entire front fits, add all solutions and move to the next front
            result_list.extend(self.ranking.get_subfront(ranking_id))
            result_list.extend(
                self.sequential_truncation(
                    ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions)
                )
            )
        else:
            # If we need to split the front, use density estimator to select solutions
            for solution in current_ranked_solutions:
                result_list.append(solution)

            # Remove solutions with worst density values until we reach the desired size
            while len(result_list) > size_of_the_result_list:
                self.density_estimator.sort(result_list)

                del result_list[-1]
                self.density_estimator.compute_density_estimator(result_list)

        return result_list

    def one_shot_truncation(self, ranking_id: int, size_of_the_result_list: int) -> list[S]:
        """Select solutions using one-shot truncation based on density estimation.

        This method is similar to sequential_truncation but is more efficient as it
        doesn't recompute density estimates after each removal. It's faster but may
        be less accurate in maintaining diversity compared to sequential truncation.

        Args:
            ranking_id: The current front index to process.
            size_of_the_result_list: Number of solutions still needed to fill the population.

        Returns:
            A list of selected solutions from the current and subsequent fronts.

        Note:
            This method is typically called internally by the replace() method when
            the removal policy is set to ONE_SHOT.
        """
        current_ranked_solutions = self.ranking.get_subfront(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)

        result_list: list[S] = []

        if len(current_ranked_solutions) < size_of_the_result_list:
            # If the entire front fits, add all solutions and move to the next front
            result_list.extend(self.ranking.get_subfront(ranking_id))
            result_list.extend(
                self.one_shot_truncation(
                    ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions)
                )
            )
        else:
            # Sort solutions by density and take the best ones
            self.density_estimator.sort(current_ranked_solutions)
            i = 0
            while len(result_list) < size_of_the_result_list:
                result_list.append(current_ranked_solutions[i])
                i += 1

        return result_list


class RankingAndCrowdingDistanceReplacement(Replacement[S]):
    """Replacement operator based on non-dominated ranking and crowding distance.

    This operator combines the parent and offspring populations, ranks them using
    non-dominated sorting, and selects the best solutions based on crowding distance.
    It's a specialized version of RankingAndDensityEstimatorReplacement that's
    specifically designed for NSGA-II and similar algorithms.

    The replacement process works as follows:
    1. Combine parent and offspring populations
    2. Rank all solutions using non-dominated sorting
    3. Fill the new population with solutions from the best fronts
    4. When a front needs to be split, use crowding distance to select
       the most diverse solutions

    Args:
        ranking: The ranking strategy to use (default: FastNonDominatedRanking)
        density_estimator: The density estimator to use (default: CrowdingDistance)

    Example:
        >>> from jmetal.operator import RankingAndCrowdingDistanceReplacement
        >>> from jmetal.core.solution import FloatSolution
        >>>
        >>> # Create a replacement operator
        >>> replacement = RankingAndCrowdingDistanceReplacement()
        >>>
        >>> # Apply replacement to combine parent and offspring populations
        >>> new_population = replacement.replace(parents, offspring)
    """

    def __init__(self, ranking: Ranking = None, density_estimator: DensityEstimator = None):
        self.ranking = ranking if ranking is not None else FastNonDominatedRanking()
        self.density_estimator = (
            density_estimator if density_estimator is not None else CrowdingDistanceDensityEstimator()
        )

    def replace(self, solution_list: list[S], offspring_list: list[S]) -> list[S]:
        """Replace solutions in the population with offspring solutions.

        This method combines the parent and offspring populations, ranks them using
        non-dominated sorting, and selects the best solutions based on crowding distance.

        Args:
            solution_list: The parent population (list of solutions).
            offspring_list: The offspring population (list of solutions).

        Returns:
            A new population with the same size as solution_list containing the
            best solutions from the combined population.

        Note:
            The size of the returned population will be equal to the size of
            solution_list, not the combined size of both populations.
        """
        join_population = solution_list + offspring_list

        # Compute ranking of the combined population
        self.ranking.compute_ranking(join_population)

        # Initialize result list
        result_list: list[S] = []

        # Fill the result list with solutions from the best fronts
        front_index = 0
        while len(result_list) < len(solution_list):
            # Get the current front
            current_front = self.ranking.get_subfront(front_index)

            # If adding the entire front won't exceed the population size, add all solutions
            if len(result_list) + len(current_front) <= len(solution_list):
                result_list.extend(current_front)
                front_index += 1
            else:
                # If we can't add the entire front, use crowding distance to select the best solutions
                self.density_estimator.compute_density_estimator(current_front)
                current_front.sort(key=lambda x: x.attributes["crowding_distance"], reverse=True)

                # Add solutions until we reach the desired population size
                remaining = len(solution_list) - len(result_list)
                result_list.extend(current_front[:remaining])

        return result_list

    def get_name(self) -> str:
        """Get the name of the replacement operator.

        Returns:
            A string representing the name of this replacement operator.
        """
        return "Ranking and crowding distance replacement"


class SMSEMOAReplacement(Replacement[S]):
    """Replacement operator for the SMS-EMOA (S-Metric Selection Evolutionary Multiobjective Algorithm).

    This replacement operator combines the parent and offspring populations, ranks them
    using non-dominated sorting, keeps every front except the last one whole, and prunes
    the last front down to size by sorting it by hypervolume contribution and dropping the
    worst-contributing solutions -- the same logic as
    `jmetal.algorithm.multiobjective.smsemoa.SMSEMOA.replacement`, generalized from
    "exactly one excess solution" (true when `offspring_population_size=1`, SMS-EMOA's
    usual steady-state configuration) to any number of excess solutions.

    The hypervolume contribution of a solution is the hypervolume that would be lost if that
    solution was removed from the front. Pruning by it, front by front, is what keeps a good
    spread of solutions along the Pareto front.

    The reference point is not fixed at construction time: it is recomputed on every
    `replace()` call as the merged population's worst objective values plus an offset of
    1.0, matching the classic `SMSEMOA`'s formula.

    Args:
        ranking: The ranking strategy to use (default: `FastNonDominatedRanking`).

    Example:
        >>> from jmetal.operator import SMSEMOAReplacement
        >>>
        >>> replacement = SMSEMOAReplacement()
        >>> new_population = replacement.replace(parents, offspring)
    """

    def __init__(self, ranking: Ranking = None):
        """Initialize the SMS-EMOA replacement operator.

        Args:
            ranking: The ranking strategy to use. Defaults to `FastNonDominatedRanking`.
        """
        self.ranking = ranking if ranking is not None else FastNonDominatedRanking()

    def replace(self, solution_list: list[S], offspring_list: list[S]) -> list[S]:
        """Replace solutions in the population with offspring solutions.

        This method combines the parent and offspring populations, ranks them using
        non-dominated sorting, keeps every front but the last whole, and -- if the last
        front doesn't fit entirely -- sorts it by hypervolume contribution (descending)
        and keeps only as many of its best-contributing solutions as needed to reach the
        size of `solution_list`. Contributions are computed once, over the whole
        overflowing front, exactly as `jmetal.algorithm.multiobjective.smsemoa.SMSEMOA`'s
        own replacement does for its single-excess-solution case -- this is that same
        computation generalized to however many solutions are in excess.

        Args:
            solution_list: The parent population (list of solutions).
            offspring_list: The offspring population (list of solutions).

        Returns:
            A new population of the same size as solution_list containing the
            best solutions from the combined population.
        """
        joint_population = solution_list + offspring_list
        self.ranking.compute_ranking(joint_population)

        num_subfronts = self.ranking.get_number_of_subfronts()
        result: list[S] = []
        for i in range(num_subfronts - 1):
            result.extend(self.ranking.get_subfront(i))

        last_front = list(self.ranking.get_subfront(num_subfronts - 1))
        target_size = len(solution_list)

        if len(result) + len(last_front) <= target_size:
            result.extend(last_front)
            return result

        reference_point = (
            np.max([s.objectives for s in joint_population], axis=0) + 1.0
        ).tolist()
        hv_estimator: HypervolumeContributionDensityEstimator[S] = (
            HypervolumeContributionDensityEstimator(reference_point=reference_point)
        )
        hv_estimator.compute_density_estimator(last_front)
        last_front.sort(key=lambda s: s.attributes["hv_contribution"], reverse=True)

        remaining_slots = target_size - len(result)
        result.extend(last_front[:remaining_slots])

        return result
