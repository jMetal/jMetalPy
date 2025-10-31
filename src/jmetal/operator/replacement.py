from enum import Enum
from typing import List, TypeVar

from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.density_estimator import DensityEstimator
from jmetal.util.density_estimator import HypervolumeContributionDensityEstimator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.ranking import Ranking

S = TypeVar("S")


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


class RankingAndDensityEstimatorReplacement:
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
        self, ranking: Ranking, density_estimator: DensityEstimator, removal_policy=RemovalPolicyType.ONE_SHOT
    ):
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.removal_policy = removal_policy

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
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
            result_list = self.sequential_truncation(0, len(solution_list))
        else:
            result_list = self.one_shot_truncation(0, len(solution_list))

        return result_list

    def sequential_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
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

        result_list: List[S] = []

        if len(current_ranked_solutions) < size_of_the_result_list:
            # If the entire front fits, add all solutions and move to the next front
            result_list.extend(self.ranking.get_subfront(ranking_id))
            result_list.extend(
                self.sequential_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
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

    def one_shot_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
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

        result_list: List[S] = []

        if len(current_ranked_solutions) < size_of_the_result_list:
            # If the entire front fits, add all solutions and move to the next front
            result_list.extend(self.ranking.get_subfront(ranking_id))
            result_list.extend(
                self.one_shot_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            # Sort solutions by density and take the best ones
            self.density_estimator.sort(current_ranked_solutions)
            i = 0
            while len(result_list) < size_of_the_result_list:
                result_list.append(current_ranked_solutions[i])
                i += 1

        return result_list

class RankingAndCrowdingDistanceReplacement:
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
    def __init__(
        self, ranking: Ranking = None, density_estimator: DensityEstimator = None
    ):
        self.ranking = ranking if ranking is not None else FastNonDominatedRanking()
        self.density_estimator = density_estimator if density_estimator is not None else CrowdingDistance()

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
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
        result_list: List[S] = []
        
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
                current_front.sort(key=lambda x: x.attributes['crowding_distance'], reverse=True)
                
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


class SMSEMOAReplacement:
    """Replacement operator for the SMS-EMOA (S-Metric Selection Evolutionary Multiobjective Algorithm).
    
    This replacement operator is specifically designed for the SMS-EMOA algorithm. It works by:
    1. Combining the parent and offspring populations
    2. Performing non-dominated sorting to identify the first front
    3. Removing the solution with the smallest hypervolume contribution from the first front
    
    The hypervolume contribution of a solution is the hypervolume that would be lost if that
    solution was removed from the front. This helps maintain a good spread of solutions
    along the Pareto front.
    
    Args:
        reference_point: The reference point used for hypervolume calculation. This point
                       should be dominated by all solutions in the population.
                       
    Example:
        >>> from jmetal.operator import SMSEMOAReplacement
        >>> from jmetal.core.solution import FloatSolution
        >>> 
        >>> # Create a replacement operator with a reference point
        >>> reference_point = FloatSolution([], [], 2)  # 2 objectives
        >>> reference_point.objectives = [1.1, 1.1]  # Slightly worse than any solution
        >>> replacement = SMSEMOAReplacement(reference_point)
        >>> 
        >>> # Apply replacement to combine parent and offspring populations
        >>> new_population = replacement.replace(parents, offspring)
    """
    def __init__(self, reference_point: S):
        """Initialize the SMS-EMOA replacement operator.
        
        Args:
            reference_point: The reference point for hypervolume calculation.
        """
        self.reference_point = reference_point

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions in the population with offspring solutions.
        
        This method combines the parent and offspring populations, performs non-dominated
        sorting, and removes the solution with the smallest hypervolume contribution
        from the first front.
        
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
        # Merge populations
        population = solution_list + offspring_list

        # Compute non-dominated ranking
        ranking = FastNonDominatedRanking()
        ranking.compute_ranking(population)
        first_front = ranking.get_subfront(0)

        # Compute hypervolume contributions for first front
        hv_estimator = HypervolumeContributionDensityEstimator(reference_point=self.reference_point)
        hv_estimator.compute_density_estimator(first_front)

        # Find solution with minimum hypervolume contribution
        min_hv_solution = min(first_front, key=lambda s: s.attributes["hv_contribution"])

        # Remove the solution with minimum contribution from population
        population.remove(min_hv_solution)

        # Return truncated population
        return population
