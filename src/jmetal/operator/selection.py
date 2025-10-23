import random
from typing import List, TypeVar

import numpy as np

from jmetal.core.operator import Selection
from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator, DominanceComparator
from jmetal.util.density_estimator import CrowdingDistanceDensityEstimator
from jmetal.util.ranking import FastNonDominatedRanking

S = TypeVar("S", bound=Solution)

"""
.. module:: selection
   :platform: Unix, Windows
   :synopsis: Module implementing selection operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class RouletteWheelSelection(Selection[List[S], S]):
    """Performs roulette wheel selection.
    
    This selection operator selects solutions based on their fitness values using a roulette
    wheel mechanism. It can handle both single and multi-objective optimization by using
    the first objective value for selection. For multi-objective optimization, consider
    using a proper fitness assignment strategy first.
    
    Note: This implementation assumes all objective values are non-negative. If negative
    values are present, a proper normalization should be applied first.
    """

    def __init__(self, objective_index: int = 0):
        """Initialize the roulette wheel selection operator.
        
        Args:
            objective_index: Index of the objective to use for selection (default: 0).
                            Only used if no fitness value is present in the solution attributes.
        """
        super().__init__()
        self.objective_index = objective_index

    def execute(self, front: List[S]) -> S:
        """Select a solution using roulette wheel selection.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            The selected solution.
            
        Raises:
            ValueError: If the front is None, empty, or contains invalid fitness values.
        """
        if not front:
            raise ValueError("The front is empty")
            
        # Calculate fitness values (using first objective if no fitness attribute)
        fitness_values = []
        for solution in front:
            if hasattr(solution, 'fitness') and solution.fitness is not None:
                fitness_values.append(solution.fitness)
            else:
                # Fallback to using the specified objective
                fitness_values.append(solution.objectives[self.objective_index])
        
        # Convert to numpy array for efficient operations
        fitness_values = np.array(fitness_values, dtype=float)
        
        # Check for invalid fitness values
        if np.any(fitness_values < 0):
            raise ValueError("Negative fitness values are not supported. "
                           "Consider normalizing the fitness values first.")
        
        # If all values are zero, return a random solution
        total_fitness = np.sum(fitness_values)
        if total_fitness <= 0:
            return random.choice(front)
            
        # Calculate selection probabilities
        probabilities = fitness_values / total_fitness
        
        # Select a solution based on the probabilities
        selected_index = np.random.choice(len(front), p=probabilities)
        return front[selected_index]

    def get_name(self) -> str:
        return "Roulette wheel selection"


class BinaryTournamentSelection(Selection[List[S], S]):
    """Performs binary tournament selection between two random solutions.
    
    This selection operator randomly selects two solutions from the population
    and returns the better one according to the provided comparator.
    If the comparator returns 0 (tie), a random solution is chosen.
    
    Args:
        comparator: Comparator used to compare solutions (default: DominanceComparator)
    """
    
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super().__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        """Execute the binary tournament selection.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            The selected solution.
            
        Raises:
            ValueError: If front is None or empty.
        """
        if not front:
            raise ValueError("The front is empty")

        if len(front) == 1:
            return front[0]

        # Sample without replacement
        idx1, idx2 = random.sample(range(len(front)), 2)
        solution1 = front[idx1]
        solution2 = front[idx2]

        # Compare solutions
        comparison = self.comparator.compare(solution1, solution2)
        
        if comparison == -1:
            return solution1
        elif comparison == 1:
            return solution2
        else:  # Tie - choose randomly
            return solution1 if random.random() < 0.5 else solution2

    def get_name(self) -> str:
        return "Binary tournament selection"


class BestSolutionSelection(Selection[List[S], S]):
    def __init__(self):
        super(BestSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise ValueError("The front is None")
        if not front:
            raise ValueError("The front is empty")

        result = front[0]

        for solution in front[1:]:
            if DominanceComparator().compare(solution, result) < 0:
                result = solution

        return result

    def get_name(self) -> str:
        return "Best solution selection"


class NaryRandomSolutionSelection(Selection[List[S], S]):
    def __init__(self, number_of_solutions_to_be_returned: int = 1):
        super(NaryRandomSolutionSelection, self).__init__()
        if number_of_solutions_to_be_returned < 1:
            raise ValueError(
                "The number of solutions to be returned must be a positive integer, got {}"
                .format(number_of_solutions_to_be_returned)
            )

        self.number_of_solutions_to_be_returned = number_of_solutions_to_be_returned

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise ValueError("The front is None")
        if not front:
            raise ValueError("The front is empty")
        if len(front) < self.number_of_solutions_to_be_returned:
            raise ValueError(
                "The front size ({}) is smaller than the number of requested solutions: {}"
                .format(len(front), self.number_of_solutions_to_be_returned)
            )
        return random.sample(front, self.number_of_solutions_to_be_returned)

    def get_name(self) -> str:
        return "Nary random_search solution selection"


class DifferentialEvolutionSelection(Selection[List[S], List[S]]):
    def __init__(self, index_to_exclude: int = None):
        super(DifferentialEvolutionSelection, self).__init__()
        self.index_to_exclude = index_to_exclude

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise ValueError("The front is None")
        if not front:
            raise ValueError("The front is empty")
        if len(front) < 4:  # Need at least 4 solutions (1 base + 3 for DE/rand/1)
            raise ValueError(
                f"Differential evolution selection requires at least 4 solutions, got {len(front)}"
            )

        # If there's an index to exclude, create a new list without it
        if self.index_to_exclude is not None and 0 <= self.index_to_exclude < len(front):
            candidates = [sol for i, sol in enumerate(front) if i != self.index_to_exclude]
        else:
            candidates = list(front)
        
        # Check if we have enough candidates after exclusion
        if len(candidates) < 3:
            raise ValueError(
                f"Not enough candidates to select from (need 3, have {len(candidates)} after exclusion)"
            )
            
        # Randomly select 3 distinct solutions from the remaining candidates
        selected = random.sample(candidates, 3)
        
        return selected

    def set_index_to_exclude(self, index: int):
        self.index_to_exclude = index

    def get_name(self) -> str:
        return "Differential evolution selection"


class RandomSelection(Selection[List[S], S]):
    def __init__(self):
        super(RandomSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise ValueError("The front is None")
        if not front:
            raise ValueError("The front is empty")

        if not isinstance(front, list):
            raise ValueError("The front must be a list")

        # Check if all elements are instances of the same type as the first element
        if front and not all(isinstance(solution, front[0].__class__) for solution in front):
            raise ValueError("All elements in the front must be of the same type")

        return random.choice(front)

    def get_name(self) -> str:
        return "Random selection"


class RankingAndCrowdingDistanceSelection(Selection[List[S], List[S]]):
    def __init__(self, max_population_size: int, dominance_comparator: Comparator = DominanceComparator()):
        super(RankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise ValueError("The front is None")
        if not front:
            raise ValueError("The front is empty")
        if not isinstance(self.max_population_size, int) or self.max_population_size <= 0:
            raise ValueError("max_population_size must be a positive integer")

        # If the front is smaller than max_population_size, return the entire front
        if len(front) <= self.max_population_size:
            return front.copy()

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        crowding_distance = CrowdingDistanceDensityEstimator()
        ranking.compute_ranking(front)

        ranking_index = 0
        new_solution_list = []
        number_of_subfronts = ranking.get_number_of_subfronts()

        while len(new_solution_list) < self.max_population_size and ranking_index < number_of_subfronts:
            subfront = ranking.get_subfront(ranking_index)
            
            # If adding the entire subfront doesn't exceed max_population_size, add it all
            if len(new_solution_list) + len(subfront) <= self.max_population_size:
                new_solution_list.extend(subfront)
            else:
                # Otherwise, sort by crowding distance and add the best remaining solutions
                crowding_distance.compute_density_estimator(subfront)
                # Sort by crowding distance in descending order
                sorted_subfront = sorted(
                    subfront, 
                    key=lambda x: x.attributes.get("crowding_distance", 0.0), 
                    reverse=True
                )
                # Take only as many as needed to fill the population
                remaining = self.max_population_size - len(new_solution_list)
                new_solution_list.extend(sorted_subfront[:remaining])
            
            ranking_index += 1

        return new_solution_list

    def get_name(self) -> str:
        return "Ranking and crowding distance selection"


class RankingAndFitnessSelection(Selection[List[S], List[S]]):
    def __init__(
        self, max_population_size: int, reference_point: S, dominance_comparator: Comparator = DominanceComparator()
    ):
        super(RankingAndFitnessSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator
        self.reference_point = reference_point

    def hypesub(self, l, A, actDim, bounds, pvec, alpha, k):
        h = [0 for _ in range(l)]
        Adim = [a[actDim - 1] for a in A]
        indices_sort = sorted(range(len(Adim)), key=Adim.__getitem__)
        S = [A[j] for j in indices_sort]
        pvec = [pvec[j] for j in indices_sort]

        for i in range(1, len(S) + 1):
            if i < len(S):
                extrusion = S[i][actDim - 1] - S[i - 1][actDim - 1]
            else:
                extrusion = bounds[actDim - 1] - S[i - 1][actDim - 1]

            if actDim == 1:
                if i > k:
                    break
                if all(alpha) >= 0:
                    for p in pvec[0:i]:
                        h[p] = h[p] + extrusion * alpha[i - 1]
            else:
                if extrusion > 0:
                    h = [
                        h[j] + extrusion * self.hypesub(l, S[0:i], actDim - 1, bounds, pvec[0:i], alpha, k)[j]
                        for j in range(l)
                    ]

        return h

    def compute_hypervol_fitness_values(self, population: List[S], reference_point: S, k: int):
        points = [ind.objectives for ind in population]
        bounds = reference_point.objectives
        population_size = len(points)

        if k < 0:
            k = population_size

        actDim = len(bounds)
        pvec = range(population_size)
        alpha = []

        for i in range(1, k + 1):
            alpha.append(np.prod([float(k - j) / (population_size - j) for j in range(1, i)]) / i)

        f = self.hypesub(population_size, points, actDim, bounds, pvec, alpha, k)

        for i in range(len(population)):
            population[i].attributes["fitness"] = f[i]

        return population

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(front)

        ranking_index = 0
        new_solution_list = []

        while len(new_solution_list) < self.max_population_size:
            if len(ranking.get_subfront(ranking_index)) < self.max_population_size - len(new_solution_list):
                subfront = ranking.get_subfront(ranking_index)
                new_solution_list = new_solution_list + subfront
                ranking_index += 1
            else:
                subfront = ranking.get_subfront(ranking_index)
                parameter_K = len(subfront) - (self.max_population_size - len(new_solution_list))
                while parameter_K > 0:
                    subfront = self.compute_hypervol_fitness_values(subfront, self.reference_point, parameter_K)
                    subfront = sorted(subfront, key=lambda x: x.attributes["fitness"], reverse=True)
                    subfront = subfront[:-1]
                    parameter_K = parameter_K - 1
                new_solution_list = new_solution_list + subfront
        return new_solution_list

    def get_name(self) -> str:
        return "Ranking and fitness selection"


class BinaryTournament2Selection(Selection[List[S], S]):
    """Performs binary tournament selection with multiple comparators.
    
    This selection operator uses a list of comparators in sequence to determine
    the winner between two randomly selected solutions. The first comparator that
    can determine a winner is used. If all comparators result in a tie, a random
    solution is chosen.
    
    Args:
        comparator_list: List of comparators to use in sequence.
    """
    
    def __init__(self, comparator_list: List[Comparator]):
        super().__init__()
        if not comparator_list:
            raise ValueError("The comparator list cannot be empty")
        self.comparator_list = comparator_list

    def execute(self, front: List[S]) -> S:
        """Execute the binary tournament selection with multiple comparators.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            The selected solution.
            
        Raises:
            ValueError: If front is None, empty, or contains only one solution.
        """
        if front is None:
            raise ValueError("The front is None")
            
        if not front:
            raise ValueError("The front is empty")
            
        if len(front) == 1:
            return front[0]

        # Use the first comparator to get initial winner
        result = self.__winner(front, self.comparator_list[0])
        
        # If first comparator couldn't decide, try the rest
        if result is None and len(self.comparator_list) > 1:
            for comparator in self.comparator_list[1:]:
                result = self.__winner(front, comparator)
                if result is not None:
                    break
        
        # If no comparator could decide, choose randomly
        if result is None:
            idx = random.randint(0, len(front) - 1)
            result = front[idx]
            
        return result

    def __winner(self, front: List[S], comparator: Comparator) -> S:
        """Select a winner between two random solutions using the given comparator.
        
        Args:
            front: List of solutions to select from.
            comparator: Comparator to determine the winner.
            
        Returns:
            The winning solution, or None if it's a tie.
        """
        # Sampling without replacement
        i, j = random.sample(range(0, len(front)), 2)

        solution1 = front[i]
        solution2 = front[j]

        flag = comparator.compare(solution1, solution2)

        if flag == -1:
            result = solution1
        elif flag == 1:
            result = solution2
        else:
            result = None

        return result

    def get_name(self) -> str:
        return "Binary tournament selection (experimental)"
