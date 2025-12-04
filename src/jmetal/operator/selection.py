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


class TournamentSelection(Selection[List[S], S]):
    """Performs k-ary tournament selection.
    
    This selection operator randomly selects k solutions from the population
    and returns the best one according to the provided comparator. It's a
    generalization of binary tournament selection that allows controlling
    selection pressure through the tournament size.
    
    A larger tournament size (k) increases selection pressure, favoring
    better solutions more strongly. A smaller k provides more diversity
    but slower convergence.
    
    Args:
        tournament_size: Number of solutions to participate in each tournament (default: 2).
                        Must be at least 2.
        comparator: Comparator used to compare solutions (default: DominanceComparator).
    
    Example:
        >>> from jmetal.operator import TournamentSelection
        >>> from jmetal.util.comparator import DominanceComparator
        >>> 
        >>> # Create a tournament selection with size 5
        >>> selector = TournamentSelection(tournament_size=5)
        >>> 
        >>> # Select from a population
        >>> winner = selector.execute(population)
    """
    
    def __init__(self, tournament_size: int = 2, comparator: Comparator = DominanceComparator()):
        super().__init__()
        if tournament_size < 2:
            raise ValueError(f"Tournament size must be at least 2, got {tournament_size}")
        self.tournament_size = tournament_size
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        """Execute the k-ary tournament selection.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            The best solution among the tournament participants.
            
        Raises:
            ValueError: If front is None, empty, or smaller than tournament size.
        """
        if not front:
            raise ValueError("The front is empty")

        if len(front) == 1:
            return front[0]
        
        # Adjust tournament size if population is smaller
        effective_size = min(self.tournament_size, len(front))
        
        # Sample k solutions without replacement
        tournament_indices = random.sample(range(len(front)), effective_size)
        tournament_solutions = [front[i] for i in tournament_indices]
        
        # Find the best solution in the tournament
        winner = tournament_solutions[0]
        for i in range(1, len(tournament_solutions)):
            candidate = tournament_solutions[i]
            comparison = self.comparator.compare(candidate, winner)
            if comparison < 0:  # candidate is better
                winner = candidate
            elif comparison == 0:  # tie - randomly decide
                if random.random() < 0.5:
                    winner = candidate
        
        return winner

    def get_name(self) -> str:
        return f"Tournament selection (k={self.tournament_size})"


class BestSolutionSelection(Selection[List[S], S]):
    """Selects the best solution from a population based on dominance comparison.
    
    This selection operator returns the non-dominated solution from the population.
    If multiple solutions are non-dominated with respect to each other, it returns
    the first one encountered in the front.
    
    The comparison is done using the DominanceComparator, which follows these rules:
    - Solution A dominates solution B if A is not worse than B in all objectives
      and A is strictly better than B in at least one objective
    - If neither solution dominates the other, they are considered non-dominated
    
    Example:
        >>> from jmetal.operator import BestSolutionSelection
        >>> from jmetal.core.solution import FloatSolution
        >>> 
        >>> # Create a population of solutions
        >>> solution1 = FloatSolution([0], [1], 2)  # 2 objectives
        >>> solution1.objectives = [0.5, 0.8]
        >>> solution2 = FloatSolution([0], [1], 2)
        >>> solution2.objectives = [0.3, 0.9]
        >>> population = [solution1, solution2]
        >>> 
        >>> # Select the best solution
        >>> selector = BestSolutionSelection()
        >>> best_solution = selector.execute(population)
    """
    def __init__(self):
        """Initialize the best solution selector."""
        super(BestSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        """Select the best solution from the front.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            The best solution in the front according to dominance comparison.
            
        Raises:
            ValueError: If front is None or empty.
        """
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
        """Get the name of the selection operator.
        
        Returns:
            A string representing the name of the selection operator.
        """
        return "Best solution selection"


class NaryRandomSolutionSelection(Selection[List[S], List[S]]):
    """Performs random selection of multiple solutions from a population.
    
    This selection operator randomly selects a specified number of distinct solutions
    from the population with uniform probability. The selection is done without replacement,
    meaning each solution can be selected at most once.
    
    Args:
        number_of_solutions_to_be_returned: Number of distinct solutions to select (default: 1).
                                          Must be a positive integer.
    
    Example:
        >>> from jmetal.operator import NaryRandomSolutionSelection
        >>> 
        >>> # Select 3 random solutions
        >>> selector = NaryRandomSolutionSelection(number_of_solutions_to_be_returned=3)
        >>> selected = selector.execute(population)  # Returns List[S] with 3 solutions
    """
    
    def __init__(self, number_of_solutions_to_be_returned: int = 1):
        super(NaryRandomSolutionSelection, self).__init__()
        if number_of_solutions_to_be_returned < 1:
            raise ValueError(
                "The number of solutions to be returned must be a positive integer, got {}"
                .format(number_of_solutions_to_be_returned)
            )

        self.number_of_solutions_to_be_returned = number_of_solutions_to_be_returned

    def execute(self, front: List[S]) -> List[S]:
        """Randomly select multiple solutions from the front.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            A list of randomly selected solutions from the front.
            
        Raises:
            ValueError: If front is None, empty, or has fewer solutions than requested.
        """
        if front is None:
            raise ValueError("The front is None")
        if not front:
            raise ValueError("The front is empty")
        if len(front) < self.number_of_solutions_to_be_returned:
            raise ValueError(
                "The front size ({}) is smaller than the number of requested solutions: {}"
                .format(len(front), self.number_of_solutions_to_be_returned)
            )
        
        # Optimization: use random.choice for single selection
        if self.number_of_solutions_to_be_returned == 1:
            return [random.choice(front)]
        
        return random.sample(front, self.number_of_solutions_to_be_returned)

    def get_name(self) -> str:
        """Get the name of the selection operator.
        
        Returns:
            A string representing the name of the selection operator.
        """
        return "N-ary random solution selection"


class DifferentialEvolutionSelection(Selection[List[S], List[S]]):
    """Performs selection for differential evolution algorithms.
    
    This selection operator is specifically designed for differential evolution algorithms.
    It selects three distinct solutions from the population, with an optional index to exclude
    (typically the current solution's index to avoid self-selection).
    
    Args:
        index_to_exclude: Optional index of a solution to exclude from selection.
                         This is useful to avoid selecting the same solution as the base vector.
    """
    
    def __init__(self, index_to_exclude: int = None):
        super(DifferentialEvolutionSelection, self).__init__()
        self.index_to_exclude = index_to_exclude

    def execute(self, front: List[S]) -> List[S]:
        """Select three distinct solutions for differential evolution.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            A list containing three distinct solutions from the front.
            
        Raises:
            ValueError: If front is None, empty, or has fewer than 4 solutions.
        """
        if front is None:
            raise ValueError("The front is null")
        elif len(front) == 0:
            raise ValueError("The front is empty")
        elif len(front) < 4:
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

    def set_index_to_exclude(self, index: int) -> None:
        """Set the index of the solution to exclude from selection.
        
        Args:
            index: Index of the solution to exclude. Can be None to disable exclusion.
        """
        self.index_to_exclude = index

    def get_name(self) -> str:
        """Get the name of the selection operator.
        
        Returns:
            A string representing the name of the selection operator.
        """
        return "Differential evolution selection"


class RandomSelection(Selection[List[S], S]):
    """Performs random selection of a solution from a population.
    
    This selection operator randomly selects a single solution from the provided
    population with uniform probability. It's a simple selection method that
    doesn't consider solution quality.
    """
    
    def __init__(self):
        super(RandomSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        """Randomly select a solution from the front.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            A randomly selected solution from the front.
            
        Raises:
            ValueError: If front is None or empty.
        """
        if front is None:
            raise ValueError("The front is None")
        elif len(front) == 0:
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
    """Performs selection based on non-dominated ranking and crowding distance.
    
    This selection operator first ranks the solutions using non-dominated sorting
    and then applies crowding distance to maintain diversity within each rank.
    It's commonly used in NSGA-II and other multi-objective evolutionary algorithms.
    
    Args:
        max_population_size: Maximum number of solutions to select.
        dominance_comparator: Comparator used for non-dominated sorting.
                           Defaults to DominanceComparator().
    """
    
    def __init__(self, max_population_size: int, dominance_comparator: Comparator = DominanceComparator()):
        super(RankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator

    def execute(self, front: List[S]) -> List[S]:
        """Select solutions using non-dominated ranking and crowding distance.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            A list of selected solutions, with size up to max_population_size.
            
        Raises:
            ValueError: If front is None, empty, or max_population_size is invalid.
        """
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
        """Get the name of the selection operator.
        
        Returns:
            A string representing the name of the selection operator.
        """
        return "Ranking and crowding distance selection"


class RankingAndFitnessSelection(Selection[List[S], List[S]]):
    """Performs selection based on non-dominated ranking and hypervolume contribution.
    
    This selection operator first ranks the solutions using non-dominated sorting
    and then applies hypervolume contribution to maintain diversity within each rank.
    It's commonly used in multi-objective evolutionary algorithms that aim to
    maximize the hypervolume indicator.
    
    Args:
        max_population_size: Maximum number of solutions to select.
        reference_point: Reference point used for hypervolume calculation.
                       Should be dominated by all solutions.
        dominance_comparator: Comparator used for non-dominated sorting.
                           Defaults to DominanceComparator().
    """
    
    def __init__(
        self, max_population_size: int, reference_point: S, dominance_comparator: Comparator = DominanceComparator()
    ):
        super(RankingAndFitnessSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator
        self.reference_point = reference_point

    def hypesub(self, l: int, A: List[List[float]], actDim: int, bounds: List[float], 
               pvec: List[int], alpha: List[float], k: int) -> List[float]:
        """Recursively compute hypervolume contributions.
        
        This is a helper method for hypervolume calculation. It's an implementation
        of the Hype algorithm for hypervolume approximation.
        
        Args:
            l: Number of points.
            A: List of objective vectors.
            actDim: Current dimension being processed.
            bounds: Reference point coordinates.
            pvec: Indices of points in A.
            alpha: Weighting factors for hypervolume contribution.
            k: Number of points to consider.
            
        Returns:
            List of hypervolume contributions for each point.
        """
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

    def compute_hypervol_fitness_values(self, population: List[S], reference_point: S, k: int) -> List[S]:
        """Compute hypervolume-based fitness values for a population.
        
        This method computes the hypervolume contribution of each solution in the
        population and stores it in the solution's attributes as 'fitness'.
        
        Args:
            population: List of solutions to evaluate.
            reference_point: Reference point for hypervolume calculation.
            k: Number of points to consider for hypervolume approximation.
               If negative, uses the entire population size.
               
        Returns:
            The input population with updated fitness values in their attributes.
        """
        points = [ind.objectives for ind in population]
        bounds = reference_point.objectives
        population_size = len(points)

        if k < 0:
            k = population_size

        actDim = len(bounds)
        pvec = range(population_size)
        alpha = []

        # Calculate alpha values for weighted hypervolume contribution
        for i in range(1, k + 1):
            alpha.append(np.prod([float(k - j) / (population_size - j) for j in range(1, i)]) / i)

        # Compute hypervolume contributions
        f = self.hypesub(population_size, points, actDim, bounds, pvec, alpha, k)

        # Store fitness values in solution attributes
        for i in range(len(population)):
            if not hasattr(population[i], 'attributes') or population[i].attributes is None:
                population[i].attributes = {}
            population[i].attributes["fitness"] = f[i]

        return population

    def execute(self, front: List[S]) -> List[S]:
        """Select solutions using non-dominated ranking and hypervolume contribution.
        
        This method first performs non-dominated sorting of the input front.
        It then fills the new population with solutions from the best ranks,
        using hypervolume contribution to select solutions when a rank needs to be split.
        
        Args:
            front: List of solutions to select from.
            
        Returns:
            A list of selected solutions, with size equal to max_population_size.
            
        Raises:
            ValueError: If front is None or empty.
        """
        if front is None:
            raise ValueError("The front is None")
        elif len(front) == 0:
            raise ValueError("The front is empty")

        # Perform non-dominated sorting
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(front)

        ranking_index = 0
        new_solution_list = []

        # Fill the new population with solutions from the best ranks
        while len(new_solution_list) < self.max_population_size:
            current_rank = ranking.get_subfront(ranking_index)
            
            # If we can take all solutions from this rank without exceeding max_population_size
            if len(current_rank) <= self.max_population_size - len(new_solution_list):
                new_solution_list.extend(current_rank)
                ranking_index += 1
            else:
                # Need to select a subset of this rank using hypervolume contribution
                remaining_slots = self.max_population_size - len(new_solution_list)
                parameter_K = len(current_rank) - remaining_slots
                
                # Remove the worst solutions based on hypervolume contribution
                while parameter_K > 0:
                    current_rank = self.compute_hypervol_fitness_values(
                        current_rank, self.reference_point, parameter_K)
                    # Sort by fitness (hypervolume contribution) in descending order
                    current_rank = sorted(current_rank, 
                                       key=lambda x: x.attributes.get("fitness", 0), 
                                       reverse=True)
                    # Remove the solution with the lowest contribution
                    current_rank = current_rank[:-1]
                    parameter_K -= 1
                
                new_solution_list.extend(current_rank)
                
        return new_solution_list

    def get_name(self) -> str:
        """Get the name of the selection operator.
        
        Returns:
            A string representing the name of the selection operator.
        """
        return "Ranking and hypervolume-based selection"


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
