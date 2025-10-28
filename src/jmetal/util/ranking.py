from abc import ABC, abstractmethod
from typing import List, TypeVar, Dict

from jmetal.util.comparator import (
    Comparator,
    DominanceComparator,
    SolutionAttributeComparator,
)

S = TypeVar("S")


class Ranking(List[S], ABC):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(Ranking, self).__init__()
        self.number_of_comparisons = 0
        self.ranked_sublists = []
        self.comparator = comparator

    @abstractmethod
    def compute_ranking(self, solutions: List[S], k: int = None):
        pass

    def get_nondominated(self):
        return self.ranked_sublists[0]

    def get_subfront(self, rank: int):
        if rank >= len(self.ranked_sublists):
            raise Exception("Invalid rank: {0}. Max rank: {1}".format(rank, len(self.ranked_sublists) - 1))
        return self.ranked_sublists[rank]

    def get_number_of_subfronts(self):
        return len(self.ranked_sublists)

    @classmethod
    def get_comparator(cls) -> Comparator:
        pass


class FastNonDominatedRanking(Ranking[List[S]]):
    """Class implementing the non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]_"""

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(FastNonDominatedRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: int = None):
        """Compute ranking of solutions.
        
        Optimized implementation with improved performance:
        - Early termination when k solutions found
        - Efficient front construction
        - Minimal object allocations

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        if not solutions:
            self.ranked_sublists = []
            return self.ranked_sublists

        num_solutions = len(solutions)
        
        # number of solutions dominating solution ith
        dominating_ith = [0] * num_solutions

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(num_solutions)]

        # Optimized dominance comparison with early break
        for p in range(num_solutions - 1):
            for q in range(p + 1, num_solutions):
                dominance_test_result = self.comparator.compare(solutions[p], solutions[q])
                self.number_of_comparisons += 1

                if dominance_test_result == -1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result == 1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        # Initialize first front efficiently
        current_front = []
        for i in range(num_solutions):
            if dominating_ith[i] == 0:
                current_front.append(i)
                solutions[i].attributes["dominance_ranking"] = 0

        # Build ranked sublists incrementally with early termination
        self.ranked_sublists = []
        front_index = 0
        total_count = 0
        
        while current_front:
            # Convert indices to solutions efficiently
            front_solutions = [solutions[idx] for idx in current_front]
            self.ranked_sublists.append(front_solutions)
            
            # Early termination check
            total_count += len(current_front)
            if k and total_count >= k:
                break
                
            # Prepare next front
            next_front = []
            for p in current_front:
                for q in ith_dominated[p]:
                    dominating_ith[q] -= 1
                    if dominating_ith[q] == 0:
                        next_front.append(q)
                        solutions[q].attributes["dominance_ranking"] = front_index + 1
            
            current_front = next_front
            front_index += 1

        # Truncate if k specified
        if k and total_count > k:
            count = 0
            for i, front in enumerate(self.ranked_sublists):
                count += len(front)
                if count >= k:
                    self.ranked_sublists = self.ranked_sublists[:i + 1]
                    break

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("dominance_ranking")


class StrengthRanking(Ranking[List[S]]):
    """Class implementing a ranking scheme based on the strength ranking used in SPEA2."""

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(StrengthRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: int = None):
        """
        Compute ranking of solutions using the provided dominance comparator.
        
        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        if not solutions:
            self.ranked_sublists = []
            return self.ranked_sublists
            
        n = len(solutions)
        strength = [0] * n
        raw_fitness = [0] * n
        
        # Compute strength values (number of solutions each solution dominates)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Use the provided comparator to check if solution i dominates solution j
                if self.comparator.compare(solutions[i], solutions[j]) < 0:
                    strength[i] += 1
        
        # Compute raw fitness (sum of strengths of dominators)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if solution j dominates solution i
                if self.comparator.compare(solutions[j], solutions[i]) < 0:
                    raw_fitness[i] += strength[j]
        
        # Store raw fitness in the strength_ranking attribute and find max fitness
        max_fitness = 0
        for i in range(n):
            fitness = int(raw_fitness[i])
            solutions[i].attributes["strength_ranking"] = fitness
            if fitness > max_fitness:
                max_fitness = fitness
        
        # Group solutions by raw fitness (ascending order)
        fitness_to_solutions: Dict[int, List[S]] = {}
        for i, fit in enumerate(raw_fitness):
            if fit not in fitness_to_solutions:
                fitness_to_solutions[fit] = []
            fitness_to_solutions[fit].append(solutions[i])
        
        # Create ranked sublists sorted by fitness (ascending order)
        self.ranked_sublists = [fitness_to_solutions[f] for f in sorted(fitness_to_solutions)]

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("strength_ranking")
