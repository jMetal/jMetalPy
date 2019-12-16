from abc import ABC, abstractmethod
from typing import TypeVar, List

from jmetal.util.comparator import DominanceComparator, Comparator, SolutionAttributeComparator

S = TypeVar('S')


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
            raise Exception('Invalid rank: {0}. Max rank: {1}'.format(rank, len(self.ranked_sublists) - 1))
        return self.ranked_sublists[rank]

    def get_number_of_subfronts(self):
        return len(self.ranked_sublists)

    @classmethod
    def get_comparator(cls) -> Comparator:
        pass


class FastNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]_ """

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(FastNonDominatedRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: int = None):
        """ Compute ranking of solutions.

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]

        for p in range(len(solutions) - 1):
            for q in range(p + 1, len(solutions)):
                dominance_test_result = self.comparator.compare(solutions[p], solutions[q])
                self.number_of_comparisons += 1

                if dominance_test_result == -1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result is 1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        for i in range(len(solutions)):
            if dominating_ith[i] is 0:
                front[0].append(i)
                solutions[i].attributes['dominance_ranking'] = 0

        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] is 0:
                            front[i].append(q)
                            solutions[q].attributes['dominance_ranking'] = i

        self.ranked_sublists = [[]] * i
        for j in range(i):
            q = [0] * len(front[j])
            for m in range(len(front[j])):
                q[m] = solutions[front[j][m]]
            self.ranked_sublists[j] = q

        if k:
            count = 0
            for i, front in enumerate(self.ranked_sublists):
                count += len(front)
                if count >= k:
                    self.ranked_sublists = self.ranked_sublists[:i + 1]
                    break

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator('dominance_ranking')


class StrengthRanking(Ranking[List[S]]):
    """ Class implementing a ranking scheme based on the strength ranking used in SPEA2. """

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(StrengthRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: int = None):
        """
        Compute ranking of solutions.

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        strength: [int] = [0 for _ in range(len(solutions))]
        raw_fitness: [int] = [0 for _ in range(len(solutions))]

        # strength(i) = | {j | j < - SolutionSet and i dominate j} |
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) < 0:
                    strength[i] += 1

        # Calculate the raw fitness:
        # rawFitness(i) = |{sum strength(j) | j <- SolutionSet and j dominate i}|
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) == 1:
                    raw_fitness[i] += strength[j]

        max_fitness_value: int = 0
        for i in range(len(solutions)):
            solutions[i].attributes['strength_ranking'] = raw_fitness[i]
            if raw_fitness[i] > max_fitness_value:
                max_fitness_value = raw_fitness[i]

        # Initialize the ranked sublists. In the worst case will be max_fitness_value + 1 different sublists
        self.ranked_sublists = [[] for _ in range(max_fitness_value + 1)]

        # Assign each solution to its corresponding front
        for solution in solutions:
            self.ranked_sublists[int(solution.attributes['strength_ranking'])].append(solution)

        # Remove empty fronts
        counter = 0
        while counter < len(self.ranked_sublists):
            if len(self.ranked_sublists[counter]) == 0:
                del self.ranked_sublists[counter]
            else:
                counter += 1

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator('strength_ranking')
