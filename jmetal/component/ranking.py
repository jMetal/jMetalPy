from abc import ABCMeta, abstractmethod
from typing import TypeVar, List

from jmetal.component.comparator import DominanceComparator

S = TypeVar('S')


class Ranking(List[S]):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(Ranking, self).__init__()
        self.number_of_comparisons = 0
        self.ranked_sublists = []

    @abstractmethod
    def compute_ranking(self, solution_list: List[S]):
        pass

    def get_subfront(self, rank: int):
        if rank >= len(self.ranked_sublists):
            raise Exception('Invalid rank: {0}. Max rank: {1}'.format(rank, len(self.ranked_sublists) - 1))
        return self.ranked_sublists[rank]

    def get_number_of_subfronts(self):
        return len(self.ranked_sublists)


class FastNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the non-dominated ranking of NSGA-II. """

    def __init__(self):
        super(FastNonDominatedRanking, self).__init__()

    def compute_ranking(self, solution_list: List[S]):
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solution_list))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solution_list))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solution_list) + 1)]

        for p in range(len(solution_list) - 1):
            for q in range(p + 1, len(solution_list)):
                dominance_test_result = DominanceComparator().compare(solution_list[p], solution_list[q])
                self.number_of_comparisons += 1

                if dominance_test_result == -1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result is 1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        for i in range(len(solution_list)):
            if dominating_ith[i] is 0:
                front[0].append(i)
                solution_list[i].attributes['dominance_ranking'] = 0

        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] is 0:
                            front[i].append(q)
                            solution_list[q].attributes['dominance_ranking'] = i

        self.ranked_sublists = [[]] * i
        for j in range(i):
            q = [0] * len(front[j])
            for k in range(len(front[j])):
                q[k] = solution_list[front[j][k]]
            self.ranked_sublists[j] = q

        return self.ranked_sublists


class EfficientNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the EDS (efficient non-dominated sorting) algorithm. """

    def __init__(self):
        super(EfficientNonDominatedRanking, self).__init__()

    def compute_ranking(self, solution_list: List[S]):
        # todo
        return self.ranked_sublists
