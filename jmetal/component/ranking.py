from typing import TypeVar, List

from jmetal.component.comparator import DominanceComparator

S = TypeVar('S')


class Ranking(List[S]):

    def __init__(self):
        super().__init__()
        self.number_of_comparisons = 0
        self.ranked_sublists = []

    def compute_ranking(self, solution_list: List[S]):
        pass

    def get_subfront(self, rank: int):
        if rank >= len(self.ranked_sublists):
            raise Exception('Invalid rank: {0}. Max rank: {1}'.format(rank, len(self.ranked_sublists) - 1))
        return self.ranked_sublists[rank]

    def get_number_of_subfronts(self):
        return len(self.ranked_sublists)

    def number_of_comparisons(self) -> int:
        return self.number_of_comparisons


class FastNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the non-dominated ranking of NSGA-II. """

    def __init__(self):
        super(FastNonDominatedRanking, self).__init__()

    def compute_ranking(self, solution_list: List[S]):
        # number of solutions dominating solution ith
        dominate_me = [0 for _ in range(len(solution_list))]

        # list of solutions dominated by solution ith
        i_dominate = [[] for _ in range(len(solution_list))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solution_list) + 1)]

        for p in range(len(solution_list) - 1):
            for q in range(p + 1, len(solution_list)):
                dominance_test_result = DominanceComparator().compare(solution_list[p], solution_list[q])
                self.number_of_comparisons += 1
                if dominance_test_result is -1:
                    i_dominate[p].append(q)
                    dominate_me[q] += 1
                elif dominance_test_result is 1:
                    i_dominate[q].append(p)
                    dominate_me[p] += 1

        for i in range(len(solution_list)):
            if dominate_me[i] is 0:
                front[0].append(i)
                solution_list[i].attributes['dominance_ranking'] = 0

        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(i_dominate):
                    for q in i_dominate[p]:
                        index = q
                        dominate_me[index] -= 1
                        if dominate_me[index] is 0:
                            front[i].append(index)
                            solution_list[index].attributes['dominance_ranking'] = i

        self.ranked_sublists = [[]] * i
        for j in range(i):
            Q = [0] * len(front[j])
            for k in range(len(front[j])):
                Q[k] = solution_list[front[j][k]]
            self.ranked_sublists[j] = Q

        return self.ranked_sublists


class EfficientNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the EDS (efficient non-dominated sorting) algorithm. """

    def __init__(self):
        super(EfficientNonDominatedRanking, self).__init__()

    def compute_ranking(self, solution_list: List[S]):
        # todo
        return self.ranked_sublists
