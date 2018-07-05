import random
from typing import List, TypeVar

from jmetal.component.density_estimator import CrowdingDistance
from jmetal.core.operator import Selection
from jmetal.component.comparator import Comparator, DominanceComparator
from jmetal.component.ranking import FastNonDominatedRanking

S = TypeVar('S')

"""
.. module:: Selection operators
   :platform: Unix, Windows
   :synopsis: Module implementing selection operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class BinaryTournamentSelection(Selection[List[S], S]):
    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(BinaryTournamentSelection, self).__init__()
        self.comparator = comparator

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        if len(solution_list) == 1:
            result = solution_list[0]
        else:
            i, j = random.sample(range(0, len(solution_list)), 2)  # sampling without replacement
            solution1 = solution_list[i]
            solution2 = solution_list[j]

            flag = self.comparator.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result

    def get_name(self) -> str:
        return "Binary tournament selection"


class BestSolutionSelection(Selection[List[S], S]):

    def __init__(self):
        super(BestSolutionSelection, self).__init__()

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        result = solution_list[0]
        for solution in solution_list[1:]:
            if DominanceComparator().compare(solution, result) < 0:
                result = solution

        return result


class NaryRandomSolutionSelection(Selection[List[S], S]):

    def __init__(self, number_of_solutions_to_be_returned:int = 1):
        super(NaryRandomSolutionSelection, self).__init__()
        if number_of_solutions_to_be_returned < 0:
            raise Exception("The number of solutions to be returned must be positive integer")

        self.number_of_solutions_to_be_returned = number_of_solutions_to_be_returned

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        if len(solution_list) == 0:
            raise Exception("The solution is empty")
        if len(solution_list) < self.number_of_solutions_to_be_returned:
            raise Exception("The solution list contains less elements then requred")

        # random sampling without replacement
        return random.sample(solution_list, self.number_of_solutions_to_be_returned)


class RandomSolutionSelection(Selection[List[S], S]):

    def __init__(self):
        super(RandomSolutionSelection, self).__init__()

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        return random.choice(solution_list)


class RankingAndCrowdingDistanceSelection(Selection[List[S], List[S]]):

    def __init__(self, max_population_size: int):
        super(RankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size

    def execute(self, solution_list: List[S]) -> List[S]:
        ranking = FastNonDominatedRanking()
        crowding_distance = CrowdingDistance()
        ranking.compute_ranking(solution_list)

        ranking_index = 0
        new_solution_list = []

        while len(new_solution_list) < self.max_population_size:
            if len(ranking.get_subfront(ranking_index)) < self.max_population_size - len(new_solution_list):
                new_solution_list = new_solution_list + ranking.get_subfront(ranking_index)
                ranking_index += 1
            else:
                subfront = ranking.get_subfront(ranking_index)
                crowding_distance.compute_density_estimator(subfront)
                sorted_subfront = sorted(subfront, key=lambda x: x.attributes["crowding_distance"], reverse=True)
                for i in range((self.max_population_size - len(new_solution_list))):
                    new_solution_list.append(sorted_subfront[i])

        return new_solution_list


class BinaryTournament2Selection(Selection[List[S], S]):

    def __init__(self, comparator_list: List[Comparator]):
        super(BinaryTournament2Selection, self).__init__()
        self.comparator_list = comparator_list

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")
        elif not self.comparator_list:
            raise Exception("The list of comparators is empty")

        winner = None

        if len(solution_list) == 1:
            winner = solution_list[0]
        else:
            for comparator in self.comparator_list:
                winner = self.__winner(solution_list, comparator)
                if winner is not None:
                    break

        if not winner:
            i = random.randrange(0, len(solution_list))
            winner = solution_list[i]

        return winner

    def __winner(self, solution_list: List[S], comparator: Comparator):
        i, j = random.sample(range(0, len(solution_list)), 2)  # sampling without replacement
        solution1 = solution_list[i]
        solution2 = solution_list[j]

        flag = comparator.compare(solution1, solution2)

        if flag == -1:
            result = solution1
        elif flag == 1:
            result = solution2
        else:
            result = None

        return result

    def get_name(self):
        return "Binary tournament selection (experimental)"
