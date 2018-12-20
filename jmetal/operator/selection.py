import random
from typing import List, TypeVar

from jmetal.component.comparator import Comparator, DominanceComparator
from jmetal.component.density_estimator import CrowdingDistance
from jmetal.component.ranking import FastNonDominatedRanking
from jmetal.core.operator import Selection, R

S = TypeVar('S')

"""
.. module:: selection
   :platform: Unix, Windows
   :synopsis: Module implementing selection operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class RouletteWheelSelection(Selection[List[S], S]):
    """Performs roulette wheel selection.
    """

    def __init__(self):
        super(RouletteWheelSelection).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        maximum = sum([solution.objectives[0] for solution in front])
        rand = random.uniform(0.0, maximum)
        value = 0.0

        for solution in front:
            value += solution.objectives[0]

            if value > rand:
                return solution

        return None

    def get_name(self) -> str:
        return 'Roulette wheel selection'


class BinaryTournamentSelection(Selection[List[S], S]):

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(BinaryTournamentSelection, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            i, j = random.sample(range(0, len(front)), 2)
            solution1 = front[i]
            solution2 = front[j]

            flag = self.comparator.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result

    def get_name(self) -> str:
        return 'Binary tournament selection'


class BestSolutionSelection(Selection[List[S], S]):

    def __init__(self):
        super(BestSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        result = front[0]

        for solution in front[1:]:
            if DominanceComparator().compare(solution, result) < 0:
                result = solution

        return result

    def get_name(self) -> str:
        return 'Best solution selection'


class NaryRandomSolutionSelection(Selection[List[S], S]):

    def __init__(self, number_of_solutions_to_be_returned: int = 1):
        super(NaryRandomSolutionSelection, self).__init__()
        if number_of_solutions_to_be_returned < 0:
            raise Exception('The number of solutions to be returned must be positive integer')

        self.number_of_solutions_to_be_returned = number_of_solutions_to_be_returned

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        if len(front) == 0:
            raise Exception('The front is empty')
        if len(front) < self.number_of_solutions_to_be_returned:
            raise Exception('The front contains less elements than required')

        # random sampling without replacement
        return random.sample(front, self.number_of_solutions_to_be_returned)

    def get_name(self) -> str:
        return 'Nary random solution selection'


class DifferentialEvolutionSelection(Selection[List[S], List[S]]):

    def __init__(self):
        super(DifferentialEvolutionSelection, self).__init__()
        self.index_to_exclude = None

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')
        elif len(front) < 4:
            raise Exception('The front has less than four solutions: ' + str(len(front)))

        selected_indexes = random.sample(range(len(front)), 3)
        while self.index_to_exclude in selected_indexes:
            selected_indexes = random.sample(range(len(front)), 3)

        return [front[i] for i in selected_indexes]

    def set_index_to_exclude(self, index:int):
        self.index_to_exclude = index

    def get_name(self) -> str:
        return "Differential evolution selection"


class RandomSolutionSelection(Selection[List[S], S]):

    def __init__(self):
        super(RandomSolutionSelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        return random.choice(front)

    def get_name(self) -> str:
        return 'Random solution selection'


class RankingAndCrowdingDistanceSelection(Selection[List[S], List[S]]):

    def __init__(self, max_population_size: int, dominance_comparator: Comparator = DominanceComparator()):
        super(RankingAndCrowdingDistanceSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator

    def execute(self, front: List[S]) -> List[S]:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        crowding_distance = CrowdingDistance()
        ranking.compute_ranking(front)

        ranking_index = 0
        new_solution_list = []

        while len(new_solution_list) < self.max_population_size:
            if len(ranking.get_subfront(ranking_index)) < self.max_population_size - len(new_solution_list):
                new_solution_list = new_solution_list + ranking.get_subfront(ranking_index)
                ranking_index += 1
            else:
                subfront = ranking.get_subfront(ranking_index)
                crowding_distance.compute_density_estimator(subfront)
                sorted_subfront = sorted(subfront, key=lambda x: x.attributes['crowding_distance'], reverse=True)
                for i in range((self.max_population_size - len(new_solution_list))):
                    new_solution_list.append(sorted_subfront[i])

        return new_solution_list

    def get_name(self) -> str:
        return 'Ranking and crowding distance selection'


class BinaryTournament2Selection(Selection[List[S], S]):

    def __init__(self, comparator_list: List[Comparator]):
        super(BinaryTournament2Selection, self).__init__()
        self.comparator_list = comparator_list

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')
        elif not self.comparator_list:
            raise Exception('The comparators\' list is empty')

        winner = None

        if len(front) == 1:
            winner = front[0]
        else:
            for comparator in self.comparator_list:
                winner = self.__winner(front, comparator)
                if winner is not None:
                    break

        if not winner:
            i = random.randrange(0, len(front))
            winner = front[i]

        return winner

    def __winner(self, front: List[S], comparator: Comparator):
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
        return 'Binary tournament selection (experimental)'
