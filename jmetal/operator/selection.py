import copy
import random
import sys
from typing import List, TypeVar

import numpy as np

from jmetal.core.operator import Selection
from jmetal.util.comparator import Comparator, DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.point import ReferencePoint
from jmetal.util.ranking import FastNonDominatedRanking

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

    def set_index_to_exclude(self, index: int):
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


class EnvironmentalSelection(Selection[List[S], S]):

    def __init__(self, number_of_objectives: int, k: int):
        super(EnvironmentalSelection, self).__init__()
        self.number_of_objectives = number_of_objectives
        self.reference_points = self.generate_reference_points(self.number_of_objectives)
        self.k = k

    def generate_reference_points(self, num_objs: int, num_divisions_per_obj: int = 12):
        def gen_refs_recursive(position, num_objs, left, total, element):
            if element == num_objs - 1:
                position[element] = left / total
                return [ReferencePoint(copy.deepcopy(position))]
            else:
                res = []
                for i in range(left):
                    position[element] = i / total
                    res += gen_refs_recursive(position, num_objs, left - i, total, element + 1)
                return res

        return gen_refs_recursive([0] * num_objs, num_objs, num_objs * num_divisions_per_obj,
                                  num_objs * num_divisions_per_obj, 0)

    def find_ideal_point(self, solutions: List[S]):
        ideal_point = [float('inf')] * self.number_of_objectives

        for solution in solutions:
            for i in range(self.number_of_objectives):
                ideal_point[i] = min(ideal_point[i], solution.objectives[i])

        return ideal_point

    def ASF(self, solution: S, idx: int):
        """ Achivement Scalarization Function. """
        max_ratio = -float('inf')

        for i in range(solution.number_of_objectives):
            weight = 1.0 if idx == i else 0.000001
            max_ratio = max(max_ratio, solution.objectives[i] / weight)

        return max_ratio

    def find_extreme_points(self, solutions: List[S]):
        extreme_points = []

        for i in range(self.number_of_objectives):
            min_ASF = float('inf')
            min_solution = None

            for solution in solutions:
                asf = self.ASF(solution, i)

                if asf < min_ASF:
                    min_ASF = asf
                    min_solution = solution

            extreme_points.append(min_solution)

        return extreme_points

    def construct_hyperplane(self, solutions: List[S], extreme_points: list):
        """ Calculate the axis intersects for a set of individuals and its extremes (construct hyperplane). """
        intercepts = []
        degenerate = False

        try:
            b = [1.0] * self.number_of_objectives
            A = [s.attributes['normalized_objectives'] for s in extreme_points]
            x = np.linalg.solve(A, b)
            intercepts = [1.0 / i for i in x]
        except:
            degenerate = True

        if not degenerate:
            for i in range(self.number_of_objectives):
                if intercepts[i] < 0.001:
                    degenerate = True
                    break

        if degenerate:
            intercepts = [-float('inf')] * self.number_of_objectives

            for i in range(self.number_of_objectives):
                intercepts[i] = max([s.attributes['normalized_objectives'][i] for s in solutions]
                                    + [sys.float_info.epsilon])

        return intercepts

    def normalize_objective(self, solution: S, m: int, intercepts, ideal_point, epsilon: float = 1e-20):
        if np.abs(intercepts[m] - ideal_point[m] > epsilon):
            return solution.objectives[m] / (intercepts[m] - ideal_point[m])
        else:
            return solution.objectives[m] / epsilon

    def normalize_objectives(self, solutions: List[S], intercepts: list, ideal_point: list):
        """ Normalize objectives using the hyperplane defined by the intercepts as reference. """
        for solution in solutions:
            solution.attributes['normalized_objectives'] = \
                [self.normalize_objective(solution, i, intercepts, ideal_point) for i in
                 range(self.number_of_objectives)]

        return solutions

    def perpendicular_distance(self, direction, point):
        k = np.dot(direction, point) / np.sum(np.power(direction, 2))
        d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2))

        return np.sqrt(d)

    def associate(self, solutions: List[S], reference_points: list):
        """ Associate each solution to a reference point. """
        for solution in solutions:
            rp_dists = [(rp, self.perpendicular_distance(solution.attributes['normalized_objectives'], rp))
                        for rp in reference_points]
            best_rp, best_dist = sorted(rp_dists, key=lambda rpd: rpd[1])[0]
            solution.attributes['ref_point_distance'] = best_dist
            best_rp.associations_count += 1  # update de niche number
            best_rp.associations += [solution]

    def execute(self, solutions: List[S]) -> List[S]:
        """ Secondary environmental selection based on reference points. Corresponds to steps 13-17 of Algorithm 1.

        :param solutions: List of solutions.
        """

        # Steps 9-10 in Algorithm 1
        if len(solutions) == self.k:
            return solutions

        # Step 14 / Algorithm 2
        ideal_point = self.find_ideal_point(solutions)

        # translate points by ideal point
        for solution in solutions:
            solution.attributes['normalized_objectives'] = \
                [solution.objectives[i] - ideal_point[i] for i in range(self.number_of_objectives)]

        extreme_points = self.find_extreme_points(solutions)
        intercepts = self.construct_hyperplane(solutions, extreme_points)
        self.normalize_objectives(solutions, intercepts, ideal_point)

        # Step 15 / Algorithm 3, Step 16
        self.associate(solutions, self.reference_points)

        # Step 17 / Algorithm 4
        pop = []
        while len(pop) < self.k:
            # find niche reference point
            min_assoc_rp = min(self.reference_points, key=lambda rp: rp.associations_count)
            min_assoc_rps = [rp for rp in self.reference_points if
                             rp.associations_count == min_assoc_rp.associations_count]
            chosen_rp = min_assoc_rps[random.randint(0, len(min_assoc_rps) - 1)]

            # select cluster member
            if chosen_rp.associations:
                if chosen_rp.associations_count == 0:
                    sel = min(chosen_rp.associations, key=lambda s: s.attributes['ref_point_distance'])
                else:
                    sel = chosen_rp.associations[random.randint(0, len(chosen_rp.associations) - 1)]
                pop += [sel]

                chosen_rp.associations.remove(sel)
                chosen_rp.associations_count -= 1
            else:
                # no potential member, disregard this reference point
                self.reference_points.remove(chosen_rp)

        return pop

    def get_name(self) -> str:
        return 'Environmental selection for NSGA-III'
