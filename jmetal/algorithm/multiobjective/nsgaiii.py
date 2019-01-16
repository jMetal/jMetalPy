import copy
import math
import operator
import random
import sys
from functools import reduce
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.util.comparator import DominanceComparator, Comparator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution_list import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: NSGA-III
   :platform: Unix, Windows
   :synopsis: NSGA-III (Non-dominance Sorting Genetic Algorithm III) implementation.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NSGAIII(NSGAII):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 termination_criterion: TerminationCriterion,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = DominanceComparator()):
        """  NSGA-III implementation.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(NSGAIII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=None,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator
        self.ideal_point = [float('inf')] * problem.number_of_objectives

    def selection(self, population: List[S]):
        """ Implements NSGA-III selection as described in

        * Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
          Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
          Part I: Solving Problems With Box Constraints. IEEE Transactions on
          Evolutionary Computation, 18(4), 577–601. doi:10.1109/TEVC.2013.2281535.
        """

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(population)

        fronts = ranking.ranked_sublists

        limit = 0
        res = []
        for f, front in enumerate(fronts):
            res += front
            if len(res) > self.population_size:
                limit = f
                break

        mating_population = []
        if limit > 0:
            for f in range(limit):
                mating_population += fronts[f]

        # complete selected individuals using the reference point based approach
        mating_population += self.__niching_select(fronts[limit], self.population_size - len(mating_population))

        return mating_population

    def __niching_select(self, population: List[S], k: int):
        """ Secondary environmental selection based on reference points. Corresponds to steps 13-17 of Algorithm 1.

        :param population:
        :param k:
        :return:
        """

        # find the reference points
        reference_points = self.__generate_reference_points(len(population[0].objectives))

        # Steps 9-10 in Algorithm 1
        if len(population) == k:
            return population

        # Step 14 / Algorithm 2. Find the ideal point
        for solution in population:
            for i in range(self.problem.number_of_objectives):
                self.ideal_point[i] = min(self.ideal_point[i], solution.objectives[i])

        # translate points by ideal point to normalize objectives
        for solution in population:
            solution.attributes['normalized_objectives'] = \
                [solution.objectives[i] - self.ideal_point[i] for i in range(self.problem.number_of_objectives)]

        # find the extreme points
        extreme_points = [self.__find_extreme_points(population, i) for i in range(self.problem.number_of_objectives)]

        # calculate the axis intersects for a set of individuals and its extremes (construct hyperplane)
        intercepts = []
        degenerate = False

        try:
            b = [1.0] * self.problem.number_of_objectives
            A = [s.attributes['normalized_objectives'] for s in extreme_points]
            x = np.linalg.solve(A, b)
            intercepts = [1.0 / i for i in x]
        except:
            degenerate = True

        if not degenerate:
            for i in range(self.problem.number_of_objectives):
                if intercepts[i] < 0.001:
                    degenerate = True
                    break

        if degenerate:
            intercepts = [-float('inf')] * self.problem.number_of_objectives

            for i in range(self.problem.number_of_objectives):
                intercepts[i] = max([s.normalized_objectives[i] for s in population] + [sys.float_info.epsilon])

        # normalize objectives using the hyperplane defined by the intercepts as reference
        for solution in population:
            solution.attributes['normalized_objectives'] = \
                [solution.attributes['normalized_objectives'][i] / intercepts[i] for i in
                 range(self.problem.number_of_objectives)]

        # Step 15 / Algorithm 3, Step 16. Associate each solution to a reference point
        members, potential_members = self.__associate_to_reference_point(population, reference_points)

        # Step 17 / Algorithm 4
        excluded = set()

        while len(population) < k:
            # identify reference point with the fewest associated members
            min_indices = []
            min_count = sys.maxsize

            for i in range(len(members)):
                if i not in excluded and len(members[i]) <= min_count:
                    if len(members[i]) < min_count:
                        min_indices = []
                        min_count = len(members[i])
                    min_indices.append(i)

            # pick one randomly if there are multiple options
            min_index = random.choice(min_indices)

            # add associated solution
            if min_count == 0:
                if len(potential_members[min_index]) == 0:
                    excluded.add(min_index)
                else:
                    min_solution = self.__find_minimum_distance(potential_members[min_index],
                                                                reference_points[min_index])
                    population.append(min_solution)
                    members[min_index].append(min_solution)
                    potential_members[min_index].remove(min_solution)
            else:
                if len(potential_members[min_index]) == 0:
                    excluded.add(min_index)
                else:
                    rand_solution = random.choice(potential_members[min_index])
                    population.append(rand_solution)
                    members[min_index].append(rand_solution)
                    potential_members[min_index].remove(rand_solution)

        return population

    def __generate_reference_points(self, num_objs: int, num_divisions_per_obj: int = 4):
        """ Generates reference points for NSGA-III selection. This code is based on
        `jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>`_.
        """

        def gen_refs_recursive(position, num_objs, left, total, element):
            if element == num_objs - 1:
                position[element] = left / total
                return copy.deepcopy(position)
            else:
                res = []
                for i in range(left):
                    position[element] = i / total
                    res.append(gen_refs_recursive(position, num_objs, left - i, total, element + 1))
                return res

        return gen_refs_recursive([0] * num_objs, num_objs, num_objs * num_divisions_per_obj,
                                  num_objs * num_divisions_per_obj, 0)

    def __find_extreme_points(self, solutions: List[S], objective):
        nobjs = self.problem.number_of_objectives

        weights = [0.000001] * nobjs
        weights[objective] = 1.0

        min_index = -1
        min_value = float('inf')

        for i in range(len(solutions)):
            objectives = solutions[i].attributes['normalized_objectives']
            value = max([objectives[j] / weights[j] for j in range(nobjs)])

            if value < min_value:
                min_index = i
                min_value = value

        return solutions[min_index]

    def __associate_to_reference_point(self, solutions: List[S], reference_points):
        """ Associates individuals to reference points and calculates niche number. """
        member = [[] for _ in range(len(reference_points))]
        potential_member = [[] for _ in range(len(reference_points))]

        for t, solution in enumerate(solutions):
            min_index = -1
            min_distance = float('inf')

            for i in range(len(reference_points)):
                distance = point_line_dist(solution.attributes['normalized_objectives'], reference_points[i])

                if distance < min_distance:
                    min_index = i
                    min_distance = distance

            if t+1 != len(solutions):
                member[min_index].append(solution)
            else:
                potential_member[min_index].append(solution)

        return member, potential_member

    def __find_minimum_distance(self, solutions: List[S], reference_point):
        min_index = -1
        min_distance = float('inf')

        for i in range(len(solutions)):
            solution = solutions[i]
            distance = point_line_dist(solution.attributes['normalized_objectives'], reference_point)

            if distance < min_distance:
                min_index = i
                min_distance = distance

        return solutions[min_index]

    def get_name(self) -> str:
        return 'NSGAIII'


def magnitude(x):
    return math.sqrt(dot(x, x))


def subtract(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def multiply(s, x):
    return [s * x[i] for i in range(len(x))]


def dot(x, y):
    return reduce(operator.add, [x[i] * y[i] for i in range(len(x))], 0)


def point_line_dist(point, line):
    return magnitude(subtract(multiply(float(dot(line, point)) / float(dot(line, line)), line), point))
