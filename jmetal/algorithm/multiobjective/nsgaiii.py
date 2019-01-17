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


class ReferencePoint(list):

    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []


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
        mating_population += self.niching_selection(fronts[limit], self.population_size - len(mating_population))

        return mating_population

    def find_ideal_point(self, solutions: List[S]):
        ideal_point = [float('inf')] * self.problem.number_of_objectives

        for solution in solutions:
            for i in range(self.problem.number_of_objectives):
                ideal_point[i] = min(ideal_point[i], solution.objectives[i])

        return ideal_point

    def find_extreme_points(self, solutions: List[S], objective):
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

    def construct_hyperplane(self, solutions: List[S], extreme_points: list):
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
                intercepts[i] = max([s.attributes['normalized_objectives'][i] for s in solutions]
                                    + [sys.float_info.epsilon])

        return intercepts

    def normalize_objective(self, solution, m, intercepts, ideal_point, epsilon=1e-20):
        if np.abs(intercepts[m] - ideal_point[m] > epsilon):
            return solution.objectives[m] / (intercepts[m] - ideal_point[m])
        else:
            return solution.objectives[m] / epsilon

    def normalize_objectives(self, solutions: List[S], intercepts: list, ideal_point: list):
        for solution in solutions:
            solution.attributes['normalized_objectives'] = \
                [self.normalize_objective(solution, i, intercepts, ideal_point) for i in
                 range(self.problem.number_of_objectives)]

        return solutions

    def generate_reference_points(self, num_objs: int, num_divisions_per_obj: int = 4):
        def gen_refs_recursive(position, num_objs, left, total, element):
            if element == num_objs - 1:
                position[element] = left / total
                return ReferencePoint(copy.deepcopy(position))
            else:
                res = []
                for i in range(left):
                    position[element] = i / total
                    res.append(gen_refs_recursive(position, num_objs, left - i, total, element + 1))
                return res

        return gen_refs_recursive([0] * num_objs, num_objs, num_objs * num_divisions_per_obj,
                                  num_objs * num_divisions_per_obj, 0)

    def associate(self, solutions: List[S], reference_points: list):
        for solution in solutions:
            rp_dists = [(rp, point_line_dist(solution.attributes['normalized_objectives'], rp))
                        for rp in reference_points]
            best_rp, best_dist = sorted(rp_dists, key=lambda rpd: rpd[1])[0]
            solution.attributes['reference_point'] = best_rp
            solution.attributes['ref_point_distance'] = best_dist
            best_rp.associations_count += 1  # update de niche number
            best_rp.associations += [solution]

    def niching_selection(self, population: List[S], k: int):
        """ Secondary environmental selection based on reference points. Corresponds to steps 13-17 of Algorithm 1.

        :param population:
        :param k:
        :return:
        """

        # Steps 9-10 in Algorithm 1
        if len(population) == k:
            return population

        # Step 14 / Algorithm 2. Find the ideal point
        ideal_point = self.find_ideal_point(population)

        # translate points by ideal point
        for solution in population:
            solution.attributes['normalized_objectives'] = \
                [solution.objectives[i] - ideal_point[i] for i in range(self.problem.number_of_objectives)]

        # find the extreme points
        extreme_points = [self.find_extreme_points(population, i) for i in range(self.problem.number_of_objectives)]

        # calculate the axis intersects for a set of individuals and its extremes (construct hyperplane)
        intercepts = self.construct_hyperplane(population, extreme_points)

        # normalize objectives using the hyperplane defined by the intercepts as reference
        self.normalize_objectives(population, intercepts, ideal_point)

        # find the reference points
        reference_points = self.generate_reference_points(len(population[0].objectives))

        # Step 15 / Algorithm 3, Step 16. Associate each solution to a reference point
        self.associate(population, reference_points)

        # Step 17 / Algorithm 4
        res = []
        while len(res) < k:
            min_assoc_rp = min(reference_points, key=lambda rp: rp.associations_count)
            min_assoc_rps = [rp for rp in reference_points if rp.associations_count == min_assoc_rp.associations_count]
            chosen_rp = min_assoc_rps[random.randint(0, len(min_assoc_rps) - 1)]

            associated_inds = chosen_rp.associations

            if associated_inds:
                if chosen_rp.associations_count == 0:
                    sel = min(chosen_rp.associations, key=lambda ind: ind.attributes['ref_point_distance'])
                else:
                    sel = chosen_rp.associations[random.randint(0, len(chosen_rp.associations) - 1)]
                res += [sel]
                print(len(res), k)
                chosen_rp.associations.remove(sel)
                chosen_rp.associations_count += 1
                population.remove(sel)
            else:
                reference_points.remove(chosen_rp)

        return population

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
