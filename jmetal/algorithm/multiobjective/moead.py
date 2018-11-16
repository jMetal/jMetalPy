import math
import random
import time
from enum import Enum
from pathlib import Path
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.evaluator import SequentialEvaluator
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem

S = TypeVar('S')
R = TypeVar(List[S])


class MOEAD(GenerationalGeneticAlgorithm[S, R]):
    class FitnessFunction(Enum):
        AGG = 'AGG'
        TCHE = 'TCHE'

    class NeighbourhoodType(Enum):
        NEIGHBOR = 'NEIGHBOR'

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 output_population_size: int,
                 neighbourhood_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 ffunction_type: FitnessFunction,
                 weights_path: str = None):
        """

        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_size: Size of the neighborhood used for mating (T in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
        than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEAD, self).__init__(
            problem=problem,
            population_size=population_size,
            mating_pool_size=population_size,
            offspring_population_size=population_size,
            max_evaluations=max_evaluations,
            mutation=mutation,
            crossover=crossover,
            selection=None,
            evaluator=SequentialEvaluator[S]())

        self.ideal_point = [math.inf] * problem.number_of_objectives  # (Z vector in Zhang & Li paper)

        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions
        self.output_population_size = output_population_size
        self.ffunction_type = ffunction_type
        self.weights_path = weights_path

        self.neighbourhood = np.zeros((self.population_size, neighbourhood_size), dtype=np.int8)
        self.neighbourhood_selection_probability = neighbourhood_selection_probability
        self.neighbourhood_size = neighbourhood_size
        self.neighbourhood_type = 'NEIGHBOR'

        # Lambda vectors (Weight vectors)
        self.lambda_ = np.zeros((self.population_size, self.problem.number_of_objectives))

    def __initialize_uniform_weight(self) -> None:
        """ Precomputed weights from

        * Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II

        Downloaded from:

        * http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar
        """
        if self.problem.number_of_objectives == 2 and self.population_size <= 300:
            for n in range(self.population_size):
                aux = 1.0 * n / (self.population_size - 1)
                self.lambda_[n][0] = aux
                self.lambda_[n][1] = 1 - aux
        else:
            file_name = 'W{}D_{}.dat'.format(self.problem.number_of_objectives, self.population_size)
            file_path = self.weights_path + '/' + file_name

            if Path(file_path).is_file():
                with open(file_path) as file:
                    for index, line in enumerate(file):
                        vector = [float(x) for x in line.split()]
                        self.lambda_[index][:] = vector
            else:
                raise FileNotFoundError('Failed to initialize weights: {} not found'.format(file_path))

    def __initialize_neighbourhood(self) -> None:
        distance = np.zeros((self.population_size, self.population_size))

        for i in range(self.population_size):
            for j in range(self.population_size):
                distance[i][j] = np.linalg.norm(self.lambda_[i] - self.lambda_[j])

            indexes = np.argsort(distance[i, :])
            self.neighbourhood[i, :] = indexes[0:self.neighbourhood_size]

    def random_permutations(self, size):
        """ Picks position from 1 to size at random and increments when value is already picked.
        """
        permutations = np.empty((size,), dtype=np.int8)

        index = np.empty((size,))
        flag = np.empty((size,))

        for i in range(size):
            index[i] = i
            flag[i] = True

        counter = 0
        while counter < size:
            start = random.randint(0, size - 1)

            while True:
                if flag[start]:
                    # Add position to order of permutation.
                    permutations[counter] = index[start]
                    flag[start] = False

                    counter += 1
                    break

                if start == size - 1:
                    start = 0
                else:
                    start += 1

        return permutations

    def parents_selection(self, subproblem_id: int):
        if random.random() < self.neighbourhood_selection_probability:
            type_ = 1
        else:
            type_ = 2

        parent_one, parent_two = self.mating_selection(subproblem_id, 2, type_)
        return [self.population[parent_one], self.population[parent_two], self.population[subproblem_id]]

    def mating_selection(self, subproblem_id: int, number_of_solutions_to_select: int, type: int):
        """ Selects 'number_of_solutions_to_select' distinct parents, either from the neighbourhood (type=1) or the
        population (type=2).
        """
        parents = list()
        neighbourhood_size = len(self.neighbourhood[subproblem_id])

        while len(parents) < number_of_solutions_to_select:
            if type == 1:
                selected_parent = self.neighbourhood[subproblem_id][random.randint(0, neighbourhood_size - 1)]
            else:
                selected_parent = random.randint(0, self.population_size - 1)

            flag = True

            # Check if parent has already been selected
            for selected_id in parents:
                if selected_id == selected_parent:
                    flag = False
                    break
            if flag:
                parents.append(selected_parent)

        return parents

    def fitness_function(self, individual: R, lambda_: np.array) -> float:
        if self.ffunction_type == self.FitnessFunction.TCHE:
            max_fun = -1.0e+30
            for i in range(self.problem.number_of_objectives):
                diff = abs(individual.objectives[i] - self.ideal_point[i])

                if lambda_[i] == 0:
                    feval = 0.0001 * diff
                else:
                    feval = diff * lambda_[i]

                if feval > max_fun:
                    max_fun = feval

            fitness = max_fun
        elif self.ffunction_type == self.FitnessFunction.AGG:
            sum = 0.0
            for n_objective in range(self.problem.number_of_objectives):
                sum += individual.objectives[n_objective] * lambda_[n_objective]
            fitness = sum
        else:
            raise NotImplementedError('Unkown MOEA/D fitness function: {}'.format(self.ffunction_type))

        return fitness

    def update_ideal_point(self, ideal_point: list, individual: S) -> list:
        """ The reference point consists of the best value of each objective over the examined solutions, which is an
        approximation of the ideal point.
        """
        for i in range(self.problem.number_of_objectives):
            ideal_point[i] = min(ideal_point[i], individual.objectives[i])

        return ideal_point

    def update_neighbourhood(self, individual: S, subproblem_id: int):
        if self.neighbourhood_type == self.NeighbourhoodType.NEIGHBOR:
            size = len(self.neighbourhood[subproblem_id])
        else:
            size = len(self.population)

        permutations = self.random_permutations(size)

        times = 0

        for i in range(size):
            if self.neighbourhood_type == self.NeighbourhoodType.NEIGHBOR:
                k = self.neighbourhood[subproblem_id][permutations[i]]
            else:
                k = permutations[i]

            f1 = self.fitness_function(self.population[k], self.lambda_[k])
            f2 = self.fitness_function(individual, self.lambda_[k])

            if f2 < f1:
                self.population[k] = individual
                times += 1

            if times >= self.max_number_of_replaced_solutions:
                return

    def run(self):
        self.start_computing_time = time.time()

        self.population = self.create_initial_population()
        self.population = self.evaluate_population(self.population)

        self.init_progress()

        self.__initialize_uniform_weight()
        self.__initialize_neighbourhood()

        for individual in self.population:
            self.ideal_point = self.update_ideal_point(self.ideal_point, individual)

        while not self.is_stopping_condition_reached():
            permutation = self.random_permutations(self.population_size)

            for i in range(self.population_size):
                subproblem_id = permutation[i]

                parents = self.parents_selection(subproblem_id)

                self.crossover_operator.current_individual = self.population[subproblem_id]
                children = self.crossover_operator.execute(parents)
                child = children[0]

                child = self.mutation_operator.execute(child)

                self.evaluator.evaluate_solution(child, problem=self.problem)

                self.ideal_point = self.update_ideal_point(self.ideal_point, child)
                self.update_neighbourhood(child, subproblem_id)

            self.update_progress()

        self.total_computing_time = self.get_current_computing_time()

    def two_objectives_case(self, population: List[S], new_population_size: int):

        def scalarizing_fitness_function(solution: S, ideal_point: list, weights: list, min_weight=0.0001):
            """Chebyshev (Tchebycheff) fitness of a solution with multiple objectives.
            This function is designed to only work with minimized objectives.
            """
            nobjs = solution.number_of_objectives
            objs = solution.objectives

            return max([max(weights[i], min_weight) * (objs[i] - ideal_point[i]) for i in range(nobjs)])

        # Compute the weight vectors
        lambda_ = np.zeros((new_population_size, 2))
        for n in range(new_population_size):
            aux = 1.0 * n / (self.population_size - 1)
            lambda_[n][0] = aux
            lambda_[n][1] = 1 - aux

        # Update ideal point
        ideal_point = [math.inf] * 2
        for individual in population:
            ideal_point = self.update_ideal_point(ideal_point, individual)

        # Select the best solution for each weight vector
        new_population = []
        for i in range(new_population_size):
            current_best = population[0]
            value = scalarizing_fitness_function(current_best, ideal_point, lambda_[i])

            for j in range(1, len(population)):
                # we are looking for the best for the weight i
                aux = scalarizing_fitness_function(population[j], ideal_point, lambda_[i])
                if aux < value:
                    value = aux
                    current_best = population[j]

            new_population.append(current_best)

        return new_population

    def get_subset_of_evenly_distributed_solutions(self, population: List[S], new_population_size: int):
        if self.problem.number_of_objectives == 2:
            new_population = self.two_objectives_case(population, new_population_size)
        else:
            new_population = population

        return new_population

    def get_result(self) -> R:
        if self.population_size > self.output_population_size:
            return self.get_subset_of_evenly_distributed_solutions(self.population, self.output_population_size)
        else:
            return self.population

    def get_name(self) -> str:
        return 'Multiobjective Evolutionary Algorithm Based on Decomposition'
