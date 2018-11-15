import math
import random
import time
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.evaluator import SequentialEvaluator
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem

S = TypeVar('S')
R = TypeVar(List[S])


# Note also that weight vectors are only computed for populations of size 2 or 3. Problems with 4 or more objectives
# will requires a weights file in the "weights" directory. Weights can be downloaded from:
# http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar


class MOEAD(GenerationalGeneticAlgorithm[S, R]):

    AGG = 'AGG'
    TCHE = 'TCHE'
    NEIGHBOR = 'NEIGHBOR'

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 neighbourhood_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 delta: float,
                 nr: int,
                 function_type: str):
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

        self.ideal_point = [math.inf, math.inf]  # (Z vector in Zhang & Li paper)

        # Neighbourhood
        self.neighbourhood_size = neighbourhood_size  # (T in Zhang & Li paper)
        self.neighbourhood = np.zeros((self.population_size, self.neighbourhood_size), dtype=np.int8)
        self.neighbourhood_selection_probability = delta  # (Delta in Zhang & Li paper)
        self.neighbourhood_type = 'NEIGHBOR'

        self.maximum_number_of_replaced_solutions = nr  # (nr in Zhang & Li paper)

        # Lambda vectors (Weight vectors)
        self.lambda_ = np.zeros((self.population_size, 2))

        self.function_type = function_type

    def __initialize_uniform_weight(self):
        """ Precomputed weights from

        * Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II

        Downloaded from:

        * http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar
        """
        if self.problem.number_of_objectives == 2:
            for n in range(self.population_size):
                aux = 1.0 * n / (self.population_size - 1)
                self.lambda_[n][0] = aux
                self.lambda_[n][1] = 1 - aux
        elif self.problem.number_of_objectives == 3:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if i + j <= self.population_size:
                        k = self.population_size - i - j
                        try:
                            weight_scalars = [] * 3
                            weight_scalars[0] = i / self.population_size
                            weight_scalars[1] = j / self.population_size
                            weight_scalars[2] = k / self.population_size

                            self.lambda_.append(weight_scalars)
                        except Exception as e:
                            raise e

            # Trim number of weights to fit population size
            self.lambda_ = sorted((x for x in self.lambda_), key=lambda x: sum(x), reverse=True)
            self.lambda_ = self.lambda_[:self.population_size]
        else:
            pass

    def __initialize_neighbourhood(self) -> None:
        distance = np.zeros((self.population_size, self.population_size))

        for i in range(self.population_size):
            for j in range(self.population_size):
                distance[i][j] = np.linalg.norm(self.lambda_[i, :] - self.lambda_[j, :])

            indexes = np.argsort(distance[i, :])
            self.neighbourhood[i, :] = indexes[0:self.neighbourhood_size]

    def __initialize_ideal_point(self) -> None:
        """ The reference point consists of the best value of each objective over the examined solutions, which is an
        approximation of the ideal point.
        """
        parent_fitness = [p.objectives for p in self.population]
        self.ideal_point = [min(min(parent_fitness[i]), self.ideal_point[i]) for i in range(2)]

    def random_permutations(self, size):
        """ Picks position from 1 to size at random and increments when value is already picked
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
                else:
                    # Try next position
                    if start == size - 1:
                        start = 0
                    else:
                        start += 1

        return permutations

    def parents_selection(self, subproblem_id: int):
        parent_one, parent_two = self.mating_selection(subproblem_id, 2)
        return [self.population[parent_one], self.population[parent_two], self.population[subproblem_id]]

    def mating_selection(self, subproblem_id: int, number_of_solutions_to_select: int):
        """ Selects 'size' distinct parents, either from the neighbourhood (type=1) or the population (type=2).
        """
        if random.random() < self.neighbourhood_selection_probability:
            type_ = 1
        else:
            type_ = 2

        parents = list()
        ss = len(self.neighbourhood[subproblem_id])

        while len(parents) < number_of_solutions_to_select:
            if type_ == 1:
                parent = self.neighbourhood[subproblem_id][random.randint(0, ss - 1)]
            else:
                parent = random.randint(0, self.population_size - 1)

            flag = True

            # Check if p has been already selected
            for i in range(len(parents)):
                if parents[i] == parent:
                    flag = False
                    break
            if flag:
                parents.append(parent)

        return parents

    def fitness_function(self, individual: R, lambda_) -> float:
        if self.function_type == self.TCHE:
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
        elif self.function_type == self.AGG:
            sum = 0.0
            for n_objective in range(self.problem.number_of_objectives):
                sum += individual.objectives[n_objective] * lambda_[n_objective]
            fitness = sum
        else:
            raise NotImplementedError

        return fitness

    def update_neighbourhood(self, individual: S, subproblem_id: int):
        times = 0

        if self.neighbourhood_type == self.NEIGHBOR:
            size = len(self.neighbourhood[subproblem_id])
        else:
            size = len(self.population)

        permutations = self.random_permutations(size)

        for i in range(size):
            if self.neighbourhood_type == self.NEIGHBOR:
                k = self.neighbourhood[subproblem_id][permutations[i]]
            else:
                k = permutations[i]

            f1 = self.fitness_function(self.population[k], self.lambda_[k])
            f2 = self.fitness_function(individual, self.lambda_[k])

            if f2 < f1:
                self.population[k] = individual
                times += 1

            if times >= self.maximum_number_of_replaced_solutions:
                return

    def run(self):
        self.start_computing_time = time.time()

        self.population = self.create_initial_population()
        self.population = self.evaluate_population(self.population)

        self.init_progress()

        self.__initialize_uniform_weight()
        self.__initialize_neighbourhood()
        self.__initialize_ideal_point()

        while not self.is_stopping_condition_reached():
            permutation = self.random_permutations(self.population_size)

            for i in range(self.population_size):
                subproblem_id = permutation[i]

                parents = self.parents_selection(subproblem_id)

                #self.crossover_operator.set_current_solution(subproblem_id)
                offspring_population = self.crossover_operator.execute(parents)
                children = self.mutation_operator.execute(offspring_population[0])
                self.evaluator.evaluate_solution(children, problem=self.problem)

                self.ideal_point = children.objectives
                self.update_neighbourhood(children, subproblem_id)

            self.update_progress()

        self.total_computing_time = self.get_current_computing_time()

    def get_result(self) -> R:
        return self.population

    def get_name(self) -> str:
        return 'Multiobjective Evolutionary Algorithm Based on Decomposition'
