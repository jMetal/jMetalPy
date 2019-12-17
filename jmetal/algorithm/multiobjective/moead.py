import copy
import random
from math import ceil
from typing import TypeVar, List, Generator

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.operator import DifferentialEvolutionCrossover, NaryRandomSolutionSelection
from jmetal.util.aggregative_function import AggregativeFunction
from jmetal.util.constraint_handling import feasibility_ratio, \
    overall_constraint_violation_degree, is_feasible
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations

S = TypeVar('S')
R = List[S]


class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: DifferentialEvolutionCrossover,
                 aggregative_function: AggregativeFunction,
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 neighbor_size: int,
                 weight_files_path: str,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
               than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEAD, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=crossover,
            selection=NaryRandomSolutionSelection(2),
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion
        )
        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions
        self.fitness_function = aggregative_function
        self.neighbourhood = WeightVectorNeighborhood(
            number_of_weight_vectors=population_size,
            neighborhood_size=neighbor_size,
            weight_vector_size=problem.number_of_objectives,
            weights_path=weight_files_path
        )
        self.neighbourhood_selection_probability = neighbourhood_selection_probability
        self.permutation = None
        self.current_subproblem = 0
        self.neighbor_type = None

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)

        self.permutation = Permutation(self.population_size)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def selection(self, population: List[S]):
        self.current_subproblem = self.permutation.get_next_value()
        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            mating_population = self.selection_operator.execute(population)

        mating_population.append(population[self.current_subproblem])

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)
        self.mutation_operator.execute(offspring_population[0])

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        new_solution = offspring_population[0]

        self.fitness_function.update(new_solution.objectives)

        new_population = self.update_current_subproblem_neighborhood(new_solution, population)

        return new_population

    def update_current_subproblem_neighborhood(self, new_solution, population):
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(self.current_subproblem)
        replacements = 0

        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]

            f1 = self.fitness_function.compute(population[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

            if f2 < f1:
                population[k] = copy.deepcopy(new_solution)
                replacements += 1

            if replacements >= self.max_number_of_replaced_solutions:
                break

        return population

    def generate_permutation_of_neighbors(self, subproblem_id):
        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighborhood()[subproblem_id]
            permuted_array = copy.deepcopy(neighbors.tolist())
        else:
            permuted_array = Permutation(self.population_size).get_permutation()

        return permuted_array

    def choose_neighbor_type(self):
        rnd = random.random()

        if rnd < self.neighbourhood_selection_probability:
            neighbor_type = 'NEIGHBOR'
        else:
            neighbor_type = 'POPULATION'

        return neighbor_type

    def get_name(self):
        return 'MOEAD'

    def get_result(self):
        return self.solutions


class MOEAD_DRA(MOEAD):
    def __init__(self, problem, population_size, mutation, crossover, aggregative_function,
                 neighbourhood_selection_probability, max_number_of_replaced_solutions, neighbor_size,
                 weight_files_path, termination_criterion=store.default_termination_criteria,
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator):
        super(MOEAD_DRA, self).__init__(problem, population_size, mutation, crossover, aggregative_function,
                                        neighbourhood_selection_probability, max_number_of_replaced_solutions,
                                        neighbor_size, weight_files_path,
                                        termination_criterion=termination_criterion,
                                        population_generator=population_generator,
                                        population_evaluator=population_evaluator)

        self.saved_values = []
        self.utility = [1.0 for _ in range(population_size)]
        self.frequency = [0.0 for _ in range(population_size)]
        self.generation_counter = 0
        self.order = []
        self.current_order_index = 0

    def init_progress(self):
        super().init_progress()
        self.saved_values = [copy.copy(solution) for solution in self.solutions]

        self.evaluations = self.population_size
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)

        self.order = self.__tour_selection(10)
        self.current_order_index = 0

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self):
        super().update_progress()

        self.current_order_index += 1
        if self.current_order_index == (len(self.order)):
            self.order = self.__tour_selection(10)
            self.current_order_index = 0

        self.generation_counter += 1
        if self.generation_counter % 30 == 0:
            self.__utility_function()

    def selection(self, population: List[S]):
        self.current_subproblem = self.order[self.current_order_index]
        self.current_order_index += 1
        self.frequency[self.current_subproblem] += 1

        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == 'NEIGHBOR':
            neighbors = self.neighbourhood.get_neighbors(self.current_subproblem, population)
            mating_population = self.selection_operator.execute(neighbors)
        else:
            mating_population = self.selection_operator.execute(population)

        mating_population.append(population[self.current_subproblem])

        return mating_population

    def get_name(self):
        return 'MOEAD-DRA'

    def __utility_function(self):
        for i in range(len(self.solutions)):
            f1 = self.fitness_function.compute(self.solutions[i].objectives, self.neighbourhood.weight_vectors[i])
            f2 = self.fitness_function.compute(self.saved_values[i].objectives, self.neighbourhood.weight_vectors[i])
            delta = f2 - f1
            if delta > 0.001:
                self.utility[i] = 1.0
            else:
                utility_value = (0.95 + (0.05 * delta / 0.001)) * self.utility[i]
                self.utility[i] = utility_value if utility_value < 1.0 else 1.0

            self.saved_values[i] = copy.copy(self.solutions[i])

    def __tour_selection(self, depth):
        selected = [i for i in range(self.problem.number_of_objectives)]
        candidate = [i for i in range(self.problem.number_of_objectives, self.population_size)]

        while len(selected) < int(self.population_size / 5.0):
            best_idd = int(random.random() * len(candidate))
            best_sub = candidate[best_idd]
            for i in range(1, depth):
                i2 = int(random.random() * len(candidate))
                s2 = candidate[i2]
                if self.utility[s2] > self.utility[best_sub]:
                    best_idd = i2
                    best_sub = s2
            selected.append(best_sub)
            del candidate[best_idd]

        return selected


class MOEADIEpsilon(MOEAD):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: DifferentialEvolutionCrossover,
                 aggregative_function: AggregativeFunction,
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 neighbor_size: int,
                 weight_files_path: str,
                 termination_criterion: TerminationCriterion = StoppingByEvaluations(300000),
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather
               than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEADIEpsilon, self).__init__(
            problem=problem,
            population_size=population_size,
            mutation=mutation,
            crossover=crossover,
            aggregative_function=aggregative_function,
            neighbourhood_selection_probability=neighbourhood_selection_probability,
            max_number_of_replaced_solutions=max_number_of_replaced_solutions,
            neighbor_size=neighbor_size,
            weight_files_path=weight_files_path,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion
        )
        self.constraints = []
        self.epsilon_k = 0
        self.phi_max = -1e30
        self.epsilon_zero = 0
        self.tc = 800
        self.tao = 0.05
        self.rk = 0
        self.generation_counter = 0
        self.archive = []

    def init_progress(self) -> None:
        super().init_progress()

        # for i in range(self.population_size):
        #    self.constraints[i] = get_overall_constraint_violation_degree(self.permutation[i])
        self.constraints = [overall_constraint_violation_degree(self.solutions[i])
                            for i in range(0, self.population_size)]

        sorted(self.constraints)
        self.epsilon_zero = abs(self.constraints[int(ceil(0.05 * self.population_size))])

        if self.phi_max < abs(self.constraints[0]):
            self.phi_max = abs(self.constraints[0])

        self.rk = feasibility_ratio(self.solutions)
        self.epsilon_k = self.epsilon_zero

    def update_progress(self) -> None:
        super().update_progress()

        if self.evaluations % self.population_size == 0:
            self.update_external_archive()
            self.generation_counter += 1
            self.rk = feasibility_ratio(self.solutions)
            if self.generation_counter >= self.tc:
                self.epsilon_k = 0
            else:
                if self.rk < 0.95:
                    self.epsilon_k = (1 - self.tao) * self.epsilon_k
                else:
                    self.epsilon_k = self.phi_max * (1 + self.tao)

    def update_current_subproblem_neighborhood(self, new_solution, population):
        if self.phi_max < overall_constraint_violation_degree(new_solution):
            self.phi_max = overall_constraint_violation_degree(new_solution)

        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(self.current_subproblem)
        replacements = 0

        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]

            f1 = self.fitness_function.compute(population[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(new_solution.objectives, self.neighbourhood.weight_vectors[k])

            cons1 = abs(overall_constraint_violation_degree(self.solutions[k]))
            cons2 = abs(overall_constraint_violation_degree(new_solution))

            if cons1 < self.epsilon_k and cons2 <= self.epsilon_k:
                if f2 < f1:
                    population[k] = copy.deepcopy(new_solution)
                    replacements += 1
            elif cons1 == cons2:
                if f2 < f1:
                    population[k] = copy.deepcopy(new_solution)
                    replacements += 1
            elif cons2 < cons1:
                population[k] = copy.deepcopy(new_solution)
                replacements += 1

            if replacements >= self.max_number_of_replaced_solutions:
                break

        return population

    def update_external_archive(self):
        feasible_solutions = []
        for solution in self.solutions:
            if is_feasible(solution):
                feasible_solutions.append(copy.deepcopy(solution))

        if len(feasible_solutions) > 0:
            feasible_solutions = feasible_solutions + self.archive
            ranking = FastNonDominatedRanking()
            ranking.compute_ranking(feasible_solutions)

            first_rank_solutions = ranking.get_subfront(0)
            if len(first_rank_solutions) <= self.population_size:
                self.archive = []
                for solution in first_rank_solutions:
                    self.archive.append(copy.deepcopy(solution))
            else:
                crowding_distance = CrowdingDistance()
                while len(first_rank_solutions) > self.population_size:
                    crowding_distance.compute_density_estimator(first_rank_solutions)
                    first_rank_solutions = sorted(first_rank_solutions, key=lambda x: x.attributes['crowding_distance'],
                                                  reverse=True)
                    first_rank_solutions.pop()

                self.archive = []
                for solution in first_rank_solutions:
                    self.archive.append(copy.deepcopy(solution))

    def get_result(self):
        return self.archive


class Permutation:

    def __init__(self, length: int):
        self.counter = 0
        self.length = length
        self.permutation = np.random.permutation(length)

    def get_next_value(self):
        next_value = self.permutation[self.counter]
        self.counter += 1

        if self.counter == self.length:
            self.permutation = np.random.permutation(self.length)
            self.counter = 0

        return next_value

    def get_permutation(self):
        return self.permutation.tolist()
