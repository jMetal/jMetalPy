import random
import copy
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.operator import DifferentialEvolutionCrossover, PolynomialMutation, NaryRandomSolutionSelection
from jmetal.util.aggregative_function import AggregativeFunction, Tschebycheff
from jmetal.util.neighborhood import WeightNeighborhood, WeightVectorNeighborhood
from jmetal.util.solution_list import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations

S = TypeVar('S')
R = List[S]

class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: DifferentialEvolutionCrossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
                 aggregative_function: AggregativeFunction=Tschebycheff(2),
                 neighbourhood_selection_probability: float=0.9,
                 max_number_of_replaced_solutions: int=2,
                 neighbor_size: int=20,
                 termination_criterion: TerminationCriterion = StoppingByEvaluations(150000),
                 weight_files_path: str = "../../resources/MOEAD_weights",
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
            mutation=PolynomialMutation(1.0/problem.number_of_variables, 20.0),
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
        self.permutation = Permutation(population_size)
        self.current_subproblem: int = 0
        self.neighbor_type: str = ""
        self.neighbor_size = neighbor_size

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        for solution in self.solutions:
            self.fitness_function.update(solution.objectives)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def selection(self, population: List[S]):
        self.current_subproblem = self.permutation.get_next_value()
        self.neighbor_type = self.choose_neighbor_type()

        if self.neighbor_type == "NEIGHBOR":
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
        size = self.source_of_neighbors_size(subproblem_id)
        if self.neighbor_type == "NEIGHBOR":
            neighbors = self.neighbourhood.get_neighborhood()[subproblem_id]
            permuted_array = copy.deepcopy(neighbors.tolist())
        else:
            permuted_array = Permutation(size).get_permutation()
        return permuted_array

    def source_of_neighbors_size(self, subproblem_id):
        if self.neighbor_type == "NEIGHBOR":
            #ns = self.neighbourhood.get_neighbors(subproblem_id, self.p)
            return self.neighbor_size
        else:
            return self.population_size

    def choose_neighbor_type(self):
        rnd = random.random()

        if rnd < self.neighbourhood_selection_probability:
            neighbor_type = "NEIGHBOR"
        else:
            neighbor_type = "POPULATION"

        return neighbor_type

    def get_name(self):
        return "MOEAD"

    def get_result(self):
        return self.solutions

################
class Permutation():
    def __init__(self, length: int):
        self.length = length
        self.permutation = np.random.permutation(length)
        self.counter = 0

    def get_next_value(self):
        next = self.permutation[self.counter]
        self.counter += 1

        if self.counter == self.length:
            self.permutation = np.random.permutation(self.length)
            self.counter = 0

        return next

    def get_permutation(self):
        return self.permutation.tolist()
