from typing import List, TypeVar

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import EpsilonIndicator
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")


class IBEA(GeneticAlgorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        kappa: float,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
    ):
        """Epsilon IBEA implementation as described in

        * Zitzler, Eckart, and Simon Künzli. "Indicator-based selection in multiobjective search."
        In International Conference on Parallel Problem Solving from Nature, pp. 832-842. Springer,
        Berlin, Heidelberg, 2004.

        https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84

        IBEA is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The multi-objective search in IBEA is guided by a fitness associated to every solution,
        which is in turn controlled by a binary quality indicator. This implementation uses the so-called
        additive epsilon indicator, along with a binary tournament mating selector.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param kappa: Weight in the fitness computation.
        """

        selection = BinaryTournamentSelection(
            comparator=SolutionAttributeComparator(key="fitness", lowest_is_best=False)
        )
        self.kappa = kappa

        super(IBEA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
        )

    def compute_fitness_values(self, population: List[S], kappa: float) -> List[S]:
        for i in range(len(population)):
            population[i].attributes["fitness"] = 0

            for j in range(len(population)):
                if j != i:
                    # I_ε(j, i): how much solution j "dominates" solution i
                    # EpsilonIndicator(reference).compute(front) computes I_ε(front, reference)
                    # So to get I_ε(j, i) we set i as reference and j as front
                    ref = np.array([population[i].objectives])
                    sol = np.array([population[j].objectives])
                    population[i].attributes["fitness"] += -np.exp(
                        -EpsilonIndicator(ref).compute(sol) / self.kappa
                    )
        return population

    def evaluate(self, population: List[S]):
        evaluated = super().evaluate(population)
        return self.compute_fitness_values(evaluated, self.kappa)

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        join_population = population + offspring_population
        join_population_size = len(join_population)
        join_population = self.compute_fitness_values(join_population, self.kappa)

        while join_population_size > self.population_size:
            current_fitnesses = [individual.attributes["fitness"] for individual in join_population]
            index_worst = current_fitnesses.index(min(current_fitnesses))

            for i in range(join_population_size):
                # When removing worst, subtract its contribution I_ε(worst, i) from each fitness
                ref = np.array([join_population[i].objectives])
                worst = np.array([join_population[index_worst].objectives])
                join_population[i].attributes["fitness"] += np.exp(
                    -EpsilonIndicator(ref).compute(worst) / self.kappa
                )

            join_population.pop(index_worst)
            join_population_size = join_population_size - 1

        return join_population

    def result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "Epsilon-IBEA"
