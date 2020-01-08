from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, DynamicAlgorithm
from jmetal.core.problem import Problem, DynamicProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import DifferentialEvolutionCrossover, RankingAndCrowdingDistanceSelection
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.comparator import Comparator, DominanceComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = List[S]


class GDE3(EvolutionaryAlgorithm[FloatSolution, FloatSolution]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 cr: float,
                 f: float,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 k: float = 0.5,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        super(GDE3, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size)
        self.dominance_comparator = dominance_comparator
        self.selection_operator = DifferentialEvolutionSelection()
        self.crossover_operator = DifferentialEvolutionCrossover(cr, f, k)

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        mating_pool = []

        for i in range(self.population_size):
            self.selection_operator.set_index_to_exclude(i)
            selected_solutions = self.selection_operator.execute(self.solutions)
            mating_pool = mating_pool + selected_solutions

        return mating_pool

    def reproduction(self, mating_pool: List[S]) -> List[S]:
        offspring_population = []
        first_parent_index = 0

        for solution in self.solutions:
            self.crossover_operator.current_individual = solution
            parents = mating_pool[first_parent_index:first_parent_index + 3]
            first_parent_index += 3

            offspring_population.append(self.crossover_operator.execute(parents)[0])

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[FloatSolution]) -> List[List[FloatSolution]]:
        tmp_list = []

        for solution1, solution2 in zip(self.solutions, offspring_population):
            result = self.dominance_comparator.compare(solution1, solution2)
            if result == -1:
                tmp_list.append(solution1)
            elif result == 1:
                tmp_list.append(solution2)
            else:
                tmp_list.append(solution1)
                tmp_list.append(solution2)

        join_population = population + offspring_population

        return RankingAndCrowdingDistanceSelection(
            self.population_size, dominance_comparator=self.dominance_comparator
        ).execute(join_population)

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_result(self) -> List[FloatSolution]:
        return self.solutions

    def get_name(self) -> str:
        return 'GDE3'


class DynamicGDE3(GDE3, DynamicAlgorithm):

    def __init__(self,
                 problem: DynamicProblem,
                 population_size: int,
                 cr: float,
                 f: float,
                 termination_criterion: TerminationCriterion,
                 k: float = 0.5,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = DominanceComparator()):
        super(DynamicGDE3, self).__init__(
            problem, population_size, cr, f, termination_criterion, k,
            population_generator, population_evaluator, dominance_comparator)

        self.completed_iterations = 0

    def restart(self) -> None:
        pass

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.evaluations += self.offspring_population_size

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            self.observable.notify_all(**observable_data)

            self.restart()
            self.init_progress()

            self.completed_iterations += 1
