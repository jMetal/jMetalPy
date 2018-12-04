import logging
import threading
import time
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List

from jmetal.component.evaluator import Evaluator
from jmetal.component.generator import Generator
from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criteria import TerminationCriteria

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :platform: Unix, Windows
   :synopsis: Templates for algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Algorithm(Generic[S, R], threading.Thread):

    __metaclass__ = ABCMeta

    def __init__(self,
                 problem: Problem[S],
                 pop_generator: Generator[R],
                 pop_evaluator: Evaluator[S],
                 termination_criteria: TerminationCriteria):
        """ :param problem: The problem to solve.
        :param pop_generator: Generator of solutions.
        :param pop_evaluator: Evaluator of solutions.
        :param max_evaluations: Maximum number of evaluations/iterations.
        """
        threading.Thread.__init__(self)
        self.problem = problem
        self.pop_generator = pop_generator
        self.pop_evaluator = pop_evaluator
        self.termination_criteria = termination_criteria

        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        if not self.pop_generator:
            self.pop_generator = store.default_generator
        if not self.pop_evaluator:
            self.pop_evaluator = store.default_evaluator
        if not self.termination_criteria:
            self.termination_criteria = store.default_termination_criteria

        self.observable = store.default_observable
        self.observable.register(self.termination_criteria)

    @abstractmethod
    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        pass

    @abstractmethod
    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        pass

    def evaluate(self, solutions: List[S]) -> List[S]:
        """ Evaluate the individual fitness of new individuals. """
        return self.pop_evaluator.evaluate(solutions, self.problem)

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        LOGGER.debug('Initializing progress')
        self.init_progress()

        try:
            LOGGER.debug('Running main loop until termination criteria is met')
            while not self.termination_criteria.is_met:
                self.step()
                self.update_progress()
        except KeyboardInterrupt:
            LOGGER.warning('Interrupted by keyboard')

        self.total_computing_time = time.time() - self.start_computing_time

    def get_observable_data(self) -> dict:
        """ Get observable data, with the information that will be send to all observers each time. """
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': [],
            'COMPUTING_TIME': time.time() - self.start_computing_time,
        }

    @abstractmethod
    def get_result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class EvolutionaryAlgorithm(Algorithm[S, R]):

    __metaclass__ = ABCMeta

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 pop_generator: Generator[R],
                 pop_evaluator: Evaluator[S],
                 termination_criteria: TerminationCriteria):
        super(EvolutionaryAlgorithm, self).__init__(
            problem=problem,
            pop_generator=pop_generator,
            pop_evaluator=pop_evaluator,
            termination_criteria=termination_criteria
        )
        self.population = []
        self.population_size = population_size

    @abstractmethod
    def selection(self, population: List[S]) -> List[S]:
        """ Select the best-fit individuals for reproduction (parents). """
        pass

    @abstractmethod
    def reproduction(self, population: List[S]) -> List[S]:
        """ Breed new individuals through crossover and mutation operations to give birth to offspring. """
        pass

    @abstractmethod
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ Replace least-fit population with new individuals. """
        pass

    def init_progress(self) -> None:
        self.population = [self.pop_generator.new(self.problem) for _ in range(self.population_size)]
        self.population = self.evaluate(self.population)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self) -> None:
        mating_population = self.selection(self.population)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)
        self.population = self.replacement(self.population, offspring_population)

    def update_progress(self) -> None:
        self.evaluations += self.population_size

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.population
        self.observable.notify_all(**observable_data)

    @abstractmethod
    def get_result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]]):

    __metaclass__ = ABCMeta

    def __init__(self,
                 problem: Problem,
                 swarm_size: int,
                 swarm_generator: Generator[FloatSolution],
                 swarm_evaluator: Evaluator[FloatSolution],
                 termination_criteria: TerminationCriteria):
        super(ParticleSwarmOptimization, self).__init__(
            problem=problem,
            pop_generator=swarm_generator,
            pop_evaluator=swarm_evaluator,
            termination_criteria=termination_criteria
        )
        self.swarm = []
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        self.swarm = [self.pop_generator.new(self.problem) for _ in range(self.swarm_size)]
        self.swarm = self.evaluate(self.swarm)

        self.initialize_velocity(self.swarm)
        self.initialize_particle_best(self.swarm)
        self.initialize_global_best(self.swarm)

    def step(self):
        self.update_velocity(self.swarm)
        self.update_position(self.swarm)
        self.perturbation(self.swarm)
        self.swarm = self.evaluate(self.swarm)
        self.update_global_best(self.swarm)
        self.update_particle_best(self.swarm)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.swarm
        self.observable.notify_all(**observable_data)

    @abstractmethod
    def get_result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
