import threading
import time
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List

from jmetal.config import store
from jmetal.core.generator import Generator
from jmetal.core.problem import Problem
from jmetal.core.evaluator import Evaluator
from jmetal.core.solution import FloatSolution

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :platform: Unix, Windows
   :synopsis: Templates for algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Algorithm(Generic[S, R], threading.Thread):

    __metaclass__ = ABCMeta

    def __init__(self, problem: Problem[S], generator: Generator[R], evaluator: Evaluator[S], max_evaluations: int):
        """ :param problem: The problem to solve.
        :param generator: Generator of solutions.
        :param evaluator: Evaluator of solutions.
        :param max_evaluations: Maximum number of evaluations/iterations.
        """
        threading.Thread.__init__(self)
        self.problem = problem
        self.generator = generator
        self.evaluator = evaluator
        self.max_evaluations = max_evaluations

        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.observable = store['default_observable']

        if not self.generator:
            self.generator = store['default_generator']
        if not self.evaluator:
            self.evaluator = store['default_evaluator']

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

    @property
    def current_computing_time(self) -> float:
        return time.time() - self.start_computing_time

    @property
    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def evaluate_all(self, solutions: List[S]) -> List[S]:
        """ Evaluate the individual fitness of new individuals. """
        return self.evaluator.evaluate(solutions, self.problem)

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()
        self.init_progress()

        while not self.is_stopping_condition_reached:
            self.step()
            self.update_progress()

        self.total_computing_time = self.current_computing_time

    def get_name(self) -> str:
        return self.__class__.__name__


class EvolutionaryAlgorithm(Algorithm[S, R]):

    __metaclass__ = ABCMeta

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 population_generator: Generator[R],
                 max_evaluations: int,
                 evaluator: Evaluator[S]):
        super(EvolutionaryAlgorithm, self).__init__(
            problem=problem,
            generator=population_generator,
            max_evaluations=max_evaluations,
            evaluator=evaluator
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
        self.evaluations = self.population_size

        self.population = [self.generator.new(self.problem) for _ in range(self.population_size)]
        self.population = self.evaluate_all(self.population)

    def step(self) -> None:
        mating_population = self.selection(self.population)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate_all(offspring_population)
        self.population = self.replacement(self.population, offspring_population)

    def update_progress(self) -> None:
        self.evaluations += self.population_size

        observable_data = {
            'problem': self.problem,
            'population': self.population,
            'evaluations': self.evaluations,
            'computing time': self.current_computing_time,
        }

        self.observable.notify_all(**observable_data)


class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]]):

    __metaclass__ = ABCMeta

    def __init__(self,
                 problem: Problem,
                 swarm_size: int,
                 swarm_generator: Generator[FloatSolution],
                 max_evaluations: int,
                 evaluator: Evaluator[FloatSolution]):
        super(ParticleSwarmOptimization, self).__init__(
            problem=problem,
            generator=swarm_generator,
            max_evaluations=max_evaluations,
            evaluator=evaluator
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

        self.swarm = [self.generator.new(self.problem) for _ in range(self.swarm_size)]
        self.swarm = self.evaluate_all(self.swarm)

        self.initialize_velocity(self.swarm)
        self.initialize_particle_best(self.swarm)
        self.initialize_global_best(self.swarm)

    def step(self):
        self.update_velocity(self.swarm)
        self.update_position(self.swarm)
        self.perturbation(self.swarm)
        self.swarm = self.evaluate_all(self.swarm)
        self.update_global_best(self.swarm)
        self.update_particle_best(self.swarm)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size

        observable_data = {
            'problem': self.problem,
            'population': self.swarm,
            'evaluations': self.evaluations,
            'computing time': self.current_computing_time,
        }

        self.observable.notify_all(**observable_data)
