import threading
import time
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List

from jmetal.core.solution import FloatSolution
from jmetal.core.observable import DefaultObservable

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

    def __init__(self):
        threading.Thread.__init__(self)
        self.evaluations = 0
        self.max_evaluations = 100
        self.start_computing_time = 0
        self.total_computing_time = 0
        self.observable = DefaultObservable()

    def get_evaluations(self) -> int:
        return self.evaluations

    def get_current_computing_time(self) -> float:
        return time.time() - self.start_computing_time

    @abstractmethod
    def get_result(self) -> R:
        """ :return: Final population.
        """
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class EvolutionaryAlgorithm(Algorithm[S, R]):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(EvolutionaryAlgorithm, self).__init__()
        self.population = []

    @abstractmethod
    def create_initial_population(self) -> List[S]:
        pass

    @abstractmethod
    def evaluate_population(self, population: List[S]) -> List[S]:
        pass

    @abstractmethod
    def init_progress(self) -> None:
        pass

    @abstractmethod
    def is_stopping_condition_reached(self) -> bool:
        pass

    @abstractmethod
    def selection(self, population: List[S]) -> List[S]:
        pass

    @abstractmethod
    def reproduction(self, population: List[S]) -> List[S]:
        pass

    @abstractmethod
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        pass

    @abstractmethod
    def update_progress(self):
        pass

    def run(self):
        """* Step One: Generate the initial population of individuals randomly. (First generation)
        * Step Two: Evaluate the fitness of each individual in that population
        * Step Three: Repeat the following regenerational steps until termination

            1. Select the best-fit individuals for reproduction. (Parents)
            2. Breed new individuals through crossover and mutation operations to give birth to offspring.
            3. Evaluate the individual fitness of new individuals.
            4. Replace least-fit population with new individuals.

        .. note::
            To develop an EA, all the abstract the methods used in the run() method must be implemented.
        """

        self.start_computing_time = time.time()

        self.population = self.create_initial_population()
        self.population = self.evaluate_population(self.population)
        self.init_progress()

        while not self.is_stopping_condition_reached():
            mating_population = self.selection(self.population)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate_population(offspring_population)
            self.population = self.replacement(self.population, offspring_population)
            self.update_progress()

        self.total_computing_time = self.get_current_computing_time()


class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]]):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(ParticleSwarmOptimization, self).__init__()
        self.swarm = []

    @abstractmethod
    def init_progress(self) -> None:
        pass

    @abstractmethod
    def update_progress(self) -> None:
        pass

    @abstractmethod
    def is_stopping_condition_reached(self) -> bool:
        pass

    @abstractmethod
    def create_initial_swarm(self) -> List[FloatSolution]:
        pass

    @abstractmethod
    def evaluate_swarm(self, swarm: List[FloatSolution]) -> List[FloatSolution]:
        pass

    @abstractmethod
    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    def run(self):
        self.start_computing_time = time.time()

        self.swarm = self.create_initial_swarm()
        self.swarm = self.evaluate_swarm(self.swarm)
        self.initialize_velocity(self.swarm)
        self.initialize_particle_best(self.swarm)
        self.initialize_global_best(self.swarm)
        self.init_progress()

        while not self.is_stopping_condition_reached():
            self.update_velocity(self.swarm)
            self.update_position(self.swarm)
            self.perturbation(self.swarm)
            self.swarm = self.evaluate_swarm(self.swarm)
            self.update_global_best(self.swarm)
            self.update_particle_best(self.swarm)
            self.update_progress()

        self.total_computing_time = self.get_current_computing_time()
