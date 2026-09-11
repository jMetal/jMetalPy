"""
This module defines the core algorithm interfaces for optimization in JMetalPy.

It provides abstract base classes for different types of optimization algorithms,
including evolutionary algorithms and particle swarm optimization, with support
for both single-objective and multi-objective optimization problems.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from jmetal.core.observer import Observable
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.logger import get_logger
from jmetal.util.observable import DefaultObservable

# Initialize module logger
logger = get_logger(__name__)

# Type variables for generic algorithm implementation
S = TypeVar("S")  # Type of the solutions
R = TypeVar("R")  # Type of the result returned by the algorithm


def thread_rng_into_operators(rng: np.random.Generator | None, *operators) -> None:
    """Best-effort propagation of an algorithm's `rng` into its operators.

    Mirrors `jmetal.component.algorithm.evolutionary_algorithm.EvolutionaryAlgorithm.
    _thread_rng_into_components` for the classic algorithm hierarchy. Does nothing when
    `rng` is None, so a classic algorithm built without an explicit `rng` keeps its exact
    prior behavior (operators fall back to their own defaults, population creation keeps
    consuming the global `random`/`numpy.random` state). When `rng` is given, it is only
    assigned to operators exposing a `rng`/`_rng` attribute that is still `None` -- an
    operator the caller already seeded explicitly is left untouched.
    """
    if rng is None:
        return
    for operator in operators:
        if operator is None:
            continue
        if hasattr(operator, "rng") and operator.rng is None:
            operator.rng = rng
        elif hasattr(operator, "_rng") and operator._rng is None:
            operator._rng = rng


@runtime_checkable
class AlgorithmProtocol(Protocol[R]):
    """Structural contract satisfied by any algorithm, classic or component-based.

    Defined independently of `threading.Thread`. Consumers that only need
    "something that can run and report progress" -- e.g.
    `jmetal.lab.experiment.Job` -- can type against this instead of requiring a
    `threading.Thread` subclass. Every `Algorithm` subclass and
    `jmetal.component.algorithm.evolutionary_algorithm.EvolutionaryAlgorithm`
    already satisfy it.
    """

    observable: Observable
    total_computing_time: float

    def run(self) -> None:
        """Run the algorithm to completion."""
        ...

    def result(self) -> R:
        """Return the algorithm's result."""
        ...

    def get_name(self) -> str:
        """Return the algorithm's name."""
        ...

    def observable_data(self) -> dict:
        """Return the data broadcast to observers on each progress update."""
        ...


class Algorithm(Generic[S, R], ABC):
    """Abstract base class for all optimization algorithms in JMetalPy.

    This class serves as the foundation for implementing various optimization
    algorithms, implementing the template method pattern through its abstract
    methods. It does not inherit from `threading.Thread`: nothing in jMetalPy calls
    `start()`/`join()` on an algorithm -- `run()` is always called directly -- so
    that inheritance only added unpicklable internal state (locks, `Event` objects)
    that `jmetal.lab.experiment.Job` had to work around to send algorithms across a
    process boundary via `ProcessPoolExecutor`. Use `run_in_thread()` below if an
    algorithm genuinely needs to run in the background.

    Attributes:
        solutions: List of solutions found by the algorithm.
        evaluations: Number of solution evaluations performed.
        start_computing_time: Timestamp when the algorithm started running.
        total_computing_time: Total time taken by the algorithm (in seconds).
        observable: Observer pattern implementation for monitoring algorithm progress.
    """

    def __init__(self):
        """Initialize the algorithm with default values."""
        self.solutions: list[S] = []
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0
        self.observable = DefaultObservable()

    @abstractmethod
    def create_initial_solutions(self) -> list[S]:
        """Creates the initial list of solutions of a metaheuristic."""
        pass

    @abstractmethod
    def evaluate(self, solution_list: list[S]) -> list[S]:
        """Evaluates a solution list."""
        pass

    @abstractmethod
    def init_progress(self) -> None:
        """Initialize the algorithm."""
        pass

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """The stopping condition is met or not."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Performs one iteration/step of the algorithm's loop."""
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """Update the progress after each iteration."""
        pass

    @abstractmethod
    def observable_data(self) -> dict:
        """Get observable data, with the information that will be seng to all observers each time."""
        pass

    def run(self):
        """Execute the algorithm."""
        self.start_computing_time = time.time()

        logger.debug("Creating initial set of solutions...")
        self.solutions = self.create_initial_solutions()

        logger.debug("Evaluating solutions...")
        self.solutions = self.evaluate(self.solutions)

        logger.debug("Initializing progress...")
        self.init_progress()

        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        logger.debug("Finished!")

        self.total_computing_time = time.time() - self.start_computing_time

    @abstractmethod
    def result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class DynamicAlgorithm(Algorithm[S, R], ABC):
    """Abstract base class for algorithms that can handle dynamic optimization problems.

    Dynamic optimization problems are those where the fitness function, constraints,
    or other problem characteristics may change over time. This class extends the
    base Algorithm with methods to handle such changes.

    Subclasses must implement the restart method to define how the algorithm should
    respond to changes in the problem definition.
    """

    @abstractmethod
    def restart(self) -> None:
        """Restart the algorithm in response to changes in the problem.

        This method is called when a change in the problem is detected. Implementations
        should reset or adapt the algorithm's state to handle the new problem conditions.
        """
        pass


class EvolutionaryAlgorithm(Algorithm[S, R], ABC):
    """Abstract base class for evolutionary algorithms.

    This class implements the core structure of an evolutionary algorithm, including
    the evolutionary cycle of selection, reproduction, and replacement. Subclasses
    must implement the specific selection, reproduction, and replacement strategies.

    Attributes:
        problem: The optimization problem to solve.
        population_size: Number of solutions in the population.
        offspring_population_size: Number of offspring solutions generated each generation.
    """

    def __init__(self, problem: Problem[S], population_size: int, offspring_population_size: int):
        """Initialize the evolutionary algorithm.

        Args:
            problem: The optimization problem to solve.
            population_size: Number of solutions in the population.
            offspring_population_size: Number of offspring solutions to generate each generation.
        """
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

    @abstractmethod
    def selection(self, population: list[S]) -> list[S]:
        """Select the best-fit individuals for reproduction (parents)."""
        pass

    @abstractmethod
    def reproduction(self, population: list[S]) -> list[S]:
        """Breed new individuals through crossover and mutation operations to give birth to offspring."""
        pass

    @abstractmethod
    def replacement(self, population: list[S], offspring_population: list[S]) -> list[S]:
        """Replace least-fit population with new individuals."""
        pass

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def init_progress(self) -> None:
        self.evaluations = self.population_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f"{self.get_name()}.{self.problem.name()}"


class ParticleSwarmOptimization(Algorithm[FloatSolution, list[FloatSolution]], ABC):
    """Abstract base class for Particle Swarm Optimization (PSO) algorithms.

    This class implements the core structure of a PSO algorithm, where a population
    of candidate solutions (particles) move through the search space according to
    simple mathematical formulae over the particle's position and velocity.

    Attributes:
        problem: The optimization problem to solve.
        swarm_size: Number of particles in the swarm.
    """

    def __init__(self, problem: Problem[S], swarm_size: int):
        """Initialize the PSO algorithm.

        Args:
            problem: The optimization problem to solve.
            swarm_size: Number of particles in the swarm.
        """
        super().__init__()
        self.problem = problem
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_velocity(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_global_best(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_velocity(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_particle_best(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_global_best(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: list[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: list[FloatSolution]) -> None:
        pass

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f"{self.get_name()}.{self.problem.name()}"


def run_in_thread(algorithm: AlgorithmProtocol) -> threading.Thread:
    """Run an algorithm in a background thread.

    Algorithm no longer inherits from `threading.Thread`, so `algorithm.run()` must
    be called directly or, if background execution is genuinely needed -- e.g. a
    live-plotting loop driven from the main thread while the algorithm keeps
    running -- via this explicit helper instead.

    Args:
        algorithm: Any object satisfying `AlgorithmProtocol`.

    Returns:
        The `threading.Thread` running `algorithm.run()`, already started.
    """
    thread = threading.Thread(target=algorithm.run)
    thread.start()

    return thread
