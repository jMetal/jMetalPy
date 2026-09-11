import pickle
import threading

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.algorithm import run_in_thread
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations


def _build_nsgaii(max_evaluations: int = 40) -> NSGAII:
    problem = ZDT1()
    return NSGAII(
        problem=problem,
        population_size=10,
        offspring_population_size=10,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables(), distribution_index=20
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )


class TestAlgorithmPickling:
    """Algorithm no longer inherits from threading.Thread, so pickling needs no
    special-casing -- it's a plain object now. This guards against a regression (e.g.
    reintroducing Thread, or another unpicklable dependency) since
    jmetal.lab.experiment.Experiment sends whole algorithm instances across a
    process boundary via ProcessPoolExecutor.
    """

    def test_should_pickle_an_unstarted_algorithm(self):
        algorithm = _build_nsgaii()

        blob = pickle.dumps(algorithm)
        restored = pickle.loads(blob)

        assert restored.problem.name() == algorithm.problem.name()
        assert restored.population_size == algorithm.population_size

    def test_should_run_after_a_pickle_round_trip(self):
        algorithm = _build_nsgaii(max_evaluations=40)

        restored = pickle.loads(pickle.dumps(algorithm))
        restored.run()

        assert restored.evaluations >= 40
        assert len(restored.result()) == 10


class TestAlgorithmDoesNotInheritFromThread:
    def test_algorithm_is_not_a_thread(self):
        algorithm = _build_nsgaii()

        assert not isinstance(algorithm, threading.Thread)


class TestRunInThread:
    def test_runs_the_algorithm_to_completion_in_a_background_thread(self):
        algorithm = _build_nsgaii(max_evaluations=40)

        thread = run_in_thread(algorithm)
        thread.join()

        assert not thread.is_alive()
        assert algorithm.evaluations >= 40
        assert len(algorithm.result()) == 10

    def test_returns_an_already_started_thread(self):
        algorithm = _build_nsgaii(max_evaluations=40)

        thread = run_in_thread(algorithm)

        assert isinstance(thread, threading.Thread)
        thread.join()

    def test_works_with_any_object_satisfying_the_algorithm_protocol(self):
        # run_in_thread() only needs a run() method -- it doesn't require an
        # Algorithm subclass, matching AlgorithmProtocol's structural typing.
        class Counter:
            def __init__(self):
                self.calls = 0

            def run(self):
                self.calls += 1

        counter = Counter()

        thread = run_in_thread(counter)
        thread.join()

        assert counter.calls == 1
