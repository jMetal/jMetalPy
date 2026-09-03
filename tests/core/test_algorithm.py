import pickle

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
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
    """Algorithm inherits from threading.Thread (see core/algorithm.py), whose instance
    state includes unpicklable locks. jmetal.lab.experiment.Experiment sends whole
    algorithm instances across a process boundary via ProcessPoolExecutor, so this must
    work even though nothing actually starts these instances as real threads.
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
