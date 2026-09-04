"""Tests for AlgorithmProtocol, the threading.Thread-independent algorithm contract."""

import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.core.algorithm import AlgorithmProtocol
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations


class TestAlgorithmProtocol:
    def test_a_classic_algorithm_satisfies_the_protocol(self):
        problem = ZDT1()
        algorithm = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=40),
        )

        assert isinstance(algorithm, AlgorithmProtocol)

    def test_a_component_based_algorithm_satisfies_the_protocol(self):
        problem = ZDT1()
        algorithm = build_nsgaii(
            problem,
            population_size=10,
            offspring_population_size=10,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
        )

        assert isinstance(algorithm, AlgorithmProtocol)

    def test_component_based_observable_data_uses_the_legacy_keys(self):
        problem = ZDT1()
        algorithm = build_nsgaii(
            problem,
            population_size=10,
            offspring_population_size=10,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            mutation=PolynomialMutation(probability=0.1, distribution_index=20),
            termination=TerminationByEvaluations(max_evaluations=10),
            rng=np.random.default_rng(1),
        )

        algorithm.run()
        data = algorithm.observable_data()

        assert set(data.keys()) == {"PROBLEM", "EVALUATIONS", "SOLUTIONS", "COMPUTING_TIME"}

    def test_an_object_missing_a_method_does_not_satisfy_the_protocol(self):
        class Incomplete:
            observable = None
            total_computing_time = 0.0

            def run(self):
                pass

            def result(self):
                pass

            def get_name(self):
                return "incomplete"

            # No observable_data().

        assert not isinstance(Incomplete(), AlgorithmProtocol)
