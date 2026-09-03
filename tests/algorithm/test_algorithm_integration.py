import pytest

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.quality_indicator import HyperVolume
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations


@pytest.fixture
def zdt1_mutation_and_crossover():
    problem = ZDT1()
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    return problem, mutation, crossover


@pytest.mark.smoke
class TestAlgorithmsRunWithoutErrors:
    """Quick sanity check that each algorithm completes a short run, no correctness
    assertions beyond that -- see TestAlgorithmsReachExpectedHypervolume for those.
    """

    def test_should_nsgaii_complete_a_short_run(self, zdt1_mutation_and_crossover):
        problem, mutation, crossover = zdt1_mutation_and_crossover

        NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=mutation,
            crossover=crossover,
            termination_criterion=StoppingByEvaluations(max_evaluations=1000),
        ).run()

    def test_should_smpso_complete_a_short_run(self, zdt1_mutation_and_crossover):
        problem, mutation, _ = zdt1_mutation_and_crossover

        SMPSO(
            problem=problem,
            swarm_size=100,
            mutation=mutation,
            leaders=CrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=1000),
        ).run()


@pytest.mark.integration
class TestAlgorithmsReachExpectedHypervolume:
    """Full-length runs on ZDT1 with standard settings, checked against a known-good
    hypervolume floor rather than just "it ran".
    """

    def test_should_nsgaii_exceed_the_expected_hypervolume_on_zdt1(self):
        problem = ZDT1()
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=25000),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([front[i].objectives for i in range(len(front))])

        assert value > 0.65

    def test_should_smpso_exceed_the_expected_hypervolume_on_zdt1(self):
        problem = ZDT1()
        algorithm = SMPSO(
            problem=problem,
            swarm_size=100,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
            leaders=CrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=25000),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([front[i].objectives for i in range(len(front))])

        assert value >= 0.655
