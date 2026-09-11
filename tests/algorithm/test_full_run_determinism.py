"""Full-run, same-seed-twice, identical-results determinism tests.

Closes the coverage gap identified in the Fase 0-3 RNG-reproducibility cleanup: most
classic algorithm classes had *some* rng-related test (initial population, a single
operator call, object identity) but not a direct end-to-end proof that building the
same algorithm twice with the same seed and running each to completion produces
identical results.

Deliberately excluded (not a coverage gap, but a structural one): DistributedNSGAII
(requires an external dask-style `client` argument, none is wired up anywhere in this
test suite) and SMPSORP (its constructor spawns a background thread that blocks
reading from stdin, so it cannot be instantiated in an automated test without mocking
thread startup).

Also excluded: running DynamicNSGAII, DynamicGDE3, or DynamicSMPSO to completion via a
plain `.run()` call. Their `stopping_condition_is_met()` override never returns a value
(always `None`, which is falsy) once the termination criterion fires -- it just restarts
and keeps going -- so `while not self.stopping_condition_is_met()` in `Algorithm.run()`
never exits and the call hangs forever, even against a static problem. That's a
pre-existing algorithm-correctness bug, not an rng-reproducibility gap, so it's left
alone here.
"""

import numpy as np

from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.hype import HYPE
from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.moead import MOEADIEpsilon
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.algorithm.singleobjective.simulated_annealing import SimulatedAnnealing
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import Sphere, ZDT1
from jmetal.util.aggregation_function import WeightedSum
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9
from jmetal.util.termination_criterion import StoppingByEvaluations


def _objectives(algorithm) -> list[tuple]:
    result = algorithm.result()
    if isinstance(result, list):
        return sorted(tuple(s.objectives) for s in result)
    return [tuple(result.objectives)]


def _zdt1_mutation_crossover(seed):
    problem = ZDT1()
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(),
        distribution_index=20,
        rng=np.random.default_rng(seed),
    )
    crossover = SBXCrossover(probability=1.0, distribution_index=20, rng=np.random.default_rng(seed))
    return problem, mutation, crossover


class TestNSGAIIFamilyDeterminism:
    def test_nsgaii(self):
        def build(seed):
            problem, mutation, crossover = _zdt1_mutation_crossover(seed)
            return NSGAII(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestMOCellDeterminism:
    def test_mocell(self):
        def build(seed):
            problem, mutation, crossover = _zdt1_mutation_crossover(seed)
            return MOCell(
                problem=problem,
                population_size=16,
                neighborhood=C9(rows=4, columns=4),
                archive=CrowdingDistanceArchive(16),
                mutation=mutation,
                crossover=crossover,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestSMSEMOADeterminism:
    def test_smsemoa(self):
        def build(seed):
            problem, mutation, crossover = _zdt1_mutation_crossover(seed)
            return SMSEMOA(
                problem=problem,
                population_size=10,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestSPEA2Determinism:
    def test_spea2(self):
        def build(seed):
            problem, mutation, crossover = _zdt1_mutation_crossover(seed)
            return SPEA2(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestHYPEDeterminism:
    def test_hype(self):
        def build(seed):
            problem, mutation, crossover = _zdt1_mutation_crossover(seed)
            reference_point = FloatSolution([0], [1], problem.number_of_objectives())
            reference_point.objectives = [1.0, 1.0]
            return HYPE(
                problem=problem,
                reference_point=reference_point,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestIBEADeterminism:
    def test_ibea(self):
        def build(seed):
            problem, mutation, crossover = _zdt1_mutation_crossover(seed)
            return IBEA(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
                kappa=1.0,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestGDE3FamilyDeterminism:
    def test_gde3(self):
        def build(seed):
            return GDE3(
                problem=ZDT1(),
                population_size=10,
                cr=0.5,
                f=0.5,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestMoeadIEpsilonDeterminism:
    def test_moead_i_epsilon(self):
        def build(seed):
            return MOEADIEpsilon(
                problem=ZDT1(),
                population_size=10,
                mutation=PolynomialMutation(
                    probability=0.1, distribution_index=20, rng=np.random.default_rng(seed)
                ),
                crossover=DifferentialEvolutionCrossover(
                    CR=1.0, F=0.5, rng=np.random.default_rng(seed)
                ),
                aggregation_function=WeightedSum(),
                neighbourhood_selection_probability=0.9,
                max_number_of_replaced_solutions=2,
                neighbor_size=4,
                weight_files_path="unused-for-2-objectives",
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)


class TestSingleObjectiveAlgorithmsDeterminism:
    def test_genetic_algorithm(self):
        def build(seed):
            problem = Sphere(number_of_variables=5)
            return GeneticAlgorithm(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=PolynomialMutation(
                    probability=0.1, distribution_index=20, rng=np.random.default_rng(seed)
                ),
                crossover=SBXCrossover(
                    probability=0.9, distribution_index=20, rng=np.random.default_rng(seed)
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)

    def test_evolution_strategy(self):
        def build(seed):
            problem = Sphere(number_of_variables=5)
            return EvolutionStrategy(
                problem=problem,
                mu=10,
                lambda_=10,
                elitist=True,
                mutation=PolynomialMutation(
                    probability=1.0 / problem.number_of_variables(),
                    rng=np.random.default_rng(seed),
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)

    def test_local_search(self):
        def build(seed):
            problem = Sphere(number_of_variables=5)
            return LocalSearch(
                problem=problem,
                mutation=PolynomialMutation(
                    probability=1.0, distribution_index=20, rng=np.random.default_rng(seed)
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=50),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)

    def test_random_search(self):
        def build(seed):
            return RandomSearch(
                problem=Sphere(number_of_variables=5),
                termination_criterion=StoppingByEvaluations(max_evaluations=50),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)

    def test_simulated_annealing(self):
        def build(seed):
            problem = Sphere(number_of_variables=5)
            return SimulatedAnnealing(
                problem=problem,
                mutation=PolynomialMutation(
                    probability=1.0, distribution_index=20, rng=np.random.default_rng(seed)
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=50),
                rng=np.random.default_rng(seed),
            )

        a, b = build(42), build(42)
        a.run()
        b.run()
        assert _objectives(a) == _objectives(b)
