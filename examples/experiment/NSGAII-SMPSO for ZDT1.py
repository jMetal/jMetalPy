from jmetal.algorithm import NSGAII
from jmetal.component.comparator import RankingAndCrowdingDistanceComparator
from jmetal.operator import NullMutation, SBX, BinaryTournamentSelection
from jmetal.problem import ZDT1, ZDT2
from jmetal.component.quality_indicator import HyperVolume
from jmetal.util.laboratory import Experiment

algorithm = [
    (NSGAII, {'population_size': 100, 'max_evaluations': 25000, 'mutation': NullMutation(), 'crossover': SBX(1.0, 20),
              'selection': BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())}),
    (NSGAII, {'population_size': 100, 'max_evaluations': 25000, 'mutation': NullMutation(), 'crossover': SBX(0.7, 20),
              'selection': BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())})

]
problem = [(ZDT1, {}), (ZDT2, {}), (ZDT2, {})]

study = Experiment(algorithm, problem, n_runs=2)
study.run()

metric = [HyperVolume(reference_point=[1, 1])]

study.compute_metrics(metric)
