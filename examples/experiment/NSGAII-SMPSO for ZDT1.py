from jmetal.algorithm import NSGAII, SMPSO
from jmetal.component.archive import CrowdingDistanceArchive
from jmetal.component.comparator import RankingAndCrowdingDistanceComparator
from jmetal.operator import NullMutation, SBX, BinaryTournamentSelection
from jmetal.problem import ZDT1, ZDT2
from jmetal.component.quality_indicator import HyperVolume
from jmetal.util.laboratory import experiment, display

algorithm = [
    (NSGAII, {'population_size': 100, 'max_evaluations': 25000, 'mutation': NullMutation(), 'crossover': SBX(1.0, 20),
              'selection': BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())}),
    (NSGAII(population_size=100, max_evaluations=25000, mutation=NullMutation(), crossover=SBX(1.0, 20),
            selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()), problem=ZDT1()), {}),
    (SMPSO, {'swarm_size': 100, 'max_evaluations': 25000, 'mutation': NullMutation(),
             'leaders': CrowdingDistanceArchive(100)})
]
metric = [HyperVolume(reference_point=[1, 1])]
problem = [(ZDT1, {}), (ZDT2, {})]

results = experiment(algorithm, metric, problem)
display(results)