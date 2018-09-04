from jmetal.algorithm import NSGAII
from jmetal.component.comparator import RankingAndCrowdingDistanceComparator
from jmetal.operator import NullMutation, SBX, BinaryTournamentSelection, Polynomial
from jmetal.problem import ZDT1, ZDT2
from jmetal.component.quality_indicator import HyperVolume
from jmetal.util.laboratory import Experiment

# Configure experiment
problem_list = [ZDT1(), ZDT2()]
algorithm_list = []

for problem in problem_list:
    algorithm_list.append(
        ('NSGAII_A',
         NSGAII(
             problem=problem,
             population_size=100,
             max_evaluations=25000,
             mutation=NullMutation(),
             crossover=SBX(probability=1.0, distribution_index=20),
             selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
         ))
    )
    algorithm_list.append(
        ('NSGAII_B',
         NSGAII(
             problem=problem,
             population_size=100,
             max_evaluations=25000,
             mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
             crossover=SBX(probability=1.0, distribution_index=20),
             selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
         ))
    )

study = Experiment(algorithm_list, n_runs=2)
study.run()

# Compute quality indicators
metric_list = [HyperVolume(reference_point=[1, 1])]

print(study.compute_metrics(metric_list))