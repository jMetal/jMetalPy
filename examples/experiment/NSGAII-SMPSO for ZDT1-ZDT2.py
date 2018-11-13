import pandas as pd

from jmetal.algorithm import NSGAII, SMPSO
from jmetal.operator import NullMutation, SBX, BinaryTournamentSelection, Polynomial
from jmetal.problem import ZDT1, ZDT2
from jmetal.component import CrowdingDistanceArchive, RankingAndCrowdingDistanceComparator, HyperVolume
from jmetal.util.laboratory import Experiment


# Configure experiment
problem_list = [ZDT1(), ZDT2()]
metric_list = [HyperVolume(reference_point=[1, 1])]
algorithm_list = []

for problem in problem_list:
    algorithm_list.append({
        'label': 'NSGAII_A',
        'algorithm': NSGAII(
             problem=problem,
             population_size=100,
             max_evaluations=10000,
             mutation=NullMutation(),
             crossover=SBX(probability=1.0, distribution_index=20),
             selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
        )}
    )
    algorithm_list.append({
        'label': 'NSGAII_B',
        'algorithm': NSGAII(
             problem=problem,
             population_size=100,
             max_evaluations=10000,
             mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
             crossover=SBX(probability=1.0, distribution_index=20),
             selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
        )}
    )
    algorithm_list.append({
        'label': 'SMPSO',
        'algorithm': SMPSO(
             problem=problem,
             swarm_size=100,
             max_evaluations=5000,
             mutation=Polynomial(probability=0.5, distribution_index=20),
             leaders=CrowdingDistanceArchive(100)
        )}
    )

study = Experiment(base_directory='./ex', algorithm_list=algorithm_list, problem_list=problem_list,
                   metric_list=metric_list, n_runs=1)
study.run()
results = study.compute_metrics()

print(results)

mean_results = results.groupby(['problem', 'metric']).mean()
median_results = results.groupby(['problem', 'metric']).median()

print(mean_results)
print(median_results)
print(results.groupby(['problem', 'metric']).min())
print(results.groupby(['problem', 'metric']).max())
print(mean_results.xs('Hypervolume', level='metric'))

print(study.convert_to_latex(mean_results.xs('Hypervolume', level='metric')))
