from jmetal.algorithm import SMPSO
from jmetal.component import CrowdingDistanceArchive
from jmetal.operator import Polynomial, NullMutation
from jmetal.problem import ZDT1, ZDT2
from jmetal.component.quality_indicator import HyperVolume
from jmetal.util.laboratory import Experiment

# Configure experiment
problem_list = [ZDT1(), ZDT2()]
algorithm_list = []

for problem in problem_list:
    algorithm_list.append(
        ('SMPSO_A',
         SMPSO(
             problem=problem,
             swarm_size=100,
             max_evaluations=25000,
             mutation=Polynomial(probability=0.5, distribution_index=20),
             leaders=CrowdingDistanceArchive(100)
         ))
    )
    algorithm_list.append(
        ('SMPSO_B',
         SMPSO(
             problem=problem,
             swarm_size=100,
             max_evaluations=25000,
             mutation=NullMutation(),
             leaders=CrowdingDistanceArchive(100)
         ))
    )
    algorithm_list.append(
        ('SMPSO_C',
         SMPSO(
             problem=problem,
             swarm_size=100,
             max_evaluations=25000,
             mutation=NullMutation(),
             leaders=CrowdingDistanceArchive(100)
         ))
    )

study = Experiment(algorithm_list, n_runs=1)
study.run()

# Compute quality indicators
metric_list = [HyperVolume(reference_point=[1, 1])]

print(study.compute_metrics(metric_list))
