import pandas as pd

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import SBX, BinaryTournamentSelection, Polynomial
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.component import RankingAndCrowdingDistanceComparator, HyperVolume, CrowdingDistanceArchive
from jmetal.util.laboratory import Experiment, Job, convert_to_latex, compute_statistical_analysis
from jmetal.util.solution_list import read_front
from jmetal.util.termination_criteria import StoppingByEvaluations


def configure_experiment(problems: list, n_run: int):
  jobs = []

  for run in range(n_run):
    for problem in problems:
      jobs.append(
        Job(
          algorithm=NSGAII(
            problem=problem,
            population_size=100,
            mating_pool_size=100,
            offspring_size=100,
            mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBX(probability=1.0, distribution_index=20),
            selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
            termination_criteria=StoppingByEvaluations(max=5000)
          ),
          label='NSGAIIa',
          run=run,
        )
      )
      jobs.append(
        Job(
          algorithm=NSGAII(
            problem=problem,
            population_size=100,
            mating_pool_size=100,
            offspring_size=100,
            mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=SBX(probability=1.0, distribution_index=20),
            selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
            termination_criteria=StoppingByEvaluations(max=5000)
          ),
          label='NSGAIIb',
          run=run
        )
      )
      jobs.append(
        Job(
          algorithm=SMPSO(
            problem=problem,
            swarm_size=100,
            mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
            leaders=CrowdingDistanceArchive(100),
            termination_criteria=StoppingByEvaluations(max=5000)
          ),
          label='SMPSO',
          run=run
        )
      )

  return jobs


if __name__ == '__main__':
  zdt1_problem = ZDT1()
  zdt1_problem.reference_front = read_front(file_path='../../resources/reference_front/ZDT1.pf')

  zdt2_problem = ZDT2()
  zdt2_problem.reference_front = read_front(file_path='../../resources/reference_front/ZDT2.pf')

  zdt3_problem = ZDT3()
  zdt3_problem.reference_front = read_front(file_path='../../resources/reference_front/ZDT3.pf')

  jobs = configure_experiment(problems=[zdt1_problem, zdt2_problem, zdt3_problem], n_run= 5)

  experiment = Experiment(jobs=jobs)
  experiment.run()

  # Compute quality indicators
  df = experiment.compute_quality_indicator(qi=HyperVolume([1.0, 1.0]))
  print(df)

  # Generate a table with Median and Interquartile range.
  median = df.groupby(level=0).median()
  iqr = df.groupby(level=0).quantile(0.75) - df.groupby(level=0).quantile(0.25)
  table = median.applymap('{:.2e}'.format) + '_{' + iqr.applymap('{:.2e}'.format) + '}'

  # Add statistical analysis
  significance = compute_statistical_analysis(df)
  table = pd.concat([table, significance], axis=1)

  print(table)

  # Convert to LaTeX
  print(convert_to_latex(table, caption='Experiment'))
