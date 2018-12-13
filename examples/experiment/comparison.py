import pandas as pd

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component import RankingAndCrowdingDistanceComparator, HyperVolume, CrowdingDistanceArchive
from jmetal.component.quality_indicator import GenerationalDistance
from jmetal.operator import SBX, BinaryTournamentSelection, Polynomial
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.util.laboratory import Experiment, Job, convert_to_latex, compute_statistical_analysis, \
    compute_quality_indicator, create_tables_from_experiment
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
                    algorithm_tag='NSGAIIa',
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
                    algorithm_tag='NSGAIIb',
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
                    algorithm_tag='SMPSO',
                    run=run
                )
            )

    return jobs


if __name__ == '__main__':
    base_directory = 'data'
    reference_fronts = '/home/benhid/Proyectos/jMetalPy/resources/reference_front'

    zdt1_problem, zdt2_problem, zdt3_problem = ZDT1(), ZDT2(), ZDT3()
    jobs = configure_experiment(problems=[zdt1_problem, zdt2_problem, zdt3_problem], n_run=5)

    experiment = Experiment(base_directory=base_directory, jobs=jobs)
    experiment.run()

    compute_quality_indicator(input_data=base_directory,
                              reference_fronts=reference_fronts,
                              quality_indicators=[HyperVolume([1.0, 1.0]), GenerationalDistance(None)])

    nsgaii_a = create_tables_from_experiment(input_data='./data/NSGAIIa')

    # Generate a table with Median and Interquartile range.
    median = nsgaii_a.groupby(level=0).median()
    iqr = nsgaii_a.groupby(level=0).quantile(0.75) - nsgaii_a.groupby(level=0).quantile(0.25)
    table = median.applymap('{:.2e}'.format) + '_{' + iqr.applymap('{:.2e}'.format) + '}'

    # Add statistical analysis
    significance = compute_statistical_analysis(nsgaii_a)
    table = pd.concat([table, significance], axis=1)

    print(table)

    # Convert to LaTeX
    print(convert_to_latex(table, caption='Experiment'))
