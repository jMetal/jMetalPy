import pandas as pd

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component import RankingAndCrowdingDistanceComparator, HyperVolume, CrowdingDistanceArchive
from jmetal.component.critical_distance import CDplot
from jmetal.component.quality_indicator import GenerationalDistance
from jmetal.operator import SBX, BinaryTournamentSelection, Polynomial, NullMutation
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jmetal.util.laboratory import Experiment, Job, convert_table_to_latex, compute_statistical_analysis, \
    compute_quality_indicator, create_tables_from_data
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
                        crossover=SBX(probability=0.8, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criteria=StoppingByEvaluations(max=25000)
                    ),
                    algorithm_tag='NSGAIIa',
                    problem_tag=problem.get_name(),
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
                        mutation=NullMutation(),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criteria=StoppingByEvaluations(max=25000)
                    ),
                    algorithm_tag='NSGAIIb',
                    problem_tag=problem.get_name(),
                    run=run
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
                        termination_criteria=StoppingByEvaluations(max=25000)
                    ),
                    algorithm_tag='NSGAIIc',
                    problem_tag=problem.get_name(),
                    run=run
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        mating_pool_size=100,
                        offspring_size=100,
                        mutation=Polynomial(probability=0.2, distribution_index=20),
                        crossover=SBX(probability=0.2, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criteria=StoppingByEvaluations(max=25000)
                    ),
                    algorithm_tag='NSGAIId',
                    problem_tag=problem.get_name(),
                    run=run
                )
            )

    return jobs


if __name__ == '__main__':
    base_directory = 'data'
    reference_fronts = 'reference_front'

    zdt1, zdt2, zdt3, zdt4, zdt6 = ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()
    jobs = configure_experiment(problems=[zdt1, zdt2, zdt3, zdt4, zdt6], n_run=25)

    experiment = Experiment(base_dir=base_directory, jobs=jobs)
    experiment.run()
    
    compute_quality_indicator(input_data=base_directory,
                              reference_fronts=reference_fronts,
                              quality_indicators=[HyperVolume([1.0, 1.0]), GenerationalDistance(None)])

    nsgaii_a = create_tables_from_data(input_data='data/NSGAIIa/', output_filename='results')

    # Generate a table with Median and Interquartile range
    median = nsgaii_a.groupby(level=0).median()
    iqr = nsgaii_a.groupby(level=0).quantile(0.75) - nsgaii_a.groupby(level=0).quantile(0.25)
    table = median.applymap('{:.2e}'.format) + '_{' + iqr.applymap('{:.2e}'.format) + '}'

    # Add statistical analysis
    significance = compute_statistical_analysis(nsgaii_a)
    table = pd.concat([table, significance], axis=1)

    # Convert to LaTeX
    print(convert_table_to_latex(table, caption='Experiment'))

    # Plot CD
    results_hv = []
    labels = ['NSGAIIa', 'NSGAIIb', 'NSGAIIc', 'NSGAIId']
    for algorithm in labels:
        values = create_tables_from_data(input_data='data/' + algorithm).groupby(level=0).median()['QI.GD'].tolist()
        results_hv.append(values)

    CDplot(results_hv, alg_names=labels)
