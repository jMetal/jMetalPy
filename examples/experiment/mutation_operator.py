from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import NullMutation, SBX, BinaryTournamentSelection, Polynomial
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.component import RankingAndCrowdingDistanceComparator, HyperVolume, ComputingTime
from jmetal.util.laboratory import Experiment, Job


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
                        max_evaluations=5000,
                        mutation=NullMutation(),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
                    ),
                    label='NSGAIIa',
                    problem_name=problem.get_name(),
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
                        max_evaluations=5000,
                        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
                    ),
                    label='NSGAIIb',
                    problem_name=problem.get_name(),
                    run=run
                )
            )

    return jobs


if __name__ == '__main__':
    problems = [ZDT1(), ZDT2(), ZDT3()]
    jobs = configure_experiment(problems=problems, n_run=30)

    experiment_with_mutation_operator = Experiment(base_directory='C:\\Users\\makis\\Proyectos\\jMetalPy\\examples', jobs=jobs)
    experiment_with_mutation_operator.run()

    metrics = [HyperVolume(reference_point=[1, 1]), ComputingTime()]
    data = experiment_with_mutation_operator.compute_metrics(metrics)

    print(data)
    print(data.groupby(['problem', 'metric']).mean())
    print(data.groupby(['problem', 'metric']).median())

    ss = data.xs(('ZDT1', 'Hypervolume'), level=('problem', 'metric'))
    print(ss)

    statistic, pvalue = Experiment.compute_statistical_analysis([ss['NSGAIIa'].tolist(), ss['NSGAIIb'].tolist()])
    print(statistic, pvalue)
