from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.quality_indicator import EpsilonIndicator, NonIndicator
from jmetal.operator import NullMutation, SBX, BinaryTournamentSelection, Polynomial
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.component import RankingAndCrowdingDistanceComparator, HyperVolume
from jmetal.util.laboratory import Experiment, Job
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
                        mutation=NullMutation(),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criteria=StoppingByEvaluations(max=500)
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
                        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criteria=StoppingByEvaluations(max=500)
                    ),
                    label='NSGAIIb',
                    problem_name=problem.get_name(),
                    run=run
                )
            )

    return jobs


if __name__ == '__main__':
    problems = [ZDT1(), ZDT2(), ZDT3()]
    jobs = configure_experiment(problems=problems, n_run=3)

    experiment_with_mutation_operator = Experiment(base_directory='./', jobs=jobs)
    experiment_with_mutation_operator.run()

    metrics = [HyperVolume(reference_point=[1, 1]), NonIndicator()]
    data = experiment_with_mutation_operator.compute_metrics(metrics)

    print(data.info())
    print(data.columns)
    print(data['Hypervolume'])
    print(data['Hypervolume', 'NSGAIIa'])

    median = data.groupby(level=0).median()
    mean = data.groupby(level=0).mean()

    print(median)

    test_values = data['Test']
    print(test_values.groupby(level=0).mean())
