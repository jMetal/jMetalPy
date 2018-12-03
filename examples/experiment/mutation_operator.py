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

    experiment = Experiment(jobs=jobs)
    experiment.run()

    results = experiment.compute_quality_indicator(qi=NonIndicator())
    print(results)

    median = results.groupby(level=0).median()
    iqr = results.groupby(level=0).quantile(0.75) - results.groupby(level=0).quantile(0.25)
    table = median.applymap('{:.2e}'.format) + '_{' + iqr.applymap('{:.2e}'.format) + '}'

    print(table)
    print(Experiment.convert_to_latex(table))
