from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.quality_indicator import GenerationalDistance, HyperVolume
from jmetal.operator import SBXCrossover, BinaryTournamentSelection, PolynomialMutation, NullMutation
from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.util.laboratory import Experiment, Job, generate_summary_from_experiment
from jmetal.util.termination_criterion import StoppingByEvaluations


def configure_experiment(problems: list, n_run: int):
    jobs = []

    for run in range(n_run):
        for problem in problems:
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        crossover=SBXCrossover(probability=0.8, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criterion=StoppingByEvaluations(max=25000)
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
                        offspring_population_size=100,
                        mutation=NullMutation(),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criterion=StoppingByEvaluations(max=25000)
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
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criterion=StoppingByEvaluations(max=25000)
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
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=0.2, distribution_index=20),
                        crossover=SBXCrossover(probability=0.2, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
                        termination_criterion=StoppingByEvaluations(max=25000)
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

    experiment = Experiment(output_dir=base_directory, jobs=jobs)
    experiment.run()
    
    generate_summary_from_experiment(
        input_dir=base_directory,
        reference_fronts=reference_fronts,
        quality_indicators=[HyperVolume([1.0, 1.0]), GenerationalDistance(None)])
