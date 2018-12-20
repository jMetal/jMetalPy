from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component import RankingAndCrowdingDistanceComparator, ProgressBarObserver, VisualizerObserver
from jmetal.operator import SBX, Polynomial, BinaryTournamentSelection
from jmetal.problem import ZDT1
from jmetal.util.solution_list import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    progress_bar = ProgressBarObserver(max=max_evaluations)
    algorithm.observable.register(observer=progress_bar)
    #algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.'+ algorithm.get_name() + "." + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
