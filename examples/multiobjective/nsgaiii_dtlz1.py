from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution_list import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization import Plot

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT1.pf')

    max_evaluations = 25000
    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='NSGAIII-ZDT1', axis_labels=problem.obj_labels, reference_front=problem.reference_front)
    plot_front.plot([algorithm.get_result()], labels=['Pareto front'], filename='NSGAIII-ZDT1.eps')

    # Save results to file
    print_function_values_to_file(front, 'FUN.NSGAIII.ZDT1')
    print_variables_to_file(front, 'VAR.NSGAIII.ZDT1')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
