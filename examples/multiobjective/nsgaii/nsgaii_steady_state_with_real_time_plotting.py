from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution import read_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

"""
Program to configure and run the steady-state NSGA-II algorithm with a real-time plotting observer. The display 
update frequency is set to 100 evaluations.
"""

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=100))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(title='Pareto front approximation. Problem: ' + problem.get_name(),
                      reference_front=problem.reference_front,
                      axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(title='Pareto front approximation. Problem: ' + problem.get_name(),
                                 reference_front=problem.reference_front,
                                 axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
