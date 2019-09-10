from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.solutions.comparator import GDominanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/{}.pf'.format(problem.get_name()))

    reference_point = [0.4, 0.6]

    max_evaluations = 25000
    algorithm = SPEA2(
        problem=problem,
        population_size=40,
        offspring_population_size=40,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        dominance_comparator=GDominanceComparator(reference_point)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=(reference_point)))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels, reference_point=reference_point, reference_front=problem.reference_front)
    plot_front.plot(front, label='gSPEA2-ZDT1', filename='gSPEA2-ZDT1')

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels, reference_point=reference_point, reference_front=problem.reference_front)
    plot_front.plot(front, label='gSPEA2-ZDT1', filename='gSPEA2-ZDT1')

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.'+ algorithm.get_name() + "." + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
