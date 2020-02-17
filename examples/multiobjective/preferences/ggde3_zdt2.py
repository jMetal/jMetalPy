from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.solutions.comparator import GDominanceComparator

from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.problem import ZDT2
from jmetal.util.observer import VisualizerObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT2()
    problem.reference_front = read_solutions(filename='resources/reference_front/{}.pf'.format(problem.get_name()))

    max_evaluations = 25000
    reference_point = [0.2, 0.5]

    algorithm = GDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=GDominanceComparator(reference_point)
    )

    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))