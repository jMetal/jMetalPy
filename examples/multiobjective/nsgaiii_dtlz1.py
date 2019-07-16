from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot, InteractivePlot

if __name__ == '__main__':
    problem = DTLZ1()
    problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ1.3D.pf')

    max_evaluations = 10000
    algorithm = NSGAIII(
        problem=problem,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    progress_bar = ProgressBarObserver(max=max_evaluations)
    algorithm.observable.register(observer=progress_bar)
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=1000))

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
