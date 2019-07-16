from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import InteractivePlot, Plot

if __name__ == '__main__':
    problem = DTLZ1(number_of_objectives=5)

    max_evaluations = 25000
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()

    label = algorithm.get_name() + "." + problem.get_name()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front,
                      axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front,
                                 axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))

