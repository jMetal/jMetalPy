from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.graphic import InteractivePlot
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DTLZ1(number_of_objectives=5)

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=25000)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=25000))

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = InteractivePlot(plot_title='SMPSO-DTLZ1-5', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    pareto_front.plot(front)
    pareto_front.export_to_html(filename='SMPSO-DTLZ1-5')

    pareto_front = InteractivePlot(plot_title='SMPSO-DTLZ1-5-norm', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    pareto_front.plot(front, normalize=True)
    pareto_front.export_to_html(filename='SMPSO-DTLZ1-5-norm')

    # Save variables to file
    print_function_values_to_file(front, 'FUN.SMPSO.DTLZ1-5')
    print_variables_to_file(front, 'VAR.SMPSO.DTLZ1-5')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
