from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.component import ProgressBarObserver, CrowdingDistanceArchiveWithReferencePoint
from jmetal.operator import Polynomial
from jmetal.problem import ZDT1
from jmetal.util.graphic import InteractivePlot
from jmetal.util.solution_list import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations


if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    swarm_size = 100

    reference_points = [[0.5, 0.5], [0.2, 0.8]]
    archives_with_reference_points = []

    for point in reference_points:
        archives_with_reference_points.append(
            CrowdingDistanceArchiveWithReferencePoint(swarm_size, point)
        )

    algorithm = SMPSORP(
        problem=problem,
        swarm_size=swarm_size,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        reference_points=reference_points,
        leaders=archives_with_reference_points,
        termination_criteria=StoppingByEvaluations(max=25000)
    )

    progress_bar = ProgressBarObserver(max=25000)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = InteractivePlot(plot_title='SMPSORP-ZDT1', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.update(reference_points, legend='reference points')
    pareto_front.export_html(filename='SMPSORP-ZDT1')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
