from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.operator import PolynomialMutation
from jmetal.problem import ZDT4, ZDT1
from jmetal.util.archive import CrowdingDistanceArchiveWithReferencePoint
<<<<<<< HEAD:examples/multiobjective/smpso/smpsorp_zdt4.py
from jmetal.util.solution import read_solutions, print_variables_to_file, print_function_values_to_file
=======
from jmetal.util.solutions import read_solutions

if __name__ == '__main__':
    problem = ZDT4()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT4.pf')

from jmetal.util.observer import VisualizerObserver
from jmetal.util.solutions import read_solutions, print_function_values_to_file, print_variables_to_file
>>>>>>> develop:examples/multiobjective/preferences/smpsorp_zdt4.py
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import InteractivePlot, Plot

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    swarm_size = 100

    reference_point = [[0.1, 0.8],[0.6, 0.1]]
    archives_with_reference_points = []

    for point in reference_point:
        archives_with_reference_points.append(
            CrowdingDistanceArchiveWithReferencePoint(int(swarm_size / len(reference_point)), point)
        )

    max_evaluations = 50000
    algorithm = SMPSORP(
        problem=problem,
        swarm_size=swarm_size,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        reference_points=reference_point,
        leaders=archives_with_reference_points,
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
