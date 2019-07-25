Examples of running PSOs
========================

SMPSO with standard settings
------------------------------------

.. code-block:: python

<<<<<<< HEAD
    from jmetal.algorithm.multiobjective.smpso import SMPSO
    from jmetal.component import ProgressBarObserver, CrowdingDistanceArchive
    from jmetal.operator import Polynomial
    from jmetal.problem import DTLZ1
    from jmetal.util.graphic import FrontPlot

    problem = DTLZ1(number_of_objectives=5)

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100)
    )

    algorithm.run()
    front = algorithm.get_result()

    pareto_front = FrontPlot(plot_title='SMPSO-DTLZ1-5', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.to_html(filename='SMPSO-DTLZ1-5')
=======
    problem = DTLZ1(number_of_objectives=5)

    max_evaluations = 25000
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()


SMPSO/RP with standard settings
------------------------------------

.. code-block:: python

<<<<<<< HEAD
    from jmetal.algorithm.multiobjective.smpso import SMPSORP
    from jmetal.component import CrowdingDistanceArchiveWithReferencePoint
    from jmetal.operator import Polynomial
    from jmetal.problem import ZDT1
    from jmetal.util.graphic import FrontPlot
    from jmetal.util.solution_list import read_front


    def points_to_solutions(points):
        solutions = []
        for i, _ in enumerate(points):
            point = problem.create_solution()
            point.objectives = points[i]
            solutions.append(point)

        return solutions

    problem = ZDT1()
    problem.reference_front = read_front(file_path='/resources/reference_front/{}.pf'.format(problem.get_name()))

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
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        reference_points=reference_points,
        leaders=archives_with_reference_points
    )

    algorithm.run()
    front = algorithm.get_result()

    pareto_front = FrontPlot(plot_title='SMPSORP-ZDT1', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.update(points_to_solutions(reference_points), legend='reference points')
    pareto_front.to_html(filename='SMPSORP-ZDT1')
=======
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/{}.pf'.format(problem.get_name()))

    swarm_size = 100

    reference_point = [[0.1, 0.8],[0.8, 0.2]]
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
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f

    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()
