SMPSO
========================

Examples
------------------------------------

Standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.smpso import SMPSO
    from jmetal.operator import PolynomialMutation
    from jmetal.problem import DTLZ1
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.util.termination_criterion import StoppingByEvaluations

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


Dynamic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.smpso import DynamicSMPSO
    from jmetal.operator import PolynomialMutation
    from jmetal.problem.multiobjective.fda import FDA2
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.util.observable import TimeCounter
    from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = FDA2()

    time_counter = TimeCounter(delay=15)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicSMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('front_vis'))
    algorithm.observable.register(observer=WriteFrontToFileObserver('front_files'))

    algorithm.run()

Reference-point based (SMPSO/RP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.smpso import SMPSORP
    from jmetal.operator import PolynomialMutation
    from jmetal.problem import ZDT4
    from jmetal.util.archive import CrowdingDistanceArchiveWithReferencePoint
    from jmetal.util.solution import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT4()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT4.pf')

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


API
--------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.smpso
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: R, change_reference_point
