Examples of running PSOs
========================

SMPSO with standard settings
------------------------------------

.. code-block:: python

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


SMPSO/RP with standard settings
------------------------------------

.. code-block:: python

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

    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()
