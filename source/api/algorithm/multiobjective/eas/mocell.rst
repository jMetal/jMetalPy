MOCell
========================

Examples
------------------------------------

.. code-block:: python

    from jmetal.algorithm.multiobjective.mocell import MOCell
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT4
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.util.neighborhood import C9
    from jmetal.util.solution import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT4()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT4.pf')

    max_evaluations = 25000
    algorithm = MOCell(
        problem=problem,
        population_size=100,
        neighborhood=C9(10, 10),
        archive=CrowdingDistanceArchive(100),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

API
---------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.mocell
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: R
