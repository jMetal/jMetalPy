SPEA2
========================

Examples
------------------------------------

Standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.spea2 import SPEA2
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT1
    from jmetal.util.solutions import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    max_evaluations = 20000
    algorithm = SPEA2(
        problem=problem,
        population_size=40,
        offspring_population_size=40,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

Preference point-based (gSPEA2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.spea2 import SPEA2
    from jmetal.lab.visualization import Plot, InteractivePlot
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT1
    from jmetal.util.solutions import read_solutions
    from jmetal.util.solutions.comparator import GDominanceComparator
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    reference_point = [0.4, 0.6]

    max_evaluations = 25000
    algorithm = SPEA2(
        problem=problem,
        population_size=40,
        offspring_population_size=40,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        dominance_comparator=GDominanceComparator(reference_point)
    )

    algorithm.run()
    front = algorithm.get_result()

API
--------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.spea2
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: R
