IBEA
========================

Examples
------------------------------------

.. code-block:: python

    from jmetal.algorithm.multiobjective.ibea import IBEA
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT1
    from jmetal.util.solutions import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    algorithm = IBEA(
        problem=problem,
        kappa=1.,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(25000)
    )

    algorithm.run()
    front = algorithm.get_result()

API
-------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.ibea
   :members:
   :undoc-members:
   :show-inheritance: