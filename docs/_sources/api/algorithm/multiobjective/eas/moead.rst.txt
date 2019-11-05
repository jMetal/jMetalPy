MOEA/D
========================

Examples
------------------------------------

Standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.moead import MOEAD
    from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
    from jmetal.problem import LZ09_F2
    from jmetal.util.aggregative_function import Tschebycheff
    from jmetal.util.solutions import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = LZ09_F2()
    problem.reference_front = read_solutions(filename='resources/reference_front/LZ09_F2.pf')

    max_evaluations = 150000

    algorithm = MOEAD(
        problem=problem,
        population_size=300,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

Epsilon (MOEADIEpsilon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.moead import MOEADIEpsilon
    from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
    from jmetal.problem.multiobjective.lircmop import LIRCMOP2
    from jmetal.util.aggregative_function import Tschebycheff
    from jmetal.util.solutions import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = LIRCMOP2()
    problem.reference_front = read_solutions('resources/reference_front/LIRCMOP2.pf')

    max_evaluations = 300000

    algorithm = MOEADIEpsilon(
        problem=problem,
        population_size=300,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

API
--------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.moead
   :members:
   :undoc-members:
   :show-inheritance: