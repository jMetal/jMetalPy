OMOPSO
========================

Examples
------------------------------------

.. code-block:: python

    from jmetal.algorithm.multiobjective.omopso import OMOPSO
    from jmetal.operator import UniformMutation
    from jmetal.operator.mutation import NonUniformMutation
    from jmetal.problem import ZDT1
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.util.solution import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    mutation_probability = 1.0 / problem.number_of_variables
    max_evaluations = 25000
    swarm_size = 100

    algorithm = OMOPSO(
        problem=problem,
        swarm_size=swarm_size,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.5),
        non_uniform_mutation=NonUniformMutation(mutation_probability, perturbation=0.5,
                                                max_iterations=int(max_evaluations / swarm_size)),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

API
---------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.omopso
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: R
