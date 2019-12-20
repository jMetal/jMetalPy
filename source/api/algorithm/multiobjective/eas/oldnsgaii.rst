NSGA-II
========================

Examples
------------------------------------

Standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT1
    from jmetal.util.solution import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

Distributed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: This requires some extra dependencies

.. code-block:: python

    from dask.distributed import Client
    from distributed import LocalCluster

    from examples.multiobjective.parallel.zdt1_modified import ZDT1Modified
    from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
    from jmetal.operator import PolynomialMutation, SBXCrossover
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1Modified()

    # setup Dask client
    client = Client(LocalCluster(n_workers=24))

    ncores = sum(client.ncores().values())
    print(f'{ncores} cores available')

    # creates the algorithm
    max_evaluations = 25000

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        number_of_cores=ncores,
        client=client
    )

    algorithm.run()
    front = algorithm.get_result()

Dynamic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
    from jmetal.operator import PolynomialMutation, SBXCrossover
    from jmetal.problem.multiobjective.fda import FDA2
    from jmetal.util.observable import TimeCounter
    from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = FDA2()

    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('front_vis'))
    algorithm.observable.register(observer=WriteFrontToFileObserver('front_files'))

    algorithm.run()

Preference point-based (gNSGA-II)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT2
    from jmetal.util.solutions import read_solutions
    from jmetal.util.solutions.comparator import GDominanceComparator
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT2()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT2.pf')

    reference_point = [0.2, 0.5]

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        dominance_comparator=GDominanceComparator(reference_point),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()


API
------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.nsgaii
   :members:
   :undoc-members:
   :show-inheritance:
