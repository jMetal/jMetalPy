GDE3
========================

Examples
------------------------------------

Standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.gde3 import GDE3
    from jmetal.problem import ZDT1
    from jmetal.util.solution import read_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    algorithm = GDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(25000)
    )

    algorithm.run()
    front = algorithm.get_result()

Dynamic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.gde3 import DynamicGDE3
    from jmetal.problem.multiobjective.fda import FDA2
    from jmetal.util.observable import TimeCounter
    from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = FDA2()

    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    algorithm = DynamicGDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max=500)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('front_plot'))
    algorithm.observable.register(observer=WriteFrontToFileObserver('front_files'))

    algorithm.run()

Preference point-based (gGDE3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from jmetal.algorithm.multiobjective.gde3 import GDE3
    from jmetal.problem import ZDT2
    from jmetal.util.solutions import read_solutions
    from jmetal.util.solutions.comparator import GDominanceComparator
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT2()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT2.pf')

    max_evaluations = 25000

    algorithm = GDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        dominance_comparator=GDominanceComparator([0.5, 0.5])
    )

    algorithm.run()
    front = algorithm.get_result()

API
-------------------------------------------

.. automodule:: jmetal.algorithm.multiobjective.gde3
   :members:
   :undoc-members:
   :show-inheritance: