Getting Started
================

Welcome to jMetalPy! This section will help you get up and running quickly.

Installation
------------

Via pip:

.. code-block:: console

    $ pip install jmetalpy  # or "jmetalpy[distributed]"

.. note:: Alternative installation options:

    .. code-block:: console

        $ pip install "jmetalpy[core]"        # Core components only
        $ pip install "jmetalpy[docs]"        # Documentation building
        $ pip install "jmetalpy[distributed]" # Parallel computing
        $ pip install "jmetalpy[complete]"    # All dependencies

Via source code:

.. code-block:: console

    $ git clone https://github.com/jMetal/jMetalPy.git
    $ python setup.py install

Quick Start
-----------

Here's a simple example to get you started with jMetalPy:

.. code-block:: python

    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.operator.selection import BinaryTournamentSelection
    from jmetal.problem import ZDT1
    from jmetal.util.termination_criterion import StoppingByEvaluations

    # Define the problem
    problem = ZDT1()

    # Configure the algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    # Run the algorithm
    algorithm.run()
    solutions = algorithm.get_result()

    # Print results
    print(f"Found {len(solutions)} solutions")

First Steps
-----------

.. toctree::
   :maxdepth: 2
   :caption: Essential tutorials:

   tutorials/your-first-optimization
   tutorials/understanding-problems
   tutorials/choosing-algorithms
   tutorials/analyzing-results

What's Next?
------------

Once you've completed the quick start:

1. **Explore the User Guide** for comprehensive tutorials and examples
2. **Browse the API Reference** for detailed technical documentation  
3. **Check Advanced Topics** for specialized use cases
4. **Join the Community** - contribute or ask questions

Key Concepts
------------

Before diving deeper, familiarize yourself with these core concepts:

- **Problems**: Define what you want to optimize
- **Algorithms**: Methods to find optimal solutions
- **Operators**: Building blocks for algorithms (crossover, mutation, selection)
- **Quality Indicators**: Metrics to evaluate solution quality
- **Experiments**: Framework for systematic algorithm comparison