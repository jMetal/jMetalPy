.. jMetalPy documentation master file, created by
   sphinx-quickstart on Fri May  4 10:10:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

jMetalPy: Python version of the jMetal framework
====================================

.. warning:: Documentation is in process!! Some information may be missing.

Installation
------------

Via pip:

.. code-block:: console

    $ pip install jmetalpy

Via Github:

.. code-block:: console

    $ git clone https://github.com/jMetal/jMetalPy.git
    $ pip install -r requirements.txt

Basic Usage
-----------

.. code-block:: python

    problem = ZDT1()
    algorithm = NSGAII[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(1.0/problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()))

    algorithm.run()
    result = algorithm.get_result()

