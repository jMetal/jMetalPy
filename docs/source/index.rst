.. jMetalPy documentation master file, created by
   sphinx-quickstart on Fri May  4 10:10:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

jMetalPy: Python version of the jMetal framework
================================================

.. warning:: Documentation is WIP!! Some information may be missing.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples
   contributing
   about
   api/jmetal

Installation steps
------------------------

Via pip:

.. code-block:: console

    $ pip install jmetalpy

Via Github:

.. code-block:: console

    $ git clone https://github.com/jMetal/jMetalPy.git
    $ pip install -r requirements.txt
    $ python setup.py install

Features
------------------------
The current release of jMetalPy (v0.5.1) contains the following components:

* Algorithms: random search, NSGA-II, SMPSO, SMPSO/RP
* Benchmark problems: ZDT1-6, DTLZ1-2, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random)
* Quality indicators: hypervolume
* Density estimator: crowding distance
* Graphics: Pareto front plotting (2 or more objectives)
* Laboratory: Experiment class for performing studies.

