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
The current release of jMetalPy (v0.5.5) contains the following components:

* Algorithms: random search, NSGA-II, SMPSO, SMPSO/RP.
* Benchmark problems:
  * Singleobjective:  unconstrained (OneMax, Sphere, SubsetSum).
  * Multiobjective: ZDT1-6, DTLZ1-2, LZ09, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2, SubsetSum), constrained (Srinivas, Tanaka).
* Encodings: real, binary.
* A full range of genetic operators.
* Quality indicators: hypervolume.
* Experiment class for performing studies.
* Pareto front plotting for problems with two or more objectives (as scatter plot/parallel coordinates).

.. figure:: visualization.png
   :alt: Visualization