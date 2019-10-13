.. jMetalPy documentation master file, created by
   sphinx-quickstart on Fri Oct 11 10:56:26 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

jMetalPy: Python version of the jMetal framework
================================================

.. warning:: Documentation is work in progress!! Some information may be missing or incomplete.


Content
------------------------

.. toctree::
   :maxdepth: 1

   tutorials
   multiobjective.algorithms
   singleobjective.algorithms
   operators
   problems
   contributing
   about

Installation steps
------------------------

Via pip:

.. code-block:: console

    $ pip install jmetalpy

Via source code:

.. code-block:: console

    $ git clone https://github.com/jMetal/jMetalPy.git
    $ pip install -r requirements.txt
    $ python setup.py install

Features
------------------------
The current release of jMetalPy (v1.5.1) contains the following components:

* Algorithms: local search, genetic algorithm, evolution strategy, simulated annealing, random search, NSGA-II, NSGA-III, SMPSO, OMOPSO, MOEA/D, MOEA/D-DRA, MOEA/D-IEpsilon, GDE3, SPEA2, HYPE, IBEA. Preference articulation-based algorithms (G-NSGA-II, G-GDE3, G-SPEA2, SMPSO/RP); Dynamic versions of NSGA-II, SMPSO, and GDE3.
* Parallel computing based on Apache Spark and Dask.
* Benchmark problems: ZDT1-6, DTLZ1-2, FDA, LZ09, LIR-CMOP, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary, permutations.
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random).
* Quality indicators: hypervolume, additive epsilon, GD, IGD.
* Pareto front plotting for problems with two or more objectives (as scatter plot/parallel coordinates/chordplot) in real-time, static or interactive.
* Experiment class for performing studies either alone or alongside jMetal.
* Pairwise and multiple hypothesis testing for statistical analysis, including several frequentist and Bayesian testing methods, critical distance plots and posterior diagrams.


Cite us
------------------------

.. code-block:: LaTeX

   @article{DBLP:journals/corr/abs-1903-02915,
     author    = {Antonio Ben{\'{\i}}tez{-}Hidalgo and
                  Antonio J. Nebro and
                  Jos{\'{e}} Garc{\'{\i}}a{-}Nieto and
                  Izaskun Oregi and
                  Javier Del Ser},
     title     = {jMetalPy: a Python Framework for Multi-Objective Optimization with
                  Metaheuristics},
     journal   = {CoRR},
     volume    = {abs/1903.02915},
     year      = {2019},
     url       = {http://arxiv.org/abs/1903.02915},
     archivePrefix = {arXiv},
     eprint    = {1903.02915},
     timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
     biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-02915},
     bibsource = {dblp computer science bibliography, https://dblp.org}
   }
