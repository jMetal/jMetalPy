jMetalPy: Python version of the jMetal framework
================================================

.. warning:: Documentation is work in progress!! Some information may be missing or incomplete.

.. table::

   +---------------------+----------+
   | **Target doc**      |  v1.5.3  |
   +---------------------+----------+

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

    $ pip install jmetalpy  # or "jmetalpy[distributed]"

.. note:: Alternatively, you can use one of these instead:

    .. code-block:: console

        $ pip install "jmetalpy[core]"  # Install core components of the framework (equivalent to `pip install jmetalpy`)
        $ pip install "jmetalpy[docs]"  # Install requirements for building docs
        $ pip install "jmetalpy[distributed]"  # Install requirements for parallel/distributed computing
        $ pip install "jmetalpy[complete]"  # Install all dependencies

Via source code:

.. code-block:: console

    $ git clone https://github.com/jMetal/jMetalPy.git
    $ python setup.py install

Summary of features
------------------------
The current release of jMetalPy (v1.5.3) contains the following components:

* Algorithms: local search, genetic algorithm, evolution strategy, simulated annealing, random search, NSGA-II, NSGA-III, SMPSO, OMOPSO, MOEA/D, MOEA/D-DRA, MOEA/D-IEpsilon, GDE3, SPEA2, HYPE, IBEA. Preference articulation-based algorithms (G-NSGA-II, G-GDE3, G-SPEA2, SMPSO/RP); Dynamic versions of NSGA-II, SMPSO, and GDE3.
* Parallel computing based on Apache Spark and Dask.
* Benchmark problems: ZDT1-6, DTLZ1-2, FDA, LZ09, LIR-CMOP, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary, permutations.
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random).
* Quality indicators: hypervolume, additive epsilon, GD, IGD.
* Pareto front approximation plotting in real-time, static or interactive.
* Experiment class for performing studies either alone or alongside jMetal.
* Pairwise and multiple hypothesis testing for statistical analysis, including several frequentist and Bayesian testing methods, critical distance plots and posterior diagrams.


Cite us
------------------------

.. code-block:: LaTeX

   @article{BENITEZHIDALGO2019100598,
      title = "jMetalPy: A Python framework for multi-objective optimization with metaheuristics",
      journal = "Swarm and Evolutionary Computation",
      pages = "100598",
      year = "2019",
      issn = "2210-6502",
      doi = "https://doi.org/10.1016/j.swevo.2019.100598",
      url = "http://www.sciencedirect.com/science/article/pii/S2210650219301397",
      author = "Antonio Benítez-Hidalgo and Antonio J. Nebro and José García-Nieto and Izaskun Oregi and Javier Del Ser",
      keywords = "Multi-objective optimization, Metaheuristics, Software framework, Python, Statistical analysis, Visualization",
   }
