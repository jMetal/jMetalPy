jMetalPy: Python version of the jMetal framework
================================================

jMetalPy is a Python framework for multi-objective optimization with metaheuristics. 
It provides a comprehensive set of algorithms, problems, and utilities for solving 
complex optimization problems.

.. note:: 
   📚 **New to jMetalPy?** Start with the :doc:`getting-started` guide for a quick introduction.

.. warning:: 
   Documentation of jMetal 1.7.1 is a work in progress! Some information may be missing or outdated.

.. table::

   +---------------------+----------+
   | **Target doc**      |  v1.7.1  |
   +---------------------+----------+

Content
------------------------

.. toctree::
   :maxdepth: 1

   getting-started
   user-guide
   api-reference
   advanced-topics
   contributing
   about

Key Features
------------------------

jMetalPy (v1.7.1) provides:

**Algorithms**
   Local search, genetic algorithms, evolution strategies, simulated annealing, NSGA-II, NSGA-III, 
   SMPSO, OMOPSO, MOEA/D, SMS-EMOA, GDE3, SPEA2, HYPE, IBEA, and preference-based variants.

**Problem Types**
   Benchmark problems (ZDT, DTLZ, FDA, LZ09, RE, RWA), constrained and unconstrained benchmark problems.

**Analysis Tools**
   Quality indicators (hypervolume, IGD, IGD+, epsilon), statistical testing, visualization, 
   and experimental frameworks.

**Advanced Features**
   Parallel computing (Apache Spark, Dask), real-time plotting, dynamic algorithms, 
   and integration with jMetal.

Quick Example
-------------

.. code-block:: python

   from jmetal.algorithm.multiobjective.nsgaii import NSGAII
   from jmetal.problem import ZDT1
   
   problem = ZDT1()
   algorithm = NSGAII(problem=problem, population_size=100)
   algorithm.run()
   solutions = algorithm.get_result()

Community & Support
-------------------

- 📖 **Documentation**: Comprehensive guides and API reference
- 💬 **Issues**: Report bugs and request features on GitHub
- 🤝 **Contributing**: Help improve jMetalPy - see :doc:`contributing`
- 📄 **Citation**: If you use jMetalPy in research, please cite our paper


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
