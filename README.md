# jMetalPy: Python version of the jMetal framework

I have just started a new project called jMetalPy. The initial idea is not to write the whole jMetal proyect in Python, but to "have fun": I'm starting with Python, and to learn this programming language I think that using jMetal as a case study would be nice.

Any ideas about how the structure the project, coding style, useful tools (I'm using PyCharm), or links to related projects are welcome. The starting point is the jMetal architecture:

![jMetal architecture](resources/jMetal5UML.png)


---


# Last changes (July 4th 2017)
* The package organization has been simplified to make it more "Python-ish". The former oarganization was a clone of the original Java-based jMetal project.
* The [`EvolutionaryAlgorithm`](https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/algorithm.py) class interits from `threading.Thread`, the any evolutionary algorithm can run as a thread. This class also contains an `Observable` field, so observer entities can register and will notified with algorithm information. 
* [Four examples](https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/algorithm.py) of configuring and running three different single-objective algorithms are provided.

# Current status (July 4th 2017)
The current implementation contains the following features: 
* Algorithms
  * Multi-objective
    * NSGA-II
  * Single-objective
    * (mu+lamba)Evolution Strategy
    * (mu,lamba)Evolution Strategy
    * Generational Genetic algorithm
* Problems (multi-objective)
  * Kursawe
  * Fonseca
  * Schaffer
  * Viennet2
* Problems (single-objective)
  * Sphere
  * OneMax
* Encoding
  * Float
  * Binary
* Operators
  * SBX Crossover
  * Single Point Crossover
  * Polynomial Mutation
  * Bit Flip Mutation
  * Simple Random Mutation
  * Null Mutation
  * Uniform Mutation
  * Binary Tournament Selection

# TODO list
* The documentation of the code is not complete
* Inclusion of integration tests for the algorithm implementations
* Full coverage with unit tests

# Contributing
  Contributions to the jMetalPy project are welcome. Please, take into account the following guidelines if possible:

  - [Git WorkFlow](resources/pages/workflow_git.md)
  - [Follow style guide for python code: PEP8](resources/pages/code_style.md)
  - [Object-oriented programming](resources/pages/poo.md)
  - [Incorporate the new features of Python 3.5](resources/pages/features_python3.md)
  - [Respect the initial structure](resources/pages/project_structure.md)
  - [How to create auto documentation using compatible code](resources/pages/auto_doc.md)
