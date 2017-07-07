# jMetalPy: Python version of the jMetal framework
[![Build Status](https://travis-ci.org/jMetal/jMetalPy.svg?branch=master)](https://travis-ci.org/jMetal/jMetalPy)

I have just started a new project called jMetalPy. The initial idea is not to write the whole jMetal proyect in Python, but to "have fun": I'm starting with Python, and to learn this programming language I think that using jMetal as a case study would be nice.

Any ideas about how the structure the project, coding style, useful tools (I'm using PyCharm), or links to related projects are welcome (see [CONTRIBUTING.md](https://github.com/jMetal/jMetalPy/blob/master/CONTRIBUTING.md)). The starting point is the jMetal architecture:

![jMetal architecture](resources/jMetal5UML.png)

---

# Usage
Examples of configuring and running all the included algorithms are located in the [jmetal.runner](https://github.com/jMetal/jMetalPy/tree/master/jmetal/runner) folder.

## Dependencies
With Python 3.6 installed, run:
```Bash
$ pip install -r requirements.txt
```

Also, some tests may need [hamcrest](https://github.com/hamcrest/PyHamcrest) in order to work:
```Bash
$ pip install PyHamcrest==1.9.0
```

# History
## Last changes (July 7th 2017)
* New methods for plotting the solution list (`plot_scatter` and `plot_scatter_real_time`).
* New decorator for computing execution time of any method. Usage: `from jmetal.util.time import get_time_of_exectuion` and add `@get_time_of_execution` before any method or function.
* Several improvements regarding [PEP8](https://github.com/jMetal/jMetalPy/blob/master/resources/pages/code_style.md) code style guide.
* Updated [TODO.md](https://github.com/jMetal/jMetalPy/blob/master/TODO.cmd) and added [CONTRIBUTING.cmd](https://github.com/jMetal/jMetalPy/blob/master/jmetal/CONTRIBUTING.cmd) file. 
* Updated requirements.

## Last changes (July 4th 2017)
* The algorithm [NSGA-II](https://github.com/jMetal/jMetalPy/blob/master/jmetal/algorithm/multiobjective/nsgaii.py) has been implemented
* Examples of configuring and running all the included algorithms are located in the [jmetal.runner](https://github.com/jMetal/jMetalPy/tree/master/jmetal/runner) package.

## Last changes (June 1st 2017)
* The package organization has been simplified to make it more "Python-ish". The former oarganization was a clone of the original Java-based jMetal project.
* The [`EvolutionaryAlgorithm`](https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/algorithm.py) class interits from `threading.Thread`, so any evolutionary algorithm can run as a thread. This class also contains an `Observable` field, allowing observer entities to register to be notified with algorithm information. 
* [Four examples](https://github.com/jMetal/jMetalPy/tree/master/jmetal/runner) of configuring and running three different single-objective algorithms are provided.

## Current status (as for July 4th 2017)
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

# Contributing
Please read [CONTRIBUTING.md](https://github.com/jMetal/jMetalPy/blob/master/CONTRIBUTING.md) for details of how to contribute to the project.

# License
This project is licensed under the terms of the MIT - see the [LICENSE](https://github.com/jMetal/jMetalPy/blob/master/LICENSE) file for details.
