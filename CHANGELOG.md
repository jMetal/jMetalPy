# History
## Last changes (July 12th 2017)
* The time of execution and evaluation number now are shown in the live plot. Furthermo::
* Several fixes regarding wrong imports and unused attributes.

## Last changes (July 11th 2017)
* Now It's possible to get to directly access the coords (x,y) of a point in a live plot by a mouse click. ~~Note: This still needs some changes in order to work properly.~~

## Last changes (July 9th 2017)
* New class for [graphics](jmetal/util/graphic.py).
* New [observer](jmetal/component/observer.py) for graphical display of algorithm. 
* Added [CHANGELOG](CHANGELOG.md) file.

## Last changes (July 7th 2017)
* New methods for plotting the solution list (`plot_scatter` and `plot_scatter_real_time`).
* New decorator for computing execution time of any method. Usage: [`from jmetal.util.time import get_time_of_exectuion`](jmetal/util/time.py) and add `@get_time_of_execution` before any method or function.
* Several improvements regarding [PEP8](resources/pages/code_style.md) code style guide.
* Updated [TODO.md](TODO.md) and added [CONTRIBUTING.cmd](CONTRIBUTING.md) file. 
* Updated requirements.

## Last changes (July 4th 2017)
* The algorithm [NSGA-II](jmetal/algorithm/multiobjective/nsgaii.py) has been implemented
* Examples of configuring and running all the included algorithms are located in the [jmetal.runner](https://github.com/jMetal/jMetalPy/tree/master/jmetal/runner) package.

## Last changes (June 1st 2017)
* The package organization has been simplified to make it more "Python-ish". The former oarganization was a clone of the original Java-based jMetal project.
* The [`EvolutionaryAlgorithm`](jmetal/core/algorithm.py) class interits from `threading.Thread`, so any evolutionary algorithm can run as a thread. This class also contains an `Observable` field, allowing observer entities to register to be notified with algorithm information. 
* [Four examples](jmetal/runner) of configuring and running three different single-objective algorithms are provided.

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