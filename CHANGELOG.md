# Changelog

* [Current]
  * Added the multi-objective TSP problem
  * Added the Average Hausdorff Distance (AHD) quality indicator.
  * Added the ZDT benchmark problems. Contributed by Nicolás Rodríguez Uribe.
* [1.9.0]
  * Add RE benchmark
  * Refactor algorithm SPEA2
  * Refactor classes Solution, BinarySolution, and FloatSolution
  * Added Unimodal Normal Distribution Crossover (UNDX), BLX-Alpha, BLX-Alpha-Beta and Arithmetic crossover operators.
  * Added Levy flight and power-law mutation operators.
* [v.1.8.0]
  * Add the SMS-EMOA algorithm (based on [moocore](https://multi-objective.github.io/moocore/python/).)
  * Add the IGD+ quality indicator
  * Add new bencharmk problems:
    * [eqdtlz.py](https://github.com/jMetal/jMetalPy/blob/main/src/jmetal/problem/multiobjective/eqdtlz.py)
    * [misc.py](https://github.com/jMetal/jMetalPy/blob/main/src/jmetal/problem/multiobjective/misc.py)
    * Their reference fronts have been obtained with the [Reference Set Generator](https://doi.org/10.3390/math13101626) method.
  * The project structure has been changed from [flat to src](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html).
  * The Hypervolume quality indicator implementation relies now on the [moocore project](https://multi-objective.github.io/moocore/python/).
* [v1.7.0] Add RWA benchmark, refactor classes BinarySolution and BinaryProblem.
* [v1.6.0] Refactor class Problem, the single-objective genetic algorithm can solve constrained problems, performance improvements in NSGA-II, generation of Latex tables summarizing the results of the Wilcoxon rank sum test, added a notebook folder with examples.
* [v1.5.7] Use of linters for catching errors and formatters to fix style, minor bug fixes.
* [v1.5.6] Removed warnings when using Python 3.8.
* [v1.5.5] Minor bug fixes.
* [v1.5.4] Refactored quality indicators to accept numpy array as input parameter.
* [v1.5.4] Added [CompositeSolution](https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/solution.py#L111) class to support mixed combinatorial problems. [#69](https://github.com/jMetal/jMetalPy/issues/69)
