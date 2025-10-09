![jMetalPy](docs/source/jmetalpy.png)

[![CI](https://github.com/jMetal/jMetalPy/actions/workflows/ci.yml/badge.svg)](https://github.com/jMetal/jMetalPy/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/jMetalPy.svg)](https://pypi.org/project/jMetalPy/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.swevo.2019.100598-blue)](https://doi.org/10.1016/j.swevo.2019.100598)
[![PyPI License](https://img.shields.io/pypi/l/jMetalPy.svg)](https://pypi.org/project/jMetalPy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A paper introducing jMetalPy is available at: https://doi.org/10.1016/j.swevo.2019.100598

### Table of Contents
- [Installation](#installation)
- [Usage](#hello-world-)
- [Features](#features)
- [Changelog](#changelog)
- [License](#license)

## Installation

You can install the latest version of jMetalPy with `pip`, 

```console
pip install jmetalpy  # or "jmetalpy[distributed]"
```

<details><summary><b>Notes on installing with <tt>pip</tt></b></summary>
<p>

jMetalPy includes features for parallel and distributed computing based on [pySpark](https://spark.apache.org/docs/latest/api/python/index.html) and [Dask](https://dask.org/).

These (extra) dependencies are *not* automatically installed when running `pip`, which only comprises the core functionality of the framework (enough for most users):

```console
pip install jmetalpy
```

This is the equivalent of running: 

```console
pip install "jmetalpy[core]"
```

Other supported commands are listed next:

```console
pip install "jmetalpy[dev]"  # Install requirements for development
pip install "jmetalpy[distributed]"  # Install requirements for parallel/distributed computing
pip install "jmetalpy[complete]"  # Install all requirements
```

</p>
</details>

## Hello, world! 👋

Examples of configuring and running all the included algorithms are located [in the documentation](https://jmetal.github.io/jMetalPy).

```python
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations

problem = ZDT1()

algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=25000)
)

algorithm.run()
```

We can then proceed to explore the results:

```python
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file

front = get_non_dominated_solutions(algorithm.result())

# save to files
print_function_values_to_file(front, 'FUN.NSGAII.ZDT1')
print_variables_to_file(front, 'VAR.NSGAII.ZDT1')
```

Or visualize the Pareto front approximation produced by the algorithm:

```python
from jmetal.lab.visualization import Plot

plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='NSGAII-ZDT1', filename='NSGAII-ZDT1', format='png')
```

<img src=docs/source/_static/NSGAII-ZDT1.png width=450 alt="Pareto front approximation">

## Features
The current release of jMetalPy (v1.8.0) contains the following components:

* Algorithms: local search, genetic algorithm, evolution strategy, simulated annealing, random search, NSGA-II, NSGA-III, SMPSO, OMOPSO, MOEA/D, SMS-EMOA, MOEA/D-DRA, MOEA/D-IEpsilon, GDE3, SPEA2, HYPE, IBEA. Preference articulation-based algorithms (G-NSGA-II, G-GDE3, G-SPEA2, SMPSO/RP); Dynamic versions of NSGA-II, SMPSO, and GDE3.
* Parallel computing based on Apache Spark and Dask.
* Benchmark problems: ZDT1-6, DTLZ1-2, FDA, LZ09, LIR-CMOP, RWA, RE, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary, permutations.
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random).
* Quality indicators: hypervolume, additive epsilon, GD, IGD, IGD+.
* Pareto front approximation plotting in real-time, static or interactive.
* Experiment class for performing studies either alone or alongside [jMetal](https://github.com/jMetal/jMetal).
* Pairwise and multiple hypothesis testing for statistical analysis, including several frequentist and Bayesian testing methods, critical distance plots and posterior diagrams.

| ![Scatter plot 2D](docs/source/_static/2D.gif) | ![Scatter plot 3D](docs/source/_static/3D.gif) |
|-------------- | ----------------  |
| ![Parallel coordinates](docs/source/_static/p-c.gif) | ![Interactive chord plot](docs/source/_static/chordplot.gif) |

## Changelog

* [v.1.8.0] 
  * Add the SMS-EMOA algorithm (based on [moocore](https://multi-objective.github.io/moocore/python/).) 
  * Add the IGD+ quality indicator
  * Add new bencharmk problems:
    * [eqdtlz.py](https://github.com/jMetal/jMetalPy/blob/main/src/jmetal/problem/multiobjective/eqdtlz.py)
    * [misc.py](https://github.com/jMetal/jMetalPy/blob/main/src/jmetal/problem/multiobjective/misc.py)
    * Their reference fronts have been obtained with the [Reference Set Generator](https://doi.org/10.3390/math13101626)  method.
  * The project structure has been changed from [flat to src](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html). 
  * The Hypervolume quality indicator implementation relies now on the [moocore project](https://multi-objective.github.io/moocore/python/).
* [v1.7.0] Add RWA benchmark, refactor classes BinarySolution and BinaryProblem.
* [v1.6.0] Refactor class Problem, the single-objective genetic algorithm can solve constrained problems, performance improvements in NSGA-II, generation of Latex tables summarizing the results of the Wilcoxon rank sum test, added a notebook folder with examples.
* [v1.5.7] Use of linters for catching errors and formatters to fix style, minor bug fixes.
* [v1.5.6] Removed warnings when using Python 3.8.
* [v1.5.5] Minor bug fixes.
* [v1.5.4] Refactored quality indicators to accept numpy array as input parameter.
* [v1.5.4] Added [CompositeSolution](https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/solution.py#L111) class to support mixed combinatorial problems. [#69](https://github.com/jMetal/jMetalPy/issues/69)

## License

This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.
