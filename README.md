<p align="center">
  <br/>
  <img src=source/jmetalpy.png alt="jMetalPy">
  <br/>
</p>

[![Build Status](https://img.shields.io/travis/jMetal/jMetalPy/master.svg?style=flat-square)](https://travis-ci.org/jMetal/jMetalPy)
[![Documentation](https://img.shields.io/badge/docs-online-success?style=flat-square)](https://jmetal.github.io/jMetalPy/index.html)
[![PyPI License](https://img.shields.io/pypi/l/jMetalPy.svg?style=flat-square)]()
[![PyPI version](https://img.shields.io/pypi/v/jMetalPy.svg?style=flat-square)]()
[![PyPI Python version](https://img.shields.io/pypi/pyversions/jMetalPy.svg?style=flat-square)]()

A paper introducing jMetalPy is available at: https://doi.org/10.1016/j.swevo.2019.100598

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

### Installation

You can install the latest version of jMetalPy with `pip`, 
including parallel and distributed computing dependencies such as pySpark and Dask:

```console
pip install "jmetalpy[complete]"
```

You can also install the core functionality of the framework (which is often enough for most users):

```console
pip install "jmetalpy[core]"
```

Or simply:

```console
pip install jmetalpy
```

### Usage

```python
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.solution import get_non_dominated_solutions, read_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

problem = ZDT1()
problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max=25000)
)

algorithm.run()
front = get_non_dominated_solutions(algorithm.get_result())

# save front to file
print_function_values_to_file(front, 'FUN.NSGAII.ZDT1')
print_variables_to_file(front, 'VAR.NSGAII.ZDT1')
```

Examples of configuring and running all the included algorithms are located [in the documentation](https://jmetal.github.io/jMetalPy/multiobjective.algorithms.html).

### Features
The current release of jMetalPy (v1.5.2) contains the following components:

* Algorithms: local search, genetic algorithm, evolution strategy, simulated annealing, random search, NSGA-II, NSGA-III, SMPSO, OMOPSO, MOEA/D, MOEA/D-DRA, MOEA/D-IEpsilon, GDE3, SPEA2, HYPE, IBEA. Preference articulation-based algorithms (G-NSGA-II, G-GDE3, G-SPEA2, SMPSO/RP); Dynamic versions of NSGA-II, SMPSO, and GDE3.
* Parallel computing based on Apache Spark and Dask.
* Benchmark problems: ZDT1-6, DTLZ1-2, FDA, LZ09, LIR-CMOP, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary, permutations.
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random).
* Quality indicators: hypervolume, additive epsilon, GD, IGD.
* Pareto front approximation plotting in real-time, static or interactive.
* Experiment class for performing studies either alone or alongside [jMetal](https://github.com/jMetal/jMetal).
* Pairwise and multiple hypothesis testing for statistical analysis, including several frequentist and Bayesian testing methods, critical distance plots and posterior diagrams.

<p align="center">
  <br/>
  <img src=source/_static/2D.gif width=600 alt="Scatter plot 2D">
  <br/>
  <img src=source/_static/3D.gif width=600 alt="Scatter plot 3D">
  <br/>
  <img src=source/_static/p-c.gif width=600 alt="Parallel coordinates">
  <br/>
  <br/>
  <img src=source/_static/chordplot.gif width=400 alt="Interactive chord plot">
  <br/>
</p>

### License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.