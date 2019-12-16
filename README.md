<p align="center">
  <br/>
  <img src=source/jmetalpy.png alt="jMetalPy">
  <br/>
</p>

# jMetalPy: Python version of the jMetal framework
[![Build Status](https://img.shields.io/travis/jMetal/jMetalPy/master.svg?style=flat-square)](https://travis-ci.org/jMetal/jMetalPy)
[![Documentation](https://img.shields.io/badge/docs-online-success?style=flat-square)](https://jmetal.github.io/jMetalPy/index.html)
[![PyPI License](https://img.shields.io/pypi/l/jMetalPy.svg?style=flat-square)]()
[![PyPI Python version](https://img.shields.io/pypi/pyversions/jMetalPy.svg?style=flat-square)]()

A paper introducing JMetalPy is available at: https://doi.org/10.1016/j.swevo.2019.100598

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation
To download jMetalPy just clone the Git repository hosted in GitHub:

```console
git clone https://github.com/jMetal/jMetalPy.git
python setup.py install
```

Alternatively, you can install it with `pip`:

```console
pip install jmetalpy
```

## Usage
Examples of configuring and running all the included algorithms are located [in the `examples` folder](examples).

## Features
The current release of jMetalPy (v1.5.2) contains the following components:

* Algorithms: local search, genetic algorithm, evolution strategy, simulated annealing, random search, NSGA-II, NSGA-III, SMPSO, OMOPSO, MOEA/D, MOEA/D-DRA, MOEA/D-IEpsilon, GDE3, SPEA2, HYPE, IBEA. Preference articulation-based algorithms (G-NSGA-II, G-GDE3, G-SPEA2, SMPSO/RP); Dynamic versions of NSGA-II, SMPSO, and GDE3.
* Parallel computing based on Apache Spark and Dask.
* Benchmark problems: ZDT1-6, DTLZ1-2, FDA, LZ09, LIR-CMOP, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary, permutations.
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random).
* Quality indicators: hypervolume, additive epsilon, GD, IGD.
* Pareto front plotting for problems with two or more objectives (as scatter plot/parallel coordinates/chordplot) in real-time, static or interactive.
* Experiment class for performing studies either alone or alongside jMetal.
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

## License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.