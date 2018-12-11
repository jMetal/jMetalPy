<p align="center">
  <br/>
  <img src=docs/source/jmetalpy.png alt="jMetalPy">
  <br/>
</p>

# Python version of the jMetal framework
[![Build Status](https://img.shields.io/travis/jMetal/jMetalPy.svg?style=flat-square)](https://travis-ci.org/jMetal/jMetalPy)
[![Read the Docs](https://img.shields.io/readthedocs/jmetalpy.svg?style=flat-square)](https://readthedocs.org/projects/jmetalpy/)
[![PyPI License](https://img.shields.io/pypi/l/jMetalPy.svg?style=flat-square)]()
[![PyPI Python version](https://img.shields.io/pypi/pyversions/jMetalPy.svg?style=flat-square)]()
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jMetal/jMetalPy/develop)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation
To download jMetalPy just clone the Git repository hosted in GitHub:
```bash
$ git clone https://github.com/jMetal/jMetalPy.git
$ python setup.py install
```

Alternatively, you can install it with `pip`:
```bash
$ pip install jmetalpy
```

## Usage
Examples of configuring and running all the included algorithms are located [in the docs](https://jmetalpy.readthedocs.io/en/latest/examples.html).

## Features
The current release of jMetalPy (v0.5.5) contains the following components:

* Algorithms: random search, GA, EA, [NSGA-II](https://jmetalpy.readthedocs.io/en/latest/examples/ea.html#nsga-ii), [SMPSO](https://jmetalpy.readthedocs.io/en/latest/examples/pso.html#smpso), [SMPSO/RP](https://jmetalpy.readthedocs.io/en/latest/examples/pso.html#smpso-rp), [MOEA/D](https://jmetalpy.readthedocs.io/en/latest/examples/ea.html#moea-d) (and steady-state variants)
* Benchmark problems:
  * Singleobjective:  unconstrained (OneMax, Sphere, SubsetSum).
  * Multiobjective: ZDT1-6, DTLZ1-2, LZ09, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2, SubsetSum), constrained (Srinivas, Tanaka).
* Encodings: real, binary.
* A full range of genetic operators.
* Quality indicators: hypervolume.
* [Experiment class for performing studies](https://jmetalpy.readthedocs.io/en/latest/examples/experiment.html).
* Pareto front plotting for problems with two or more objectives (as scatter plot/parallel coordinates).

<p align="center">
  <br/>
  <img src=docs/source/2D.gif width=600 alt="Scatter plot 2D">
  <br/>
  <img src=docs/source/3D.gif width=600 alt="Scatter plot 3D">
  <br/>
  <img src=docs/source/p-c.gif width=600 alt="Parallel coordinates">
  <br/>
</p>

## License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.
