<p align="center">
  <br/>
  <img src=resources/jmetalpy.png alt="jMetalPy">
  <br/>
</p>

# jMetalPy: Python version of the jMetal framework
[![Build Status](https://img.shields.io/travis/jMetal/jMetalPy.svg?style=flat-square)](https://travis-ci.org/jMetal/jMetalPy)
[![Read the Docs](https://img.shields.io/readthedocs/jmetalpy.svg?style=flat-square)](https://readthedocs.org/projects/jmetalpy/)
[![PyPI License](https://img.shields.io/pypi/l/jMetalPy.svg?style=flat-square)]()
[![PyPI Python version](https://img.shields.io/pypi/pyversions/jMetalPy.svg?style=flat-square)]()

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
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
The current release of jMetalPy (v0.5.0) contains the following components:

* Algorithms: random search, NSGA-II, SMPSO, SMPSO/RP
* Problems: ZDT1-6, DTLZ1-2, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crosover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random)
* Quality indicators: hypervolume
* Density estimator: crowding distance
* Graphics: 2D/3D plotting in real-time

## Contributing
Please read [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute to the project.

## License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.
