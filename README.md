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

> jMetalPy is currently under heavy development!. The current version is 0.5.0

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

Alternatively, you can install with `pip`:
```bash
$ pip install jmetalpy
```

## Usage
Examples of configuring and running all the included algorithms are located [in the docs](http://jmetalpy.readthedocs.io/en/develop/examples.html).

## Features
The current release of jMetalPy contains the following components:

* Algorithms: Random search, NSGA-II, SMPSO, SMPSO/RP
* Problems: ZDT1-6, DTLZ1-2
* Encodings: real, binary
* Operators: binary tournament, single-point crossover, SBX crossover, bit-blip mutation, polynomial mutation, uniform mutation
* Quality indicators: hypervolume
* Density estimator: crowding distance
* Graphics: 2D/3D plotting in real-time

## Contributing
Please read [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute to the project.

## License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.
