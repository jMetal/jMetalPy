# jMetalPy: Python version of the jMetal framework
[![Build Status](https://travis-ci.org/jMetal/jMetalPy.svg?branch=master)](https://travis-ci.org/jMetal/jMetalPy)

> jMetalPy is currently under heavy development!  

I started a new project called jMetalPy in February 2017. The initial idea was not to write the whole jMetal proyect in Python but to use it as a practical study to learn that programming language, although due to the interest of some researchers the goal of an usable jMetal version in Python is an ongoing work.

Any ideas about how the structure the project, coding style, useful tools (I'm using PyCharm), or links to related projects are welcome (see [CONTRIBUTING](https://github.com/jMetal/jMetalPy/blob/master/CONTRIBUTING.md)). The starting point is the jMetal architecture:

![jMetal architecture](resources/jMetal5UML.png)

---


# Table of Contents
- [Usage](#usage)
	- [Dependencies](#dependencies)
- [History](#history)
	- [Last changes (July 11th 2017)](#last-changes-july-11th-2017)
- [Contributing](#contributing)
- [License](#license)


# Usage
Examples of configuring and running all the included algorithms are located in the [jmetal.runner](https://github.com/jMetal/jMetalPy/tree/master/jmetal/runner) folder.

## Dependencies
With Python 3.6 installed, run:
```Bash
$ git clone https://github.com/jMetal/jMetalPy.git
$ pip install -r requirements.txt
```


# History
See [CHANGELOG](CHANGELOG.md) for full version history.

## Last changes (July 11th 2017)
* Now It's possible to get to directly access the coords (x,y) of a point in a live plot by a mouse click. Note: This still needs some changes in order to work properly.

# Contributing
Please read [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute to the project.

# License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.
