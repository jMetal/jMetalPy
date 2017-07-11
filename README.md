# jMetalPy: Python version of the jMetal framework
[![Build Status](https://travis-ci.org/jMetal/jMetalPy.svg?branch=master)](https://travis-ci.org/jMetal/jMetalPy)

> jMetalPy is currently under heavy development!  

I have just started a new project called jMetalPy. The initial idea is not to write the whole jMetal proyect in Python, but to "have fun": I'm starting with Python, and to learn this programming language I think that using jMetal as a case study would be nice.

Any ideas about how the structure the project, coding style, useful tools (I'm using PyCharm), or links to related projects are welcome (see [CONTRIBUTING](https://github.com/jMetal/jMetalPy/blob/master/CONTRIBUTING.md)). The starting point is the jMetal architecture:

![jMetal architecture](resources/jMetal5UML.png)

---

# Table of Contents
- [Usage](#usage)
	- [Dependencies](#dependencies)
- [History](#history)
	- [Last changes (July 9th 2017)](#last-changes-july-9th-2017)
- [Contributing](#contributing)
- [License](#license)


# Usage
Examples of configuring and running all the included algorithms are located in the [jmetal.runner](https://github.com/jMetal/jMetalPy/tree/master/jmetal/runner) folder.

## Dependencies
With Python 3.6 installed, run:
```Bash
$ git clone https://github.com/benhid/jMetalPy.git
$ pip install -r requirements.txt
```

Also, some tests may need [hamcrest](https://github.com/hamcrest/PyHamcrest) in order to work:
```Bash
$ pip install PyHamcrest==1.9.0
$ python -m unittest discover
```

# History
See [CHANGELOG](CHANGELOG.md) for full version history.

## Last changes (July 11th 2017)
* Now It's possible to get to directly access the coords (x,y) of a point in a live plot by a mouse click. Note: This still needs some changes in order to work properly.

# Contributing
Please read [CONTRIBUTING](CONTRIBUTING.md) for details on how to contribute to the project.

# License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.
