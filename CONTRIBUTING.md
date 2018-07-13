# Development guidelines

Contributions to the jMetalPy project are welcome. Please, take into account the following guidelines (all developers should follow these guidelines):

- [Git WorkFlow](resources/pages/workflow_git.md)
- [Follow style guide for python code: PEP8](resources/pages/code_style.md)
- [Object-oriented programming](resources/pages/poo.md)
- [Incorporate the new features of Python 3.5](resources/pages/features_python3.md)
- [Respect the initial structure](resources/pages/project_structure.md)
- [How to create auto documentation using compatible code](resources/pages/auto_doc.md)
- [Performance analysis of Python](resources/pages/profiling.md)

# Documentation

To generate the documentation, install [Sphinx](http://www.sphinx-doc.org/en/master/) by running:

```bash
$ pip install sphinx
$ pip install sphinx_rtd_theme
```

And then `cd` to `/docs` and run:

```bash
$ sphinx-apidoc -f -o source/ ../jmetal/
$ make html
```