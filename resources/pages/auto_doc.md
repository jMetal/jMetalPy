# Create automatic documentation files with Sphinx

First, you need to know how to correctly document your code.

## How to write docs directly from source code

It is **important** to follow these simple rules in order to automatically create good documentation for the project.

When you create a new module file (testDoc.py in this example), you should mention it using this format:

```python
"""
.. module:: testDoc
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Andrew Carter <andrew@invalid.com>


"""


class testDoc(object):
    """We use this as a public class example class.

    This class is ruled by the very trendy important method :func:`public_fn_with_sphinxy_docstring`.

    .. note::

       An example of intersphinx is this: you **cannot** use :mod:`pickle` on this class.

    """

    def __init__(self, foo: str, bar: str):
        """A really simple class.

        Args:
           foo (str): We all know what foo does.
           bar (str): Really, same as foo.

        """
        self.__foo = foo
        self.__bar = bar
```

This code snippet generates the following documentation:
<br/>
<br/>
![jMetal architecture](../../resources/class_header.png)

Now, you can document your methods using the following sintax:

 ```python
    def public_fn_with_sphinxy_docstring(self, name: str, state: bool = False) -> int:
        """This function does something.

        :param name: The name to use.
        :type name: str.
        :param state: Current state to be in.
        :type state: bool.
        :returns:  int -- the return code.
        :raises: AttributeError, KeyError

        """
        return 0

    def public_fn_without_docstring(self):
        return True
```

And the produced output doc will be:

![jMetal architecture](../../resources/method_way_sphinx.png)

As you may notice, if you don't use any docstring, the method documentation will be empty.

In addition, if you only use "::members", even though you have a docstring, it won't be imported in the documentation.

For example, this chunk of code does not produce any output doc:

 ```python
    def __private_fn_with_docstring(self, foo: str, bar: str = 'baz', foobarbas: int = None) -> int:
        """I have a docstring, but won't be imported if you just use ``:members:``.
        """
        return 20
```
## How to compile and produce the docs

After you have properly commented your code, you are now able to generate the documentation.

> Note: You need to have previously installed the Sphinx dependency.

In order to do that, from the root directory of **jMetalPy** you have to change your directory to the **auto-docs** one.

```sh
$ cd auto-docs/
```
Now, you can generate all the .rst files (which is the documentation project "skeleton").

```sh
$ sphinx-apidoc -f -o source/ ../jmetal/
```

After that, if you want to produce your docs in *html* format you onle have to run this shell command.

```sh
$ make html
```
Your documentation is now served in this path

```
jMetalPy/auto-docs/build/html
```

Inside that folder it exists a file called *index.html*. If you open it using a web browser, you will be able to visualize the generated docs.

You should clean this folder if you want to perform another compilation. You can perform that operation with this command:

```sh
$ make clean
```

> Note: Whenever you create another module or file (.py), if you want to add it to the docs, you have to re-run *sphinx-apidoc -f -o source/ ../jmetal/* inside the *auto-docs* folder.