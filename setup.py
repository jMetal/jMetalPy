from os.path import abspath, dirname, join

from setuptools import find_packages, setup

basedir = abspath(dirname(__file__))

with open(join(basedir, "README.md"), encoding="utf-8") as f:
    README = f.read()

with open(join(basedir, "jmetal", "__init__.py"), "r") as f:
    version_marker = "__version__ ="
    for line in f:
        if line.startswith(version_marker):
            _, VERSION = line.split(version_marker)
            VERSION = VERSION.strip().strip('"')
            break
    else:
        raise RuntimeError("Version not found on __init__")

install_requires = [
    "tqdm",
    "numpy>=1.16.0",
    "pandas>=0.24.2",
    "plotly>=3.3.0",
    "matplotlib>=3.0.2",
    "scipy>=1.3.0",
    "statsmodels>=0.9.0",
]
extras_require = {
    "core": install_requires,
    "dev": install_requires + ["isort", "black", "mypy"],
    "distributed": install_requires + ["dask[complete]>=1.2.2", "distributed>=1.28.1", "pyspark>=2.4.0"],
}
extras_require["complete"] = {v for req in extras_require.values() for v in req}
tests_require = ["mockito", "PyHamcrest"]

setup(
    name="jmetalpy",
    version=VERSION,
    description="Python version of the jMetal framework",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Antonio J. Nebro",
    author_email="antonio@lcc.uma.es",
    maintainer="Antonio J. Nebro, Antonio Benitez-Hidalgo",
    maintainer_email="antonio@lcc.uma.es, antoniobenitez@lcc.uma.es",
    license="MIT",
    url="https://github.com/jMetal/jMetalPy",
    packages=find_packages(exclude=["test_"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require,
)
