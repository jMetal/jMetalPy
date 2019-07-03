<<<<<<< HEAD
from setuptools import find_packages
from setuptools import setup

setup(
    name='jmetalpy',
    version='0.9.0',
    description='Python version of the jMetal framework',
=======
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

basedir = abspath(dirname(__file__))

with open(join(basedir, 'README.md'), encoding='utf-8') as f:
    README = f.read()

setup(
    name='jmetalpy',
    version='1.0.5',
    description='Python version of the jMetal framework',
    long_description=README,
    long_description_content_type='text/markdown',
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
    author='Antonio J. Nebro',
    author_email='antonio@lcc.uma.es',
    maintainer='Antonio J. Nebro, Antonio Benitez-Hidalgo',
    maintainer_email='antonio@lcc.uma.es, antonio.b@uma.es',
    license='MIT',
    url='https://github.com/jMetal/jMetalPy',
    packages=find_packages(exclude=['test_']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=[
        'tqdm',
<<<<<<< HEAD
        'numpy==1.16.0',
        'pandas==0.23.4',
        'scipy==1.1.0',
        'pyspark==2.4.3',
        'ipython',
        'holoviews==1.10.9',
        'plotly==3.3.0',
        'matplotlib==3.0.2',
        'statsmodels==0.9.0',
        'dask[complete]==1.0.0'
=======
        'ipython',
        'scipy==1.1.0',
        'numpy>=1.16.0',
        'pandas>=0.24.2',
        'pyspark>=2.4.0',
        'plotly>=3.3.0',
        'matplotlib>=3.0.2',
        'statsmodels>=0.9.0',
        'dask[complete]==1.2.2',
        'distributed==1.28.1'
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
    ],
    tests_require=[
        'mockito',
        'PyHamcrest',
        'mock'
    ]
)
