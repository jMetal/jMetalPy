from setuptools import find_packages
from setuptools import setup

setup(
    name='jmetalpy',
    version='0.5.5',
    description='Python version of the jMetal framework',
    author='Antonio J. Nebro',
    author_email='antonio@lcc.uma.es',
    maintainer='Antonio J. Nebro, Antonio Benitez-Hidalgo',
    maintainer_email='antonio@lcc.uma.es, antonio.b@uma.es',
    license='MIT',
    url='https://github.com/jMetal/jMetalPy',
    long_description=open('README.md').read(),
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
        'numpy==1.13.1',
        'pandas==0.23.4',
        'scipy==1.1.0',
        'ipython',
        'holoviews==1.10.9',
        'plotly==3.3.0',
        'matplotlib==3.0.2',
        'statsmodels'
    ],
    tests_require=[
        'mockito',
        'PyHamcrest',
        'mock'
    ]
)
