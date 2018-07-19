from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='jmetalpy',
    version='0.5.0',
    description='JMetalPy. Python version of the jMetal framework',
    author='Antonio J. Nebro',
    author_email='antonio@lcc.uma.es',
    maintainer='Antonio J. Nebro',
    maintainer_email='antonio@lcc.uma.es',
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
        'numpy',
        'pandas',
        'plotly',
        'matplotlib==2.0.2',
    ],
    tests_require=[
        'mockito'
        'PyHamcrest',
    ]
)
