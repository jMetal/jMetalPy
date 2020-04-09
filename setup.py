from os.path import abspath, dirname, join

from setuptools import find_packages, setup

basedir = abspath(dirname(__file__))

with open(join(basedir, 'README.md'), encoding='utf-8') as f:
    README = f.read()

install_requires = [
    'tqdm',
    'numpy>=1.16.0',
    'pandas>=0.24.2',
    'plotly>=3.3.0',
    'matplotlib>=3.0.2',
    'scipy>=1.3.0',
    'statsmodels>=0.9.0'
]
extras_require = {
    'core': install_requires,
    'docs': install_requires + ['jupyter', 'nbsphinx'],
    'distributed': install_requires + ['dask[complete]>=1.2.2', 'distributed>=1.28.1', 'pyspark>=2.4.0']
}
extras_require['complete'] = {v for req in extras_require.values() for v in req}

setup(
    name='jmetalpy',
    version='1.5.5',
    description='Python version of the jMetal framework',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Antonio J. Nebro',
    author_email='antonio@lcc.uma.es',
    maintainer='Antonio J. Nebro, Antonio Benitez-Hidalgo',
    maintainer_email='antonio@lcc.uma.es, antonio.benitez@lcc.uma.es',
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
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=[
        'mockito',
        'PyHamcrest',
        'mock'
    ]
)
