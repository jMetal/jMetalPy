from setuptools import setup, find_packages

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
    classifiers=[
      'Development Status :: 3 - Alpha',

      'Intended Audience :: Science/Research',

      'License :: OSI Approved :: MIT License',

      'Topic :: Scientific/Engineering :: Artificial Intelligence',

      'Programming Language :: Python :: 3.6'],

    packages=find_packages(exclude=['test_']),
)
