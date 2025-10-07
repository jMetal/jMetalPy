Quality Indicator CLI
======================

The Quality Indicator CLI is a command-line interface for computing quality indicators between two fronts (solution front and reference front). This tool provides an easy way to evaluate the performance of multi-objective optimization algorithms.

Features
--------

This CLI tool supports the following quality indicators:

- **Additive Epsilon (epsilon)**: Measures the minimum additive factor needed to weakly dominate the reference front
- **Inverted Generational Distance (igd)**: Measures the average distance from reference points to the nearest solution
- **Inverted Generational Distance Plus (igdplus)**: IGD variant using dominance-based distance calculation
- **Hypervolume (hv)**: Volume of objective space dominated by the front
- **Normalized Hypervolume (nhv)**: Hypervolume normalized by the reference front's hypervolume
- **All indicators**: Compute all indicators at once

Installation
------------

The CLI is included with jMetalPy. No additional installation is required.

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

    python -m jmetal.util.quality_indicator_cli <front.csv> <reference.csv> <indicator> [options]

Examples
~~~~~~~~

**Compute IGD between two fronts:**

.. code-block:: bash

    python -m jmetal.util.quality_indicator_cli front.csv reference.csv igd

**Compute all indicators with custom reference point:**

.. code-block:: bash

    python -m jmetal.util.quality_indicator_cli front.csv reference.csv all --ref-point 2.0,2.0

**Normalize fronts and output as JSON:**

.. code-block:: bash

    python -m jmetal.util.quality_indicator_cli front.csv reference.csv all --normalize --format json

**Compute epsilon indicator only:**

.. code-block:: bash

    python -m jmetal.util.quality_indicator_cli front.csv reference.csv epsilon

Options
~~~~~~~

- ``--normalize``: Normalize both fronts using reference_only strategy
- ``--ref-point V1,V2,...``: Custom reference point for HV/NHV (overrides auto-generation)
- ``--format {text,json}``: Output format (default: text)
- ``--margin M``: Margin added when auto-building reference point (default: 0.1)
- ``-h, --help``: Show help message

File Format
-----------

CSV files should contain numeric data with one solution per row and one objective per column.

**Example front.csv:**

.. code-block:: text

    0.0,1.0
    0.2,0.8
    0.4,0.6
    0.6,0.4
    0.8,0.2
    1.0,0.0

**Example reference.csv:**

.. code-block:: text

    0.1,0.9
    0.3,0.7
    0.5,0.5
    0.7,0.3
    0.9,0.1

Output
------

Text Format (default)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Result (epsilon): 0.1
    Result (igd): 0.1414213562373095
    Result (igdplus): 0.1
    Result (hv): 0.84
    Result (nhv): -0.037037037037037

JSON Format
~~~~~~~~~~~

.. code-block:: json

    {
      "epsilon": 0.1,
      "igd": 0.1414213562373095,
      "igdplus": 0.1,
      "hv": 0.84,
      "nhv": -0.037037037037037
    }

Important Notes
---------------

Reference Points
~~~~~~~~~~~~~~~~

- **HV and NHV** require a reference point that is dominated by all solutions in the front
- If no reference point is provided, one is automatically generated using the maximum values of the reference front plus a margin
- For normalized data, the default reference point is ``[1.1, 1.1, ...]``

Normalization
~~~~~~~~~~~~~

- Uses "reference_only" strategy: normalizes both fronts based on the bounds of the reference front
- Useful when fronts have different scales or when you want to focus on relative performance

Normalized Hypervolume (NHV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Calculated as: ``NHV = 1 - HV(front) / HV(reference)``
- Can be negative if the solution front dominates the reference front
- Values closer to 0 indicate better performance

Error Handling
--------------

The CLI provides informative error messages for common issues:

- File not found
- Invalid CSV format
- Dimension mismatches between fronts
- Invalid reference point format
- Missing reference points for HV/NHV indicators

Integration with jMetalPy
-------------------------

This CLI tool is built on top of jMetalPy's quality indicator implementations and can be used:

- As a standalone tool for evaluating algorithm results
- In experimental pipelines and scripts
- For comparing different optimization runs
- In continuous integration systems for performance monitoring

Practical Example
-----------------

Let's walk through a complete example using the CLI:

1. **Generate sample data files:**

.. code-block:: python

    import numpy as np
    
    # Create a sample solution front
    front = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
    np.savetxt('my_front.csv', front, delimiter=',')
    
    # Create a reference front (e.g., true Pareto front)
    reference = np.array([[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0]])
    np.savetxt('reference_front.csv', reference, delimiter=',')

2. **Compute all quality indicators:**

.. code-block:: bash

    python -m jmetal.util.quality_indicator_cli my_front.csv reference_front.csv all --format json

3. **Expected output:**

.. code-block:: json

    {
      "epsilon": 0.1,
      "igd": 0.1414213562373095,
      "igdplus": 0.1,
      "hv": 0.84,
      "nhv": 0.15
    }

This provides a comprehensive evaluation of your algorithm's performance compared to the reference front.

Technical Implementation
------------------------

The CLI tool implements the exact same algorithms as jMetal for consistency and compatibility:

- **IGD formula**: ``IGD = (Σ(d^pow))^(1/pow) / N`` where ``d`` is the minimum distance from each reference point to the solution front
- **Hypervolume**: Uses the WFG algorithm for efficient computation
- **Epsilon indicator**: Implements the additive epsilon metric for convergence assessment

This ensures that results are directly comparable with other tools in the multi-objective optimization community.