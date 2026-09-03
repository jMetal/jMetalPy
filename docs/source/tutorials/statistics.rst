Statistical analysis
========================

:py:mod:`jmetal.lab.statistical_test` implements frequentist and Bayesian tests for comparing
algorithms across multiple problems and runs: the Friedman, Friedman aligned-rank, and Quade tests;
post-hoc p-value adjustment procedures (Bonferroni-Dunn, Holm, Hochberg, Holland, Finner, Li,
Shaffer, Nemenyi); Bayesian sign and signed-rank tests; and critical-distance plots. See
:doc:`experiment` for how to get from a set of algorithm runs to the tidy summary these functions
expect as input.

.. note::
   This module may be superseded by `SAES <https://github.com/jMetal/SAES>`_, a dedicated
   statistical-analysis package under the same jMetal organization, once it becomes installable
   alongside jMetalPy. See ``MODERNIZATION.md`` for the current status of that decision.

API
---

.. toctree::
   :maxdepth: 2

   /api/jmetal.lab.statistical_test