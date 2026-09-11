# Statistical analysis

`jmetal.lab.statistical_test` implements frequentist and Bayesian tests for comparing algorithms
across multiple problems and runs: the Friedman, Friedman aligned-rank, and Quade tests; post-hoc
p-value adjustment procedures (Bonferroni-Dunn, Holm, Hochberg, Holland, Finner, Li, Shaffer,
Nemenyi); Bayesian sign and signed-rank tests; and critical-distance plots. See
[Experiments](experiment.md) for how to get from a set of algorithm runs to the tidy summary these
functions expect as input.

!!! note
    This module may be superseded by [SAES](https://github.com/jMetal/SAES), a dedicated
    statistical-analysis package under the same jMetal organization, once it becomes installable
    alongside jMetalPy -- currently blocked on a SAES release with a relaxed numpy pin.

## API

::: jmetal.lab.statistical_test.functions

::: jmetal.lab.statistical_test.apv_procedures

::: jmetal.lab.statistical_test.bayesian

::: jmetal.lab.statistical_test.critical_distance
