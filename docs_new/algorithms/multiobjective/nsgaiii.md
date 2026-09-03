# NSGA-III

[NSGA-III](https://doi.org/10.1109/TEVC.2013.2281535) extends NSGA-II to many-objective problems:
instead of the crowding distance, it uses a set of predefined reference directions to guide
selection and maintain diversity in high-dimensional objective spaces.

```python
--8<-- "examples/multiobjective/nsgaiii/nsgaiii_dtlz2.py"
```
