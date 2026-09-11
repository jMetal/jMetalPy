# Changelog

* [Current]
  * Added the multi-objective TSP problem
  * Added the Average Hausdorff Distance (AHD) quality indicator.
  * Added the ZDT benchmark problems. Contributed by Nicolás Rodríguez Uribe.
  * Added a component-based architecture (`jmetal.component`) as an alternative to the
    classic algorithm hierarchy: factory functions `build_nsgaii()`, `build_smsemoa()`,
    `build_moead()` and `build_moead_de()` assemble an algorithm from six independently
    swappable pieces (solution creation, evaluation, termination, selection, variation,
    replacement) instead of subclassing. Verified to reproduce the classic algorithms'
    results exactly given the same seed (MOEA/D-DE excepted, which is fully reproducible
    from a single seed instead -- see below).
  * Added support for external archives (bounded and unbounded) in the component-based
    algorithms, mirroring jMetal Java's `EvolutionaryAlgorithmWithArchive`.
  * Added `rng` (an injectable `numpy.random.Generator`) to `Problem.create_solution()`
    and several operators, enabling fully single-seed-reproducible runs for the
    component-based MOEA/D and MOEA/D-DE.
  * Performance: the Hypervolume, IGD+ and additive epsilon quality indicators, and
    non-dominated sorting, now delegate to
    [moocore](https://multi-objective.github.io/moocore/python/) for faster computation.
  * Performance: unbounded-archive updates are now batched (`Archive.add_batch()`),
    up to ~30x faster on large archives.
  * `Algorithm` no longer inherits from `threading.Thread` (nothing in jMetalPy called
    `.start()`/`.join()` on one; this also removes the previous unpicklable-state
    workaround needed for `MultiprocessEvaluator`).
  * Fixed `Experiment.run()` submitting jobs to the process pool sequentially instead of
    in parallel.
  * Fixed `Algorithm` not being picklable across process boundaries in some cases.
  * Fixed the GDE3 ZDT1 basic example hanging forever.
  * Fixed `IBEA.create_initial_solutions()` not forwarding its `rng` to the population
    generator, and `GDE3` not forwarding its `rng` to its internal
    `DifferentialEvolutionCrossover`, both of which silently broke single-seed
    reproducibility for those two algorithms.
  * Fixed the FDA1-FDA5 dynamic problems (`jmetal.problem.multiobjective.fda`), which
    predated the current `Problem` interface and could not be instantiated; FDA4 and
    FDA5 also declared only 2 objective directions for their 3 objectives.
  * Migrated the documentation from Sphinx to MkDocs.
* [1.9.0]
  * Add RE benchmark
  * Refactor algorithm SPEA2
  * Refactor classes Solution, BinarySolution, and FloatSolution
  * Added Unimodal Normal Distribution Crossover (UNDX), BLX-Alpha, BLX-Alpha-Beta and Arithmetic crossover operators.
  * Added Levy flight and power-law mutation operators.
* [v.1.8.0]
  * Add the SMS-EMOA algorithm (based on [moocore](https://multi-objective.github.io/moocore/python/).)
  * Add the IGD+ quality indicator
  * Add new bencharmk problems:
    * [eqdtlz.py](https://github.com/jMetal/jMetalPy/blob/main/src/jmetal/problem/multiobjective/eqdtlz.py)
    * [misc.py](https://github.com/jMetal/jMetalPy/blob/main/src/jmetal/problem/multiobjective/misc.py)
    * Their reference fronts have been obtained with the [Reference Set Generator](https://doi.org/10.3390/math13101626) method.
  * The project structure has been changed from [flat to src](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html).
  * The Hypervolume quality indicator implementation relies now on the [moocore project](https://multi-objective.github.io/moocore/python/).
* [v1.7.0] Add RWA benchmark, refactor classes BinarySolution and BinaryProblem.
* [v1.6.0] Refactor class Problem, the single-objective genetic algorithm can solve constrained problems, performance improvements in NSGA-II, generation of Latex tables summarizing the results of the Wilcoxon rank sum test, added a notebook folder with examples.
* [v1.5.7] Use of linters for catching errors and formatters to fix style, minor bug fixes.
* [v1.5.6] Removed warnings when using Python 3.8.
* [v1.5.5] Minor bug fixes.
* [v1.5.4] Refactored quality indicators to accept numpy array as input parameter.
* [v1.5.4] Added [CompositeSolution](https://github.com/jMetal/jMetalPy/blob/master/jmetal/core/solution.py#L111) class to support mixed combinatorial problems. [#69](https://github.com/jMetal/jMetalPy/issues/69)
