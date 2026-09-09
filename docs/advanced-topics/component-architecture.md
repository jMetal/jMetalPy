# Component-Based Architecture

`jmetal.component` assembles algorithms from small, independently testable components instead of
defining them by subclassing. It mirrors jMetal Java's `jmetal-component` module and lives
alongside `jmetal.algorithm`, which keeps working unchanged -- nothing here replaces the classic
algorithms, and no existing example, notebook, or user code is affected.

This page currently covers evolutionary algorithms (EA/MOEA). A PSO template and catalogue are
planned but not implemented yet.

For a hands-on walkthrough of every configuration described below -- final fronts, quality
indicator values, and the observer pattern in action -- see the
[`notebooks/NSGAIIComponentBased.ipynb`](https://github.com/jMetal/jMetalPy/blob/main/notebooks/NSGAIIComponentBased.ipynb)
and
[`notebooks/SMSEMOAComponentBased.ipynb`](https://github.com/jMetal/jMetalPy/blob/main/notebooks/SMSEMOAComponentBased.ipynb)
notebooks.

## Why

The classic algorithms hard-wire their steps as methods on a class hierarchy: `GeneticAlgorithm`
defines `create_initial_solutions()`, `evaluate()`, `selection()`, `reproduction()`, and
`NSGAII.replacement()` overrides the base class's replacement step. Swapping one step for another
means subclassing. A component-based algorithm instead takes six collaborators as constructor
arguments, so any of them can be swapped independently without touching the algorithm's control
flow at all.

## The six components

| Component | Responsibility | Protocol method(s) |
|---|---|---|
| `SolutionsCreation` | Creates the initial population | `create() -> list[S]` |
| `Evaluation` | Evaluates a population against a problem | `evaluate(list[S]) -> list[S]`, `computed_evaluations() -> int` |
| `Termination` | Decides when to stop | `is_met(status: Mapping) -> bool` |
| `Selection` | Builds a mating pool from the population | `select(list[S]) -> list[S]` |
| `Variation` | Turns a mating pool into offspring | `variate(...) -> list[S]`, `mating_pool_size() -> int`, `offspring_population_size() -> int` |
| `Replacement` | Selects survivors from parents and offspring | `replace(current, offspring) -> list[S]` |

Each is a [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)
(except `Replacement`, an `ABC` -- see below):
structural typing means any object exposing the right method already satisfies the contract, with
no need to inherit from anything. A bare function wrapped in a small object, a `lambda`, or an
existing class all work.

`Replacement` is the one exception: the three replacement strategies that already existed in
`jmetal.operator.replacement` (`RankingAndDensityEstimatorReplacement`,
`RankingAndCrowdingDistanceReplacement`, `SMSEMOAReplacement`) only shared their `replace()` method
by convention, with no common base to type against. `jmetal.operator.replacement.Replacement` is a
minimal `ABC` that fixes that, and `jmetal.component.catalogue.ea.replacement` re-exports it
directly -- no adapter needed, since the existing classes already match the shape the component
model expects.

## The `EvolutionaryAlgorithm` template

`jmetal.component.algorithm.evolutionary_algorithm.EvolutionaryAlgorithm` is a direct translation
of jMetal Java's `EvolutionaryAlgorithm` template. Its `run()` is exactly:

```python
population = solutions_creation.create()
population = evaluation.evaluate(population)
while not termination.is_met(state):
    mating_population = selection.select(population)
    offspring_population = variation.variate(population, mating_population)
    offspring_population = evaluation.evaluate(offspring_population)
    population = replacement.replace(population, offspring_population)
```

Unlike `jmetal.core.algorithm.Algorithm`, it does not inherit from `threading.Thread`: nothing in
jMetalPy calls `start()`/`join()` on an algorithm, so the template skips that coupling instead of
carrying it for no benefit.

## Building NSGA-II

`build_nsgaii()` is a factory *function*, not a chainable builder class. jMetal Java's
`NSGAIIBuilder` uses `.setX().build()` because Java has no keyword arguments; Python already does,
so a function with keyword-only overrides gives the same flexibility without an intermediate
mutable object:

```python
from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1

problem = ZDT1()
algorithm = build_nsgaii(
    problem,
    population_size=100,
    offspring_population_size=100,
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
)

algorithm.run()
front = algorithm.result()
```

Every default -- `selection`, `variation`, `replacement`, `termination` -- can be overridden with a
keyword argument, and every existing observer in `jmetal.util.observer`
(`ProgressBarObserver`, `VisualizerObserver`, `WriteFrontToFileObserver`, ...) works against
`algorithm.observable` unchanged, since `AlgorithmState.as_dict()` produces the same
`"PROBLEM"`/`"EVALUATIONS"`/`"SOLUTIONS"`/`"COMPUTING_TIME"`-keyed mapping the classic algorithms
already notify with.

## Building SMS-EMOA

SMS-EMOA is very similar to NSGA-II: `build_smsemoa()` follows the exact same factory-function
pattern and reuses four of the six components unchanged (`RandomSolutionsCreation`,
`SequentialEvaluation`, `TerminationByEvaluations`, `CrossoverAndMutationVariation`). It only
differs in mating selection -- `RandomSelection` rather than tournament, since SMS-EMOA relies on
its replacement strategy alone to drive convergence -- and in replacement -- `SMSEMOAReplacement`,
which ranks the merged population, keeps every front but the last whole, and prunes the last front
by hypervolume contribution rather than crowding distance:

```python
from jmetal.component.algorithm.multiobjective.smsemoa import build_smsemoa
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1

problem = ZDT1()
algorithm = build_smsemoa(
    problem,
    population_size=100,
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
)

algorithm.run()
front = algorithm.result()
```

Unlike `build_nsgaii()`, there is no `offspring_population_size` parameter: SMS-EMOA is
steady-state by definition (Beume et al., 2007) and always produces exactly one offspring per
generation, matching `jmetal.algorithm.multiobjective.smsemoa.SMSEMOA`, which hardcodes the same
value. Every other default -- `selection`, `variation`, `replacement`, `termination` -- can be
overridden the same way as `build_nsgaii()`'s, and `archive=` is supported identically (see
[External archives](#external-archives) below) -- both factories share the same
`EvolutionaryAlgorithm` template, so nothing archive-specific needed to change.

## Building MOEA/D

`build_moead()` (classic, any crossover) and `build_moead_de()` (differential evolution) build on
the exact same `EvolutionaryAlgorithm` template as `build_nsgaii()`/`build_smsemoa()` -- no
modification needed there. What MOEA/D actually needs that NSGA-II/SMS-EMOA don't: `Selection`,
`Variation` and `Replacement` all need to agree, every iteration, on *which subproblem* is being
processed and whether this iteration's mating pool and replacement scan are scoped to that
subproblem's neighborhood or the whole population -- neither concept exists in the generic
`select()`/`variate()`/`replace()` signatures, nor should it (it's specific to MOEA/D, not a general
evolutionary-algorithm concern). `catalogue/ea/moead.py`'s `MOEADContext` is a small object
constructed once per run and passed by reference into `MOEADSelection`, `MOEADReplacement` and (for
the DE variant) `DifferentialEvolutionCrossoverVariation`, each of which reads it without the
template needing to know it exists -- mirroring how jMetal Java's own component-based MOEA/D
(`MOEADBuilder`/`MOEADDEBuilder` in `jmetal-component`) shares a `SequenceGenerator<Integer>` the
same way:

```python
from jmetal.component.algorithm.multiobjective.moead import build_moead, build_moead_de
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1

problem = ZDT1()

# Classic: any crossover, defaults to PenaltyBoundaryIntersection aggregation.
algorithm = build_moead(
    problem,
    population_size=100,
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
)

# MOEA/D-DE: differential-evolution crossover, defaults to Tschebycheff aggregation.
algorithm = build_moead_de(
    problem,
    population_size=100,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
    cr=1.0,
    f=0.5,
)

algorithm.run()
front = algorithm.result()
```

Both reuse `RandomSolutionsCreation`, `SequentialEvaluation`/`SequentialEvaluationWithArchive` and
`TerminationByEvaluations` unchanged, and support `archive=` identically to `build_nsgaii()`/
`build_smsemoa()`. `population_size` doubles as the number of subproblems (one weight vector per
population slot); for 3+ objectives the weight vectors are read from a file in `weight_files_path`
(default: this repository's bundled `resources/MOEAD_weights/`) -- 2-objective weight vectors are
generated analytically and need no file.

**A correction on record.** This page, and `MODERNIZATION.md`, previously carried a note claiming
MOEA/D "does not fit the component model well," attributed to unverified Java design notes. Direct
investigation of `jmetal-component` found the opposite -- `MOEADBuilder`/`MOEADDEBuilder` already
exist there, building the same generic template with no modification -- so the note was corrected
rather than repeated here.

**A step further on reproducibility.** Every MOEA/D-specific random decision (which subproblem,
which neighborhood-vs-population scope, mating-pool sampling, the replacement scan) draws
exclusively from the `rng` passed to `build_moead()`/`build_moead_de()`, never from the global
`random`/`numpy.random` state -- a deliberate departure from the classic
`jmetal.algorithm.multiobjective.moead.MOEAD`, which mixes three incompatible random sources (global
`random`, global *legacy* `numpy.random`, and each operator's own `np.random.Generator`) and can
therefore never be made reproducible from a single seed. Getting there also meant reaching one level
deeper than `build_nsgaii()`/`build_smsemoa()` did: `Problem.create_solution()` gained an optional
`rng` parameter (defaulting to the same global-state behavior as before, so nothing already relying
on `random.seed()` breaks), and `RandomSolutionsCreation` now forwards its own `rng` into it -- so
population creation is reproducible from `rng=` too, for all three factories, not just MOEA/D's.
One caveat remains: `crossover` (classic variant only) and `mutation` are always your own operators,
constructed outside these factories, so their reproducibility is in your hands the same way it
already is for `build_nsgaii()`/`build_smsemoa()` -- pass them their own matching `rng` too.

The consequence: no execution-level equivalence test against the classic `MOEAD` is possible for
`build_moead_de()` the way `test_nsgaii_equivalence.py`/`test_smsemoa_equivalence.py` compare against
their classic counterparts (see [Verified behavioral equivalence](#verified-behavioral-equivalence)
below) -- two runs seeded "the same way" draw from genuinely different random number generators, so
they cannot produce identical fronts. What's verified instead: `test_moead_replacement_structural_equivalence.py`
checks that `MOEADReplacement`'s replace/keep decisions exactly match the classic algorithm's
`update_current_subproblem_neighborhood()` given identical inputs (population, offspring, current
subproblem, scope) -- the logic that matters, isolated from randomness -- plus
`test_moead_integration.py`'s hypervolume-floor checks for both variants (`build_moead()` has no
classic SBX-based counterpart in jMetalPy to compare against in the first place -- the existing
classic `MOEAD` class is already MOEA/D-DE despite its name, and always has been).

## Reproducibility

No classic jMetalPy algorithm accepts a seed. `EvolutionaryAlgorithm` accepts an optional
`rng: np.random.Generator`, shared with any component that already declares a plain `rng`
attribute:

```python
import numpy as np

algorithm = build_nsgaii(..., rng=np.random.default_rng(42))
```

Not every operator is RNG-aware yet -- `SBXCrossover` and `PolynomialMutation` already accept an
explicit `rng`, but `TournamentSelection` and `Problem.create_solution()` still draw from Python's
global `random` module. Component-based algorithms are reproducible today when built with
RNG-aware operators and a reseeded global `random` state; migrating the remaining operators to the
injectable-`rng` pattern is ongoing, tracked in `MODERNIZATION.md`.

## External archives

An external archive collects solutions independently of the population: every evaluated solution
is copied into it, regardless of what the population/replacement strategy keeps or discards. This
mirrors jMetal Java's `SequentialEvaluationWithArchive` + `EvolutionaryAlgorithmWithArchive` pair,
but as a plain constructor parameter rather than a subclass:

```python
from jmetal.util.archive import CrowdingDistanceArchive

algorithm = build_nsgaii(
    problem, population_size=100, offspring_population_size=100,
    crossover=crossover, mutation=mutation,
    archive=CrowdingDistanceArchive(maximum_size=100),  # bounded
)
algorithm.run()
front = algorithm.result()  # returns the archive's contents, not the final population
```

Any `jmetal.util.archive.Archive` works, bounded (`CrowdingDistanceArchive`, `DistanceBasedArchive`,
...) or unbounded (`NonDominatedSolutionsArchive`). The population still drives
selection/replacement as usual -- the archive is a pure addition, useful in particular for
multi-modal problems like ZDT4, where it guards against the population converging on a local
Pareto front. See `examples/component/nsgaii_crowding_distance_archive_zdt4.py` and
`examples/component/nsgaii_unbounded_archive_dtlz2.py`.

**`result()`'s size.** An unbounded archive can accumulate far more solutions than the population
size -- into the thousands over a full run. `result()` accounts for this: if the archive holds more
solutions than the population size, it reduces it to exactly that many via
`distance_based_subset_selection_robust` before returning, rather than handing back the whole
(potentially huge) archive. This mirrors jMetal Java's `BestSolutionsArchive`, which wraps an
otherwise-unbounded archive the same way. A bounded archive (`CrowdingDistanceArchive`, ...) never
exceeds the population size to begin with, so this is a no-op for it -- `result()` returns its
contents directly.

`SequentialEvaluationWithArchive` feeds the archive a whole generation at a time via
`Archive.add_batch()` rather than one solution at a time. For `NonDominatedSolutionsArchive`, whose
`add()` is O(n) per call, this replaces many individual insertions with a single
`moocore.is_nondominated()` call per generation -- on the DTLZ2 example above (an unbounded archive
growing into the thousands over 40000 evaluations), this took the run from ~43s to ~1.4s.

## Discovering what's available: the catalogue reference

Before this architecture existed, answering "what can I configure in NSGA-II?" meant reading Java
source by hand. `jmetal.component.catalogue_info.describe_catalogue()` answers it from the running
code instead: which component slots exist, which implementations are available for each, and each
implementation's control parameters (name, type, default). It deliberately stops there -- no ranges
or distributions to *explore* those parameters, which is an automatic-configuration concern kept out
of scope for now.

```python
from jmetal.component.catalogue_info import describe_catalogue

for slot, implementations in describe_catalogue().items():
    print(slot)
    for implementation in implementations:
        print(" ", implementation.name, [p.name for p in implementation.parameters])
```

Everything below is derived from the real classes via `inspect`, not hand-maintained, so it cannot
drift out of sync with the code the way a separate parameter list could. Current output for the
NSGA-II catalogue:

### SolutionsCreation

**`RandomSolutionsCreation`**

| Parameter | Type | Default |
|---|---|---|
| `problem` | `Problem[~S]` | required |
| `number_of_solutions_to_create` | `int` | required |
| `rng` | `Generator \| None` | None |

### Evaluation

**`SequentialEvaluation`**

| Parameter | Type | Default |
|---|---|---|
| `problem` | `Problem[~S]` | required |
| `evaluator` | `Optional[Evaluator[~S]]` | None |

### Termination

**`TerminationByEvaluations`**

| Parameter | Type | Default |
|---|---|---|
| `max_evaluations` | `int` | required |

### Selection

**`TournamentSelection`**

| Parameter | Type | Default |
|---|---|---|
| `selection_operator` | `TournamentSelection` | required |
| `mating_pool_size` | `int` | required |

**`RandomSelection`**

| Parameter | Type | Default |
|---|---|---|
| `selection_operator` | `RandomSelection` | required |
| `mating_pool_size` | `int` | required |

**`MOEADSelection`**

| Parameter | Type | Default |
|---|---|---|
| `context` | `MOEADContext` | required |
| `neighbourhood` | `WeightVectorNeighborhood` | required |
| `number_of_parents` | `int` | required |
| `selection_operator` | `NaryRandomSolutionSelection \| None` | None |

### Variation

**`CrossoverAndMutationVariation`**

| Parameter | Type | Default |
|---|---|---|
| `offspring_population_size` | `int` | required |
| `crossover` | `Crossover` | required |
| `mutation` | `Mutation` | required |

**`DifferentialEvolutionCrossoverVariation`**

| Parameter | Type | Default |
|---|---|---|
| `context` | `MOEADContext` | required |
| `crossover` | `DifferentialEvolutionCrossover` | required |
| `mutation` | `Mutation` | required |

### Replacement

**`RankingAndDensityEstimatorReplacement`**

| Parameter | Type | Default |
|---|---|---|
| `ranking` | `Ranking` | required |
| `density_estimator` | `DensityEstimator` | required |
| `removal_policy` | `RemovalPolicyType` | RemovalPolicyType.ONE_SHOT |

**`RankingAndCrowdingDistanceReplacement`**

| Parameter | Type | Default |
|---|---|---|
| `ranking` | `Ranking` | None |
| `density_estimator` | `DensityEstimator` | None |

**`SMSEMOAReplacement`**

| Parameter | Type | Default |
|---|---|---|
| `ranking` | `Ranking` | None |

**`MOEADReplacement`**

| Parameter | Type | Default |
|---|---|---|
| `context` | `MOEADContext` | required |
| `neighbourhood` | `WeightVectorNeighborhood` | required |
| `aggregation_function` | `AggregationFunction` | required |
| `max_number_of_replaced_solutions` | `int` | required |

### Crossover (`jmetal.operator.crossover`)

**`SBXCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `distribution_index` | `float` | 20.0 |
| `repair_operator` | `Callable[[float, float, float], float] \| FloatRepairOperator \| None` | ClampFloatRepair() |
| `rng` | `Generator \| None` | None |

**`IntegerSBXCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `distribution_index` | `float` | 20.0 |
| `rng` | `Generator \| None` | None |

**`BLXAlphaCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | 0.9 |
| `alpha` | `float` | 0.5 |
| `repair_operator` | `Callable[[float, float, float], float] \| None` | None |
| `rng` | `Generator \| None` | None |

**`BLXAlphaBetaCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | 0.9 |
| `alpha` | `float` | 0.5 |
| `beta` | `float` | 0.5 |
| `repair_operator` | `Callable[[float, float, float], float] \| None` | None |
| `rng` | `Generator \| None` | None |

**`PMXCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `rng` | `Generator \| None` | None |

**`CXCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |

**`SPXCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `rng` | `Generator \| None` | None |

**`DifferentialEvolutionCrossover`**

| Parameter | Type | Default |
|---|---|---|
| `CR` | `float` | required |
| `F` | `float` | required |
| `K` | `float` | 0.5 |
| `rng` | `Generator \| None` | None |

### Mutation (`jmetal.operator.mutation`)

**`PolynomialMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | 0.01 |
| `distribution_index` | `float` | 20.0 |
| `repair_operator` | `Callable[[float, float, float], float] \| None` | None |
| `rng` | `Generator \| None` | None |

**`IntegerPolynomialMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `distribution_index` | `float` | 20.0 |
| `repair_operator` | `Callable[[float, int, int], int] \| None` | None |
| `rng` | `Generator \| None` | None |

**`BitFlipMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |

**`PermutationSwapMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `rng` | `Generator \| None` | None |

**`ScrambleMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `rng` | `Generator \| None` | None |

**`SimpleRandomMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `rng` | `Generator \| None` | None |

**`UniformMutation`**

| Parameter | Type | Default |
|---|---|---|
| `probability` | `float` | required |
| `perturbation` | `float` | 0.5 |
| `repair_operator` | `Callable[[float, float, float], float] \| None` | None |
| `rng` | `Generator \| None` | None |

This table is generated from `describe_catalogue()`'s output and should be regenerated whenever
`jmetal.component.catalogue_info.CATALOGUE` changes (a new component lands, or a constructor
signature changes) -- it is not kept in sync automatically.

## Verified behavioral equivalence

`build_nsgaii(...)`/`build_smsemoa(...)` and their classic `NSGAII(...)`/`SMSEMOA(...)` counterparts
produce **identical** final populations on ZDT1 and DTLZ2 given the same seed -- both reuse the exact
same operator classes and `Problem.create_solution()`, so identical random state produces identical
results. This is checked by dedicated tests
(`tests/component/algorithm/multiobjective/test_nsgaii_equivalence.py`,
`test_smsemoa_equivalence.py`) and is the acceptance criterion for this architecture: adopting
components changes *how* an algorithm is assembled, not *what* it computes.

MOEA/D is the exception: `build_moead()`/`build_moead_de()` deliberately draw all their own
randomness from a single `rng`, never the classic `MOEAD`'s mix of global `random`/legacy
`numpy.random`/per-operator generators, so no seed makes their fronts bit-identical to the classic
algorithm's. See [Building MOEA/D](#building-moead) above for what's verified in its place.

## Package layout

```text
src/jmetal/component/
├── algorithm/
│   ├── algorithm_state.py              # AlgorithmState
│   ├── evolutionary_algorithm.py       # EvolutionaryAlgorithm
│   └── multiobjective/
│       ├── nsgaii.py                   # build_nsgaii()
│       ├── smsemoa.py                  # build_smsemoa()
│       └── moead.py                    # build_moead(), build_moead_de()
└── catalogue/
    ├── common/
    │   ├── solutions_creation.py
    │   ├── evaluation.py
    │   └── termination.py
    └── ea/
        ├── selection.py
        ├── variation.py
        ├── replacement.py
        └── moead.py                    # MOEADContext and MOEA/D's Selection/Variation/Replacement
```

This covers NSGA-II, SMS-EMOA and MOEA/D. Further MOEAs (SPEA2, MOCell) reusing the same catalogue,
and a PSO template and catalogue, are planned -- see `MODERNIZATION.md`.
