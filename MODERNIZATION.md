# jMetalPy modernization tracker

Tracks two workstreams ahead of a jMetalPy 2.0:

- **L0 — Hygiene** (branch `chore/l0-hygiene`): fix concrete, verified defects — a broken
  getting-started path, a non-parallel experiment runner, an integration test CI never runs,
  ~230 MB of generated artifacts committed to the repo, and stale packaging/doc metadata.
- **L1 — Component-based core** (branch `feat/component-architecture`, branched from `main` once
  L0 is merged): add `jmetal.component`, a component-based algorithm architecture living alongside
  the existing `jmetal.algorithm`, mirroring what Java jMetal did with `jmetal-component`. Non-
  breaking; existing code, examples, and notebooks keep working unchanged.

Each checkbox below is scoped to be one atomic commit, per `GIT_GUIDELINES.md` (Conventional
Commits, one logical change per commit, `make lint` and `make test` clean before committing). Boxes
are checked in the same commit that completes them.

Education tooling and the SoftwareX write-up are explicitly out of scope for this tracker.

---

## L0 — Hygiene (`chore/l0-hygiene`)

### Tooling baseline

Found while starting this branch, not in the original scope — recorded here because it blocked
the "clean lint before every commit" rule from `GIT_GUIDELINES.md` for everything that follows.

- [x] `fix: use PEP 604 unions in isinstance checks to satisfy ruff UP038` — `ruff>=0.6.0` resolves
  to 0.12.0 today, which flags two tuple-form `isinstance()` calls
  (`src/jmetal/operator/repair.py:51`, `src/jmetal/util/density_estimator.py:146`) not covered by
  the ignore list; `make lint`/CI lint currently fail on `main` because of this.

Also note for future commands in this repo: `make test` needs the `jmetalpy` conda env
(`moocore` is not on the base env's `python`) — `conda activate jmetalpy` before `make lint`/`make
test`.

### Getting-started documentation

The quick-start snippets are broken; `README.md` and `examples/` already use the correct API, so
this is fixing a divergence, not guessing at intent.

- [x] `docs: fix broken API calls in the quick-start snippets` — `docs/source/getting-started.rst:52,60`, `docs/source/index.rst:61,63`; also fixed the stale `python setup.py install` in `getting-started.rst` and the missing required `offspring_population_size`/`mutation`/`crossover` args in `index.rst`'s example. Both snippets executed end-to-end to confirm.
- [x] `docs: update the experiment tutorial to the current problem API` — `docs/source/tutorials/experiment.rst:37,66` (parens), the 3 `StoppingByEvaluations(max=...)` calls, the hardcoded `/home/user/...` reference-front path (wrong dir name too — `reference_front` vs the real `reference_fronts`), and the sample CSV header/rows (`IndicatorName`/`ExecutionId` were swapped). Executed the full 3-algorithm × 3-problem tutorial end-to-end (reduced scale) to confirm — this is what surfaced the three source bugs below, which were blocking it independently of the doc typos.

  **Found while verifying — three real bugs in `generate_summary_from_experiment`'s call path, not just docs:**
  - `src/jmetal/lab/experiment.py`: `hasattr(indicator, "reference_fronts")` (plural) checked for an attribute that no indicator has — every indicator sets `self.reference_front` (singular). The per-problem reference-front auto-load this function exists for never actually fired.
  - `src/jmetal/lab/experiment.py`: the loaded reference front and the solutions passed to `indicator.compute(...)` were plain Python lists, not `np.ndarray` — `AttributeError` on `.size`/`.shape` inside `compute()`.
  - `src/jmetal/core/quality_indicator.py`: `InvertedGenerationalDistance`, `InvertedGenerationalDistancePlus`, `AverageHausdorffDistance`, and `AdditiveEpsilonIndicator` (`EpsilonIndicator`) all raised `ValueError` in `__init__` if `reference_front` was `None`, contradicting their own `= None` default — unlike `GenerationalDistance`, whose own existing test (`test_none_reference_front_should_raise_error`, `tests/core/test_quality_indicator.py:147`) already documents deferring the check to `compute()` as the intended pattern. `generate_summary_from_experiment` is built entirely around constructing indicators without a reference front and assigning one per problem, so 4 of jMetalPy's 5 real indicators could never actually be used that way — only `GenerationalDistance` worked. Relaxed all four to match `GenerationalDistance`.
  - 8 pre-existing tests in `tests/core/test_quality_indicator.py` asserted the old (eager) behavior for those same four classes and had to be updated to assert the deferred one — same pattern `GenerationalDistance`'s own test already used. Added `tests/core/test_quality_indicator.py::TestIndicatorsWithDeferredReferenceFrontValidation` (parametrized across all 5 indicators) and a new `tests/lab/test_experiment.py` (previously zero tests existed for anything in `lab/`).
- [x] `docs: rewrite the problem tutorial against the post-1.6 Problem API` — `docs/source/tutorials/problem.rst` rewritten to mirror the real, shipped `SubsetSum` classes (`jmetal.problem.{singleobjective,multiobjective}.unconstrained.SubsetSum`) instead of inventing a fix: `number_of_variables`/`number_of_objectives`/`number_of_constraints` as methods (not `__init__` attributes), `super().__init__()` with no args, `name()` not `get_name()`, and `solution.bits` (a numpy bool array) rather than `solution.variables[0]` — the old tutorial's list-of-lists model of `BinarySolution.variables` no longer matches the current flat-array implementation. Both code blocks executed and cross-checked against the shipped classes' output.
- [x] `docs: generate algorithm API pages from the maintained examples` — replaced the 10 actually-linked notebooks (of 17; the other 7 were orphaned, not in any toctree, and dropped outright) with `literalinclude`-based `.rst` pages pointing at real `examples/multiobjective/` scripts. Every linked script was executed end-to-end (HYPE smoke-tested at reduced scale — it's legitimately slow at 25k evals, not broken) before being cited.

  **Found and fixed along the way, beyond the notebooks themselves:**
  - `examples/multiobjective/gde3/gde3_zdt1.py` used `StoppingByKeyboard()` — hangs forever with no TTY, unlike every sibling "basic" example. Switched to `StoppingByEvaluations`.
  - Three examples had filenames contradicting the problem they actually solve: `ibea_zdt1.py` (really DTLZ1) → `ibea_dtlz1.py`, `moead_dtlz2.py` (really DTLZ1) → `moead_dtlz1.py`, `smpso_zdt4.py` (really ZDT3) → `smpso_zdt3.py`. Confirmed nothing referenced the old names before renaming.
  - **The real `sphinx-build` doesn't just warn, it hard-crashes** (exit 2) on the first notebook: `nbsphinx` needs a system `pandoc` binary, which isn't installed in the project's own recommended `jmetalpy` conda env, and isn't documented anywhere as a prerequisite. Since removing the 17 notebooks left zero `.ipynb` references anywhere in the docs, dropped `nbsphinx` entirely rather than chase the pandoc gap — see below.
  - That also exposed that the `docs` extra never declared `sphinx` itself, only relying on `nbsphinx` to pull it in transitively — `pip install jmetalpy[docs]` would have stopped installing Sphinx at all once `nbsphinx` was removed. Added `sphinx>=7.0.0` directly.
  - `sphinx.ext.napoleon` (translates the Google-style docstrings `CODING_GUIDELINES.md` mandates into RST) was never enabled, so most of the API reference was rendering broken (`Unexpected indentation` etc. from docutils trying to parse `Args:`/`Returns:` blocks as raw RST). Enabling it dropped the real `sphinx-build` warning count from 141 to 95.
  - **Verified end-to-end with a real `sphinx-build`, not just spot-checks**: before any of this, the build couldn't get past ~15% of the source tree. It now completes: `build succeeded, 95 warnings` (exit 0).

  **New follow-up items surfaced, not fixed here (out of scope for this one task):**
  - [ ] `fix(docs): fix malformed RST list/indentation in docstrings` — napoleon fixed most autodoc parsing errors but ~20 remain across `operator/{crossover,mutation,selection}.py`, `core/problem.py`, `util/distance.py`, `lab/statistical_test/{functions,apv_procedures}.py`, `lab/visualization/posterior.py` (numbered/bulleted lists inside docstrings missing the blank-line-before-list or consistent indentation RST needs). One docstring at a time, not mechanical.
  - [ ] `docs: fix 5 broken cross-references and 2 orphaned pages` — `api/util/{archive,distance}.rst` reference `../distance`/`../comparator`/`../normalization`/`../archive`/`../../algorithm/multiobjective` as documents that don't exist at those paths; `tutorials.rst` and `api/jmetal.lab.statistical_test.rst` exist but aren't in any toctree.
  - [ ] `docs: fix "Title underline too short" in advanced-topics/advanced-selection-strategies.rst:55`
  - [ ] `docs: fix Problem's docstring double-registering Attributes: entries` — `tutorials/problem.rst`'s automodule of `jmetal.core.problem` produces 3 `duplicate object description` warnings for `Problem.directions`/`labels`/`reference_front` (same napoleon-Attributes:-vs-autodoc'd-member class of bug fixed for `solution.rst` in the next item, but this instance lives inside a tutorial page rather than a dedicated reference page)
- [x] `docs: write the missing api/core and api/util reference pages` — 7 pages referenced by `docs/source/api-reference.rst`, following the project's dominant plain-`automodule` convention (not `archive.rst`/`distance.rst`'s much heavier hand-written style, which turned out to be the exception, not the norm — confirmed by checking `api/operator/*.rst`, `api/problem/*.rst`). `problem`, `observer`, and `evaluator` already had autodoc blocks inside their `tutorials/*.rst` pages; marked the new pages' directives `:no-index:` rather than duplicating that content, and pointed each at its tutorial. Also found and fixed, with the same `:no-index:` pattern, a real pre-existing bug on `solution.rst`: `Solution`/`FloatSolution`/`IntegerSolution`'s docstrings double-register `Attributes:` entries against their actual autodoc'd class attributes. Verified with `sphinx-build`: 95 warnings before, 90 after — a net improvement despite adding 7 pages.
- [x] `docs: prune toctree entries pointing at non-existent pages` — 29 dead links across `getting-started.rst`, `user-guide.rst`, `advanced-topics.rst`. Redirected to a real close-enough page where one exists (e.g. `understanding-problems`/`choosing-algorithms`/`analyzing-results` → `tutorials/problem`/`multiobjective.algorithms`/`tutorials/experiment`); removed the entry where nothing close exists rather than force a misleading redirect (all of `advanced-topics.rst`'s Distributed Computing/Custom Development/Integration/Research Applications/Performance Optimization subsections — 16 of the 29 — had zero real content behind any entry). Verified with `sphinx-build`: 90 warnings before, 61 after.
- [ ] `docs: document the statistical analysis module` — replace the `ToDo` in `docs/source/tutorials/statistics.rst`

### Experiment runner

- [x] `fix(lab): submit jobs to the process pool instead of running them eagerly` — `src/jmetal/lab/experiment.py:75`, plus propagating worker exceptions via `future.result()` instead of discarding them, in the same commit (they were the same one-line-cause bug: `executor.submit(job.execute(output_path))` runs eagerly *and* is why the futures were never checked).

  **Turned out to need two more fixes before it could work at all — `Job` (which holds a live `Algorithm`) wasn't picklable, so nothing could actually be sent to a worker process:**
  - `Algorithm` inherits `threading.Thread` (nothing ever calls `.start()`/`.join()` on one — confirmed again here). Thread's own instance state carries a lock, an `Event`, a stderr stream, and an internal excepthook closure, none of which pickle. Added `Algorithm.__getstate__`/`__setstate__` (`src/jmetal/core/algorithm.py`) that diffs against a fresh, never-started `Thread`'s attributes rather than hardcoding names.
  - Separately, 8 crossover operators (`PMXCrossover`, `SBXCrossover`, `IntegerSBXCrossover`, `BLXAlphaCrossover`, `BLXAlphaBetaCrossover`, `ArithmeticCrossover`, `UnimodalNormalDistributionCrossover`, `DifferentialEvolutionCrossover`) defaulted `self._rng` to the bare `numpy.random` **module** (not a `Generator`) whenever no explicit `rng` was passed — the common case, since no example passes one. Modules aren't picklable either. Fixed all 8 to `np.random.default_rng()`, matching the pattern `SPXCrossover`/`CXCrossover` in the same file already used correctly.
  - One of `tests/operator/test_crossover.py`'s existing tests relied on the module-sharing bug to pass (seeded the global `numpy` RNG and expected two separate instances to read from it) — updated to inject an explicit, identically-seeded `Generator` into each instance instead.
  - Verified with real, not synthetic, numbers: 12 NSGA-II jobs at `m_workers=8` complete in **4.24x less wall-clock time** than `m_workers=1` — genuine multi-core speedup, not just "technically concurrent" (a `ThreadPoolExecutor` alternative was measured first and rejected: only 1.19x on 4 workers, since most of the evolutionary loop is GIL-bound pure Python).

### Adopt SAES for statistical analysis, retire the redundant half of `lab`

**Frozen (2026-09-03): SAES is undergoing a deep refactor upstream, so integrating against its
current shape now would mean redoing this work later.** Do not resume any of the checklist items
below, and do not touch the SAES repo, until a new SAES release lands on PyPI with the numpy 2
fix included — re-evaluate the plan against that release's actual API before picking this back up,
since the refactor may change more than just the numpy pin.

Compared `src/jmetal/lab/` against the real code of `github.com/jMetal/SAES` (checked out locally
at `/Users/ajnebro/Softw/SAES`), not just its description. SAES is Nebro & Carreira's own successor
project for cross-algorithm statistical analysis, and its `apv_procedures.py` implements the same 8
post-hoc adjustment procedures as jMetalPy's module of the same name — this is a genuine, verified
duplication, not a superficial one.

**Redundant, to remove** — SAES covers these at least as well, plus adds t-test/ANOVA tables
jMetalPy never had:
- `src/jmetal/lab/statistical_test/{functions,apv_procedures,bayesian,critical_distance}.py` → `SAES/statistical_tests/*`, `SAES/plots/cdplot.py`
- `Experiment.generate_boxplot` → `SAES/plots/boxplot.py::Boxplot`
- `Experiment.generate_latex_tables`, `generate_median_and_wilcoxon_latex_tables`, `compute_wilcoxon`, `compute_mean_indicator` → `SAES/latex_generation/stats_table.py` (`MeanMedian`, `Wilcoxon`, `WilcoxonPivot`, `Friedman`, `FriedmanPValues`)
- Side effect: `critical_distance.py` is the *only* user of `statsmodels` in the whole codebase — removing it drops `statsmodels` from the core (required) dependencies entirely.

**Not redundant — kept as-is, no action:**
- `lab/visualization/{plotting,interactive,streaming}.py` — live/single-run visualization wired to
  the Observer pattern (`VisualizerObserver`, `PlotFrontToFileObserver`). SAES only analyzes
  already-completed multi-run results already written to disk; it has nothing for execution-time
  monitoring.
- `lab/visualization/chord_plot.py` — no SAES equivalent.
- `lab/visualization/posterior.py::plot_posterior` — script-callable, returns a figure directly;
  SAES's Bayesian output (`SAES/html/html_generator.py::notebook_bayesian`) instead shells out to
  `papermill`/`nbconvert` to render a whole notebook-based HTML report. Different usage pattern, not
  a clean substitute.
- `Job`, `Experiment`, `generate_summary_from_experiment` — SAES never runs algorithms, it only
  consumes CSVs already on disk. This is the part of `lab` that produces those CSVs; it stays
  (including the `Experiment.run()` parallelism fix below, independent of this decision).

**Decided: remove `generate_kolmogorov_smirnov_latex_tables` outright.** SAES has no
Kolmogorov-Smirnov test (only Friedman/Friedman-aligned/Quade/Wilcoxon/t-test/ANOVA), so this is a
real capability loss, not pure de-duplication — accepted anyway, since nothing in `tests/` covers it
today.

**Two blockers found, one resolved (in the SAES repo, not yet released):**
1. **License**: SAES is GPLv3 (`/Users/ajnebro/Softw/SAES/LICENSE`); jMetalPy is MIT. `saes` must
   never be a core/required dependency — optional extra only, documented as pulling in GPLv3 code.
2. ~~**numpy conflict**: SAES pins `numpy<2`, jMetalPy requires `numpy>=2.2.5`.~~ **Fixed, on a
   local branch in `/Users/ajnebro/Softw/SAES`, not merged or released.** Confirmed by a dedicated
   agent session (separate repo, not tracked step-by-step here): `np.reshape(..., newshape=...)` in
   `non_parametrical.py:174` and `apv_procedures.py:194` needed the shape argument passed
   **positionally**, not as `shape=` — that keyword doesn't exist on numpy 1.23.x, so `shape=` would
   have traded the numpy-2 failure for a numpy-1 one. Verified across numpy 1.23.5/2.2.6/2.5.1
   directly. Pin relaxed to `numpy>=1.23` (matches `matplotlib==3.9.2`'s own floor, not a guess).
   Along the way, found and fixed an unrelated, pre-existing bug that would break any fresh SAES
   install today regardless of numpy: `scikit-posthocs==0.10.0`'s unpinned `statsmodels` dependency
   imports a shim `statsmodels>=0.15.0` removed; pinned `statsmodels<0.15`. Full suite: 85/85,
   confirmed two independent ways. Branch `fix/numpy2-compatibility`, 2 commits, **local only, not
   pushed** — merging/releasing is Antonio's call on that repo, not done here.
   Also flagged (not fixed, correctly out of scope for that task): 5 gitignored `.ipynb` files under
   SAES's `tests/htmls/` were accidentally committed in an earlier SAES commit; and SAES's own `main`
   CI is currently red but on an unrelated, pre-existing failure.

**Still blocked below until the SAES fix above is actually merged and released to PyPI** — do not
wire `saes` into `pyproject.toml` before then, since PyPI still only has the old `numpy<2` 1.5.0:
- [ ] `refactor(lab): remove the statistical-test modules superseded by SAES` — `statistical_test/{functions,apv_procedures,bayesian,critical_distance}.py`
- [ ] `refactor(lab)!: remove the Experiment methods superseded by SAES` — `generate_boxplot`, `generate_latex_tables`, `generate_median_and_wilcoxon_latex_tables`, `compute_wilcoxon`, `compute_mean_indicator`
- [ ] `refactor(lab)!: remove generate_kolmogorov_smirnov_latex_tables` — no SAES equivalent; accepted capability loss
- [ ] `build: drop statsmodels from the core dependencies` — only used by the removed `critical_distance.py`
- [ ] `build: add saes as an optional extra, never a core dependency` — blocked until the SAES fix above is published to PyPI with a relaxed numpy pin
- [ ] `docs: point the experiment tutorial and examples at SAES for statistical analysis` — `docs/source/tutorials/experiment.rst`, `examples/experiment/{statistical_analysis.py,generateKolmogorovSmirnovLatexTables.py,generateMedianAndWilcoxonLatexTables.py}`
- [ ] `docs: document the GPLv3 licensing implication of the optional saes extra` — README

### Tests

- [x] `test(algorithm): rename the integration test file so pytest discovers it` — `tests/algorithm/ittest_algorithm.py` → `test_algorithm_integration.py`. All 4 tests passed immediately once discovered.
- [x] `test(algorithm): migrate the integration tests to pytest conventions` — `unittest.TestCase` → `test_should_<behavior>`, done in the same commit as the rename (the file's old content wasn't worth preserving as an intermediate state).
- [x] `test: apply the declared smoke/slow/integration markers` — applied `@pytest.mark.smoke`/`@pytest.mark.integration` to the two classes in the file above, which map directly onto that distinction. Other test files weren't audited for `slow`/`integration` candidates — leaving the markers declared and now precedented rather than removed.

### Repository artifacts

- [x] `chore: untrack generated result artifacts and ignore them` — `results/` (194 MB, 80 files, confirmed unreferenced by any source/test/example/doc before removing). Kept on disk, just untracked; `.gitignore` now has `/results/`.
- [x] `chore: untrack notebook checkpoints and experiment output artifacts` — `notebooks/.ipynb_checkpoints/` (4 files) plus `examples/experiment/{boxplot,latex}/` and 3 more top-level generated files found while checking (`QualityIndicatorSummary.csv`, `cdplot.eps`, `posterior.eps`) — 286 files, ~5.3 MB total. Confirmed only `.py` scripts remain tracked in `examples/experiment/`.
- [ ] `chore: drop stale tuning entries from .gitignore` — leftovers from the removed `jmetal/tuning` package
- [ ] `ci: publish the documentation from a gh-pages branch`
- [ ] `chore: untrack the built HTML site from docs/` — 311 files, 27 MB

### Packaging and metadata

- [x] `docs: sync Sphinx conf release, copyright and src-layout path` — `docs/source/conf.py:6,14`. `release` was `'1.7.0'` (package is 1.9.0), `sys.path` pointed at the repo root from before the `src/` layout move. Author/copyright synced against `pyproject.toml`'s actual authors list. Verified: same 61-warning build.
- [x] `build: move mockito to the test extra` — only used by `tests/util/test_comparator.py`; removed from base `dependencies`, `core`, `docs`, `distributed`, `complete`.
- [x] `build: unify the moocore version pin across dependency groups` — `>=0.1.8` base vs `>=0.1.9` extras, unified on `>=0.1.9`. Verified the package still builds and passes `twine check`.
- [x] `build: expose the quality indicator CLI as a console script` — added `jmetalpy-quality-indicator`, verified with a real install + `--help`. Documented alongside the existing `python -m` form.
- [x] `docs: add CITATION.cff` — matches `pyproject.toml`'s authors and the existing SWEVO BibTeX in `about.rst` field for field; validated as parseable YAML.
- [x] `docs: add CONTRIBUTING.md pointing at the coding and git guidelines` — also rewrote the stale `docs/source/contributing.rst` (a master/develop/feature/hotfix git-flow this repo never used, Python 3.6, and 8 screenshot references that point at files which don't exist in the repo) to point at the new file instead. `sphinx-build`: 61 warnings before, 39 after.
- [ ] `docs: extract the changelog from README into CHANGELOG.md`
- [ ] `docs: list the full algorithm and problem catalogue in the README` — currently omits MOCell, WFG1-9, ZCAT1-20, DTLZ3-7, eqDTLZ, `misc.py`

### CI

- [ ] `ci: build the documentation with warnings as errors`
- [ ] `ci: execute the notebooks and examples`
- [ ] `ci: report test coverage`

---

## L1 — Component-based core (`feat/component-architecture`)

### Phase 1 — EA template and NSGA-II

- [ ] `refactor(operator): introduce a Replacement ABC for the existing replacement classes` — `RankingAndDensityEstimatorReplacement`, `RankingAndCrowdingDistanceReplacement`, `SMSEMOAReplacement` in `src/jmetal/operator/replacement.py` only share `replace()` by convention today
- [ ] `feat(component): add the component protocols for the EA catalogue`
- [ ] `feat(component): add SolutionsCreation, Evaluation and Termination components`
- [ ] `feat(component): add MatingPoolSelection, Variation and Replacement components`
- [ ] `feat(component): add AlgorithmState with a backwards-compatible observer payload`
- [ ] `feat(component): add the EvolutionaryAlgorithm template`
- [ ] `feat(component): thread an injectable random generator and seed through the template`
- [ ] `feat(component): add NSGAIIBuilder`
- [ ] `test(component): assert builder NSGA-II matches the classic one for a fixed seed` — acceptance test for Phase 1: identical fronts on ZDT1 and DTLZ2
- [ ] `test(component): assert run reproducibility, including with MultiprocessEvaluator`
- [ ] `docs: document the component-based architecture`

### Phase 1b — decouple from threading.Thread

Verified nobody calls `algorithm.start()`/`.join()` anywhere in `src/`, `examples/`, `tests/`,
`notebooks/`, or `docs/source` — this is dead coupling, not a used feature.

- [ ] `refactor(core): define an algorithm protocol independent of threading.Thread`
- [ ] `refactor(lab): type Job against the algorithm protocol`
- [ ] `refactor(core)!: stop inheriting from threading.Thread` — add an explicit `run_in_thread()` helper for the live-plotting use case

### Phase 2 — widen the catalogue

- [ ] `feat(component): add SMSEMOABuilder`
- [ ] `feat(component): add SPEA2Builder`
- [ ] `feat(component): add MOCellBuilder`
- [ ] `feat(component): add GeneticAlgorithmBuilder`

### Phase 3 — PSO

- [ ] `feat(component): add the PSO catalogue`
- [ ] `feat(component): add the ParticleSwarmOptimization template`
- [ ] `feat(component): add SMPSOBuilder`

**Known risk:** MOEA/D does not fit the component model well — Java jMetal's own docs
(`jmetal-component` design notes) admit needing complex, tightly-coupled components for it. Not
attempted in the phases above.

---

## Verification

**L0**
- `make lint` and `make test` clean, with the renamed integration test actually running.
- CI builds the docs with `-W` (a broken toctree fails the build).
- CI executes notebooks and examples; the documented quick-start runs in CI.
- New `Experiment.run()` test: `m_workers > 1` genuinely uses multiple processes, and an exception
  in one job propagates instead of being silently dropped.
- `git ls-files | xargs du -ch` confirms the ~230 MB reduction in tracked generated artifacts.
- `pip install .` and the quality-indicator CLI work from the installed package.

**L1**
- **Behavioural equivalence test** (Phase 1 acceptance criterion): given the same seed,
  `NSGAIIBuilder(...).build()` and the classic `NSGAII` produce identical fronts on ZDT1 and DTLZ2.
- **Reproducibility**: two runs with the same `seed` give identical results, including with
  `MultiprocessEvaluator` — not true for any algorithm today.
- Existing observers (`ProgressBarObserver`, `VisualizerObserver`, `WriteFrontToFileObserver`, ...)
  work unchanged against the new template.
- The full test suite stays green — the "nothing breaks" criterion is self-verifying.
