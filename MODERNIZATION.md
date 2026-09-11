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
  - [x] `fix(docs): fix malformed RST list/indentation in docstrings` — all of them, across `operator/{crossover,mutation,selection}.py`, `core/problem.py`, `util/distance.py`, `lab/statistical_test/{functions,apv_procedures}.py`, `lab/visualization/posterior.py`. `apv_procedures.py` turned out to have the same fragile NumPy-style-with-colon pattern in all 8 of its functions (only 2 happened to warn) — converted all 8 to Google style rather than leave 6 "working by luck".
  - [x] `docs: fix 5 broken cross-references and 2 orphaned pages` — `../distance`/`../archive` were one directory level too deep (fixed to same-directory refs); `../comparator`/`../normalization` pointed at pages that were never written (dropped, nothing to link to); `../../algorithm/multiobjective` fixed to the real page path. `tutorials.rst` removed (exact duplicate of `user-guide.rst`'s listing, referenced from nowhere); `api/jmetal.lab.statistical_test.rst` linked from `tutorials/statistics.rst`, replacing its `ToDo` placeholder.
  - [x] `docs: fix "Title underline too short" in advanced-topics/advanced-selection-strategies.rst:55`
  - [x] `docs: fix Problem's docstring double-registering Attributes: entries` — moved `Problem`'s and `OnTheFlyFloatProblem`'s `Attributes:` prose into inline `#:` comments at each assignment, the actual source autodoc discovers, instead of listing them twice.

  **`sphinx-build -b html -W`: exit 0, zero warnings.** Started this session as a hard crash (missing `pandoc`) before reading 15% of the source tree.
- [x] `docs: write the missing api/core and api/util reference pages` — 7 pages referenced by `docs/source/api-reference.rst`, following the project's dominant plain-`automodule` convention (not `archive.rst`/`distance.rst`'s much heavier hand-written style, which turned out to be the exception, not the norm — confirmed by checking `api/operator/*.rst`, `api/problem/*.rst`). `problem`, `observer`, and `evaluator` already had autodoc blocks inside their `tutorials/*.rst` pages; marked the new pages' directives `:no-index:` rather than duplicating that content, and pointed each at its tutorial. Also found and fixed, with the same `:no-index:` pattern, a real pre-existing bug on `solution.rst`: `Solution`/`FloatSolution`/`IntegerSolution`'s docstrings double-register `Attributes:` entries against their actual autodoc'd class attributes. Verified with `sphinx-build`: 95 warnings before, 90 after — a net improvement despite adding 7 pages.
- [x] `docs: prune toctree entries pointing at non-existent pages` — 29 dead links across `getting-started.rst`, `user-guide.rst`, `advanced-topics.rst`. Redirected to a real close-enough page where one exists (e.g. `understanding-problems`/`choosing-algorithms`/`analyzing-results` → `tutorials/problem`/`multiobjective.algorithms`/`tutorials/experiment`); removed the entry where nothing close exists rather than force a misleading redirect (all of `advanced-topics.rst`'s Distributed Computing/Custom Development/Integration/Research Applications/Performance Optimization subsections — 16 of the 29 — had zero real content behind any entry). Verified with `sphinx-build`: 90 warnings before, 61 after.
- [x] `docs: document the statistical analysis module` — replaced the `ToDo` in `docs/source/tutorials/statistics.rst` with a real description of what `jmetal.lab.statistical_test` does and a link to its (previously orphaned) auto-generated API page, plus a note that it may be superseded by SAES. Done as part of fixing the orphaned-page warning, not a separate pass.

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
- [x] `chore: drop stale tuning entries from .gitignore` — `db.sqlite3`, `optuna/optuna_nsgaii.db`, `*_tuned_config.json`, `validation_results/`, `tuning_output/`, `*.prof`; confirmed nothing in the current codebase references any of them.
- [x] `ci: publish the documentation from a gh-pages branch` — resolved by the MkDocs migration below (item "promote the MkDocs staging tree to docs/" and its `gh-deploy` job): `origin/gh-pages` exists and is current (confirmed 2026-09-11 via `git fetch origin gh-pages`; its latest commit deploys the same commit just pushed to `main`). Settings → Pages is confirmed switched to serve from it.
- [x] `chore: untrack the built HTML site from docs/` — resolved the same way: `docs/` on `main` is MkDocs Markdown source, not built HTML (confirmed empty `find docs -iname '*.html'`).

### Packaging and metadata

- [x] `docs: sync Sphinx conf release, copyright and src-layout path` — `docs/source/conf.py:6,14`. `release` was `'1.7.0'` (package is 1.9.0), `sys.path` pointed at the repo root from before the `src/` layout move. Author/copyright synced against `pyproject.toml`'s actual authors list. Verified: same 61-warning build.
- [x] `build: move mockito to the test extra` — only used by `tests/util/test_comparator.py`; removed from base `dependencies`, `core`, `docs`, `distributed`, `complete`.
- [x] `build: unify the moocore version pin across dependency groups` — `>=0.1.8` base vs `>=0.1.9` extras, unified on `>=0.1.9`. Verified the package still builds and passes `twine check`.
- [x] `build: expose the quality indicator CLI as a console script` — added `jmetalpy-quality-indicator`, verified with a real install + `--help`. Documented alongside the existing `python -m` form.
- [x] `docs: add CITATION.cff` — matches `pyproject.toml`'s authors and the existing SWEVO BibTeX in `about.rst` field for field; validated as parseable YAML.
- [x] `docs: add CONTRIBUTING.md pointing at the coding and git guidelines` — also rewrote the stale `docs/source/contributing.rst` (a master/develop/feature/hotfix git-flow this repo never used, Python 3.6, and 8 screenshot references that point at files which don't exist in the repo) to point at the new file instead. `sphinx-build`: 61 warnings before, 39 after.
- [x] `docs: extract the changelog from README into CHANGELOG.md` — moved verbatim, README now points at it.
- [x] `docs: list the full algorithm and problem catalogue in the README` — also found missing while verifying against the real class inventory: UF (CEC'09), the multi-objective TSP, Osyczka2/Binh2, NHV and AHD quality indicators, and several crossover/mutation operators (PMX, CX, BLX-Alpha(-Beta), arithmetic, UNDX, differential evolution; non-uniform, Levy flight, power-law). Fixed a "bit-blip" → "bit-flip" typo along the way.

### CI

- [x] `ci: build the documentation with warnings as errors` — new `.github/workflows/docs.yml`, matching the existing lint/test/build workflows. Added a Docs badge to the README too.
- [x] `ci: execute the notebooks and examples` — new `.github/workflows/notebooks.yml` runs the 4 lighter tutorial notebooks via `nbconvert --execute` plus 9 representative example scripts (same set linked from the algorithm API pages). Two heavy ones excluded on purpose and documented in the workflow: `NSGAIISolvingZCAT_3obj.ipynb` (20 ZCAT problems × 100k evaluations — verified correct via a reduced scratch copy, just too slow for CI) and `hype_zdt1.py` (HYPE's exact-hypervolume fitness assignment, several minutes at documented settings). Verified every command locally before committing.
- [x] `ci: report test coverage` — `test.yml` now runs with `--cov` and uploads `coverage.xml` as a build artifact (Python 3.12 leg). No external service (Codecov etc.) wired up — that needs an account/token, a separate decision. Current total: 43%, confirming `lab/`/`algorithm/` are the weak spots already known.

---

## Documentation toolchain — migrate Sphinx → MkDocs (decided, in progress)

**Decision:** replace Sphinx (custom `guzzle` theme, RST) with **MkDocs + Material for MkDocs +
mkdocstrings[python]**. Reasons:

- `mkdocstrings[python]` parses Google-style docstrings directly (via `griffe`) — `CODING_GUIDELINES.md`
  already mandates that style, so no docstring rewriting is needed.
- Material for MkDocs is a mature, actively maintained theme configured entirely in `mkdocs.yml` —
  no more maintaining a bundled custom theme (`docs/source/_templates/guzzle/`).
- Content moves from RST to Markdown — lower friction for future contributors.
- CI story is simpler and directly equivalent to what L0 just built: `mkdocs build --strict` fails
  the build on any warning (broken link, missing page — the same thing `sphinx-build -W` now gates),
  and `mkdocs gh-deploy` (or the official `mkdocs-material` GitHub Actions recipe) publishes to
  `gh-pages` in one step — which also finally resolves the deferred `ci: publish the documentation
  from a gh-pages branch` / `chore: untrack the built HTML site from docs/` items above, since the
  new toolchain needs that branch-based deploy anyway.
- Considered and rejected as a smaller step: **Sphinx + MyST-Parser** (write Markdown, keep Sphinx/
  autodoc/napoleon and the CI already wired up). Less migration work, but keeps the custom theme and
  Sphinx's heavier configuration surface — doesn't address the actual maintenance burden.

**Real cost, not hidden:** every `.rst` page fixed during L0 (tutorials, advanced-topics, api
reference, the two hand-written `archive.rst`/`distance.rst` pages) needs re-authoring in Markdown.
Content, not just format, in the two hand-written pages — Sphinx-specific directives
(`.. autoclass::`, `.. toctree::`, `:doc:`) have no 1:1 Markdown equivalent and need real rework
against `mkdocstrings`/`mkdocs-nav` conventions.

**Sequencing — recommendation: migrate before starting L1, not after.**

1. L1 will produce new documentation of its own (a component-architecture tutorial, API pages for
   the new `jmetal.component` package). Migrating first means that content gets written once, in the
   final format — migrating after L1 means writing it in RST now and re-migrating it later, doubling
   that slice of the work for no benefit.
2. Full context on every page's current content and structure is fresh *right now* (L0 just read and
   fixed every one of them) — that context decays. Doing the migration while it's cheap to get right
   is better than reconstructing it later.
3. The migration is orthogonal to L1's architecture work — no technical dependency runs either
   direction, so there's no efficiency loss in sequencing it first; it's purely about not paying for
   the same content twice.

The counter-case — do L1 first, since it's the substantively higher-value work and a tooling swap
is infrastructure, not user-facing capability — is reasonable too; recorded here so the tradeoff is
visible, not just the recommendation.

**Decided: migrate now, before L1.** Antonio wants documentation treated as a priority, not an
afterthought bolted on once L1 lands — confirms the recommendation above. Sphinx is being dropped
entirely, not kept as a fallback.

### Migration checklist

- [x] `build: add mkdocs, mkdocs-material, mkdocstrings[python] as a docs dependency group`
- [x] `docs: scaffold mkdocs.yml and the new content structure`
- [x] `docs: migrate the tutorials to Markdown` — `problem`, `observer`, `evaluator`, `visualization`, `experiment`, `statistics`, `quality_indicators_cli`
- [x] `docs: migrate advanced-topics to Markdown` — `distance-based-archive`, `custom-archives`, `advanced-selection-strategies`
- [x] `docs: rebuild the api reference on mkdocstrings` — replaces every `automodule`/`autoclass` page (`api/core/*`, `api/util/*`, `api/operator/*`, `api/problem/*`, `api/algorithm/*`, `api/jmetal.lab.statistical_test.rst`)
- [x] `docs: migrate archive.rst and distance.rst`, preserving their hand-written content (performance notes, worked examples) — not just their `automodule` blocks
- [x] `docs: migrate getting-started, user-guide, api-reference, advanced-topics, contributing, about, index to Markdown nav`
- [x] `ci: replace the Sphinx docs workflow with mkdocs build --strict`
- [x] `chore: remove docs/source's Sphinx config and the bundled guzzle theme`
- [x] `docs: update CONTRIBUTING.md/README references from .rst to the new structure`
- [x] `docs: promote the MkDocs staging tree to docs/` — replaced the old Sphinx-built HTML at `docs/` root with the migrated Markdown source; resolves the L0 "untrack the built HTML site from docs/" item too
- [x] `ci: deploy via mkdocs gh-deploy to a gh-pages branch` — build job unchanged, new deploy job gated to pushes on `main`
- [x] **Manual step (user, one-time):** switch the repo's Pages source in Settings → Pages to "Deploy from a branch" / `gh-pages` — confirmed done 2026-09-11.

---

## L1 — Component-based core (`feat/component-architecture`)

Scope for this round: **MOEAs only, starting with NSGA-II.** PSO (catalogue and template) is
deliberately out of scope until NSGA-II is implemented and validated — see Phase 3.

Package layout, verified against `jmetal-component`
(`org.uma.jmetal.component`) in the Java codebase:

```text
src/jmetal/component/
├── algorithm/
│   ├── evolutionary_algorithm.py       # EvolutionaryAlgorithm template, a direct translation of
│   │                                    # algorithm/EvolutionaryAlgorithm.java's run()
│   └── multiobjective/
│       ├── nsgaii.py                   # build_nsgaii()
│       └── smsemoa.py                  # build_smsemoa() (Phase 2 adds spea2.py, mocell.py)
└── catalogue/
    ├── common/
    │   ├── solutions_creation.py       # Protocol SolutionsCreation + RandomSolutionsCreation
    │   ├── evaluation.py               # Protocol Evaluation + SequentialEvaluation
    │   └── termination.py              # Protocol Termination + TerminationByEvaluations
    └── ea/
        ├── selection.py                # Protocol Selection + TournamentSelection
        ├── variation.py                # Protocol Variation + CrossoverAndMutationVariation
        └── replacement.py              # Replacement ABC + RankingAndDensityEstimatorReplacement
```

Design decisions (Python idioms over literal Java translation, confirmed with the user):
`typing.Protocol` for the single-method components (`SolutionsCreation`, `Selection`, `Termination`)
instead of `ABC` — a plain function already satisfies the contract, no subclassing ceremony; `ABC`
only where implementations share real state or behavior (`Replacement`, given the three existing
classes in `operator/replacement.py` share no base today). A factory function
(`build_nsgaii(problem, population_size, offspring_population_size, crossover, mutation, *,
selection=None, variation=None, replacement=None, termination=None, rng=None)`) instead of a
chainable `NSGAIIBuilder` class — Java's builder pattern exists to simulate keyword arguments, which
Python already has natively. Every component and factory function ships with its Google-style
docstring and its unit test in the same commit, not as a follow-up.

### Phase 1 — EA template and NSGA-II

- [x] `refactor(operator): introduce a Replacement ABC for the existing replacement classes` — `RankingAndDensityEstimatorReplacement`, `RankingAndCrowdingDistanceReplacement`, `SMSEMOAReplacement` in `src/jmetal/operator/replacement.py` only share `replace()` by convention today
- [x] `feat(component): add the SolutionsCreation, Evaluation and Termination protocols and defaults` — `catalogue/common/{solutions_creation,evaluation,termination}.py`
- [x] `feat(component): add the Selection, Variation and Replacement protocols and defaults` — `catalogue/ea/{selection,variation,replacement}.py`
- [x] `feat(component): add AlgorithmState with a backwards-compatible observer payload`
- [x] `feat(component): add the EvolutionaryAlgorithm template`
- [x] `feat(component): thread an injectable random generator and seed through the template`
- [x] `feat(component): add build_nsgaii()` — factory function in `algorithm/multiobjective/nsgaii.py`, not a builder class
- [x] `test(component): assert build_nsgaii matches the classic NSGAII for a fixed seed` — acceptance test for Phase 1: identical fronts on ZDT1 and DTLZ2, verified passing
- [x] `test(component): assert run reproducibility, including with MultiprocessEvaluator` — verified passing, including with a real `MultiprocessEvaluator`
- [x] `docs: document the component-based architecture` — `docs/advanced-topics/component-architecture.md`

**Phase 1 complete.** All nine checklist items landed as separate commits on
`feat/component-architecture`, each with lint/tests green. The acceptance criterion holds: given the
same seed, `build_nsgaii(...)` and the classic `NSGAII(...)` produce identical final populations on
both ZDT1 and DTLZ2.

**Ad-hoc addition — catalogue introspection.** `jmetal.component.catalogue_info.describe_catalogue()`
answers "what components exist, what implementations are available, what are each implementation's
control parameters (name, type, default)" by inspecting the real classes (`inspect.signature`), not
from a hand-maintained description — deliberately narrower than Evolver's YAML parameter spaces (no
ranges/distributions to explore those parameters; that's an auto-configuration concern, out of scope
for now). Surfaced a real bug: nine `jmetal.operator.mutation` constructors declared `rng: object`
instead of `rng: np.random.Generator`, fixed in the same round. Documented in
`docs/advanced-topics/component-architecture.md`.

**Ad-hoc addition — external archives.** `EvolutionaryAlgorithm` and `build_nsgaii()` gained an
`archive: Archive | None` parameter, paired with `SequentialEvaluationWithArchive` (an `Evaluation`
decorator that copies every evaluated solution into the archive). Mirrors jMetal Java's
`SequentialEvaluationWithArchive` + `EvolutionaryAlgorithmWithArchive`, but as a constructor
parameter rather than a subclass. Verified with three integration tests (bounded
`CrowdingDistanceArchive` on ZDT4, HV > 0.60 at 20000 evaluations; steady-state ZDT1 with
`offspring_population_size=1`, HV > 0.63; unbounded `NonDominatedSolutionsArchive` on DTLZ2,
HV > 0.35 at 40000 evaluations) and four manually-run examples under `examples/component/` with
inspected front plots. Documented in `docs/advanced-topics/component-architecture.md`.

**Ad-hoc addition — `Archive.add_batch()`.** The unbounded-archive-on-DTLZ2 case was originally too
slow (~43s) for a `pytest` test: `NonDominatedSolutionsArchive.add()` is O(n) per call, and
`SequentialEvaluationWithArchive` called it once per evaluated solution instead of using the whole
generation it already had on hand. Added `Archive.add_batch()` (default: loop over `add()`; overridden
on `NonDominatedSolutionsArchive` with a single `moocore.is_nondominated()` call over the combined
archive + incoming batch) and switched `SequentialEvaluationWithArchive` to call it once per
generation. DTLZ2 case: ~43s → ~1.4s, same archive size and hypervolume. `BoundedArchive`/
`CrowdingDistanceArchive` untouched (the default loop; that case was never the bottleneck).

**Correctness fix — `result()` with an unbounded archive.** `EvolutionaryAlgorithm.result()` was
returning the whole archive whenever one was set, including for `NonDominatedSolutionsArchive`,
which can grow into the thousands over a run -- unlike jMetal Java's `BestSolutionsArchive`, which
wraps an unbounded archive but reduces it to the population size via distance-based subset selection
before returning. Fixed to match: `result()` now calls the already-existing
`distance_based_subset_selection_robust` (`jmetal/util/archive.py`) whenever the archive holds more
solutions than the population size, a no-op for bounded archives (`CrowdingDistanceArchive`, ...),
which never exceed that size to begin with. As a side effect, this also fixed the DTLZ2 3D plot in
`notebooks/NSGAIIComponentBased.ipynb`, which had been unreadable (an opaque ~9500-point scatter
renders as a solid blob) -- 100 well-distributed points display cleanly with the same
`jmetal.lab.visualization.Plot` the other sections already use, no custom plotting code needed.

**Analysis on record (not yet acted on): full NSGA-II-Double parameter-space feasibility.** Compared
jMetal Java's `NSGAIIDouble.yaml` (Evolver) against jMetalPy operator-by-operator. Verdict: nothing
hits an architectural wall. Already present: most crossover/mutation operators (`SBXCrossover`,
`BLXAlphaCrossover`, `BLXAlphaBetaCrossover`, `ArithmeticCrossover`, `UnimodalNormalDistributionCrossover`,
`PolynomialMutation`, `UniformMutation`, `NonUniformMutation`, `LevyFlightMutation`, `PowerLawMutation`),
all three repair strategies (`RandomUniformRepair`, `ClampFloatRepair`, `BoundSwapRepair` — the last
matches Java's "round"/wrap-to-opposite-bound exactly), and the archive machinery above. Missing but
feasible, roughly in order of effort: a `KNNDistanceArchive` (trivial — `KNearestNeighborDensityEstimator`
already exists, just needs an ~8-line `BoundedArchive` wrapper like `CrowdingDistanceArchive`);
`latinHypercubeSampling`/`sobol`/`cauchy`/`oppositionBased`/`scatterSearch` solutions-creation strategies
(jMetalPy only has random creation today; LHS and Sobol have direct `scipy.stats.qmc` support);
`wholeArithmetic`/`laplace`/`fuzzyRecombination`/`PCX` crossovers, `linkedPolynomial` mutation, and
`boltzmann`/`ranking`/`stochasticUniversalSampling` selection strategies (all self-contained, 100-220
line ports); `spatialSpreadDeviationArchive`/`angleArchive` (need new density estimators, medium
effort). Not scheduled — recorded here so the next pass doesn't re-derive it.

### Phase 1b — decouple from threading.Thread

Verified nobody calls `algorithm.start()`/`.join()` anywhere in `src/`, `examples/`, `tests/`,
`notebooks/`, or `docs/source` — this is dead coupling, not a used feature. `core/algorithm.py`
already has a `__getstate__`/`__setstate__` pair that strips `threading.Thread`'s unpicklable
internals so `Algorithm` survives `ProcessPoolExecutor`; removing the `Thread` inheritance lets that
whole workaround be deleted too.

- [x] `refactor(core): define an algorithm protocol independent of threading.Thread` — `AlgorithmProtocol`, satisfied by both the classic hierarchy and `EvolutionaryAlgorithm`
- [x] `refactor(lab): type Job against the algorithm protocol` — verified end-to-end with a component-based algorithm run through `Experiment`/`ProcessPoolExecutor`
- [x] `refactor(core)!: stop inheriting from threading.Thread` — added `run_in_thread()`; also dropped `LocalSearch`'s and `SimulatedAnnealing`'s redundant direct `threading.Thread` inheritance, which would otherwise have been left half-initialized once `Algorithm.__init__` stopped calling `Thread.__init__`

**Phase 1b complete.** `Algorithm` and every subclass are plain objects now — no `threading.Thread`,
no `__getstate__`/`__setstate__` workaround. Verified: `LocalSearch` and `SimulatedAnnealing` still
run and pickle correctly; the full suite (925 tests) and lint are green.

### Phase 2 — widen the catalogue (remaining MOEAs)

**Scope decision (2026-09):** the component package ships for 2.0 with exactly three algorithms —
NSGA-II, MOEA/D (classic and DE), and SMS-EMOA. `build_spea2()`, `build_mocell()`, and
`build_genetic_algorithm()` are out of scope for this release, not merely deferred; removed from
this checklist rather than left as pending boxes. Phase 3 (PSO) is out of scope for the same reason
— see its section below.

- [x] `fix(operator): implement the full multi-front SMS-EMOA replacement algorithm` — `SMSEMOAReplacement.replace()` only pruned front 0 with a fixed constructor-time reference point; now keeps every front but the last whole and prunes the last by hypervolume contribution, with a reference point recomputed per call, matching the classic `SMSEMOA`
- [x] `feat(component): add the RandomSelection selection component` — `catalogue/ea/selection.py`, SMS-EMOA's default mating selection
- [x] `feat(component): add build_smsemoa()` — factory function in `algorithm/multiobjective/smsemoa.py`, no `offspring_population_size` parameter (SMS-EMOA is steady-state by definition, always 1)
- [x] `test(component): assert build_smsemoa matches the classic SMSEMOA for a fixed seed` — identical fronts on ZDT1 and DTLZ2, verified passing
- [x] `docs: document build_smsemoa() in the component architecture page`
- [x] `feat(core): add an optional rng to Problem.create_solution()` — prerequisite for MOEA/D single-seed reproducibility, see below
- [x] `feat(component): thread rng through RandomSolutionsCreation`
- [x] `feat(operator): add an optional rng to NaryRandomSolutionSelection`
- [x] `docs: correct the outdated MOEA/D component-fit risk note in MODERNIZATION.md` — see Phase 3's correction note
- [x] `feat(component): add MOEADContext and the pluggable subproblem sequence generator` — `catalogue/ea/moead.py`
- [x] `feat(component): add MOEADSelection`
- [x] `feat(component): add DifferentialEvolutionCrossoverVariation`
- [x] `feat(component): add MOEADReplacement`
- [x] `feat(component): add build_moead()` — classic, any crossover, `PenaltyBoundaryIntersection` by default
- [x] `feat(component): add build_moead_de()` — differential evolution, `Tschebycheff` by default
- [x] `test(component): assert MOEADReplacement matches the classic replacement logic given identical inputs` — structural equivalence, not full-run (see below)
- [x] `test(component): reach expected hypervolume floors with build_moead and build_moead_de` — ZDT1 and DTLZ2, both variants
- [x] `docs: document build_moead()/build_moead_de() in the component architecture page`

**SMS-EMOA complete.** SMS-EMOA reuses `RandomSolutionsCreation`, `SequentialEvaluation`,
`TerminationByEvaluations` and `CrossoverAndMutationVariation` unchanged from NSGA-II — only
selection and replacement needed new/fixed code. `SMSEMOAReplacement`
(`src/jmetal/operator/replacement.py`) turned out to be a real, unnoticed bug rather than a reusable
component: it only ever pruned front 0 of the ranked merged population, so a dominated solution in a
later front could survive while a non-dominated one from front 0 was discarded, and its reference
point was fixed at construction time instead of tracking the population. It was unused in production
(only its own tests exercised it), so fixed in place rather than duplicated. The corrected version
generalizes the classic `SMSEMOA.replacement()`'s single-excess-solution truncation (keep every
front but the last, sort the last front by hypervolume contribution descending, keep as many as
still fit) to any number of excess solutions. Getting this bit-for-bit equivalent to the classic
algorithm surfaced a subtlety beyond set-membership: `RandomSelection` picks mating-pool members by
list index, so the *order* of the replaced population matters, not just its contents — an earlier
iterative-removal draft that preserved original front order (rather than re-sorting by hypervolume
contribution after truncation, like the classic implementation does) produced a different-but-valid
population that silently diverged from the classic algorithm after ~25 generations. Verified with
the same equivalence-test methodology as Phase 1 (`test_smsemoa_equivalence.py`, ZDT1 and DTLZ2).

**MOEA/D complete — classic and MOEA/D-DE.** Both `build_moead()` and `build_moead_de()` build on
the unmodified `EvolutionaryAlgorithm` template (see Phase 3's correction note below for why that
was ever in doubt). The one genuinely new piece: `MOEADContext`
(`catalogue/ea/moead.py`) — a small object, constructed once per run and passed by reference into
`MOEADSelection`, `MOEADReplacement` and (DE variant only) `DifferentialEvolutionCrossoverVariation`,
carrying the one thing none of the six generic component protocols carry and shouldn't: which
subproblem the current iteration is processing, and whether its mating pool/replacement scan is
scoped to that subproblem's neighborhood or the whole population. Mirrors jMetal Java's own
`SequenceGenerator<Integer>`, shared the same way between `MOEADBuilder`/`MOEADDEBuilder`'s
components. `build_moead()`'s classic path needed no new `Variation` at all — plain
`CrossoverAndMutationVariation` already works with SBX, since only the DE variant needs a third
"parent" (the current subproblem's own solution) that isn't part of the sampled mating pool.

A real, corrected misconception surfaced along the way: jMetalPy's existing (non-component)
`jmetal.algorithm.multiobjective.moead.MOEAD` — despite the plain name — is *already* MOEA/D-DE
(`crossover` is typed `DifferentialEvolutionCrossover` and required); there has never been a classic
SBX-based MOEA/D in jMetalPy. Cross-checked directly against jMetal Java's own legacy (non-component)
`jmetal-algorithm` module: its `MOEAD.java` has the identical situation — a constructor accepting a
generic `CrossoverOperator<DoubleSolution>` that gets force-cast to `DifferentialEvolutionCrossover`
in the same line, so passing SBX there would throw `ClassCastException`, and its own runnable example
only ever constructs a DE crossover. So the classic/DE split `build_moead()`/`build_moead_de()`
provide didn't exist anywhere in the jMetal ecosystem outside `jmetal-component`/Evolver before this.

**A deliberate reproducibility redesign, not a like-for-like port.** The classic `MOEAD` mixes three
incompatible random sources: the global `random` module, *global legacy* `numpy.random` (direct
`np.random.permutation()` calls, not a `Generator` instance — used by its `Permutation` helper), and
each operator's own `np.random.Generator`. No seed reconciles a `Generator` (PCG64) with
`random`/legacy `numpy.random` (Mersenne Twister) — they're different algorithms — so no amount of
seeding can make it reproducible from one value. `build_moead()`/`build_moead_de()` were built
"rng-clean" instead: every MOEA/D-specific random decision draws exclusively from the shared `rng`,
matching the project's already-declared direction ("migrating the remaining operators to the
injectable-`rng` pattern is ongoing," Phase 1's notes) rather than replicating legacy behavior.
Getting there required going one level deeper than `build_nsgaii()`/`build_smsemoa()` did:
`Problem.create_solution()` gained an optional `rng` (10 explicit problem overrides plus
`FloatProblem`/`IntegerProblem`'s two shared implementations touched; ~130+ other problem classes
inherit the fix for free), and `RandomSolutionsCreation` now forwards its own `rng` into it — which,
via `build_nsgaii()`'s/`build_smsemoa()`'s existing `rng=` parameter, makes their population creation
reproducible too, as a free side effect. One subtlety caught by running the existing equivalence
tests immediately after a naive first attempt: `EvolutionaryAlgorithm._thread_rng_into_components()`
would have auto-injected the algorithm's own `rng` into `RandomSolutionsCreation` even when the user
passed none, silently switching every unseeded `build_nsgaii()`/`build_smsemoa()` call onto a
different random source for population creation and breaking
`test_nsgaii_equivalence.py`/`test_smsemoa_equivalence.py`. Fixed by excluding `solutions_creation`
from that auto-injection loop — population creation only becomes `rng`-reproducible when a factory
explicitly forwards its own `rng` parameter into it, which `build_nsgaii()`/`build_smsemoa()`/
`build_moead()`/`build_moead_de()` all now do.

The cost of this choice: no execution-level equivalence test against the classic `MOEAD` is possible
for `build_moead_de()`, unlike `test_nsgaii_equivalence.py`/`test_smsemoa_equivalence.py`'s
bit-identical-fronts standard — two runs seeded "the same way" draw from genuinely different PRNG
streams. Verified instead with two narrower, still-meaningful checks:
`test_moead_replacement_structural_equivalence.py` (`MOEADReplacement`'s replace/keep decisions match
the classic `update_current_subproblem_neighborhood()` exactly, given identical inputs — the logic
that matters, isolated from randomness) and `test_moead_integration.py` (hypervolume floors on ZDT1
and DTLZ2 for both variants, each fully reproducible from its one fixed seed — no
`random.seed()`/`np.random.seed()` involved, verified directly by running each test's assembly twice
and comparing fronts).

**Analysis on record (not yet acted on): a shared `rng=` for the classic algorithm hierarchy.**
Raised during MOEA/D's reproducibility design (in the form of a proposed project-wide
`RandomGenerator` class) and deliberately scoped out as too large for this round, but worth acting on
separately: `Algorithm`/`GeneticAlgorithm` and their subclasses (`NSGAII`, `SMSEMOA`, `MOEAD`, ...)
have no constructor-level `rng`, unlike the component template's `EvolutionaryAlgorithm(rng=...)` +
`_thread_rng_into_components()`. A **class-level singleton** version of this (a `RandomGenerator`
with `_generator` as a class attribute, mutated via a `.seed()` classmethod) was considered and
rejected: it would reintroduce exactly the problem this project already fixed away from once before
(operators used to fall back to the bare, shared `numpy.random` module — unpicklable across
`ProcessPoolExecutor` workers, and coupling unrelated operators' random-draw counts to each other —
fixed to per-instance `np.random.default_rng()`, see the note above). The instance-based version
(`rng=` as a plain constructor parameter, threaded to `population_generator`/`selection_operator`/
`crossover_operator`/`mutation_operator` the same way `_thread_rng_into_components()` already does
for components) would be a real, low-risk improvement — just project-wide in scope rather than
MOEA/D-specific, so left for a dedicated pass.

### Phase 3 — PSO (out of scope for 2.0)

Per the Phase 2 scope decision above, the component package ships for 2.0 with NSGA-II, MOEA/D and
SMS-EMOA only. The PSO catalogue, its `ParticleSwarmOptimization` template, and `build_smpso()`
are not planned for this release.

**Correction (2026-09):** this section previously claimed "MOEA/D does not fit the component model
well — Java jMetal's own docs admit needing complex, tightly-coupled components for it." Fresh
investigation of `jmetal-component` disproves this: `MOEADBuilder`/`MOEADDEBuilder` exist there and
build a plain `EvolutionaryAlgorithm<S>`, the same generic template `NSGAIIBuilder`/
`SMSEMOABuilder` use, with no modification — and `docs/component.rst` documents MOEA/D with no
"hard to fit" caveat. No design note backing the original claim was found; it appears to have been
an unverified assumption. See Phase 2's MOEA/D entry for what actually makes it fit: a small shared
per-iteration context object, mirroring the `SequenceGenerator<Integer>` `MOEADBuilder`/
`MOEADDEBuilder` pass by reference between `Selection`, `Variation` and `Replacement` — not a
protocol change, and not "complex, tightly-coupled components."

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
