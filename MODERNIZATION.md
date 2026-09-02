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

- [ ] `docs: fix broken API calls in the quick-start snippets` — `docs/source/getting-started.rst:52,60`, `docs/source/index.rst:61,63`
- [ ] `docs: update the experiment tutorial to the current problem API` — `docs/source/tutorials/experiment.rst:37,66`
- [ ] `docs: rewrite the problem tutorial against the post-1.6 Problem API` — `docs/source/tutorials/problem.rst`
- [ ] `docs: generate algorithm API pages from the maintained examples` — replace the 17 stale `docs/source/api/algorithm/multiobjective/{eas,psos}/*.ipynb`
- [ ] `docs: write the missing api/core and api/util reference pages` — 7 pages referenced by `docs/source/api-reference.rst`
- [ ] `docs: prune toctree entries pointing at non-existent pages` — remaining ~29 dead links across `getting-started.rst`, `user-guide.rst`, `advanced-topics.rst`
- [ ] `docs: document the statistical analysis module` — replace the `ToDo` in `docs/source/tutorials/statistics.rst`

### Experiment runner

- [ ] `fix(lab): submit jobs to the process pool instead of running them eagerly` — `src/jmetal/lab/experiment.py:75`
- [ ] `fix(lab): propagate worker exceptions instead of discarding the futures`
- [ ] `test(lab): cover Experiment.run parallelism and error propagation`

### Tests

- [ ] `test(algorithm): rename the integration test file so pytest discovers it` — `tests/algorithm/ittest_algorithm.py`
- [ ] `test(algorithm): migrate the integration tests to pytest conventions` — `unittest.TestCase` → `test_should_<behavior>`
- [ ] `test: apply the declared smoke/slow/integration markers` — or remove them from `pyproject.toml:44-48` if unused by design

### Repository artifacts

- [ ] `chore: untrack generated result artifacts and ignore them` — `results/` (194 MB)
- [ ] `chore: untrack notebook checkpoints and experiment output artifacts` — `notebooks/.ipynb_checkpoints/`, `examples/experiment/{boxplot,latex}/`
- [ ] `chore: drop stale tuning entries from .gitignore` — leftovers from the removed `jmetal/tuning` package
- [ ] `ci: publish the documentation from a gh-pages branch`
- [ ] `chore: untrack the built HTML site from docs/` — 311 files, 27 MB

### Packaging and metadata

- [ ] `docs: sync Sphinx conf release, copyright and src-layout path` — `docs/source/conf.py:6,14`
- [ ] `build: move mockito to the test extra` — only used by `tests/util/test_comparator.py`
- [ ] `build: unify the moocore version pin across dependency groups` — `>=0.1.8` base vs `>=0.1.9` extras
- [ ] `build: expose the quality indicator CLI as a console script` — `src/jmetal/util/quality_indicator_cli.py` has no `[project.scripts]` entry
- [ ] `docs: add CITATION.cff`
- [ ] `docs: add CONTRIBUTING.md pointing at the coding and git guidelines`
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
