import numpy as np
import pytest

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.core.quality_indicator import EpsilonIndicator, GenerationalDistance
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations

FUN_CONTENTS = "0.0 1.0\n0.5 0.5\n1.0 0.0\n"
REFERENCE_FRONT_CONTENTS = "0.0 1.0\n0.5 0.5\n1.0 0.0\n"


def _make_nsgaii_job(run: int, population_size: int = 10, max_evaluations: int = 40) -> Job:
    problem = ZDT1()
    algorithm = NSGAII(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables(), distribution_index=20
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    return Job(algorithm=algorithm, algorithm_tag="NSGAII", problem_tag="ZDT1", run=run)


class TestExperimentRun:
    def test_should_run_jobs_across_multiple_processes_and_collect_real_results(
        self, tmp_path
    ):
        jobs = [_make_nsgaii_job(run) for run in range(2)]
        experiment = Experiment(output_dir=str(tmp_path), jobs=jobs, m_workers=2)

        experiment.run()

        assert len(experiment.job_data) == 2
        for data in experiment.job_data:
            # A stale/never-run algorithm would report 0 evaluations -- this is the
            # exact bug that shipped when Experiment.run() executed jobs eagerly in
            # the parent instead of actually submitting them to the pool.
            assert data["EVALUATIONS"] >= 40
            assert len(data["SOLUTIONS"]) == 10
        for run in range(2):
            assert (tmp_path / "NSGAII" / "ZDT1" / f"FUN.{run}.tsv").exists()

    def test_should_propagate_an_exception_raised_inside_a_job(self, tmp_path):
        # Force Job.execute()'s output directory creation to fail deterministically:
        # a real, plain file sitting where the algorithm_tag directory needs to go.
        (tmp_path / "NSGAII").write_text("not a directory")
        jobs = [_make_nsgaii_job(run=0)]
        experiment = Experiment(output_dir=str(tmp_path), jobs=jobs, m_workers=1)

        with pytest.raises((NotADirectoryError, FileExistsError)):
            experiment.run()


class TestJobWithAComponentBasedAlgorithm:
    """Job is typed against AlgorithmProtocol, not the threading.Thread-based
    Algorithm ABC, specifically so that jmetal.component algorithms -- which don't
    inherit from either -- work here too, including across the process boundary
    Experiment.run() sends jobs through.
    """

    def test_should_run_a_component_based_algorithm_across_multiple_processes(self, tmp_path):
        def make_job(run: int) -> Job:
            problem = ZDT1()
            algorithm = build_nsgaii(
                problem,
                population_size=10,
                offspring_population_size=10,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                mutation=PolynomialMutation(
                    probability=1.0 / problem.number_of_variables(), distribution_index=20
                ),
                termination=TerminationByEvaluations(max_evaluations=40),
                rng=np.random.default_rng(run),
            )
            return Job(algorithm=algorithm, algorithm_tag="NSGAII", problem_tag="ZDT1", run=run)

        jobs = [make_job(run) for run in range(2)]
        experiment = Experiment(output_dir=str(tmp_path), jobs=jobs, m_workers=2)

        experiment.run()

        assert len(experiment.job_data) == 2
        for data in experiment.job_data:
            assert data["EVALUATIONS"] >= 40
            assert len(data["SOLUTIONS"]) == 10
        for run in range(2):
            assert (tmp_path / "NSGAII" / "ZDT1" / f"FUN.{run}.tsv").exists()


@pytest.fixture
def experiment_layout(tmp_path, monkeypatch):
    """Lay out <input_dir>/<algorithm>/<problem>/FUN.0.tsv plus a matching
    <problem>.pf reference front, matching what Experiment.run() produces and what
    generate_summary_from_experiment expects to walk.
    """
    monkeypatch.chdir(tmp_path)

    input_dir = tmp_path / "data"
    problem_dir = input_dir / "NSGAII" / "ZDT1"
    problem_dir.mkdir(parents=True)
    (problem_dir / "FUN.0.tsv").write_text(FUN_CONTENTS)
    (problem_dir / "TIME.0").write_text("0.01")

    reference_fronts_dir = tmp_path / "reference_fronts"
    reference_fronts_dir.mkdir()
    (reference_fronts_dir / "ZDT1.pf").write_text(REFERENCE_FRONT_CONTENTS)

    return input_dir, reference_fronts_dir


class TestGenerateSummaryFromExperiment:
    def test_should_load_the_reference_front_from_disk_for_indicators_that_need_one(
        self, experiment_layout
    ):
        input_dir, reference_fronts_dir = experiment_layout
        indicator = GenerationalDistance()

        generate_summary_from_experiment(
            input_dir=str(input_dir),
            quality_indicators=[indicator],
            reference_fronts=str(reference_fronts_dir),
        )

        assert indicator.reference_front is not None
        assert isinstance(indicator.reference_front, np.ndarray)

    def test_should_write_a_summary_row_per_indicator_without_raising(
        self, experiment_layout
    ):
        input_dir, reference_fronts_dir = experiment_layout

        generate_summary_from_experiment(
            input_dir=str(input_dir),
            quality_indicators=[GenerationalDistance(), EpsilonIndicator()],
            reference_fronts=str(reference_fronts_dir),
        )

        summary = (input_dir.parent / "QualityIndicatorSummary.csv").read_text()
        rows = [line for line in summary.splitlines() if line]

        assert rows[0] == "Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue"
        indicator_names = {row.split(",")[3] for row in rows[1:]}
        assert indicator_names == {"Time", "GD", "EP"}
        # The solution front exactly matches the reference front, so both
        # distance-based indicators should report (approximately) zero.
        for row in rows[1:]:
            if row.split(",")[3] in ("GD", "EP"):
                assert float(row.split(",")[4]) == pytest.approx(0.0, abs=1e-9)
