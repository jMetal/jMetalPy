import numpy as np
import pytest

from jmetal.core.quality_indicator import EpsilonIndicator, GenerationalDistance
from jmetal.lab.experiment import generate_summary_from_experiment

FUN_CONTENTS = "0.0 1.0\n0.5 0.5\n1.0 0.0\n"
REFERENCE_FRONT_CONTENTS = "0.0 1.0\n0.5 0.5\n1.0 0.0\n"


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
