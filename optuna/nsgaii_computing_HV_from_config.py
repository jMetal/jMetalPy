import json
from pathlib import Path

import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import BLXAlphaCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, ZDT4
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import NormalizedHyperVolume

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "best_nsgaii_zdt1_params.json"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de configuracion: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def build_crossover(params: dict):
    crossover_type = params.get("crossover_type", "sbx")
    probability = float(params["crossover_probability"])

    if crossover_type == "sbx":
        distribution_index = float(params["crossover_eta"])
        return SBXCrossover(probability=probability, distribution_index=distribution_index)

    if crossover_type == "blxalpha":
        alpha = float(params["blx_alpha"])
        return BLXAlphaCrossover(probability=probability, alpha=alpha)

    raise ValueError(f"Tipo de cruce no soportado en config: {crossover_type}")


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    params = cfg["best_params"]
    population_size = int(cfg.get("population_size", 100))
    max_evaluations = int(cfg.get("max_evaluations", 25000))
    ref_point_offset = float(cfg.get("ref_point_offset", 0.0))

    reference_front = read_solutions(str(ROOT_DIR / "resources/reference_fronts/ZDT1.pf"))
    reference_front_objectives = np.array([s.objectives for s in reference_front])

    algorithm = NSGAII(
        problem=ZDT4(),
        population_size=population_size,
        offspring_population_size=int(params["offspring_population_size"]),
        mutation=PolynomialMutation(
            probability=float(params["mutation_probability"]), distribution_index=float(params["mutation_eta"])
        ),
        crossover=build_crossover(params),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()

    front = get_non_dominated_solutions(algorithm.result())
    # Construct the NormalizedHyperVolume using the reference front and
    # the scalar offset. This keeps behavior consistent with other examples
    # and ensures the internal HyperVolume is derived from the front.
    normalized_hv_indicator = NormalizedHyperVolume(
        reference_front=reference_front_objectives,
        reference_point_offset=ref_point_offset,
    )
    # Compute and cache the hypervolume of the reference front
    normalized_hv_indicator.set_reference_front(reference_front_objectives)
    hv_value = normalized_hv_indicator.compute(np.array([s.objectives for s in front]))

    # Save results to file
    print_function_values_to_file(front, "FUN.Opt." + algorithm.label)
    print_variables_to_file(front, "VAR.Opt." + algorithm.label)


    print("Algorithm:", algorithm.get_name())
    print("Config file:", CONFIG_PATH)
    print("Parametros:", params)
    print("Normalized Hypervolume:", hv_value)
