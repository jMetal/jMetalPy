import json
from pathlib import Path

import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, ZDT4
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import HyperVolume

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "best_nsgaii_zdt1_params.json"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de configuracion: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    params = cfg["best_params"]
    population_size = 100
    max_evaluations = 25000
    ref_point_offset = float(cfg.get("ref_point_offset", 0.0))

    reference_front = read_solutions(str(ROOT_DIR / "resources/reference_fronts/ZDT1.pf"))
    ref_point = np.max([s.objectives for s in reference_front], axis=0) + ref_point_offset

    algorithm = NSGAII(
        problem=ZDT4(),
        population_size=population_size,
        offspring_population_size=int(params["offspring_population_size"]),
        mutation=PolynomialMutation(
            probability=float(params["mutation_probability"]), distribution_index=float(params["mutation_eta"])
        ),
        crossover=SBXCrossover(
            probability=float(params["crossover_probability"]), distribution_index=float(params["crossover_eta"])
        ),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()

    front = get_non_dominated_solutions(algorithm.result())
    hv = HyperVolume(reference_point=ref_point)
    hv_value = hv.compute(np.array([s.objectives for s in front]))

    # Save results to file
    print_function_values_to_file(front, "FUN.Opt." + algorithm.label)
    print_variables_to_file(front, "VAR.Opt." + algorithm.label)


    print("Algorithm:", algorithm.get_name())
    print("Config file:", CONFIG_PATH)
    print("Parametros:", params)
    print("Hypervolume:", hv_value)
