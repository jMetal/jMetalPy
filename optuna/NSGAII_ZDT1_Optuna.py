import json
import time
from pathlib import Path

import numpy as np
import optuna
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, ZDT4
from jmetal.util.solution import get_non_dominated_solutions, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import HyperVolume

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "best_nsgaii_zdt1_params.json"

reference_front = read_solutions(str(ROOT_DIR / "resources/reference_fronts/ZDT1.pf"))
ref_point = np.max([s.objectives for s in reference_front], axis=0) + 0.1  # offset

POPULATION_SIZE = 100
MAXIMUM_EVALUATIONS = 20000


def objective(trial: optuna.Trial) -> float:
    problem = ZDT4()
    pop_size = POPULATION_SIZE
    off_size = trial.suggest_int("offspring_population_size", 1, 200)
    pc = trial.suggest_float("crossover_probability", 0.7, 1.0)
    eta_c = trial.suggest_float("crossover_eta", 5.0, 400.0)
    pm = trial.suggest_float("mutation_probability",
                             1.0 / problem.number_of_variables(), 0.5)
    eta_m = trial.suggest_float("mutation_eta", 5.0, 400.0)
    max_eval = MAXIMUM_EVALUATIONS

    print(
        f"Trial {trial.number} params -> offspring_size={off_size}, "
        f"pc={pc:.4f}, eta_c={eta_c:.2f}, pm={pm:.4f}, eta_m={eta_m:.2f}"
    )

    algo = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=off_size,
        mutation=PolynomialMutation(probability=pm, distribution_index=eta_m),
        crossover=SBXCrossover(probability=pc, distribution_index=eta_c),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_eval),
    )
    algo.run()
    front = get_non_dominated_solutions(algo.result())
    hv = HyperVolume(reference_point=ref_point)
    objs = np.array([s.objectives for s in front])
    return hv.compute(objs)

sampler = optuna.samplers.NSGAIISampler(
    population_size=50,
    crossover_prob=0.9,
    mutation_prob=0.1,
    seed=42,
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)
start = time.perf_counter()
number_of_trials = 300
study.optimize(objective, n_trials=number_of_trials)
elapsed = time.perf_counter() - start
print("Best HV:", study.best_value)
print("Best params:", study.best_params)
print(f"Tiempo Optuna (s): {elapsed:.2f}")

payload = {
    "best_value": float(study.best_value),
    "best_params": study.best_params,
    "population_size": POPULATION_SIZE,
    "max_evaluations": MAXIMUM_EVALUATIONS,
    "ref_point_offset": 0.1,
    "n_trials": number_of_trials,
    "elapsed_seconds": elapsed,
}
with CONFIG_PATH.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Configuracion guardada en {CONFIG_PATH}")
