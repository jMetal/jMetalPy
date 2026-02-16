"""Validación de la mejor configuración de NSGA-II encontrada por tuning.

Este programa carga la mejor configuración del estudio de tuning y ejecuta
NSGA-II con esos parámetros, guardando los frentes y generando gráficas.

Uso:
    python tune_nsgaii_zdt1_validation.py
"""

import json
import logging
from pathlib import Path

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

# Configurar logging para omitir mensajes DEBUG e INFO
logging.getLogger("jmetal").setLevel(logging.WARNING)


def load_best_config(summary_path: Path) -> dict:
    """Carga la mejor configuración desde el archivo de resumen."""
    with open(summary_path, "r") as f:
        summary = json.load(f)
    return summary["best_config"]


if __name__ == "__main__":
    print("=" * 60)
    print("VALIDACIÓN DE NSGA-II CON CONFIGURACIÓN OPTIMIZADA")
    print("=" * 60)
    
    # Cargar la mejor configuración del tuning
    summary_path = Path(__file__).parent.parent.parent.parent / "artifacts_zdt1_tuning" / "study_nsgaii_zdt1" / "summary.json"
    
    if not summary_path.exists():
        # Configuración por defecto si no existe el archivo
        print("\nArchivo de resumen no encontrado. Usando configuración por defecto del tuning.")
        best_config = {
            "crossover.probability": 0.9735750169653813,
            "crossover.distribution_index": 22.238415028016057,
            "mutation.probability": 0.08828442097960408,
            "mutation.distribution_index": 20.007944686206947,
        }
    else:
        best_config = load_best_config(summary_path)
        print(f"\nConfiguración cargada desde: {summary_path.name}")
    
    # Mostrar configuración al principio
    print("\n" + "-" * 60)
    print("CONFIGURACIÓN DEL ALGORITMO")
    print("-" * 60)
    print(f"  Problema: ZDT1")
    print(f"  Población: 100")
    print(f"  Evaluaciones: 25000")
    print("\n  Parámetros optimizados:")
    for key, value in best_config.items():
        print(f"    {key}: {value:.6f}")
    print("-" * 60)
    
    # Configurar el problema
    problem = ZDT1()
    problem.reference_front = read_solutions(filename="resources/reference_fronts/ZDT1.pf")
    
    # Configurar operadores con los parámetros óptimos
    crossover = SBXCrossover(
        probability=best_config["crossover.probability"],
        distribution_index=best_config["crossover.distribution_index"],
    )
    mutation = PolynomialMutation(
        probability=best_config["mutation.probability"],
        distribution_index=best_config["mutation.distribution_index"],
    )
    
    # Configurar y ejecutar NSGA-II
    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    
    print("\nEjecutando NSGA-II...")
    algorithm.run()
    print("Ejecución completada.")
    
    front = get_non_dominated_solutions(algorithm.result())
    
    # Guardar resultados
    output_prefix = "FUN.NSGAII_tuned.ZDT1"
    print_function_values_to_file(front, output_prefix)
    print_variables_to_file(front, output_prefix.replace("FUN", "VAR"))
    
    # Generar gráficas
    png = save_plt_to_file(front, output_prefix, out_dir='.', html_plotly=True)
    print(f"\nResultados guardados:")
    print(f"  Frente: {output_prefix}")
    print(f"  Variables: {output_prefix.replace('FUN', 'VAR')}")
    print(f"  Gráfica: {png}")
    
    print(f"\nAlgoritmo: {algorithm.get_name()}")
    print(f"Problema: {problem.name()}")
    print(f"Tiempo de cómputo: {algorithm.total_computing_time:.2f} segundos")
    print(f"Soluciones en el frente: {len(front)}")
