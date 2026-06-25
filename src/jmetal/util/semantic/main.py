from pathlib import Path

try:
    from .queries import (
        load_graph,
        get_position_constraints,
        get_route_constraints
    )
    from .constraints import (
        position_constraint,
        route_constraint
    )
except ImportError:
    # Allow running this file directly: python src/jmetal/util/semantic/main.py
    import sys

    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from jmetal.util.semantic.queries import (
        load_graph,
        get_position_constraints,
        get_route_constraints
    )
    from jmetal.util.semantic.constraints import (
        position_constraint,
        route_constraint
    )




def main():

    ontology_path = (
        Path(__file__).resolve().parents[4] /
        "resources" /
        "ontologies" /
        "traffic-tsp.owl"
    )

    graph = load_graph(str(ontology_path))

    print("Triples:", len(graph))

    position_constraints = get_position_constraints(graph)

    print("\nPosition constraints")

    for c in position_constraints:
        print(c)

    route_constraints_list = get_route_constraints(graph)

    print("\nRoute constraints")

    for c in route_constraints_list:
        print(c)

    evaluators = []

    for c in position_constraints:

        if c.position.isdigit():

            evaluators.append(
                position_constraint(
                    target_position=int(c.position) - 1,
                    building_id=c.building_id
                )
            )

    for c in route_constraints_list:

        evaluators.append(
            route_constraint(
                c.first_building,
                c.second_building
            )
        )

    solution = [4, 1, 3, 2, 5]

    print("\nEvaluation")

    for evaluator in evaluators:
        print(evaluator(solution))


if __name__ == "__main__":
    main()