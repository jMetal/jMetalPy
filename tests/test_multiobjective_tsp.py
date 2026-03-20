import math

from jmetal.problem.multiobjective.multiobjective_tsp import MultiObjectiveTSP


def write_tsplib(path, coords):
    with open(path, "w") as fh:
        fh.write("NAME: test\n")
        fh.write(f"DIMENSION: {len(coords)}\n")
        fh.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            fh.write(f"{i} {x} {y}\n")


def test_multiobjective_tsp_basic(tmp_path):
    coords1 = [(0, 0), (1, 0), (0, 1)]
    coords2 = [(0, 0), (2, 0), (0, 2)]

    f1 = tmp_path / "inst1.tsp"
    f2 = tmp_path / "inst2.tsp"

    write_tsplib(f1, coords1)
    write_tsplib(f2, coords2)

    problem = MultiObjectiveTSP([str(f1), str(f2)])

    # create a specific solution [0,1,2]
    solution = problem.create_solution()
    solution.variables = [0, 1, 2]
    solution = problem.evaluate(solution)

    assert problem.number_of_objectives() == 2
    # compute expected objectives using the same rounding rule
    def tour_length(coords):
        dist = 0
        for i in range(len(coords) - 1):
            dx = coords[i][0] - coords[i + 1][0]
            dy = coords[i][1] - coords[i + 1][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist += int(d + 0.5)
        # close tour
        dx = coords[0][0] - coords[-1][0]
        dy = coords[0][1] - coords[-1][1]
        d = math.sqrt(dx * dx + dy * dy)
        dist += int(d + 0.5)
        return dist

    assert solution.objectives[0] == tour_length(coords1)
    assert solution.objectives[1] == tour_length(coords2)


def test_resolves_resources_filename():
    # use the short name that exists under resources/TSP_instances
    problem = MultiObjectiveTSP(["test.tsp"])
    assert problem.number_of_variables() == 4
    sol = problem.create_solution()
    assert len(sol.variables) == 4
