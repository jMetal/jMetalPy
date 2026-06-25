def position_constraint(
        target_position: int,
        building_id: int
):

    def evaluate(solution: list[int]):

        if not solution:
            return -1.0

        if building_id not in solution:
            return -float(len(solution))

        if target_position < 0:
            return -float(abs(target_position))

        if target_position >= len(solution):
            return -float(target_position - (len(solution) - 1))

        if solution[target_position] == building_id:
            return 0.0

        penalty = abs(
            target_position -
            solution.index(building_id)
        )

        return -float(penalty)

    return evaluate

def route_constraint(
        first_building: int,
        second_building: int
):

    def evaluate(solution: list[int]):

        if not solution:
            return -1.0

        if first_building not in solution or second_building not in solution:
            return -float(len(solution))

        first_idx = solution.index(first_building)
        second_idx = solution.index(second_building)

        if second_idx == first_idx + 1:
            return 0.0

        return -float(abs(first_idx - second_idx))

    return evaluate