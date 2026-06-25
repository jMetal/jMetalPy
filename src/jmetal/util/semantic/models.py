from dataclasses import dataclass


@dataclass(slots=True)
class PositionConstraint:
    building_id: int
    position: str


@dataclass(slots=True)
class RouteConstraint:
    first_building: int
    second_building: int