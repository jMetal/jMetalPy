from rdflib import Graph
from .models import PositionConstraint
from .models import RouteConstraint



TRAFFIC = """
PREFIX traffic: <http://www.khaos.uma.es/perception/traffic/khaosteam#>
"""


def load_graph(path: str) -> Graph:

    graph = Graph()

    graph.parse(path)

    return graph

def get_position_constraints(graph):

    query = TRAFFIC + """

    SELECT ?buildingId ?positionValue
    WHERE {

        ?deliveryRoute traffic:hasPreference ?pref .

        ?pref traffic:hasPositionInPreference ?position .

        ?position traffic:hasCityObject ?building .

        ?building traffic:hasId ?buildingId .

        ?position traffic:hasPositionValue ?positionValue .
    }
    """

    results = graph.query(query)

    return [
        PositionConstraint(
            building_id=int(row.buildingId),
            position=str(row.positionValue)
        )
        for row in results
    ]
def get_route_constraints(graph):

    query = TRAFFIC + """

    SELECT ?firstId ?secondId
    WHERE {

        ?pref traffic:hasPositionInPreference ?firstPos .
        ?pref traffic:hasPositionInPreference ?secondPos .

        ?firstPos traffic:isFirst true .

        ?firstPos traffic:hasNext ?secondPos .

        ?firstPos traffic:hasCityObject ?b1 .
        ?secondPos traffic:hasCityObject ?b2 .

        ?b1 traffic:hasId ?firstId .
        ?b2 traffic:hasId ?secondId .
    }
    """

    results = graph.query(query)

    return [
        RouteConstraint(
            first_building=int(row.firstId),
            second_building=int(row.secondId)
        )
        for row in results
    ]