"""Introspection over the component catalogue.

Answers three questions without reading source code by hand: which component
slots exist, which implementations are available for each, and what control
parameters (name, type, default) each implementation takes. Deliberately stops
there -- it does not describe *ranges* or *distributions* to explore those
parameters, which is an automatic-configuration concern kept out of scope for now
(see MODERNIZATION.md's L1 notes).

Everything here is derived from the real classes via `inspect`, not from a
hand-maintained description, so it cannot drift out of sync with the code the way
a separate parameter list could.
"""

import inspect
import re
from dataclasses import dataclass, field

from jmetal.component.catalogue.common.evaluation import SequentialEvaluation
from jmetal.component.catalogue.common.solutions_creation import RandomSolutionsCreation
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.component.catalogue.ea.replacement import (
    RankingAndCrowdingDistanceReplacement,
    RankingAndDensityEstimatorReplacement,
    SMSEMOAReplacement,
)
from jmetal.component.catalogue.ea.selection import RandomSelection, TournamentSelection
from jmetal.component.catalogue.ea.variation import CrossoverAndMutationVariation
from jmetal.operator.crossover import (
    BLXAlphaBetaCrossover,
    BLXAlphaCrossover,
    CXCrossover,
    DifferentialEvolutionCrossover,
    IntegerSBXCrossover,
    PMXCrossover,
    SBXCrossover,
    SPXCrossover,
)
from jmetal.operator.mutation import (
    BitFlipMutation,
    IntegerPolynomialMutation,
    PermutationSwapMutation,
    PolynomialMutation,
    ScrambleMutation,
    SimpleRandomMutation,
    UniformMutation,
)

CATALOGUE: dict[str, list[type]] = {
    "solutions_creation": [RandomSolutionsCreation],
    "evaluation": [SequentialEvaluation],
    "termination": [TerminationByEvaluations],
    "selection": [TournamentSelection, RandomSelection],
    "variation": [CrossoverAndMutationVariation],
    "replacement": [
        RankingAndDensityEstimatorReplacement,
        RankingAndCrowdingDistanceReplacement,
        SMSEMOAReplacement,
    ],
    "crossover": [
        SBXCrossover,
        IntegerSBXCrossover,
        BLXAlphaCrossover,
        BLXAlphaBetaCrossover,
        PMXCrossover,
        CXCrossover,
        SPXCrossover,
        DifferentialEvolutionCrossover,
    ],
    "mutation": [
        PolynomialMutation,
        IntegerPolynomialMutation,
        BitFlipMutation,
        PermutationSwapMutation,
        ScrambleMutation,
        SimpleRandomMutation,
        UniformMutation,
    ],
}
"""Component slot name -> the implementations currently available for it.

Deliberately an explicit registry, not a module scan: which classes count as
"the" implementations of a slot is an editorial decision (e.g. `crossover` lists
jmetal.operator's curated public crossover operators, not every class in that
module), and an explicit list keeps that decision visible and easy to extend when
a new component lands (see MODERNIZATION.md's Phase 2).
"""

_QUALIFIED_NAME = re.compile(r"(?:[a-zA-Z_][a-zA-Z0-9_]*\.)+")


@dataclass
class ParameterInfo:
    """One constructor parameter of a component implementation.

    Attributes:
        name: The parameter's name.
        type: Its type annotation, formatted for readability (module-qualified
            names stripped, e.g. `Crossover` rather than `jmetal.core.operator.Crossover`).
            `"Any"` if the parameter has no annotation.
        default: Its default value, formatted as text, or `None` if the parameter
            is required.
    """

    name: str
    type: str
    default: str | None


@dataclass
class ComponentInfo:
    """One implementation of a component slot and its control parameters.

    Attributes:
        name: The implementation's class name.
        parameters: Its constructor parameters, in declaration order.
    """

    name: str
    parameters: list[ParameterInfo] = field(default_factory=list)


def _format_type(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    return _QUALIFIED_NAME.sub("", inspect.formatannotation(annotation))


def _format_default(default: object) -> str | None:
    if default is inspect.Parameter.empty:
        return None
    text = str(default)
    if " object at 0x" in text:
        # No custom __str__/__repr__ on the default instance -- fall back to its
        # class name rather than an unreadable memory address.
        return f"{type(default).__name__}()"
    return text


def describe_component(implementation: type) -> ComponentInfo:
    """Describe one component implementation's constructor parameters.

    Args:
        implementation: The class to describe, e.g. `SBXCrossover`.

    Returns:
        Its name and control parameters (name, type, default), derived from its
        `__init__` signature via `inspect`.
    """
    signature = inspect.signature(implementation.__init__)
    parameters = [
        ParameterInfo(
            name=name,
            type=_format_type(parameter.annotation),
            default=_format_default(parameter.default),
        )
        for name, parameter in signature.parameters.items()
        if name != "self"
    ]

    return ComponentInfo(name=implementation.__name__, parameters=parameters)


def describe_catalogue() -> dict[str, list[ComponentInfo]]:
    """Describe every implementation of every component slot in `CATALOGUE`.

    Returns:
        Each slot name mapped to the `ComponentInfo` of every implementation
        registered for it, in `CATALOGUE`'s order.
    """
    return {
        slot: [describe_component(implementation) for implementation in implementations]
        for slot, implementations in CATALOGUE.items()
    }
