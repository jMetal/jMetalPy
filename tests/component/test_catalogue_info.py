"""Tests for the component catalogue introspection utilities."""

from jmetal.component.catalogue.common.solutions_creation import RandomSolutionsCreation
from jmetal.component.catalogue.ea.variation import CrossoverAndMutationVariation
from jmetal.component.catalogue_info import (
    CATALOGUE,
    ComponentInfo,
    ParameterInfo,
    describe_catalogue,
    describe_component,
)
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType


class TestDescribeComponent:
    def test_lists_every_constructor_parameter_except_self(self):
        info = describe_component(RandomSolutionsCreation)

        assert [p.name for p in info.parameters] == ["problem", "number_of_solutions_to_create"]

    def test_a_required_parameter_has_no_default(self):
        info = describe_component(RandomSolutionsCreation)

        problem_param = next(p for p in info.parameters if p.name == "problem")

        assert problem_param.default is None

    def test_a_parameter_with_a_default_reports_it_as_text(self):
        info = describe_component(SBXCrossover)

        distribution_index = next(p for p in info.parameters if p.name == "distribution_index")

        assert distribution_index.default == "20.0"

    def test_an_enum_default_is_formatted_by_name_not_by_repr(self):
        info = describe_component(RankingAndDensityEstimatorReplacement)

        removal_policy = next(p for p in info.parameters if p.name == "removal_policy")

        assert removal_policy.default == str(RemovalPolicyType.ONE_SHOT)
        assert "object at 0x" not in removal_policy.default

    def test_a_default_instance_without_a_custom_repr_falls_back_to_its_class_name(self):
        info = describe_component(SBXCrossover)

        repair_operator = next(p for p in info.parameters if p.name == "repair_operator")

        assert repair_operator.default == "ClampFloatRepair()"

    def test_type_annotations_are_stripped_of_their_module_path(self):
        info = describe_component(CrossoverAndMutationVariation)

        crossover_param = next(p for p in info.parameters if p.name == "crossover")

        assert crossover_param.type == "Crossover"
        assert "jmetal" not in crossover_param.type

    def test_returns_the_class_name(self):
        info = describe_component(SBXCrossover)

        assert info.name == "SBXCrossover"

    def test_returns_a_component_info_instance(self):
        assert isinstance(describe_component(SBXCrossover), ComponentInfo)
        assert all(
            isinstance(p, ParameterInfo) for p in describe_component(SBXCrossover).parameters
        )


class TestDescribeCatalogue:
    def test_describes_every_registered_slot(self):
        catalogue = describe_catalogue()

        assert set(catalogue.keys()) == set(CATALOGUE.keys())

    def test_describes_every_implementation_registered_for_a_slot(self):
        catalogue = describe_catalogue()

        replacement_names = {info.name for info in catalogue["replacement"]}

        assert replacement_names == {cls.__name__ for cls in CATALOGUE["replacement"]}

    def test_every_implementation_in_the_catalogue_is_importable_and_describable(self):
        # Guards against a stale entry in CATALOGUE (e.g. a renamed or removed class)
        # going unnoticed: describe_component() would raise before this assertion.
        for implementations in CATALOGUE.values():
            for implementation in implementations:
                describe_component(implementation)
