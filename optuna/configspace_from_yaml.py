"""Convert a parameter-space YAML (like `NSGAIIDouble.yaml`) to a
ConfigSpace `ConfigurationSpace`.

This helper implements a pragmatic mapping sufficient for typical
parameter-space specs used in jMetalPy experiments. It supports:
- `categorical` (with `values` as list or dict)
- `integer` with `range`
- `double` with `range`
- `conditionalParameters` and `globalSubParameters` inside categorical branches

It produces hyperparameter names using double-underscore separators
so they are unique and easy to map back when building algorithm objects.

Requires: `ConfigSpace` (pip install ConfigSpace)
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC


def _hp_name(prefix: Optional[str], name: str) -> str:
    return f"{prefix}__{name}" if prefix else name


def _add_numeric_hp(cs: CS.ConfigurationSpace, name: str, descr: Dict[str, Any]):
    r = descr.get("range")
    if not r or len(r) < 2:
        raise ValueError(f"numeric parameter {name} must define a range")
    low, high = r[0], r[1]
    if descr.get("type") == "integer":
        hp = CSH.UniformIntegerHyperparameter(name, lower=int(low), upper=int(high))
    else:
        hp = CSH.UniformFloatHyperparameter(name, lower=float(low), upper=float(high))
    cs.add_hyperparameter(hp)
    return hp


def _add_categorical(cs: CS.ConfigurationSpace, name: str, descr: Dict[str, Any]):
    values = descr.get("values")
    # values may be list or dict
    if isinstance(values, list):
        hp = CSH.CategoricalHyperparameter(name, choices=values)
        cs.add_hyperparameter(hp)
        return hp
    elif isinstance(values, dict):
        choices = list(values.keys())
        hp = CSH.CategoricalHyperparameter(name, choices=choices)
        cs.add_hyperparameter(hp)
        # handle globalSubParameters (same-level params applied regardless of branch)
        global_sub = descr.get("globalSubParameters") or {}
        for gk, gv in (global_sub.items() if isinstance(global_sub, dict) else []):
            _convert_entry(cs, gv, _hp_name(name, gk))
        # each branch may define conditionalParameters
        for choice_key, branch in values.items():
            branch_desc = branch.get("conditionalParameters") if isinstance(branch, dict) else None
            if branch_desc:
                # recursively add branch hyperparameters and add conditions
                for bk, bv in branch_desc.items():
                    child_name = _hp_name(name, bk)
                    hp = _convert_entry(cs, bv, child_name)
                    if hp is not None:
                        # create an EqualsCondition linking child to parent==choice_key
                        parent_hp = cs.get_hyperparameter(name)
                        cs.add_condition(CSC.EqualsCondition(hp, parent=parent_hp, value=choice_key))
        return hp
    else:
        # fallback: treat as categorical with keys
        hp = CSH.CategoricalHyperparameter(name, choices=list(values))
        cs.add_hyperparameter(hp)
        return hp


def _convert_entry(cs: CS.ConfigurationSpace, descr: Dict[str, Any], name: str):
    typ = descr.get("type")
    if typ == "categorical":
        return _add_categorical(cs, name, descr)
    elif typ in ("integer", "double"):
        return _add_numeric_hp(cs, name, descr)
    else:
        # nested dict without explicit type: recurse into fields
        if isinstance(descr, dict):
            # create a dummy group by adding sub-parameters with prefix name
            for k, v in descr.items():
                _convert_entry(cs, v, _hp_name(name, k))
        return None


def yaml_to_configspace(spec: Dict[str, Any], cs_name: str = "jmetal_space") -> CS.ConfigurationSpace:
    """Convert a loaded YAML spec (dict) to a ConfigSpace.ConfigurationSpace.

    Args:
        spec: parsed YAML mapping (top-level dict)
        cs_name: optional name for the ConfigurationSpace

    Returns:
        ConfigSpace.ConfigurationSpace
    """
    cs = CS.ConfigurationSpace(name=cs_name)

    for key, descr in spec.items():
        try:
            _convert_entry(cs, descr, key)
        except Exception as exc:
            # surface the parameter that failed for easier debugging
            raise RuntimeError(f"Error converting parameter '{key}': {exc}") from exc

    return cs


if __name__ == "__main__":
    # simple demo (requires PyYAML). Run: python optuna/configspace_from_yaml.py <yamlfile>
    import sys
    import yaml

    if len(sys.argv) < 2:
        print("Usage: python configspace_from_yaml.py <spec.yaml>")
        sys.exit(2)

    path = sys.argv[1]
    with open(path, "r") as fh:
        spec = yaml.safe_load(fh)

    cs = yaml_to_configspace(spec)
    print(cs)
