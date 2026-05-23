from .collision_optimization import *

_METRIC_EXPORTS = {
    "PlanningMetric": "planning_metrics",
}


def __getattr__(name):
    module_name = _METRIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
