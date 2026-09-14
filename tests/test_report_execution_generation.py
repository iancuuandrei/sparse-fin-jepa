"""Execution v5 cannot masquerade as the earlier TCA-restart contract."""

import pytest

from execsim.ml.paper.evaluation_execution import _validate_inheritance_generation


@pytest.mark.parametrize("report_only", [False, True])
@pytest.mark.parametrize(
    "fault", [None, "execution_schema", "inheritance_schema", "frontier", "stages"]
)
def test_execution_generation_requires_exact_inheritance_contract(report_only, fault):
    stages = ["evaluate-forecast", "evaluate-representation"]
    if report_only:
        stages.append("run-tca")
    frontier = "report" if report_only else "run-tca"
    execution = {
        "schema_version": "paper-evaluation-execution-v5"
        if report_only
        else "paper-evaluation-execution-v4",
        "inherited_stages": stages,
        "invalidation_frontier": frontier,
    }
    inheritance = {
        "schema_version": "paper-evaluation-stage-inheritance-v3"
        if report_only
        else "paper-evaluation-stage-inheritance-v2",
        "inherited_stages": stages,
        "invalidation_frontier": frontier,
    }
    if fault == "execution_schema":
        execution["schema_version"] = (
            "paper-evaluation-execution-v4" if report_only else "paper-evaluation-execution-v5"
        )
    elif fault == "inheritance_schema":
        inheritance["schema_version"] = (
            "paper-evaluation-stage-inheritance-v2"
            if report_only
            else "paper-evaluation-stage-inheritance-v3"
        )
    elif fault == "frontier":
        inheritance["invalidation_frontier"] = "run-tca" if report_only else "report"
    elif fault == "stages":
        execution["inherited_stages"] = ["evaluate-forecast"]
    if fault is None:
        _validate_inheritance_generation(execution, inheritance)
    else:
        with pytest.raises(ValueError, match="schema/frontier"):
            _validate_inheritance_generation(execution, inheritance)
