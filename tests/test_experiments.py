from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pandas as pd

from execsim.data.scenarios import ScenarioConfig, generate_scenario
from execsim.experiments import ExperimentRunner, ExperimentSpec
from execsim.policies import DEPLOYABLE_POLICY_NAMES


def _multi_day_bars() -> pd.DataFrame:
    frames = []
    for day, shape in (
        (date(2026, 3, 12), "uniform"),
        (date(2026, 3, 13), "front_loaded"),
        (date(2026, 3, 16), "back_loaded"),
    ):
        frames.append(
            generate_scenario(
                ScenarioConfig(
                    symbol="AAPL",
                    session_date=day,
                    n_buckets=4,
                    base_volume=100,
                    volume_scenario=shape,
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_experiment_runner_is_reproducible_and_uses_one_engine(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        symbols=("AAPL",),
        trade_dates=(date(2026, 3, 16),),
        quantities=(10,),
        sides=("buy",),
        strategies=DEPLOYABLE_POLICY_NAMES,
        start_time=time(9, 30),
        end_time=time(9, 34),
        planned_participation_rate=0.5,
        hard_participation_rate=0.5,
        risk_aversions=(0.01,),
        temporary_impacts=(0.1,),
    )
    runner = ExperimentRunner(spec, {"AAPL": _multi_day_bars()}, tmp_path)
    first = runner.run(write_outputs=False)
    second = runner.run(write_outputs=False)

    deterministic_columns = [
        column for column in first.results.columns if column != "optimizer_time_seconds"
    ]
    pd.testing.assert_frame_equal(
        first.results[deterministic_columns], second.results[deterministic_columns]
    )
    assert first.run_id == second.run_id
    assert set(first.results["strategy"]) == set(DEPLOYABLE_POLICY_NAMES)
    assert not first.decision_trace.empty


def test_experiment_writes_durable_outputs_and_oracle_is_explicit(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        symbols=("AAPL",),
        trade_dates=(date(2026, 3, 16),),
        quantities=(10,),
        strategies=("twap", "oracle-vwap"),
        start_time=time(9, 30),
        end_time=time(9, 34),
        include_oracle=True,
        risk_aversions=(0.0,),
    )
    output = ExperimentRunner(spec, {"AAPL": _multi_day_bars()}, tmp_path).run()

    assert output.paths["run_results"].is_file()
    assert output.paths["report"].is_file()
    oracle = output.results.loc[output.results["strategy"].str.startswith("oracle")]
    assert oracle["evaluation_only"].all()
    assert "demonstration" in output.paths["report"].read_text(encoding="utf-8")


def test_experiment_reuses_mpc_workspaces_across_compatible_units(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        symbols=("AAPL",),
        trade_dates=(date(2026, 3, 13), date(2026, 3, 16)),
        quantities=(10,),
        strategies=("mpc",),
        start_time=time(9, 30),
        end_time=time(9, 34),
        planned_participation_rate=0.5,
        hard_participation_rate=0.5,
        risk_aversions=(0.01,),
        temporary_impacts=(0.1,),
    )

    output = ExperimentRunner(spec, {"AAPL": _multi_day_bars()}, tmp_path).run(write_outputs=False)

    first_day = output.decision_trace["trade_date"] == "2026-03-13"
    second_day = output.decision_trace["trade_date"] == "2026-03-16"
    assert not output.decision_trace.loc[first_day, "solver_workspace_reused"].any()
    unique = output.decision_trace["solver_status"] == "unique_feasible"
    assert (second_day & ~unique).any()
    assert output.decision_trace.loc[second_day & ~unique, "solver_workspace_reused"].all()
    assert unique.sum() == 2  # The last one-bucket decision on each date is exact.
    assert output.decision_trace.loc[unique, "solver_iterations"].eq(0).all()
    assert not output.decision_trace.loc[unique, "solver_workspace_reused"].any()
