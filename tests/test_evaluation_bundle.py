"""Reports become authoritative only as complete identity-bound trees."""

import pytest

from execsim.ml.paper.evaluation_artifacts import publish_bundle


def test_bundle_interruption_resume_and_corruption(tmp_path):
    destination = tmp_path / "report"
    calls = []

    def interrupted(staging):
        (staging / "partial.txt").write_text("partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        publish_bundle(destination, identity={"source": "one"}, build=interrupted)
    assert not destination.exists()

    def complete(staging):
        calls.append(1)
        (staging / "tables").mkdir()
        (staging / "tables/result.txt").write_text("complete")
        (staging / "figure.png").write_bytes(b"fixture")

    receipt = publish_bundle(destination, identity={"source": "one"}, build=complete)
    assert len(receipt["files"]) == 2
    assert publish_bundle(destination, identity={"source": "one"}, build=complete) == receipt
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        publish_bundle(destination, identity={"source": "two"}, build=complete)
    (destination / "figure.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        publish_bundle(destination, identity={"source": "one"}, build=complete)
    assert len(calls) == 1


def test_report_orchestration_reuses_only_unchanged_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from execsim.data.paper.manifests import file_sha256, write_json_atomic
    from execsim.ml.paper import orchestration

    root = tmp_path / "execution"
    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime_evaluation_root=root,
        paper_run_id="paper",
        config_hash="science",
        authorize=lambda *args, **kwargs: None,
    )
    for name in ("_require_parameter_freeze", "_require_locked_test_opened"):
        monkeypatch.setattr(orchestration, name, lambda *args: None)
    monkeypatch.setattr(orchestration, "_git_head", lambda: "source")
    monkeypatch.setattr(orchestration, "_git_tree", lambda: "tree")
    write_json_atomic(tmp_path / "selection/parameter-freeze-v1.json", {})
    names = (
        "evaluation/forecast-results",
        "evaluation/representation-accessibility",
        "evaluation/representation-date-metrics",
        "evaluation/support-regimes",
        "tca/main",
        "tca/sensitivity",
    )
    for name in names:
        path = root / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture input")
        write_json_atomic(
            path.with_suffix(".manifest.json"),
            {
                "parquet_sha256": file_sha256(path),
                "paper_config_hash": "science",
                "merge_identity": {"source_commit": "source"},
            },
        )
    calls = []

    def build(config, *, output_root):
        calls.append(1)
        directory = output_root / config.paper_run_id
        directory.mkdir()
        (directory / "REPORT.md").write_text("synthetic fixture")
        write_json_atomic(directory / "provenance.json", {"fixture": True})

    monkeypatch.setattr(orchestration, "_build_report_stage", build)
    first = orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)
    assert first["reuse"] == "created"
    assert (
        orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)[
            "reuse"
        ]
        == "validated"
    )
    assert len(calls) == 1
    for name in (
        "execution.json",
        "evaluation/representation-evaluation-manifest.json",
        "tca/manifest.json",
    ):
        write_json_atomic(root / name, {"fixture": True})
    write_json_atomic(tmp_path / "selection/locked-test-opened-v1.json", {"fixture": True})
    for fold in range(1, 4):
        for geometry in ("dense", "sparse"):
            for seed in (13, 29, 47):
                write_json_atomic(
                    tmp_path / f"representations/fold-{fold}/{geometry}/{seed}/final/manifest.json",
                    {"code_commit": "immutable-training"},
                )
        for coordinate in range(8):
            write_json_atomic(
                tmp_path / f"lightgbm/fold-{fold}/coordinate-{coordinate}/shared/manifest.json", {}
            )
    freeze = orchestration.write_final_result_freeze(config)
    assert freeze["path"] == str(root / "selection/final-result-freeze-v1.json")
    assert orchestration.write_final_result_freeze(config)["sha256"] == freeze["sha256"]
    assert (
        orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)[
            "reuse"
        ]
        == "validated"
    )
    (root / "tca/main.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Report input merge checksum"):
        orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)
    assert len(calls) == 1
