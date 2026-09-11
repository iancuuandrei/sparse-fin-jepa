import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_artifacts import merge_result_shards


def test_result_merge_is_order_independent_batched_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr("execsim.ml.paper.evaluation_artifacts.ROW_GROUP_SIZE", 2)
    frames = [
        pd.DataFrame({"method": "raw", "seed": None, "sample_id": ["b", "a"], "value": [2.0, 1.0]}),
        pd.DataFrame({"method": "dense", "seed": 13, "sample_id": ["b", "a"], "value": [4.0, 3.0]}),
    ]
    sources = {}
    for index, frame in enumerate(frames):
        path = tmp_path / f"source-{index}.parquet"
        frame.to_parquet(path, index=False)
        sources[str(index)] = (path, file_sha256(path))
    kwargs = dict(
        sources=sources,
        keys=("method", "seed", "sample_id"),
        identity={"paper_config_hash": "fixture", "source_commit": "fixed"},
        schema_version="fixture-v1",
    )
    output = tmp_path / "merged.parquet"
    receipt = merge_result_shards(output, **kwargs)
    expected = pd.concat(frames).sort_values(["method", "seed", "sample_id"]).reset_index(drop=True)
    expected.to_parquet(tmp_path / "direct.parquet", index=False)
    pd.testing.assert_frame_equal(
        pd.read_parquet(output), pd.read_parquet(tmp_path / "direct.parquet")
    )
    assert receipt["rows"] == 4
    before = output.stat().st_mtime_ns
    assert merge_result_shards(output, **kwargs) == receipt
    assert output.stat().st_mtime_ns == before
    second = tmp_path / "second.parquet"
    merge_result_shards(second, **{**kwargs, "sources": dict(reversed(list(sources.items())))})
    assert file_sha256(second) == file_sha256(output)
    with pytest.raises(ValueError, match="identity"):
        merge_result_shards(output, **{**kwargs, "identity": {"paper_config_hash": "changed"}})
    output.write_bytes(b"corrupted fixture")
    with pytest.raises(ValueError, match="checksum"):
        merge_result_shards(output, **kwargs)


@pytest.mark.parametrize("problem", ["duplicate", "overlap", "checksum", "missing"])
def test_result_merge_rejects_invalid_expected_inventory(tmp_path, problem):
    sources = {}
    for index, values in enumerate(([1, 1] if problem == "duplicate" else [1, 3], [2, 4])):
        path = tmp_path / f"{index}.parquet"
        pd.DataFrame({"id": values, "value": [1.0, 2.0]}).to_parquet(path)
        sources[str(index)] = (path, file_sha256(path))
    if problem == "checksum":
        sources["0"] = (sources["0"][0], "0" * 64)
    if problem == "missing":
        sources["0"] = (tmp_path / "absent.parquet", sources["0"][1])
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_result_shards(
            tmp_path / "merged.parquet",
            sources=sources,
            keys=("id",),
            identity={"paper_config_hash": "fixture"},
            schema_version="fixture-v1",
        )
    assert not (tmp_path / "merged.manifest.json").exists()


def test_merge_preserves_mixed_timestamp_precision_and_empty_shards(tmp_path):
    sources = {}
    expected = []
    for index, unit in enumerate(("ms", "us", "ns", "us")):
        stamp = pd.Timestamp("2024-04-12 10:30:00", tz="America/New_York")
        stamp += pd.Timedelta(index, unit=unit)
        values = [] if index == 3 else [stamp, None]
        table = pa.table(
            {
                "id": pa.array([index * 2, index * 2 + 1] if values else [], type=pa.int64()),
                "generated_at": pa.array(values, type=pa.timestamp(unit, "America/New_York")),
            }
        )
        path = tmp_path / f"{index}.parquet"
        pq.write_table(table, path)
        sources[str(index)] = (path, file_sha256(path))
        expected.extend(values)
    output = tmp_path / "merged.parquet"
    options = dict(
        sources=sources,
        keys=("id",),
        identity={"paper_config_hash": "fixture"},
        schema_version="fixture-v1",
    )
    receipt = merge_result_shards(output, **options)
    actual = pq.read_table(output)["generated_at"]
    assert actual.type == pa.timestamp("ns", "America/New_York")
    assert actual == pa.chunked_array([pa.array(expected, type=actual.type)])
    assert receipt["rows"] == 6
    assert merge_result_shards(output, **options) == receipt
    other = tmp_path / "reversed.parquet"
    merge_result_shards(other, **{**options, "sources": dict(reversed(list(sources.items())))})
    assert file_sha256(other) == file_sha256(output)


@pytest.mark.parametrize("problem", ["timezone", "naive", "type", "overflow"])
def test_timestamp_promotion_remains_fail_closed(tmp_path, problem):
    types = [pa.timestamp("ms", "America/New_York"), pa.timestamp("ns", "America/New_York")]
    if problem == "timezone":
        types[1] = pa.timestamp("us", "UTC")
    elif problem == "naive":
        types[1] = pa.timestamp("us")
    elif problem == "type":
        types[1] = pa.int64()
    sources = {}
    for index, kind in enumerate(types):
        value = 20_000_000_000_000 if problem == "overflow" and index == 0 else 1
        path = tmp_path / f"{index}.parquet"
        pq.write_table(
            pa.table({"id": [index], "generated_at": pa.array([value], type=kind)}), path
        )
        sources[str(index)] = (path, file_sha256(path))
    output = tmp_path / "merged.parquet"
    with pytest.raises((ValueError, pa.ArrowException)):
        merge_result_shards(
            output,
            sources=sources,
            keys=("id",),
            identity={"paper_config_hash": "fixture"},
            schema_version="fixture-v1",
        )
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
