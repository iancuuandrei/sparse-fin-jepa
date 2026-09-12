"""Atomic manifest publication preserves bytes and failed-attempt isolation."""

import json

import pytest

from execsim.data.paper import manifests


def test_atomic_manifest_does_not_use_shared_temporary_name(tmp_path):
    destination = tmp_path / "receipt.json"
    stale = tmp_path / "receipt.json.tmp"
    stale.write_text("unrelated retained evidence", encoding="utf-8")
    payload = {"status": "PASS", "value": 3}
    manifests.write_json_atomic(destination, payload)
    assert (
        destination.read_text(encoding="utf-8")
        == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    assert stale.read_text(encoding="utf-8") == "unrelated retained evidence"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["receipt.json", "receipt.json.tmp"]


def test_atomic_manifest_failed_replace_preserves_previous_receipt(tmp_path, monkeypatch):
    destination = tmp_path / "receipt.json"
    manifests.write_json_atomic(destination, {"version": 1})
    original = destination.read_bytes()

    def fail_replace(source, target):
        raise OSError("fixture publication failure")

    monkeypatch.setattr(manifests.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        manifests.write_json_atomic(destination, {"version": 2})
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]
