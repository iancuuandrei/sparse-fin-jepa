from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_references_existing_writing_and_implementation_specifications() -> None:
    manifest = yaml.safe_load((ROOT / "repo_manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["authority"]["implementation_standard"] == ("docs/standards/implementation.md")
    assert manifest["authority"]["implementation_specification"] == "docs/SPECIFICATIONS.md"
    assert (ROOT / manifest["authority"]["implementation_standard"]).is_file()
    assert (ROOT / manifest["authority"]["implementation_specification"]).is_file()
    assert "simulation" in manifest["areas"]
    assert "ml" in manifest["areas"]
    assert "paper_data" in manifest["areas"]
    assert "paper_sequences" in manifest["areas"]
    assert "paper_representations" in manifest["areas"]
    assert "paper_evaluation" in manifest["areas"]
    assert (ROOT / "docs/PAPER_DESIGN.md").is_file()
    assert (ROOT / "docs/PAPER_IMPLEMENTATION_REPORT.md").is_file()
    assert (ROOT / "docs/ADRs/0006-sparse-predictive-representation-paper-framework.md").is_file()


def test_repository_context_check_and_selection_are_deterministic() -> None:
    check = subprocess.run(
        [sys.executable, "scripts/repo_context.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    first = subprocess.run(
        [
            sys.executable,
            "scripts/repo_context.py",
            "--path",
            "src/execsim/simulator/core.py",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(first.args, cwd=ROOT, capture_output=True, text=True, check=False)

    assert check.returncode == 0, check.stderr
    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert '"simulation"' in first.stdout


def test_byte_exact_paper_artifacts_pin_lf_checkout_bytes() -> None:
    """Keep frozen scientific hashes invariant under platform Git settings."""
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "configs/paper/**/*.json text eol=lf" in attributes
    assert "data/manifests/paper_universe_v2.json text eol=crlf" in attributes
    assert "CORPUS_QUALITY_REPORT.md text eol=crlf" in attributes
    assert "V2_FORMATION_QUALITY_REPORT.md text eol=crlf" in attributes

    byte_exact_paths = (
        "configs/paper/sparse_jepa/design-freeze-v1.json",
        "configs/paper/sparse_jepa/safe-default-receipt-v1.json",
        "configs/paper/sparse_jepa/v1-evidence-final.json",
        "configs/paper/sparse_jepa_v2/design-freeze-v2.json",
    )
    for relative_path in byte_exact_paths:
        assert b"\r" not in (ROOT / relative_path).read_bytes(), relative_path

    freeze = yaml.safe_load(
        (ROOT / "configs/paper/sparse_jepa_v2/design-freeze-v2.json").read_text(encoding="utf-8")
    )
    formation = freeze["formation_evidence"]
    empirical_paths = {
        "universe_manifest_sha256": "data/manifests/paper_universe_v2.json",
        "formation_report_sha256": "V2_FORMATION_QUALITY_REPORT.md",
    }
    for field, relative_path in empirical_paths.items():
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == formation[field], relative_path
