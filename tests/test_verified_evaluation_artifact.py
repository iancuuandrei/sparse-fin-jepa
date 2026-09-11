import pandas as pd
import pytest

from execsim.ml.paper.evaluation_artifacts import VerifiedArtifact, publish_frames


def test_verified_artifact_rejects_changed_files_and_identity(tmp_path):
    identity = {"fold_id": "fold-1", "source_commit": "a" * 40}
    directory = tmp_path / "artifact"
    publish_frames(
        directory, identity=identity, frames={"rows.parquet": pd.DataFrame({"sample_id": ["a"]})}
    )
    verified = VerifiedArtifact(directory, identity=identity, names=("rows.parquet",))
    verified.check(directory, identity)
    with pytest.raises(ValueError, match="identity"):
        verified.check(directory, {**identity, "fold_id": "fold-2"})
    (directory / "rows.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        verified.check(directory, identity)
