"""Unified, fail-closed configuration for the locked paper experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from execsim.data.paper.manifests import file_sha256, stable_hash
from execsim.data.paper.partitions import PAPER_FOLDS

CONFIG_FILES = (
    "data.yaml",
    "sequences.yaml",
    "representation.yaml",
    "lightgbm.yaml",
    "evaluation.yaml",
    "tca.yaml",
)

RUNTIME_APPROVAL_SCHEMA = "paper-runtime-approval-v1"
RUNTIME_APPROVAL_OPERATIONS = (
    "target_acquisition",
    "historical_training",
    "locked_result_evaluation",
)

_FREEZE_IDENTITIES = {
    "sparse-jepa-v1": ("design-freeze-v1.json", "paper-design-freeze-v1", 1),
    "sparse-jepa-v2": ("design-freeze-v2.json", "paper-design-freeze-v3", 2),
}


@dataclass(frozen=True, slots=True)
class PaperRuntimeApproval:
    """Represent a runtime-only approval outside the scientific configuration."""

    approval_id: str
    approved_at_utc: str
    protocol_id: str
    paper_config_sha256: str
    approved_operations: frozenset[str]

    def approves(self, operation: str) -> bool:
        """Return whether this receipt explicitly approves one known operation."""
        if operation not in RUNTIME_APPROVAL_OPERATIONS:
            raise ValueError(f"Unknown paper operation: {operation}")
        return operation in self.approved_operations


@dataclass(frozen=True, slots=True)
class PaperRunConfig:
    """Hold all six authoritative paper configurations and their canonical hash."""

    root: Path
    sections: dict[str, dict[str, Any]]
    config_hash: str
    design_freeze: dict[str, Any]
    runtime_artifact_root: Path | None = None
    runtime_data_root: Path | None = None
    runtime_cache_root: Path | None = None
    runtime_report_root: Path | None = None
    runtime_sequence_root: Path | None = None
    runtime_embedding_root: Path | None = None
    runtime_output_root: Path | None = None

    @property
    def sequence_root(self) -> Path:
        return self.runtime_sequence_root or self.artifact_root / "sequences"

    @property
    def embedding_root(self) -> Path:
        return self.runtime_embedding_root or self.artifact_root / "embeddings"

    @property
    def lightgbm_root(self) -> Path:
        return self.runtime_output_root or self.artifact_root / "lightgbm"

    @property
    def data(self) -> dict[str, Any]:
        return self.sections["data"]

    @property
    def sequences(self) -> dict[str, Any]:
        return self.sections["sequences"]

    @property
    def representation(self) -> dict[str, Any]:
        return self.sections["representation"]

    @property
    def lightgbm(self) -> dict[str, Any]:
        return self.sections["lightgbm"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.sections["evaluation"]

    @property
    def tca(self) -> dict[str, Any]:
        return self.sections["tca"]

    @property
    def allow_network(self) -> bool:
        return bool(self.data["allow_network"])

    @property
    def allow_historical_training(self) -> bool:
        return bool(self.data["allow_historical_training"])

    @property
    def allow_full_paper_run(self) -> bool:
        return bool(self.data["allow_full_paper_run"])

    @property
    def paper_run_id(self) -> str:
        return str(self.data["paper_run_id"])

    @property
    def artifact_root(self) -> Path:
        return self.runtime_artifact_root or Path(self.data["artifact_root"])

    @property
    def report_root(self) -> Path:
        return self.runtime_report_root or Path(self.data["report_root"])

    @property
    def cache_root(self) -> Path:
        """Return the runtime-only cache root outside the scientific configuration."""
        return self.runtime_cache_root or Path(".runtime")

    def data_path(self, name: str) -> Path:
        """Resolve one configured data path below an optional portable data root."""
        configured = Path(str(self.data[name]))
        if self.runtime_data_root is None:
            return configured
        parts = configured.parts
        relative = Path(*parts[1:]) if parts and parts[0].lower() == "data" else configured
        if relative.is_absolute():
            raise ValueError(f"Cannot relocate absolute configured data path: {configured}")
        return self.runtime_data_root / relative

    def with_runtime_roots(
        self,
        *,
        artifact_root: Path | None = None,
        data_root: Path | None = None,
        cache_root: Path | None = None,
        report_root: Path | None = None,
        sequence_root: Path | None = None,
        embedding_root: Path | None = None,
        output_root: Path | None = None,
    ) -> PaperRunConfig:
        """Apply operational path relocation without changing the canonical config hash."""
        return PaperRunConfig(
            root=self.root,
            sections=self.sections,
            config_hash=self.config_hash,
            design_freeze=self.design_freeze,
            runtime_artifact_root=artifact_root,
            runtime_data_root=data_root,
            runtime_cache_root=cache_root,
            runtime_report_root=report_root,
            runtime_sequence_root=sequence_root,
            runtime_embedding_root=embedding_root,
            runtime_output_root=output_root,
        )

    def authorization_granted(
        self,
        operation: str,
        *,
        approval: PaperRuntimeApproval | None,
        cli_enabled: bool,
    ) -> bool:
        """Return true only for an identity-bound approval plus explicit CLI opt-in."""
        if operation not in RUNTIME_APPROVAL_OPERATIONS:
            raise ValueError(f"Unknown paper operation: {operation}")
        if approval is None:
            return False
        if (
            approval.protocol_id != self.paper_run_id
            or approval.paper_config_sha256 != self.config_hash
        ):
            return False
        return approval.approves(operation) and cli_enabled

    def authorize(
        self,
        operation: str,
        *,
        approval: PaperRuntimeApproval | None,
        cli_enabled: bool,
    ) -> None:
        """Require an identity-bound runtime approval and matching CLI opt-in."""
        if not self.authorization_granted(operation, approval=approval, cli_enabled=cli_enabled):
            raise PermissionError(
                f"Paper operation {operation!r} requires a matching runtime approval "
                "and explicit command-line opt-in."
            )


def load_runtime_approval(path: Path, config: PaperRunConfig) -> PaperRuntimeApproval:
    """Load a strict runtime approval bound to the current protocol/config identity."""
    approval_root = (
        config.runtime_cache_root or config.root.resolve().parents[2] / ".runtime"
    ) / "paper-approvals"
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(approval_root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise PermissionError(
            f"Paper runtime approval must be an existing file below {approval_root}."
        ) from exc
    if resolved.suffix.lower() != ".json" or not resolved.is_file():
        raise PermissionError("Paper runtime approval must be a regular JSON file.")
    if resolved.stat().st_size > 65_536:
        raise ValueError("Paper runtime approval exceeds the 64 KiB size limit.")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Paper runtime approval must contain a JSON object.")
    approvals = payload.get("approvals")
    expected_operations = set(RUNTIME_APPROVAL_OPERATIONS)
    expected_fields = {
        "schema_version",
        "approval_id",
        "approved_at_utc",
        "protocol_id",
        "paper_config_sha256",
        "approvals",
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != RUNTIME_APPROVAL_SCHEMA
        or not isinstance(payload.get("approval_id"), str)
        or not payload["approval_id"].strip()
        or not isinstance(payload.get("approved_at_utc"), str)
        or not payload["approved_at_utc"].strip()
        or not isinstance(approvals, dict)
        or set(approvals) != expected_operations
        or any(not isinstance(approvals[name], bool) for name in expected_operations)
    ):
        raise ValueError("Paper runtime approval schema is invalid.")
    try:
        approved_at = datetime.fromisoformat(payload["approved_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Paper runtime approval timestamp is invalid.") from exc
    if approved_at.tzinfo is None:
        raise ValueError("Paper runtime approval timestamp must include a timezone.")
    approval = PaperRuntimeApproval(
        approval_id=payload["approval_id"],
        approved_at_utc=payload["approved_at_utc"],
        protocol_id=str(payload.get("protocol_id", "")),
        paper_config_sha256=str(payload.get("paper_config_sha256", "")),
        approved_operations=frozenset(
            name for name in RUNTIME_APPROVAL_OPERATIONS if approvals[name]
        ),
    )
    if (
        approval.protocol_id != config.paper_run_id
        or approval.paper_config_sha256 != config.config_hash
    ):
        raise PermissionError("Paper runtime approval does not match this protocol/config.")
    return approval


def load_paper_config(path: Path) -> PaperRunConfig:
    """Load and cross-validate all six configuration files from one directory."""
    root = path if path.is_dir() else path.parent
    sections: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILES:
        file_path = root / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Paper configuration is incomplete: {file_path}")
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Paper YAML must contain a mapping: {file_path}")
        sections[file_path.stem] = payload
    paper_run_id = str(sections.get("data", {}).get("paper_run_id", ""))
    try:
        freeze_name, _, _ = _FREEZE_IDENTITIES[paper_run_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported paper protocol identity: {paper_run_id!r}") from exc
    freeze_path = root / freeze_name
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Paper design freeze is missing: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise TypeError("Paper design freeze must contain a JSON object.")
    _validate_design_freeze(root, sections, freeze_path, freeze)
    _validate_sections(sections)
    return PaperRunConfig(root, sections, stable_hash({"sections": sections}), freeze)


def _validate_design_freeze(
    root: Path,
    sections: dict[str, dict[str, Any]],
    freeze_path: Path,
    freeze: dict[str, Any],
) -> None:
    """Verify the one-time protocol receipt, its sidecar, and every frozen source."""
    protocol_id = str(sections["data"].get("paper_run_id", ""))
    try:
        _, expected_schema, expected_version = _FREEZE_IDENTITIES[protocol_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported paper protocol identity: {protocol_id!r}") from exc
    if freeze.get("schema_version") != expected_schema or freeze.get("status") != "PROTOCOL_FROZEN":
        raise ValueError("Paper design freeze identity is invalid.")
    sidecar = freeze_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Paper design freeze checksum is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != freeze_path.name:
        raise ValueError("Paper design freeze checksum sidecar is malformed.")
    if fields[0] != file_sha256(freeze_path):
        raise ValueError("Paper design freeze checksum does not match immutable bytes.")
    if protocol_id == "sparse-jepa-v1":
        _validate_archived_v1_receipts(freeze_path)
        return
    if (
        freeze.get("protocol_id") != protocol_id
        or freeze.get("protocol_version") != expected_version
        or freeze.get("paper_config_sha256") != stable_hash({"sections": sections})
    ):
        raise ValueError("Paper design freeze does not match the six normative YAML files.")
    repository_root = root.parents[2]
    documents = freeze.get("normative_document_sha256")
    if not isinstance(documents, dict) or not documents:
        raise ValueError("Paper design freeze must bind its normative documents.")
    mismatches: list[str] = [
        str(relative)
        for relative, expected in documents.items()
        if not isinstance(relative, str)
        or not isinstance(expected, str)
        or _portable_document_sha256(repository_root / relative) != expected
    ]
    if mismatches:
        raise ValueError(f"Paper design freeze normative document mismatch: {sorted(mismatches)}")
    specification = str(freeze.get("source_specification", ""))
    if (
        documents.get(specification) != freeze.get("source_specification_sha256")
        or not specification
    ):
        raise ValueError("Paper design freeze does not match its normative specification.")


def _portable_document_sha256(path: Path) -> str:
    """Hash tracked text after canonical LF normalization across Git checkouts."""
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _validate_archived_v1_receipts(freeze_path: Path) -> None:
    """Verify self-contained v1 bytes without requiring a Git object database."""
    safe_path = freeze_path.with_name("safe-default-receipt-v1.json")
    safe_sidecar = safe_path.with_suffix(".sha256")
    evidence_path = freeze_path.with_name("v1-evidence-final.json")
    evidence_sidecar = evidence_path.with_suffix(".sha256")
    if not all(
        path.is_file() for path in (safe_path, safe_sidecar, evidence_path, evidence_sidecar)
    ):
        raise ValueError("Archived v1 receipts are incomplete.")
    safe_fields = safe_sidecar.read_text(encoding="utf-8").strip().split()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    fields = evidence_sidecar.read_text(encoding="utf-8").strip().split()
    content = dict(evidence)
    claimed_bundle_hash = content.pop("bundle_content_sha256", None)
    if (
        not isinstance(evidence, dict)
        or safe_fields != [file_sha256(safe_path), safe_path.name]
        or fields != [file_sha256(evidence_path), evidence_path.name]
        or evidence.get("schema_version") != "paper-v1-terminal-evidence-v1"
        or evidence.get("protocol_id") != "sparse-jepa-v1"
        # This legacy field names the acquisition-time safe-default receipt.
        or evidence.get("artifacts", {}).get("design_freeze") != file_sha256(safe_path)
        or claimed_bundle_hash != stable_hash(content)
    ):
        raise ValueError("Archived v1 terminal evidence is invalid.")


def _validate_sections(sections: dict[str, dict[str, Any]]) -> None:
    data = sections["data"]
    sequence = sections["sequences"]
    representation = sections["representation"]
    lightgbm = sections["lightgbm"]
    evaluation = sections["evaluation"]
    tca = sections["tca"]
    protocol_id = str(data.get("paper_run_id", ""))
    expected = {
        "provider": "alpaca",
        "feed": "sip",
        "frequency": "1min",
        "adjustment": "raw",
        "timezone": "America/New_York",
        "extended_hours": False,
        "universe_size": 100,
    }
    contradictions = [name for name, value in expected.items() if data.get(name) != value]
    if contradictions:
        raise ValueError(f"Paper data configuration contradicts locked design: {contradictions}")
    if (
        sequence.get("token_minutes") != 15
        or sequence.get("session_tokens") != 26
        or sequence.get("feature_count") != representation.get("observed_feature_dim")
        or tuple(sequence.get("horizons", ())) != (1, 2, 4, 8)
        or sequence.get("context_length") != 8
    ):
        raise ValueError("Sequence and representation dimensions contradict the locked design.")
    expected_quality_protocol = (
        "resolution-aware-v2" if protocol_id == "sparse-jepa-v2" else "exact-minute-v1"
    )
    if sequence.get("quality_protocol", "exact-minute-v1") != expected_quality_protocol:
        raise ValueError("Sequence quality protocol contradicts the selected paper protocol.")
    configured_folds = evaluation.get("folds")
    expected_folds = [
        {
            "id": fold.fold_id,
            "train": [fold.train_start, fold.train_end],
            "validation": [fold.validation_start, fold.validation_end],
            "test": [fold.test_start, fold.test_end],
        }
        for fold in PAPER_FOLDS
    ]
    if configured_folds != expected_folds:
        raise ValueError("Evaluation folds do not match locked PAPER_FOLDS.")
    if lightgbm.get("main_rows") != ["ewma", "raw", "untrained_neural", "dense", "sparse"]:
        raise ValueError("LightGBM main rows contradict the locked comparison.")
    if (
        representation.get("geometries") != ["dense", "sparse"]
        or representation.get("predictor_family") != "mlp"
        or representation.get("observed_feature_dim") != 18
        or representation.get("feature_dim") != 13
        or representation.get("conditioning_dim") != 5
        or representation.get("seeds") != [13, 29, 47]
        or representation.get("max_epochs") != 40
        or representation.get("early_stopping_patience") != 6
        or representation.get("warmup_fraction") != 0.05
        or representation.get("rdm_lambda_candidates") != [0.1, 1.0, 10.0]
        or representation.get("probe_capacity_ladder") != ["affine_ridge", "mlp_64", "mlp_256"]
        or representation.get("probe_ridge_alphas") != [0.1, 1.0, 10.0]
        or representation.get("probe_mlp_epochs") != 20
        or representation.get("checkpoint_interval_steps") != 500
        or representation.get("rdm_diagnostic_sample_rows") != 2048
        or representation.get("safe_resource_bounds")
        != {
            "maximum_representation_runs": 29,
            "maximum_jepa_steps": 10_000_000,
            "maximum_shape_rows": 25_000_000,
            "maximum_embedding_bytes": 200_000_000_000,
        }
        or representation.get("future_difficulty_adaptation", {}).get("paper_matrix") is not False
    ):
        raise ValueError("Representation configuration contradicts the locked comparison.")
    sparse_target = representation.get("sparse_target", {})
    if (
        sparse_target.get("p") != 2.0
        or sparse_target.get("mu") != -0.6744897501960817
        or sparse_target.get("sigma") != 1.0
    ):
        raise ValueError("Primary sparse target must be the locked rectified Gaussian.")
    if (
        lightgbm.get("num_leaves") != [15, 31]
        or lightgbm.get("min_child_samples") != [50, 200]
        or lightgbm.get("reg_lambda") != [1.0, 10.0]
        or lightgbm.get("learning_rate") != 0.03
        or lightgbm.get("n_estimators") != 2000
        or lightgbm.get("early_stopping_rounds") != 100
    ):
        raise ValueError("LightGBM configuration contradicts the locked grid.")
    if (
        evaluation.get("bootstrap_block_dates") != 5
        or evaluation.get("bootstrap_block_sensitivity_dates") != [1, 10]
        or evaluation.get("bootstrap_repetitions") != 10_000
        or evaluation.get("confidence") != 0.95
        or evaluation.get("multiple_testing") != "holm"
        or len(evaluation.get("confirmatory_contrast_definitions", ())) != 5
    ):
        raise ValueError("Evaluation configuration contradicts locked inference.")
    if tca.get("forecast_update_minutes") != 15 or tca.get("fill_minutes") != 1:
        raise ValueError("TCA update/fill clocks contradict the locked design.")
    if (
        tca.get("window") != ["10:30", "15:30"]
        or tca.get("quantity_fraction_adv20") != 0.03
        or tca.get("planned_participation_rate") != 0.10
        or tca.get("hard_participation_rate") != 0.10
        or tca.get("risk_aversion") != 0.0
        or tca.get("tracking_penalty") != 0.0
    ):
        raise ValueError("TCA configuration contradicts the locked experiment.")
    if data.get("allow_full_paper_run") and not data.get("allow_historical_training"):
        raise ValueError("Full paper evaluation requires historical-training authorization.")
    if protocol_id == "sparse-jepa-v2":
        _validate_v2_quality_sections(data, sequence)


def _validate_v2_quality_sections(data: dict[str, Any], sequence: dict[str, Any]) -> None:
    """Reject any silent relaxation of the v2 daily/token/minute hierarchy."""
    if (
        data.get("formation_frequency") != "1Day"
        or data.get("target_frequency") != "1min"
        or not math.isclose(
            float(data.get("formation_daily_completeness_minimum", float("nan"))),
            0.95,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or data.get("quality_hierarchy")
        != ["daily_formation", "token_15min_representation", "exact_minute_tca_window"]
        or data.get("missing_minute_policy") != "never_zero_fill_or_interpolate"
        or data.get("tca_required_minutes")
        != {"start_inclusive": "10:30", "end_exclusive": "15:30", "count": 300}
    ):
        raise ValueError("V2 data configuration contradicts the resolution-quality protocol.")
    if (
        sequence.get("quality_protocol") != "resolution-aware-v2"
        or sequence.get("minimum_observed_bars_per_token") != 2
        or sequence.get("token_aggregation") != "observed_provider_bars_only"
        or sequence.get("realized_volatility") != "observed_close_log_return_sum_of_squares"
        or sequence.get("missing_minute_policy") != "never_zero_fill_or_interpolate"
        or sequence.get("primary_session_rule") != "all_26_tokens_valid_and_standard_xnys_session"
        or sequence.get("token_completeness_bands")
        != {
            "high_minimum": 0.95,
            "medium_minimum": 0.80,
            "low_maximum_exclusive": 0.80,
        }
    ):
        raise ValueError("V2 sequence configuration contradicts the token-quality protocol.")
