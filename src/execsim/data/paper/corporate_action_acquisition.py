"""Acquire and normalize point-in-time split metadata from Alpaca."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from execsim.data.paper.corporate_action_manifest import ingest_corporate_actions
from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic
from execsim.data.paper.schemas import InstrumentSymbolInterval, PaperDataConfig

_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
_ACTION_TYPES = ("forward_split", "reverse_split")
_RESPONSE_FIELDS = {
    "forward_splits": "forward_split",
    "reverse_splits": "reverse_split",
}
_OUTPUT_COLUMNS = (
    "instrument_id",
    "effective_date",
    "factor",
    "available_at",
    "source",
    "provider_action_id",
    "action_type",
    "symbol",
    "process_date",
    "old_rate",
    "new_rate",
)


class CorporateActionPageFetcher(Protocol):
    """Return one decoded provider page and non-secret response metadata."""

    def __call__(self, params: Mapping[str, str]) -> tuple[dict[str, object], dict[str, str]]: ...


def create_alpaca_corporate_action_fetcher() -> CorporateActionPageFetcher:
    """Create the authenticated Alpaca corporate-action page boundary."""
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY.")

    def fetch(params: Mapping[str, str]) -> tuple[dict[str, object], dict[str, str]]:
        request = Request(
            _ENDPOINT + "?" + urlencode(params),
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
        )
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read())
                    headers = {
                        name: value
                        for name in (
                            "X-RateLimit-Limit",
                            "X-RateLimit-Remaining",
                            "X-RateLimit-Reset",
                        )
                        if (value := response.headers.get(name)) is not None
                    }
                if not isinstance(payload, dict):
                    raise ValueError("Alpaca corporate-action response must be an object.")
                return payload, headers
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(
                    f"Alpaca corporate-action request failed with HTTP {exc.code}: {detail}"
                )
                if exc.code != 429 and exc.code < 500:
                    break
            except Exception as exc:  # network stacks expose heterogeneous failures
                last_error = exc
            if attempt < 4:
                time.sleep(min(2**attempt, 16))
        raise RuntimeError("BLOCKED: Alpaca corporate-action acquisition failed.") from last_error

    return fetch


def normalize_split_actions(
    payloads: tuple[dict[str, object], ...],
    *,
    symbol_history: tuple[InstrumentSymbolInterval, ...],
    instrument_ids: tuple[str, ...],
    effective_end: date,
) -> pd.DataFrame:
    """Normalize split rates to the repository's causal restatement convention.

    ``factor`` is ``old_rate / new_rate``. Dividing raw price by this factor and
    multiplying raw volume by it maps post-split observations to the preceding
    share basis while preserving dollar notional.
    """
    selected = frozenset(instrument_ids)
    rows: list[dict[str, object]] = []
    action_ids: set[str] = set()
    for payload in payloads:
        unknown = set(payload).difference({*_RESPONSE_FIELDS, "next_page_token"})
        if unknown:
            raise ValueError(f"Alpaca corporate-action response has unknown fields: {unknown}")
        for response_field, action_type in _RESPONSE_FIELDS.items():
            actions = payload.get(response_field, [])
            if not isinstance(actions, list):
                raise ValueError(f"Alpaca {response_field} must be a list.")
            for item in actions:
                if not isinstance(item, dict):
                    raise ValueError(f"Alpaca {response_field} contains a non-object row.")
                action_id = str(item.get("id", "")).strip()
                symbol = str(item.get("symbol", "")).strip().upper()
                if not action_id or not symbol:
                    raise ValueError("Alpaca split action is missing id or symbol.")
                if action_id in action_ids:
                    raise ValueError(f"Duplicate Alpaca corporate-action id: {action_id}")
                action_ids.add(action_id)
                effective_date = _required_date(item, "ex_date")
                process_date = _required_date(item, "process_date")
                if effective_date > effective_end:
                    continue
                old_rate = _positive_rate(item, "old_rate")
                new_rate = _positive_rate(item, "new_rate")
                instrument_id = _resolve_action_instrument(
                    symbol_history,
                    selected=selected,
                    symbol=symbol,
                    effective_date=effective_date,
                )
                available_at = datetime.combine(
                    process_date + timedelta(days=1), datetime_time.min, tzinfo=UTC
                )
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "effective_date": effective_date,
                        "factor": old_rate / new_rate,
                        "available_at": available_at,
                        "source": f"{_ENDPOINT}#{action_id}",
                        "provider_action_id": action_id,
                        "action_type": action_type,
                        "symbol": symbol,
                        "process_date": process_date,
                        "old_rate": old_rate,
                        "new_rate": new_rate,
                    }
                )
    frame = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    if frame.empty:
        return frame
    if frame.duplicated(["instrument_id", "effective_date"]).any():
        raise RuntimeError("BLOCKED: multiple split actions share one instrument/effective date.")
    return frame.sort_values(
        ["instrument_id", "effective_date", "provider_action_id"], kind="stable"
    ).reset_index(drop=True)


def acquire_split_actions(
    symbol_history: tuple[InstrumentSymbolInterval, ...],
    instrument_ids: tuple[str, ...],
    *,
    start: date,
    end: date,
    output_path: Path,
    raw_output_path: Path,
    receipt_path: Path,
    paper_config_hash: str,
    config: PaperDataConfig,
    cli_enabled: bool,
    fetch_page: CorporateActionPageFetcher | None = None,
) -> dict[str, object]:
    """Acquire all selected split actions with identity-bound idempotent receipts."""
    from execsim.data.paper.acquisition import authorize_acquisition

    authorize_acquisition(config, cli_enabled=cli_enabled)
    if start > end:
        raise ValueError("Corporate-action start must not follow end.")
    selected = frozenset(instrument_ids)
    symbols = sorted(
        {
            interval.symbol.upper()
            for interval in symbol_history
            if interval.instrument_id in selected
            and max(interval.start, start) <= min(interval.end, end)
        }
    )
    if not symbols:
        raise RuntimeError("BLOCKED: no sourced symbols exist for corporate-action acquisition.")
    history_identity = [
        {
            "instrument_id": interval.instrument_id,
            "symbol": interval.symbol.upper(),
            "start": interval.start.isoformat(),
            "end": interval.end.isoformat(),
            "source": interval.source,
        }
        for interval in sorted(
            (
                item
                for item in symbol_history
                if item.instrument_id in selected and max(item.start, start) <= min(item.end, end)
            ),
            key=lambda item: (item.instrument_id, item.start, item.end, item.symbol),
        )
    ]
    request_identity: dict[str, object] = {
        "provider": "alpaca",
        "endpoint": _ENDPOINT,
        "types": list(_ACTION_TYPES),
        "region": "us",
        "data_quality": "complete",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "instrument_ids": sorted(selected),
        "symbols": symbols,
        "symbol_history_sha256": stable_hash(history_identity),
        "paper_config_hash": paper_config_hash,
    }
    existing = (output_path.is_file(), raw_output_path.is_file(), receipt_path.is_file())
    if any(existing):
        if not all(existing):
            raise ValueError("Existing corporate-action artifact set is incomplete.")
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "complete"
            or receipt.get("request") != request_identity
            or receipt.get("source_sha256") != file_sha256(output_path)
            or receipt.get("raw_sha256") != file_sha256(raw_output_path)
        ):
            raise ValueError("Existing corporate-action artifacts are incompatible.")
        actions = ingest_corporate_actions(output_path)
        if len(actions) != receipt.get("row_count"):
            raise ValueError("Existing corporate-action row count is incompatible.")
        return receipt

    fetch = fetch_page or create_alpaca_corporate_action_fetcher()
    base_params = {
        "symbols": ",".join(symbols),
        "types": ",".join(_ACTION_TYPES),
        "region": "us",
        "data_quality": "complete",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": "1000",
        "sort": "asc",
    }
    payloads: list[dict[str, object]] = []
    rate_limit_headers: dict[str, str] = {}
    page_token: str | None = None
    while True:
        params = {**base_params, **({"page_token": page_token} if page_token else {})}
        payload, headers = fetch(params)
        payloads.append(payload)
        rate_limit_headers = headers
        next_token = payload.get("next_page_token")
        if next_token is None:
            break
        if not isinstance(next_token, str) or not next_token or len(payloads) >= 100:
            raise ValueError("Alpaca corporate-action pagination metadata is invalid.")
        page_token = next_token

    actions = normalize_split_actions(
        tuple(payloads),
        symbol_history=symbol_history,
        instrument_ids=instrument_ids,
        effective_end=end,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    actions.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    write_json_atomic(
        raw_output_path,
        {
            "schema_version": "paper-alpaca-corporate-actions-raw-v1",
            "request": request_identity,
            "pages": payloads,
        },
    )
    receipt = {
        "schema_version": "paper-corporate-action-acquisition-v1",
        "status": "complete",
        "request": request_identity,
        "row_count": len(actions),
        "pages": len(payloads),
        "source_sha256": file_sha256(output_path),
        "raw_sha256": file_sha256(raw_output_path),
        "rate_limit_headers": rate_limit_headers,
        "available_at_policy": "provider_process_date_plus_one_utc_day",
        "provider_creation_time_available": False,
        "provider_vintage_limitation": (
            "Alpaca does not guarantee corporate-action creation timestamps; "
            "process_date plus one UTC day is the conservative causal availability surrogate."
        ),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "paper_config_hash": paper_config_hash,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _resolve_action_instrument(
    symbol_history: tuple[InstrumentSymbolInterval, ...],
    *,
    selected: frozenset[str],
    symbol: str,
    effective_date: date,
) -> str:
    matches = {
        interval.instrument_id
        for interval in symbol_history
        if interval.instrument_id in selected
        and interval.symbol.upper() == symbol
        and interval.start <= effective_date <= interval.end
    }
    if len(matches) != 1:
        raise RuntimeError(
            "BLOCKED: corporate-action symbol does not resolve uniquely through sourced "
            f"history: symbol={symbol}, effective_date={effective_date}, matches={sorted(matches)}"
        )
    return next(iter(matches))


def _required_date(item: Mapping[str, Any], field: str) -> date:
    try:
        parsed = date.fromisoformat(str(item[field])[:10])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Alpaca split action has invalid {field}.") from exc
    return parsed


def _positive_rate(item: Mapping[str, Any], field: str) -> float:
    try:
        value = float(item[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Alpaca split action has invalid {field}.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Alpaca split action has invalid {field}.")
    return value
