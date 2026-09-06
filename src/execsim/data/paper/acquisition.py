"""Resumable Alpaca SIP acquisition planning with explicit network authorization."""

from __future__ import annotations

import calendar
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from execsim.data.paper.manifests import write_json_atomic
from execsim.data.paper.schemas import (
    PAPER_BAR_COLUMNS,
    AcquisitionChunk,
    AcquisitionReceipt,
    PaperDataConfig,
    ProviderResponse,
)
from execsim.data.paper.validation import (
    expected_xnys_minutes,
    validate_exact_xnys_session,
    validate_paper_bars,
)


class ChunkFetcher(Protocol):
    """Provider boundary that returns the immutable raw response bytes."""

    def __call__(self, chunk: AcquisitionChunk) -> ProviderResponse: ...


def create_alpaca_sip_fetcher() -> ChunkFetcher:
    """Create the licensed SIP-only provider adapter from environment credentials."""
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY.")
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'data' or 'paper' extra for Alpaca acquisition.") from exc

    client = StockHistoricalDataClient(api_key, api_secret)

    def fetch(chunk: AcquisitionChunk) -> ProviderResponse:
        request = StockBarsRequest(
            symbol_or_symbols=chunk.symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(chunk.start, datetime.min.time(), tzinfo=UTC),
            end=datetime.combine(chunk.end, datetime.max.time(), tzinfo=UTC),
            adjustment=Adjustment.RAW,
            feed=DataFeed.SIP,
            asof=chunk.end.isoformat(),
        )
        response = cast(Any, client.get_stock_bars(request))
        frame = _normalize_alpaca_frame(response.df.reset_index(), chunk)
        buffer = BytesIO()
        frame.to_parquet(buffer, index=False)
        return ProviderResponse(buffer.getvalue(), len(frame))

    return fetch


def _normalize_alpaca_frame(frame: pd.DataFrame, chunk: AcquisitionChunk) -> pd.DataFrame:
    """Normalize SDK output, including empty delisted-symbol responses, into corpus schema."""
    if frame.empty:
        normalized = pd.DataFrame(columns=PAPER_BAR_COLUMNS)
    else:
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise", utc=True).dt.tz_convert(
            "America/New_York"
        )
        regular = (timestamps.dt.time >= datetime.strptime("09:30", "%H:%M").time()) & (
            timestamps.dt.time <= datetime.strptime("15:59", "%H:%M").time()
        )
        normalized = frame.loc[regular].copy()
        normalized["timestamp"] = timestamps.loc[regular]
    normalized["instrument_id"] = chunk.instrument_id
    normalized["symbol"] = chunk.symbol
    return normalized


def monthly_chunks(
    instrument_id: str, symbol: str, start: date, end: date
) -> tuple[AcquisitionChunk, ...]:
    """Create stable inclusive monthly request chunks."""
    if start > end:
        raise ValueError("Acquisition start must not follow end.")
    chunks = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = date(
            cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]
        )
        chunks.append(
            AcquisitionChunk(instrument_id, symbol.upper(), max(start, cursor), min(end, month_end))
        )
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return tuple(chunks)


def authorize_acquisition(config: PaperDataConfig, *, cli_enabled: bool) -> None:
    """Require both configuration and command-line network authorization."""
    if not config.allow_network or not cli_enabled:
        raise PermissionError(
            "Paper data acquisition is disabled. Set allow_network=true and pass "
            "--enable-network after separately authorizing the download."
        )
    if config.feed != "sip":
        raise ValueError("The paper corpus requires Alpaca SIP; IEX fallback is prohibited.")


def probe_alpaca_sip_entitlement(
    config: PaperDataConfig,
    *,
    cli_enabled: bool,
    output: Path,
) -> dict[str, object]:
    """Verify old SIP minute access, raw adjustment, pagination, and an exact XNYS grid."""
    authorize_acquisition(config, cli_enabled=cli_enabled)
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY.")
    params = {
        "timeframe": "1Min",
        "start": "2021-01-04T00:00:00Z",
        "end": "2021-01-05T00:00:00Z",
        "adjustment": "raw",
        "feed": "sip",
        "asof": "2021-01-04",
        "limit": "100",
        "sort": "asc",
    }
    bars: list[dict[str, object]] = []
    pages = 0
    rate_limit: dict[str, str] = {}
    page_token: str | None = None
    while True:
        query = {**params, **({"page_token": page_token} if page_token else {})}
        request = Request(
            "https://data.alpaca.markets/v2/stocks/AAPL/bars?" + urlencode(query),
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
                rate_limit = {
                    name: value
                    for name in (
                        "X-RateLimit-Limit",
                        "X-RateLimit-Remaining",
                        "X-RateLimit-Reset",
                    )
                    if (value := response.headers.get(name)) is not None
                }
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"BLOCKED: Alpaca SIP entitlement probe failed with HTTP {exc.code}: {detail}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("bars"), list):
            raise ValueError("Alpaca SIP probe response schema is invalid.")
        bars.extend(payload["bars"])
        pages += 1
        page_token = payload.get("next_page_token")
        if page_token is None:
            break
        if not isinstance(page_token, str) or pages > 20:
            raise ValueError("Alpaca SIP probe pagination metadata is invalid.")
    frame = pd.DataFrame(bars).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "trade_count",
            "vw": "vwap",
        }
    )
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise", utc=True).dt.tz_convert(
        "America/New_York"
    )
    regular = (timestamps.dt.time >= datetime.strptime("09:30", "%H:%M").time()) & (
        timestamps.dt.time <= datetime.strptime("15:59", "%H:%M").time()
    )
    frame = frame.loc[regular].copy()
    frame["timestamp"] = timestamps.loc[regular]
    frame["instrument_id"] = "probe-aapl"
    frame["symbol"] = "AAPL"
    errors = validate_exact_xnys_session(frame)
    if errors:
        raise RuntimeError(f"BLOCKED: Alpaca SIP probe session is invalid: {errors}")
    receipt = {
        "schema_version": "paper-alpaca-sip-probe-v1",
        "status": "PASS",
        "provider": "alpaca",
        "feed": "sip",
        "adjustment": "raw",
        "symbol": "AAPL",
        "session_date": "2021-01-04",
        "regular_session_rows": len(frame),
        "pages": pages,
        "pagination_verified": pages > 1,
        "rate_limit_headers": rate_limit,
        "paper_config_hash": config.paper_config_hash,
        "probed_at_utc": datetime.now(UTC).isoformat(),
    }
    if pages <= 1:
        raise RuntimeError("BLOCKED: Alpaca SIP probe did not exercise pagination.")
    write_json_atomic(output, receipt)
    return receipt


def write_failure_receipt(
    path: Path, chunk: AcquisitionChunk, error: Exception, *, paper_config_hash: str = ""
) -> Path:
    """Persist a failed attempt without creating a false complete marker."""
    receipt = AcquisitionReceipt(
        chunk.identity,
        "failed",
        chunk.start.isoformat(),
        chunk.end.isoformat(),
        chunk.feed,
        chunk.adjustment,
        datetime.now(UTC).isoformat(),
        None,
        0,
        instrument_id=chunk.instrument_id,
        symbol=chunk.symbol,
        error=f"{type(error).__name__}: {error}",
        paper_config_hash=paper_config_hash,
    )
    return write_json_atomic(path, asdict(receipt))


def acquire_chunk(
    chunk: AcquisitionChunk,
    *,
    output_directory: Path,
    fetch: ChunkFetcher,
    config: PaperDataConfig,
    cli_enabled: bool,
    max_attempts: int = 3,
) -> AcquisitionReceipt:
    """Fetch one idempotent chunk with bounded retries and atomic completion metadata."""
    authorize_acquisition(config, cli_enabled=cli_enabled)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / f"{chunk.identity}.response"
    receipt_path = output_directory / f"{chunk.identity}.json"
    if raw_path.is_file() and receipt_path.is_file():
        from execsim.data.paper.manifests import read_json

        existing = AcquisitionReceipt(**read_json(receipt_path))
        if (
            existing.status == "complete"
            and existing.chunk_identity == chunk.identity
            and existing.instrument_id == chunk.instrument_id
            and existing.symbol == chunk.symbol
            and existing.requested_start == chunk.start.isoformat()
            and existing.requested_end == chunk.end.isoformat()
            and existing.feed == chunk.feed
            and existing.adjustment == chunk.adjustment
            and existing.paper_config_hash == config.paper_config_hash
            and existing.response_sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        ):
            observed, expected = _validate_provider_response(
                chunk, raw_path.read_bytes(), existing.row_count
            )
            if observed != existing.observed_sessions or expected != existing.expected_sessions:
                raise ValueError(
                    f"Existing acquisition coverage metadata is inconsistent: {chunk.identity}"
                )
            return existing
        raise ValueError(f"Existing acquisition chunk is inconsistent: {chunk.identity}")
    last_error: Exception | None = None
    temporary = raw_path.with_suffix(raw_path.suffix + ".part")
    for _ in range(max_attempts):
        try:
            response = fetch(chunk)
            observed_sessions, expected_sessions = _validate_provider_response(
                chunk, response.content, response.row_count
            )
            temporary.write_bytes(response.content)
            os.replace(temporary, raw_path)
            receipt = AcquisitionReceipt(
                chunk.identity,
                "complete",
                chunk.start.isoformat(),
                chunk.end.isoformat(),
                chunk.feed,
                chunk.adjustment,
                datetime.now(UTC).isoformat(),
                hashlib.sha256(response.content).hexdigest(),
                response.row_count,
                instrument_id=chunk.instrument_id,
                symbol=chunk.symbol,
                observed_sessions=observed_sessions,
                expected_sessions=expected_sessions,
                provider_request_id=response.request_id,
                paper_config_hash=config.paper_config_hash,
            )
            write_json_atomic(receipt_path, asdict(receipt))
            return receipt
        except Exception as exc:  # provider implementations expose heterogeneous failures
            last_error = exc
            temporary.unlink(missing_ok=True)
    if last_error is None:  # pragma: no cover - max_attempts validation makes this unreachable
        raise RuntimeError("Acquisition failed without an exception.")
    write_failure_receipt(
        receipt_path, chunk, last_error, paper_config_hash=config.paper_config_hash
    )
    raise RuntimeError(
        f"Acquisition failed after {max_attempts} attempts: {chunk.identity}"
    ) from last_error


def _validate_provider_response(
    chunk: AcquisitionChunk, content: bytes, declared_rows: int
) -> tuple[int, int]:
    """Validate provider Parquet schema, identity, interval, and session coverage."""
    try:
        frame = pd.read_parquet(BytesIO(content))
    except Exception as exc:
        raise ValueError("Provider response is not readable Parquet.") from exc
    if len(frame) != declared_rows or declared_rows <= 0:
        raise ValueError("Provider response row count is zero or does not match metadata.")
    required = {
        "instrument_id",
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Provider response schema is incomplete: {sorted(missing)}")
    if set(frame["instrument_id"].astype(str)) != {chunk.instrument_id}:
        raise ValueError("Provider response instrument identity does not match request.")
    if set(frame["symbol"].astype(str).str.upper()) != {chunk.symbol.upper()}:
        raise ValueError("Provider response symbol does not match request interval.")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any() or timestamps.dt.tz is None:
        raise ValueError("Provider response timestamps must be timezone-aware.")
    if str(timestamps.dt.tz) != "America/New_York":
        raise ValueError("Provider response timezone must be exactly America/New_York.")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    if local_dates.min() < chunk.start or local_dates.max() > chunk.end:
        raise ValueError("Provider response contains rows outside the exact request interval.")
    observed = 0
    for session_date, session in frame.groupby(local_dates, sort=True):
        structural_errors = validate_paper_bars(session, expected_minutes=len(session))
        if structural_errors:
            raise ValueError(
                "Provider response contains structurally invalid bars: "
                f"{session_date}: {structural_errors}"
            )
        expected = expected_xnys_minutes(session_date)
        actual = pd.DatetimeIndex(pd.to_datetime(session["timestamp"]))
        if len(actual.difference(expected)):
            raise ValueError("Provider response contains off-grid regular-session timestamps.")
        if len(expected) == 390 and not validate_exact_xnys_session(session):
            observed += 1
    calendar = __import__("exchange_calendars").get_calendar("XNYS")
    expected_sessions = len(calendar.sessions_in_range(chunk.start, chunk.end))
    if expected_sessions <= 0:
        raise ValueError("Provider response has no expected XNYS session coverage metadata.")
    return observed, expected_sessions
