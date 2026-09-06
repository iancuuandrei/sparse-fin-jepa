"""Corpus-wide, fold-safe sequence construction from validated minute bars."""

from __future__ import annotations

import multiprocessing
from bisect import bisect_left
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.corporate_actions import (
    apply_point_in_time_split_adjustment,
    point_in_time_split_factor,
)
from execsim.data.paper.manifests import file_sha256
from execsim.data.paper.partitions import fold_training_cutoff, paper_fold, resolve_fold_partition
from execsim.data.paper.resolution_quality import (
    aggregate_observed_tokens,
    assess_session_resolution_quality,
)
from execsim.data.paper.validation import validate_exact_xnys_session
from execsim.ml.sequences.builder import _aggregate_tokens, build_session_sequence
from execsim.ml.sequences.index import build_sample_index, write_sample_index
from execsim.ml.sequences.manifests import (
    SequenceManifest,
    write_sequence_manifest,
    write_sequence_record,
)
from execsim.ml.sequences.normalization import RobustFoldNormalizer
from execsim.ml.sequences.schemas import SequenceRecord

_TOKEN_CACHE_ATTR = "_execsim_paper_tokens"


@dataclass(frozen=True, slots=True)
class _HistoricalMemberWorkerState:
    """Hold immutable fold state initialized once in each spawned worker."""

    corpus_root: Path
    corporate_actions: pd.DataFrame
    fold_id: str
    cutoff: date
    spy_sessions: tuple[tuple[date, pd.DataFrame], ...]
    spy_dates: tuple[date, ...]
    spy_by_date: dict[date, pd.DataFrame]
    spy_token_cache: dict[date, pd.DataFrame]
    data_classification: str
    quality_protocol: str
    symbol_history: tuple[dict[str, Any], ...]
    spy_instrument_id: str


_HISTORICAL_MEMBER_WORKER_STATE: _HistoricalMemberWorkerState | None = None


def build_fold_sequence_corpus(
    bars: pd.DataFrame,
    *,
    universe_members: tuple[dict[str, Any], ...],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    output_root: Path,
    universe_manifest_hash: str,
    corporate_action_manifest_hash: str,
    config_hash: str,
    spy_instrument_id: str,
    data_classification: str,
    quality_protocol: str = "exact-minute-v1",
    symbol_history: tuple[dict[str, Any], ...] = (),
) -> SequenceManifest:
    """Build, normalize, index, and manifest every valid session in one fold."""
    instruments = tuple(str(member["instrument_id"]) for member in universe_members)
    if spy_instrument_id in instruments:
        raise ValueError("SPY benchmark identity must not be an execution-universe member.")
    available = set(bars["instrument_id"].astype(str).unique())
    if spy_instrument_id not in available:
        raise ValueError("SPY corpus is required for every paper sequence build.")
    missing = set(instruments).difference(available)
    if missing:
        raise ValueError(f"Universe instruments missing from raw corpus: {sorted(missing)}")
    if data_classification != "synthetic_fixture" and not symbol_history:
        raise ValueError("Historical sequence builds require sourced symbol history.")

    def load_sessions(
        instrument_id: str,
        active_fold_id: str,
        exclusions: list[dict[str, str]] | None,
    ) -> list[tuple[date, pd.DataFrame]]:
        return _validated_sessions(
            bars,
            instrument_id,
            fold_id=active_fold_id,
            exclusions=exclusions,
            quality_protocol=quality_protocol,
        )

    return _build_fold_sequence_corpus_from_sessions(
        load_sessions,
        universe_members=universe_members,
        corporate_actions=corporate_actions,
        fold_id=fold_id,
        output_root=output_root,
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        spy_instrument_id=spy_instrument_id,
        data_classification=data_classification,
        quality_protocol=quality_protocol,
        symbol_history=symbol_history,
    )


def build_fold_sequence_corpus_from_root(
    corpus_root: Path,
    *,
    universe_members: tuple[dict[str, Any], ...],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    output_root: Path,
    universe_manifest_hash: str,
    corporate_action_manifest_hash: str,
    config_hash: str,
    spy_instrument_id: str,
    data_classification: str,
    quality_protocol: str = "exact-minute-v1",
    symbol_history: tuple[dict[str, Any], ...] = (),
) -> SequenceManifest:
    """Build one fold while holding only one instrument corpus in memory."""
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"Paper corpus root does not exist: {corpus_root}")
    if data_classification != "synthetic_fixture" and not symbol_history:
        raise ValueError("Historical sequence builds require sourced symbol history.")

    def load_sessions(
        instrument_id: str,
        active_fold_id: str,
        exclusions: list[dict[str, str]] | None,
    ) -> list[tuple[date, pd.DataFrame]]:
        bars = load_corpus_instrument(corpus_root, instrument_id)
        if active_fold_id != "unresolved":
            timestamps = pd.to_datetime(bars["timestamp"])
            local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
            fold = paper_fold(active_fold_id)
            keep = (local_dates >= fold.train_start) & (local_dates <= fold.test_end)
            bars = bars.loc[keep]
        return _validated_sessions(
            bars,
            instrument_id,
            fold_id=active_fold_id,
            exclusions=exclusions,
            quality_protocol=quality_protocol,
        )

    return _build_fold_sequence_corpus_from_sessions(
        load_sessions,
        universe_members=universe_members,
        corporate_actions=corporate_actions,
        fold_id=fold_id,
        output_root=output_root,
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        spy_instrument_id=spy_instrument_id,
        data_classification=data_classification,
        quality_protocol=quality_protocol,
        symbol_history=symbol_history,
        historical_corpus_root=corpus_root.resolve(),
    )


SessionLoader = Callable[[str, str, list[dict[str, str]] | None], list[tuple[date, pd.DataFrame]]]


def _build_fold_sequence_corpus_from_sessions(
    load_sessions: SessionLoader,
    *,
    universe_members: tuple[dict[str, Any], ...],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    output_root: Path,
    universe_manifest_hash: str,
    corporate_action_manifest_hash: str,
    config_hash: str,
    spy_instrument_id: str,
    data_classification: str,
    quality_protocol: str,
    symbol_history: tuple[dict[str, Any], ...],
    historical_corpus_root: Path | None = None,
) -> SequenceManifest:
    """Build one fold from a bounded session-loader boundary."""
    cutoff = fold_training_cutoff(fold_id)
    fold_root = output_root / fold_id
    instruments = tuple(str(member["instrument_id"]) for member in universe_members)
    if spy_instrument_id in instruments:
        raise ValueError("SPY benchmark identity must not be an execution-universe member.")
    spy_sessions = [
        item
        for item in load_sessions(spy_instrument_id, fold_id, None)
        if _belongs_to_fold(fold_id, item[0])
    ]
    if not spy_sessions:
        raise ValueError("SPY corpus is required for every paper sequence build.")
    spy_dates = [item[0] for item in spy_sessions]
    spy_by_date = dict(spy_sessions)
    spy_token_cache = _token_cache(spy_sessions, quality_protocol=quality_protocol)

    records: list[tuple[str, SequenceRecord]] = []
    exclusions: list[dict[str, str]] = []
    raw_hashes: list[str] = []
    worker_count = min(8, max(1, len(universe_members)))
    if historical_corpus_root is None:

        def build_member(
            member: dict[str, Any],
        ) -> tuple[list[tuple[str, SequenceRecord]], list[dict[str, str]], list[str]]:
            member_exclusions: list[dict[str, str]] = []
            instrument_id = str(member["instrument_id"])
            sessions = [
                item
                for item in load_sessions(instrument_id, fold_id, member_exclusions)
                if _belongs_to_fold(fold_id, item[0])
            ]
            return _build_member_records(
                member,
                sessions,
                member_exclusions=member_exclusions,
                corporate_actions=corporate_actions,
                fold_id=fold_id,
                cutoff=cutoff,
                spy_sessions=spy_sessions,
                spy_dates=spy_dates,
                spy_by_date=spy_by_date,
                spy_token_cache=spy_token_cache,
                data_classification=data_classification,
                quality_protocol=quality_protocol,
                symbol_history=symbol_history,
                spy_instrument_id=spy_instrument_id,
            )

        executor: ThreadPoolExecutor | ProcessPoolExecutor
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="paper-sequence")
        member_results = executor.map(build_member, universe_members)
    else:
        state = _HistoricalMemberWorkerState(
            corpus_root=historical_corpus_root,
            corporate_actions=corporate_actions,
            fold_id=fold_id,
            cutoff=cutoff,
            spy_sessions=tuple(spy_sessions),
            spy_dates=tuple(spy_dates),
            spy_by_date=spy_by_date,
            spy_token_cache=spy_token_cache,
            data_classification=data_classification,
            quality_protocol=quality_protocol,
            symbol_history=symbol_history,
            spy_instrument_id=spy_instrument_id,
        )
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_historical_member_worker,
            initargs=(state,),
        )
        member_results = executor.map(_build_historical_member_worker, universe_members)
    with executor:
        for member_records, member_exclusions, member_hashes in member_results:
            records.extend(member_records)
            exclusions.extend(member_exclusions)
            raw_hashes.extend(member_hashes)
    training = [record for partition, record in records if partition == "train"]
    if not training:
        raise ValueError(f"No valid training sequences were produced for {fold_id}.")
    training_values = np.stack([record.features for record in training])
    training_masks = np.stack([record.token_mask for record in training])
    normalizer = RobustFoldNormalizer.fit(training_values, training_masks)
    sequence_files: list[str] = []
    index_files: list[str] = []
    counts = {"train": 0, "validation": 0, "test": 0}

    def write_record(item: tuple[str, SequenceRecord]) -> tuple[str, str, str]:
        partition, record = item
        normalized = replace(
            record,
            features=normalizer.transform(record.features[None, ...], record.token_mask[None, ...])[
                0
            ],
        )
        sequence_path = write_sequence_record(normalized, fold_root / "sessions" / partition)
        relative_sequence = str(sequence_path.relative_to(fold_root)).replace("\\", "/")
        sequence_hash = file_sha256(sequence_path)
        samples = build_sample_index(
            normalized,
            fold_id=fold_id,
            partition=partition,
            source_sequence_hash=sequence_hash,
        )
        index_path = fold_root / "indexes" / partition / f"{record.session_id}.parquet"
        write_sample_index(samples, index_path)
        relative_index = str(index_path.relative_to(fold_root)).replace("\\", "/")
        return partition, relative_sequence, relative_index

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="paper-artifact") as pool:
        written = pool.map(write_record, records)
        for partition, relative_sequence, relative_index in written:
            sequence_files.append(relative_sequence)
            index_files.append(relative_index)
            counts[partition] += 1
    return write_sequence_manifest(
        fold_root / "sequence-manifest.json",
        fold_id=fold_id,
        cutoff=cutoff.isoformat(),
        raw_hashes=tuple(raw_hashes),
        sequence_files=tuple(sequence_files),
        normalizer=normalizer,
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        index_files=tuple(index_files),
        partition_counts=counts,
        exclusions=tuple(exclusions),
        quality_protocol=quality_protocol,
    )


def _initialize_historical_member_worker(state: _HistoricalMemberWorkerState) -> None:
    """Install immutable fold state once per spawn-safe historical worker."""
    global _HISTORICAL_MEMBER_WORKER_STATE
    _HISTORICAL_MEMBER_WORKER_STATE = state


def _build_historical_member_worker(
    member: dict[str, Any],
) -> tuple[list[tuple[str, SequenceRecord]], list[dict[str, str]], list[str]]:
    """Build one historical instrument using bounded worker-local corpus reads."""
    state = _HISTORICAL_MEMBER_WORKER_STATE
    if state is None:
        raise RuntimeError("Historical sequence worker was not initialized.")
    member_exclusions: list[dict[str, str]] = []
    instrument_id = str(member["instrument_id"])
    bars = load_corpus_instrument(state.corpus_root, instrument_id)
    timestamps = pd.to_datetime(bars["timestamp"])
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    fold = paper_fold(state.fold_id)
    keep = (local_dates >= fold.train_start) & (local_dates <= fold.test_end)
    sessions = [
        item
        for item in _validated_sessions(
            bars.loc[keep],
            instrument_id,
            fold_id=state.fold_id,
            exclusions=member_exclusions,
            quality_protocol=state.quality_protocol,
        )
        if _belongs_to_fold(state.fold_id, item[0])
    ]
    return _build_member_records(
        member,
        sessions,
        member_exclusions=member_exclusions,
        corporate_actions=state.corporate_actions,
        fold_id=state.fold_id,
        cutoff=state.cutoff,
        spy_sessions=list(state.spy_sessions),
        spy_dates=list(state.spy_dates),
        spy_by_date=state.spy_by_date,
        spy_token_cache=state.spy_token_cache,
        data_classification=state.data_classification,
        quality_protocol=state.quality_protocol,
        symbol_history=state.symbol_history,
        spy_instrument_id=state.spy_instrument_id,
    )


def _build_member_records(
    member: dict[str, Any],
    sessions: list[tuple[date, pd.DataFrame]],
    *,
    member_exclusions: list[dict[str, str]],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    cutoff: date,
    spy_sessions: list[tuple[date, pd.DataFrame]],
    spy_dates: list[date],
    spy_by_date: dict[date, pd.DataFrame],
    spy_token_cache: dict[date, pd.DataFrame],
    data_classification: str,
    quality_protocol: str,
    symbol_history: tuple[dict[str, Any], ...],
    spy_instrument_id: str,
) -> tuple[list[tuple[str, SequenceRecord]], list[dict[str, str]], list[str]]:
    """Construct one member's ordered records under one immutable fold state."""
    instrument_id = str(member["instrument_id"])
    member_records: list[tuple[str, SequenceRecord]] = []
    member_hashes: list[str] = []
    member_actions = (
        corporate_actions
        if corporate_actions.empty
        else corporate_actions.loc[corporate_actions["instrument_id"].astype(str) == instrument_id]
    )
    session_token_cache = _token_cache(sessions, quality_protocol=quality_protocol)
    history: list[tuple[date, pd.DataFrame]] = []
    for session_date, session in sessions:
        partition = resolve_fold_partition(fold_id, session_date)
        prior = history[-20:]
        spy_index = bisect_left(spy_dates, session_date)
        spy_prior = spy_sessions[max(0, spy_index - 20) : spy_index]
        prior_spy_session = spy_by_date.get(session_date)
        if not prior or not spy_prior or prior_spy_session is None:
            member_exclusions.append(
                {
                    "fold_id": fold_id,
                    "instrument_id": instrument_id,
                    "session_date": session_date.isoformat(),
                    "reason": "insufficient causal stock or SPY history",
                }
            )
            history.append((session_date, session))
            continue
        symbol = _session_symbol(session)
        if symbol_history:
            _verify_sourced_symbol(
                symbol_history,
                instrument_id=instrument_id,
                session_date=session_date,
                observed_symbol=symbol,
            )
            _verify_sourced_symbol(
                symbol_history,
                instrument_id=spy_instrument_id,
                session_date=session_date,
                observed_symbol=_session_symbol(prior_spy_session),
            )
        record_cutoff = prior[-1][0]
        market_information_as_of = pd.Timestamp(session["timestamp"].iloc[0])
        if member_actions.empty:
            adjusted = session
            adjusted_previous = prior[-1][1]
        else:
            adjusted = _adjust_for_market_information(
                session,
                member_actions,
                instrument_id=instrument_id,
                market_information_as_of=market_information_as_of,
            )
            adjusted_previous = _adjust_for_market_information(
                prior[-1][1],
                member_actions,
                instrument_id=instrument_id,
                market_information_as_of=market_information_as_of,
            )
        previous_close = float(adjusted_previous["close"].iloc[-1])
        stock_seasonal = _seasonal_frame_from_token_cache(
            prior,
            session_token_cache,
            corporate_actions=member_actions if not member_actions.empty else None,
            instrument_id=instrument_id if not member_actions.empty else None,
            market_information_as_of=(
                market_information_as_of if not member_actions.empty else None
            ),
        )
        spy_seasonal = _seasonal_frame_from_token_cache(spy_prior, spy_token_cache)
        source_hash = _frame_hash(session)
        record = build_session_sequence(
            adjusted,
            instrument_id=instrument_id,
            symbol=symbol,
            source_sha256=source_hash,
            cutoff=record_cutoff.isoformat(),
            seasonal=stock_seasonal,
            spy_bars=prior_spy_session,
            spy_seasonal=spy_seasonal,
            previous_close=previous_close,
            data_classification=data_classification,
            training_cutoff=cutoff.isoformat(),
            quality_protocol=quality_protocol,
            precomputed_tokens=(
                session_token_cache[session_date] if member_actions.empty else None
            ),
            precomputed_spy_tokens=spy_token_cache[session_date],
        )
        member_records.append((partition, record))
        member_hashes.append(source_hash)
        history.append((session_date, session))
    return member_records, member_exclusions, member_hashes


def load_corpus_instrument(root: Path, instrument_id: str) -> pd.DataFrame:
    """Load only one instrument's bounded Parquet partitions from a large corpus."""
    files = sorted(
        {
            *root.glob(f"{instrument_id}-*.response"),
            *root.glob(f"{instrument_id}-*.parquet"),
            *root.glob(f"**/instrument_id={instrument_id}/**/*.parquet"),
        }
    )
    if not files:
        raise FileNotFoundError(f"No corpus partitions for instrument {instrument_id}.")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    observed = set(frame["instrument_id"].astype(str).unique())
    if observed != {instrument_id}:
        raise ValueError(
            f"Corpus partitions for {instrument_id} contain incompatible identities: {observed}"
        )
    return frame


def _validated_sessions(
    bars: pd.DataFrame,
    instrument_id: str,
    *,
    fold_id: str = "unresolved",
    exclusions: list[dict[str, str]] | None = None,
    quality_protocol: str = "exact-minute-v1",
) -> list[tuple[date, pd.DataFrame]]:
    selected = bars.loc[bars["instrument_id"].astype(str) == instrument_id].copy()
    if selected.empty:
        return []
    timestamps = pd.to_datetime(selected["timestamp"])
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    sessions: list[tuple[date, pd.DataFrame]] = []
    for session_date, session in selected.groupby(local_dates, sort=True):
        session = session.sort_values("timestamp", kind="stable").reset_index(drop=True)
        errors: tuple[str, ...]
        if quality_protocol == "resolution-aware-v2":
            quality, tokens = assess_session_resolution_quality(session, return_tokens=True)
            errors = () if quality.token_valid_full_session else (quality.invalid_token_reason,)
        elif quality_protocol == "exact-minute-v1":
            errors = validate_exact_xnys_session(session)
            tokens = None if errors else _aggregate_tokens(session)
        else:
            raise ValueError(f"Unknown paper quality protocol: {quality_protocol}")
        if errors:
            if exclusions is not None:
                exclusions.append(
                    {
                        "fold_id": fold_id,
                        "instrument_id": instrument_id,
                        "session_date": session_date.isoformat(),
                        "reason": "; ".join(errors),
                    }
                )
            continue
        if tokens is None:
            raise RuntimeError("Validated sequence session did not produce token data.")
        session.attrs[_TOKEN_CACHE_ATTR] = tokens
        sessions.append((session_date, session))
    return sessions


def _seasonal_frame(
    history: list[tuple[date, pd.DataFrame]], *, quality_protocol: str
) -> pd.DataFrame:
    return _seasonal_frame_from_token_cache(
        history, _token_cache(history, quality_protocol=quality_protocol)
    )


def _belongs_to_fold(fold_id: str, session_date: date) -> bool:
    try:
        resolve_fold_partition(fold_id, session_date)
    except ValueError:
        return False
    return True


def _token_cache(
    sessions: list[tuple[date, pd.DataFrame]], *, quality_protocol: str
) -> dict[date, pd.DataFrame]:
    """Aggregate each causal session once instead of once per future case."""
    return {
        session_date: (
            session.attrs[_TOKEN_CACHE_ATTR]
            if _TOKEN_CACHE_ATTR in session.attrs
            else aggregate_observed_tokens(session)
            if quality_protocol == "resolution-aware-v2"
            else _aggregate_tokens(session)
        )
        for session_date, session in sessions
    }


def _seasonal_frame_from_token_cache(
    history: list[tuple[date, pd.DataFrame]],
    token_cache: dict[date, pd.DataFrame],
    *,
    corporate_actions: pd.DataFrame | None = None,
    instrument_id: str | None = None,
    market_information_as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build causal seasonal rows from cached tokens with point-in-time split scaling."""
    if corporate_actions is not None and (
        instrument_id is None or market_information_as_of is None
    ):
        raise ValueError(
            "Corporate-action seasonal scaling requires identity and information time."
        )
    selected = history[-20:]
    if not selected:
        return pd.DataFrame(
            columns=("session_date", "bucket_index", "volume", "dollar_volume", "trade_count")
        )
    session_dates: list[date] = []
    volumes: list[np.ndarray] = []
    dollar_volumes: list[np.ndarray] = []
    trade_counts: list[np.ndarray] = []
    for session_date, session in selected:
        tokens = token_cache[session_date]
        factor = 1.0
        if corporate_actions is not None and not corporate_actions.empty:
            factor = point_in_time_split_factor(
                corporate_actions,
                instrument_id=str(instrument_id),
                observation_at=pd.Timestamp(session["timestamp"].iloc[-1]),
                market_information_as_of=market_information_as_of,
            )
        volume = pd.to_numeric(tokens["volume"]).to_numpy(dtype=float)
        vwap = pd.to_numeric(tokens["vwap"]).to_numpy(dtype=float)
        if len(volume) != 26:
            raise ValueError("Seasonal token cache must contain all 26 buckets.")
        session_dates.append(session_date)
        volumes.append(volume * factor)
        dollar_volumes.append(vwap * volume)
        trade_counts.append(pd.to_numeric(tokens["trade_count"]).to_numpy(dtype=float))
    count = len(session_dates)
    return pd.DataFrame(
        {
            "session_date": np.repeat(np.asarray(session_dates, dtype=object), 26),
            "bucket_index": np.tile(np.arange(26, dtype=np.int16), count),
            "volume": np.concatenate(volumes),
            "dollar_volume": np.concatenate(dollar_volumes),
            "trade_count": np.concatenate(trade_counts),
        }
    )


def _session_symbol(session: pd.DataFrame) -> str:
    symbols = tuple(session["symbol"].astype(str).str.upper().drop_duplicates())
    if len(symbols) != 1 or not symbols[0]:
        raise ValueError("Each sequence session must contain one non-empty observed symbol.")
    return symbols[0]


def _verify_sourced_symbol(
    symbol_history: tuple[dict[str, Any], ...],
    *,
    instrument_id: str,
    session_date: date,
    observed_symbol: str,
) -> None:
    matches = [
        str(item["symbol"]).upper()
        for item in symbol_history
        if str(item["instrument_id"]) == instrument_id
        and date.fromisoformat(str(item["start"])[:10])
        <= session_date
        <= date.fromisoformat(str(item["end"])[:10])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"BLOCKED: no unique sourced symbol for {instrument_id} on {session_date}."
        )
    if observed_symbol != matches[0]:
        raise ValueError(
            f"Observed symbol {observed_symbol} does not match sourced symbol {matches[0]} "
            f"for {instrument_id} on {session_date}."
        )


def _adjust_for_market_information(
    session: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    instrument_id: str,
    market_information_as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Restate a session using only actions known by the current case."""
    if actions.empty:
        return session.copy()
    from execsim.data.paper.corporate_actions import point_in_time_split_factor

    observation_at = pd.Timestamp(session["timestamp"].iloc[-1])
    factor = point_in_time_split_factor(
        actions,
        instrument_id=instrument_id,
        observation_at=observation_at,
        market_information_as_of=market_information_as_of,
    )
    return apply_point_in_time_split_adjustment(
        session, pd.Series(np.full(len(session), factor), index=session.index)
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    import hashlib

    values = frame.sort_values("timestamp", kind="stable").to_json(
        orient="records", date_format="iso", double_precision=15
    )
    return hashlib.sha256(values.encode()).hexdigest()
