"""Data ingestion: nflverse (via ``nflreadpy``) and The Odds API.

nflverse
--------
Play-by-play, schedules, weekly rosters and injury reports from
``settings.start_season`` (2016) through the current season are mirrored into
``data/raw/<dataset>/`` as Parquet, one file per season (schedules ship as one
all-seasons file). ``data/raw/_manifest.json`` records when each file was fetched.

Cache policy:

* A season is *final* once its Super Bowl has a result and
  ``season_final_grace_days`` have passed (late stat corrections land in that
  window). Final seasons are downloaded exactly once. A copy taken before the
  season became final (e.g. cached mid-season) is replaced once, then frozen.
* The in-progress season goes stale at every weekly refresh boundary (Tuesday
  12:00 UTC by default, once Monday Night Football is processed upstream) or
  when its copy is older than the dataset's ``max_age_hours``. Injuries and
  schedules use a 12-hour TTL because they change during the week.
* Writes are atomic (temp file + ``os.replace``), so a crashed download never
  leaves a truncated Parquet file behind for the app to read.

The Odds API
------------
:func:`fetch_market_win_probs` pulls h2h (moneyline) and spread prices for the
current NFL week, removes the vig per book with the multiplicative method,
aggregates books into a consensus and returns ``market_home_wp`` /
``market_away_wp`` keyed by nflverse ``game_id``.

CLI::

    python -m src.ingest                         # sync all nflverse datasets
    python -m src.ingest --datasets pbp --force  # re-download play-by-play
    python -m src.ingest --odds                  # sync, then pull market odds
    python -m src.ingest --skip-nflverse --odds  # odds only (schedules still refresh)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import nflreadpy as nfl
import polars as pl
import requests
from nflreadpy.config import CacheMode, update_config

from config import (
    DATASETS,
    ODDS_TEAM_TO_NFLVERSE,
    ConsensusMethod,
    DatasetSpec,
    OddsFormat,
    Settings,
    settings,
    setup_logging,
)

logger = logging.getLogger("src.ingest" if __name__ == "__main__" else __name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class IngestError(RuntimeError):
    """Base class for ingestion failures."""


class NflverseDownloadError(IngestError):
    """An nflverse file could not be downloaded."""

    def __init__(self, message: str, *, not_found: bool = False) -> None:
        super().__init__(message)
        self.not_found = not_found


class OddsAPIError(IngestError):
    """The Odds API returned an error or an unusable payload."""

    def __init__(self, message: str, *, status_code: int | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class MissingAPIKeyError(OddsAPIError):
    """No API key was supplied or configured."""


class OddsAPIAuthError(OddsAPIError):
    """The API key was rejected."""


class OddsAPIQuotaError(OddsAPIError):
    """The account's usage credits are exhausted."""


class OddsAPIRateLimitError(OddsAPIError):
    """Requests were throttled (HTTP 429) and retries were exhausted."""


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_z(ts: datetime) -> str:
    """Format as the ``YYYY-MM-DDTHH:MM:SSZ`` form The Odds API expects."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.write_parquet(tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def add_kickoff_utc(schedules: pl.DataFrame) -> pl.DataFrame:
    """Add ``kickoff_utc`` from nflverse ``gameday`` + ``gametime`` (US/Eastern)."""
    return schedules.with_columns(
        pl.concat_str([pl.col("gameday"), pl.col("gametime").fill_null("13:00")], separator=" ")
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone("America/New_York", ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .alias("kickoff_utc")
    )


# --------------------------------------------------------------------------- #
# Cache manifest
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CacheEntry:
    dataset: str
    season: int | None  # None for single-file datasets (schedules)
    path: str  # relative to settings.raw_dir
    fetched_at: datetime
    rows: int
    max_week: int | None
    final: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "season": self.season,
            "path": self.path,
            "fetched_at": self.fetched_at.isoformat(),
            "rows": self.rows,
            "max_week": self.max_week,
            "final": self.final,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CacheEntry:
        fetched_at = datetime.fromisoformat(raw["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        return cls(
            dataset=str(raw["dataset"]),
            season=None if raw.get("season") is None else int(raw["season"]),
            path=str(raw["path"]),
            fetched_at=fetched_at,
            rows=int(raw.get("rows", 0)),
            max_week=None if raw.get("max_week") is None else int(raw["max_week"]),
            final=bool(raw.get("final", False)),
        )


class CacheManifest:
    """JSON index of every file mirrored into ``data/raw``."""

    VERSION: Final[int] = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    @staticmethod
    def key(dataset: str, season: int | None) -> str:
        return dataset if season is None else f"{dataset}/{season}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {key: CacheEntry.from_dict(raw) for key, raw in payload.get("entries", {}).items()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Cache manifest %s is unreadable (%s); existing files will be re-adopted.", self.path, exc)
            self._entries = {}

    def get(self, dataset: str, season: int | None) -> CacheEntry | None:
        return self._entries.get(self.key(dataset, season))

    def put(self, entry: CacheEntry) -> None:
        self._entries[self.key(entry.dataset, entry.season)] = entry
        self.save()

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def save(self) -> None:
        payload = {
            "version": self.VERSION,
            "updated_at": utc_now().isoformat(),
            "entries": {key: entry.to_dict() for key, entry in sorted(self._entries.items())},
        }
        write_text_atomic(json.dumps(payload, indent=2), self.path)


# --------------------------------------------------------------------------- #
# Freshness policy
# --------------------------------------------------------------------------- #
def last_weekly_refresh(now: datetime, cfg: Settings = settings) -> datetime:
    """Most recent weekly refresh boundary at or before ``now``."""
    anchor = now.astimezone(UTC).replace(hour=cfg.weekly_refresh_hour_utc, minute=0, second=0, microsecond=0)
    anchor -= timedelta(days=(anchor.weekday() - cfg.weekly_refresh_weekday) % 7)
    if anchor > now:
        anchor -= timedelta(days=7)
    return anchor


def season_final_after(
    schedules: pl.DataFrame | None, season: int, current_season: int, cfg: Settings = settings
) -> datetime | None:
    """UTC instant after which a season's data is frozen, or ``None`` while it is in progress.

    The current season is final only once its Super Bowl has a result; checking
    "every game has a result" is not enough because playoff games are added to
    the schedule as the bracket is set. Past seasons are always over (a
    cancelled game such as 2022 BUF-CIN keeps a null result forever).
    """
    grace = timedelta(days=cfg.season_final_grace_days + 1)  # +1: gameday is a date, kickoff is late
    games = None
    if schedules is not None and schedules.height:
        games = schedules.filter(pl.col("season") == season)

    if games is not None and games.height:
        if season < current_season:
            last_day = games["gameday"].max()
        else:
            super_bowl = games.filter((pl.col("game_type") == "SB") & pl.col("result").is_not_null())
            if super_bowl.is_empty():
                return None
            last_day = super_bowl["gameday"].max()
        return datetime.fromisoformat(str(last_day)).replace(tzinfo=UTC) + grace

    if season < current_season:
        # No schedule available: the Super Bowl is always played by mid-February.
        return datetime(season + 1, 3, 1, tzinfo=UTC)
    return None


def staleness_reason(
    entry: CacheEntry | None,
    file_exists: bool,
    spec: DatasetSpec,
    final_after: datetime | None,
    now: datetime,
    cfg: Settings = settings,
) -> str | None:
    """Why a cached file must be re-downloaded, or ``None`` if it is fresh."""
    if entry is None or not file_exists:
        return "not cached"
    if final_after is not None:
        if entry.fetched_at >= final_after:
            return None  # frozen season
        if now >= final_after:
            return "cached before the season was final"
        # Season over but still inside the stat-correction grace window: treat as in progress.
    if entry.fetched_at < last_weekly_refresh(now, cfg):
        return "weekly refresh boundary passed"
    age_hours = (now - entry.fetched_at).total_seconds() / 3600
    if age_hours > spec.max_age_hours:
        return f"older than {spec.max_age_hours:g}h"
    return None


# --------------------------------------------------------------------------- #
# nflverse downloads
# --------------------------------------------------------------------------- #
def _is_http_not_found(exc: BaseException) -> bool:
    # nflreadpy re-raises requests errors as ConnectionError(...) from the original.
    response = getattr(exc.__cause__, "response", None)
    return response is not None and getattr(response, "status_code", None) == 404


def _download_nflverse(spec: DatasetSpec, season: int | None, cfg: Settings) -> pl.DataFrame:
    loader = getattr(nfl, spec.loader)
    label = spec.name if season is None else f"{spec.name} {season}"
    last_exc: Exception | None = None
    for attempt in range(1, cfg.nflverse_max_retries + 1):
        started = time.perf_counter()
        try:
            df: pl.DataFrame = loader(True if season is None else season)
        # nflreadpy raises ConnectionError (an OSError) for HTTP failures and ValueError for parse failures.
        except (OSError, ValueError) as exc:
            if _is_http_not_found(exc):
                raise NflverseDownloadError(f"{label} is not published by nflverse (HTTP 404)", not_found=True) from exc
            last_exc = exc
            if attempt < cfg.nflverse_max_retries:
                delay = cfg.nflverse_backoff_s * 2 ** (attempt - 1)
                logger.warning(
                    "Download of %s failed (attempt %d/%d): %s. Retrying in %.0fs.",
                    label,
                    attempt,
                    cfg.nflverse_max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            continue
        logger.info(
            "Downloaded %s: %s rows x %d cols in %.1fs",
            label,
            f"{df.height:,}",
            df.width,
            time.perf_counter() - started,
        )
        return df
    raise NflverseDownloadError(
        f"Failed to download {label} after {cfg.nflverse_max_retries} attempts: {last_exc}"
    ) from last_exc


def _max_week(df: pl.DataFrame) -> int | None:
    if "week" not in df.columns or df.is_empty():
        return None
    value = df["week"].max()
    return None if value is None else int(value)


class NflverseCache:
    """Incremental Parquet mirror of nflverse data under ``data/raw``."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = CacheManifest(cfg.manifest_path)
        self._schedules: pl.DataFrame | None = None
        self._schedules_unavailable = False
        # This class is the cache; nflreadpy's own in-memory copy would only double RAM.
        update_config(cache_mode=CacheMode.OFF, timeout=cfg.nflverse_timeout_s, verbose=False)

    # -- paths and manifest ------------------------------------------------- #
    @staticmethod
    def _spec(name: str) -> DatasetSpec:
        try:
            return DATASETS[name]
        except KeyError:
            raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(DATASETS)}") from None

    def path_for(self, name: str, season: int | None = None) -> Path:
        spec = self._spec(name)
        folder = self.cfg.dataset_dir(spec.name)
        if not spec.per_season:
            return folder / f"{spec.name}.parquet"
        if season is None:
            raise ValueError(f"Dataset {name!r} is stored per season; pass a season")
        return folder / f"{spec.name}_{season}.parquet"

    def _entry(self, spec: DatasetSpec, season: int | None, path: Path) -> CacheEntry | None:
        entry = self.manifest.get(spec.name, season)
        if entry is not None or not path.exists():
            return entry
        # A Parquet file the manifest doesn't know about (manifest deleted or corrupt):
        # adopt it, dated by its mtime, instead of re-downloading it blindly.
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        entry = CacheEntry(
            dataset=spec.name,
            season=season,
            path=str(path.relative_to(self.cfg.raw_dir)),
            fetched_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
            rows=int(rows),
            max_week=None,
            final=False,
        )
        logger.info("Adopted untracked cache file %s (mtime %s)", path, entry.fetched_at.isoformat())
        self.manifest.put(entry)
        return entry

    def _store(
        self, spec: DatasetSpec, season: int | None, df: pl.DataFrame, *, fetched_at: datetime, final: bool
    ) -> CacheEntry:
        path = self.path_for(spec.name, season)
        write_parquet_atomic(df, path)
        entry = CacheEntry(
            dataset=spec.name,
            season=season,
            path=str(path.relative_to(self.cfg.raw_dir)),
            fetched_at=fetched_at,
            rows=df.height,
            max_week=_max_week(df) if spec.per_season else None,
            final=final,
        )
        self.manifest.put(entry)
        logger.info("Cached %s (%s rows%s)", entry.path, f"{entry.rows:,}", ", final" if final else "")
        return entry

    def _resolve_seasons(self, spec: DatasetSpec, seasons: Iterable[int] | None, current: int) -> list[int]:
        first = max(self.cfg.start_season, spec.first_season)
        if seasons is None:
            return list(range(first, current + 1))
        requested = sorted(set(int(s) for s in seasons))
        invalid = [s for s in requested if not spec.first_season <= s <= current]
        if invalid:
            raise ValueError(f"{spec.name} is available for {spec.first_season}-{current}; invalid seasons: {invalid}")
        return requested

    # -- sync ---------------------------------------------------------------- #
    def sync_schedules(self, *, force: bool = False) -> pl.DataFrame:
        """Refresh the all-seasons schedule file if stale and return it (``season >= start_season``)."""
        spec = self._spec("schedules")
        path = self.path_for("schedules")
        entry = self._entry(spec, None, path)
        reason = "forced" if force else staleness_reason(entry, path.exists(), spec, None, utc_now(), self.cfg)
        if reason:
            logger.info("Refreshing schedules: %s", reason)
            try:
                df = _download_nflverse(spec, None, self.cfg)
                df = df.filter(pl.col("season") >= self.cfg.start_season)
                self._store(spec, None, df, fetched_at=utc_now(), final=False)
            except NflverseDownloadError as exc:
                if not path.exists():
                    self._schedules_unavailable = True
                    raise
                logger.warning(
                    "Schedule refresh failed; using cached copy from %s: %s", entry.fetched_at if entry else "?", exc
                )
        self._schedules = pl.read_parquet(path)
        self._schedules_unavailable = False
        return self._schedules

    def sync(self, name: str, seasons: Iterable[int] | None = None, *, force: bool = False) -> list[CacheEntry]:
        """Download whichever seasons of ``name`` are missing or stale. Returns the cache entries."""
        spec = self._spec(name)
        if not spec.per_season:
            self.sync_schedules(force=force)
            entry = self.manifest.get(spec.name, None)
            return [entry] if entry else []

        schedules = self._schedules
        if schedules is None and not self._schedules_unavailable:
            try:
                schedules = self.sync_schedules()
            except NflverseDownloadError as exc:
                logger.warning("Schedules unavailable (%s); season finality falls back to the calendar.", exc)

        current = nfl.get_current_season()
        entries: list[CacheEntry] = []
        for season in self._resolve_seasons(spec, seasons, current):
            path = self.path_for(spec.name, season)
            entry = self._entry(spec, season, path)
            final_after = season_final_after(schedules, season, current, self.cfg)
            reason = (
                "forced" if force else staleness_reason(entry, path.exists(), spec, final_after, utc_now(), self.cfg)
            )
            if reason is None:
                logger.debug(
                    "%s %d is fresh (fetched %s)", spec.name, season, entry.fetched_at.isoformat() if entry else "?"
                )
                if entry is not None:
                    entries.append(entry)
                continue

            logger.info("Refreshing %s %d: %s", spec.name, season, reason)
            try:
                df = _download_nflverse(spec, season, self.cfg)
            except NflverseDownloadError as exc:
                if entry is not None and path.exists():
                    logger.warning(
                        "Keeping stale %s %d from %s: %s", spec.name, season, entry.fetched_at.isoformat(), exc
                    )
                    entries.append(entry)
                    continue
                if season == current:
                    logger.warning("Skipping %s %d: %s", spec.name, season, exc)
                    continue
                raise
            fetched_at = utc_now()
            final = final_after is not None and fetched_at >= final_after
            entries.append(self._store(spec, season, df, fetched_at=fetched_at, final=final))
        return entries

    def sync_all(
        self,
        datasets: Iterable[str] | None = None,
        seasons: Iterable[int] | None = None,
        *,
        force: bool = False,
    ) -> dict[str, list[CacheEntry]]:
        """Sync schedules first (they drive season finality), then every requested dataset."""
        names = list(datasets) if datasets is not None else list(DATASETS)
        season_list = list(seasons) if seasons is not None else None
        summary: dict[str, list[CacheEntry]] = {}
        if "schedules" in names or self._schedules is None:
            try:
                self.sync_schedules(force=force and "schedules" in names)
            except NflverseDownloadError as exc:
                logger.warning("Schedules unavailable (%s); season finality falls back to the calendar.", exc)
            entry = self.manifest.get("schedules", None)
            summary["schedules"] = [entry] if entry else []
        for name in names:
            if name != "schedules":
                summary[name] = self.sync(name, season_list, force=force)
        return summary

    # -- read ---------------------------------------------------------------- #
    def read(
        self,
        name: str,
        seasons: Iterable[int] | None = None,
        *,
        columns: Sequence[str] | None = None,
        refresh: bool = True,
    ) -> pl.DataFrame:
        """Load cached data, syncing stale seasons first unless ``refresh=False``.

        ``columns`` is applied per file before concatenation (essential for the
        ~370-column play-by-play); columns a season doesn't have come back null.
        Season files are concatenated with ``diagonal_relaxed`` so schema drift
        between seasons (new columns, widened dtypes) doesn't break the read.
        """
        spec = self._spec(name)
        season_list = list(seasons) if seasons is not None else None
        if refresh:
            self.sync(name, season_list)

        if not spec.per_season:
            path = self.path_for(name)
            if not path.exists():
                raise IngestError(f"No cached {name}; run `python -m src.ingest` first")
            lf = pl.scan_parquet(path)
            if season_list is not None:
                lf = lf.filter(pl.col("season").is_in(season_list))
            if columns:
                available = lf.collect_schema().names()
                lf = lf.select([c for c in columns if c in available])
            return lf.collect()

        current = nfl.get_current_season()
        targets = self._resolve_seasons(spec, season_list, current)
        paths = [p for p in (self.path_for(name, s) for s in targets) if p.exists()]
        if not paths:
            raise IngestError(
                f"No cached {name} for seasons {targets[0]}-{targets[-1]}; run `python -m src.ingest` first"
            )
        missing = len(targets) - len(paths)
        if missing:
            logger.warning("%d of %d requested %s seasons are not cached", missing, len(targets), name)

        frames: list[pl.DataFrame] = []
        for path in paths:
            lf = pl.scan_parquet(path)
            if columns:
                available = set(lf.collect_schema().names())
                lf = lf.select([c for c in columns if c in available])
            frames.append(lf.collect())
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")

    def status(self) -> pl.DataFrame:
        """One row per cached file: what it is, when it was fetched, whether it is frozen."""
        rows = [entry.to_dict() for entry in self.manifest.entries()]
        if not rows:
            return pl.DataFrame(
                schema={
                    "dataset": pl.String,
                    "season": pl.Int64,
                    "rows": pl.Int64,
                    "max_week": pl.Int64,
                    "final": pl.Boolean,
                    "fetched_at": pl.String,
                    "path": pl.String,
                }
            )
        return (
            pl.DataFrame(rows)
            .select("dataset", "season", "rows", "max_week", "final", "fetched_at", "path")
            .sort("dataset", "season", nulls_last=True)
        )


_default_cache: NflverseCache | None = None


def get_cache() -> NflverseCache:
    """Process-wide :class:`NflverseCache` built from the default settings."""
    global _default_cache
    if _default_cache is None:
        _default_cache = NflverseCache(settings)
    return _default_cache


def get_pbp(
    seasons: Iterable[int] | None = None, columns: Sequence[str] | None = None, *, refresh: bool = True
) -> pl.DataFrame:
    return get_cache().read("pbp", seasons, columns=columns, refresh=refresh)


def get_schedules(seasons: Iterable[int] | None = None, *, refresh: bool = True) -> pl.DataFrame:
    return get_cache().read("schedules", seasons, refresh=refresh)


def get_weekly_rosters(
    seasons: Iterable[int] | None = None, columns: Sequence[str] | None = None, *, refresh: bool = True
) -> pl.DataFrame:
    return get_cache().read("rosters_weekly", seasons, columns=columns, refresh=refresh)


def get_injuries(
    seasons: Iterable[int] | None = None, columns: Sequence[str] | None = None, *, refresh: bool = True
) -> pl.DataFrame:
    return get_cache().read("injuries", seasons, columns=columns, refresh=refresh)


def current_week_games(schedules: pl.DataFrame, now: datetime | None = None, season: int | None = None) -> pl.DataFrame:
    """All games of the current NFL week, with ``kickoff_utc``.

    The current week is the earliest week that still has a game yet to kick off,
    so on Monday night (MNF in progress) it is already next week. Empty when the
    season is over.
    """
    now = now or utc_now()
    season = season if season is not None else nfl.get_current_season()
    games = add_kickoff_utc(schedules.filter(pl.col("season") == season))
    upcoming = games.filter(pl.col("result").is_null() & (pl.col("kickoff_utc") > now))
    if upcoming.is_empty():
        return games.clear()
    week = upcoming["week"].min()
    return games.filter(pl.col("week") == week).sort("kickoff_utc")


# --------------------------------------------------------------------------- #
# Odds math
# --------------------------------------------------------------------------- #
_STANDARD_NORMAL: Final = statistics.NormalDist()


def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    if odds <= -100:
        return 1.0 + 100.0 / -odds
    raise ValueError(f"Invalid American odds {odds!r}: must be <= -100 or >= +100")


def implied_probability(price: float, odds_format: OddsFormat = "american") -> float:
    """Raw (vig-inclusive) implied probability of a quoted price."""
    decimal = american_to_decimal(price) if odds_format == "american" else float(price)
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise ValueError(f"Invalid decimal odds {decimal!r}: must be > 1.0")
    return 1.0 / decimal


def devig_multiplicative(implied: Sequence[float]) -> list[float]:
    """Remove the vig by scaling every outcome's implied probability by the overround.

    ``p_fair_i = p_i / sum_j p_j``. Preserves each book's relative pricing and
    always returns probabilities that sum to 1.
    """
    if len(implied) < 2:
        raise ValueError("Need at least two outcomes to remove the vig")
    if any(not math.isfinite(p) or p <= 0.0 for p in implied):
        raise ValueError(f"Implied probabilities must be finite and positive, got {list(implied)}")
    total = math.fsum(implied)
    return [p / total for p in implied]


def spread_implied_win_prob(home_spread: float, home_cover_prob: float, sigma: float) -> float:
    """P(home wins) implied by a spread and its de-vigged cover probability.

    Models the final home margin as ``M ~ Normal(mu, sigma)``. Home covers when
    ``M + home_spread > 0``, so ``P(cover) = Phi((home_spread + mu) / sigma)`` and
    ``mu = sigma * Phi^-1(P(cover)) - home_spread``. Then ``P(win) = Phi(mu / sigma)``.
    A fallback only: the real margin distribution is lumpy around 3 and 7.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    cover = min(max(home_cover_prob, 1e-6), 1 - 1e-6)
    mu = sigma * _STANDARD_NORMAL.inv_cdf(cover) - home_spread
    return _STANDARD_NORMAL.cdf(mu / sigma)


# --------------------------------------------------------------------------- #
# The Odds API client
# --------------------------------------------------------------------------- #
_API_KEY_PATTERN: Final = re.compile(r"(apiKey=)[^&\s'\"]+", re.IGNORECASE)
_RETRYABLE_STATUS: Final = frozenset({429, 500, 502, 503, 504})


def _redact(text: str, secret: str | None = None) -> str:
    text = _API_KEY_PATTERN.sub(r"\1***", text)
    return text.replace(secret, "***") if secret else text


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


class OddsAPIClient:
    """Minimal, retrying client for The Odds API v4."""

    def __init__(
        self, api_key: str | None = None, *, cfg: Settings = settings, session: requests.Session | None = None
    ) -> None:
        key = api_key if api_key is not None else cfg.odds_api_key
        if not key or not key.strip():
            raise MissingAPIKeyError(
                "The Odds API key is missing. Set ODDS_API_KEY in your environment or in .env "
                "(see .env.example). Keys are issued at https://the-odds-api.com/."
            )
        self._api_key = key.strip()
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.requests_remaining: int | None = None
        self.requests_used: int | None = None
        self.requests_last: int | None = None

    def get_odds(
        self,
        *,
        markets: Sequence[str] | None = None,
        bookmakers: Sequence[str] | None = None,
        regions: str | None = None,
        commence_time_from: datetime | None = None,
        commence_time_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """``GET /sports/{sport}/odds``. Bookmakers take priority over regions when both apply."""
        params: dict[str, str] = {
            "markets": ",".join(markets or self.cfg.odds_markets),
            "oddsFormat": self.cfg.odds_format,
            "dateFormat": "iso",
        }
        books = tuple(bookmakers) if bookmakers is not None else self.cfg.odds_bookmakers
        if books:
            if len(books) > 10:
                logger.warning("%d bookmakers requested: every 10 books cost one extra region of credits", len(books))
            params["bookmakers"] = ",".join(books)
        else:
            params["regions"] = regions or self.cfg.odds_regions
        if commence_time_from is not None:
            params["commenceTimeFrom"] = _iso_z(commence_time_from)
        if commence_time_to is not None:
            params["commenceTimeTo"] = _iso_z(commence_time_to)

        payload = self._request(f"/sports/{self.cfg.odds_sport_key}/odds", params)
        if not isinstance(payload, list):
            raise OddsAPIError(f"Expected a list of events, got {type(payload).__name__}")
        return payload

    def _request(self, path: str, params: Mapping[str, str]) -> Any:
        url = f"{self.cfg.odds_api_base_url}{path}"
        query = {**params, "apiKey": self._api_key}
        attempts = self.cfg.odds_max_retries + 1
        logger.debug("GET %s %s", url, dict(params))

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(url, params=query, timeout=self.cfg.odds_timeout_s)
            except (requests.Timeout, requests.ConnectionError) as exc:
                detail = _redact(str(exc), self._api_key)
                if attempt == attempts:
                    # `from None`: the original exception's message contains the request URL and key.
                    raise OddsAPIError(
                        f"Network error calling The Odds API after {attempts} attempts: {detail}"
                    ) from None
                delay = self._backoff(attempt, None)
                logger.warning(
                    "Network error calling The Odds API (%s); retrying in %.1fs (%d/%d)",
                    detail,
                    delay,
                    attempt,
                    attempts,
                )
                time.sleep(delay)
                continue
            except requests.RequestException as exc:
                raise OddsAPIError(f"Request to The Odds API failed: {_redact(str(exc), self._api_key)}") from None

            self._record_quota(response)
            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError:
                    raise OddsAPIError("The Odds API returned a non-JSON body", status_code=status) from None

            message, error_code = self._error_details(response)
            if self._is_quota_error(message, error_code):
                raise OddsAPIQuotaError(
                    f"The Odds API usage credits are exhausted ({message}). "
                    "Wait for the monthly reset or upgrade the plan.",
                    status_code=status,
                    error_code=error_code,
                )
            if status in _RETRYABLE_STATUS and attempt < attempts:
                delay = self._backoff(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "The Odds API returned HTTP %d (%s); retrying in %.1fs (%d/%d)",
                    status,
                    message,
                    delay,
                    attempt,
                    attempts,
                )
                time.sleep(delay)
                continue
            if status == 429:
                raise OddsAPIRateLimitError(
                    f"Rate limited by The Odds API after {attempts} attempts: {message}",
                    status_code=status,
                    error_code=error_code,
                )
            if status in (401, 403):
                raise OddsAPIAuthError(
                    f"The Odds API rejected the API key (HTTP {status}): {message}",
                    status_code=status,
                    error_code=error_code,
                )
            raise OddsAPIError(
                f"The Odds API request failed (HTTP {status}): {message}", status_code=status, error_code=error_code
            )

        raise AssertionError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                pass
        base = self.cfg.odds_backoff_s * 2 ** (attempt - 1)
        return base + random.uniform(0, base / 4)

    def _record_quota(self, response: requests.Response) -> None:
        remaining = _int_header(response.headers, "x-requests-remaining")
        used = _int_header(response.headers, "x-requests-used")
        last = _int_header(response.headers, "x-requests-last")
        if remaining is None and used is None:
            return
        self.requests_remaining, self.requests_used, self.requests_last = remaining, used, last
        logger.info("The Odds API credits: last call=%s, used=%s, remaining=%s", last, used, remaining)
        if remaining is not None and remaining < self.cfg.odds_quota_warn_threshold:
            logger.warning("The Odds API credits are running low: %d remaining", remaining)

    def _error_details(self, response: requests.Response) -> tuple[str, str | None]:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            message = str(body.get("message") or body.get("error") or body)
            code = body.get("error_code")
            return _redact(message, self._api_key), str(code) if code else None
        return _redact((response.text or response.reason or "no body")[:300], self._api_key), None

    def _is_quota_error(self, message: str, error_code: str | None) -> bool:
        code = (error_code or "").upper()
        text = message.lower()
        return (
            any(token in code for token in ("QUOTA", "USAGE", "CREDIT"))
            or "quota" in text
            or "usage credits" in text
            or self.requests_remaining == 0
        )


# --------------------------------------------------------------------------- #
# Odds parsing and consensus
# --------------------------------------------------------------------------- #
BOOK_ODDS_SCHEMA: Final[dict[str, Any]] = {
    "event_id": pl.String,
    "commence_time": pl.Datetime("us", "UTC"),
    "home_team_name": pl.String,
    "away_team_name": pl.String,
    "home_team": pl.String,
    "away_team": pl.String,
    "bookmaker": pl.String,
    "market": pl.String,
    "last_update": pl.Datetime("us", "UTC"),
    "home_price": pl.Float64,
    "away_price": pl.Float64,
    "home_point": pl.Float64,
    "home_implied": pl.Float64,
    "away_implied": pl.Float64,
    "overround": pl.Float64,
    "home_fair_prob": pl.Float64,
    "away_fair_prob": pl.Float64,
}

CONSENSUS_SCHEMA: Final[dict[str, Any]] = {
    "game_id": pl.String,
    "season": pl.Int32,
    "week": pl.Int32,
    "event_id": pl.String,
    "commence_time": pl.Datetime("us", "UTC"),
    "home_team": pl.String,
    "away_team": pl.String,
    "market_home_wp": pl.Float64,
    "market_away_wp": pl.Float64,
    "wp_source": pl.String,
    "n_books_h2h": pl.UInt32,
    "h2h_overround": pl.Float64,
    "consensus_home_spread": pl.Float64,
    "market_home_cover_prob": pl.Float64,
    "spread_implied_home_wp": pl.Float64,
    "n_books_spread": pl.UInt32,
    "home_team_name": pl.String,
    "away_team_name": pl.String,
    "fetched_at": pl.Datetime("us", "UTC"),
}


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _devig_market(
    market_key: str,
    outcomes: Sequence[Mapping[str, Any]],
    home_name: str,
    away_name: str,
    odds_format: OddsFormat,
) -> dict[str, float | None] | None:
    by_name = {o.get("name"): o for o in outcomes}
    home, away = by_name.get(home_name), by_name.get(away_name)
    if home is None or away is None:
        return None
    home_point: float | None = None
    if market_key == "spreads":
        if home.get("point") is None:
            return None
        home_point = float(home["point"])
        priced = [home, away]  # a spread is strictly two-way
    else:
        # Include any extra leg (e.g. a Draw) so the overround is complete.
        priced = [home, away, *(o for o in outcomes if o is not home and o is not away)]
    try:
        implied = [implied_probability(float(o["price"]), odds_format) for o in priced]
        fair = devig_multiplicative(implied)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "home_price": float(home["price"]),
        "away_price": float(away["price"]),
        "home_point": home_point,
        "home_implied": implied[0],
        "away_implied": implied[1],
        "overround": math.fsum(implied) - 1.0,
        "home_fair_prob": fair[0],
        "away_fair_prob": fair[1],
    }


def parse_odds_events(
    events: Sequence[Mapping[str, Any]],
    *,
    odds_format: OddsFormat = "american",
    team_map: Mapping[str, str] = ODDS_TEAM_TO_NFLVERSE,
) -> pl.DataFrame:
    """Flatten Odds API events into one de-vigged row per (event, bookmaker, market)."""
    rows: list[dict[str, Any]] = []
    unknown_teams: set[str] = set()
    skipped = 0
    for event in events:
        home_name, away_name = event.get("home_team"), event.get("away_team")
        if not home_name or not away_name:
            skipped += 1
            continue
        for name in (home_name, away_name):
            if name not in team_map:
                unknown_teams.add(name)
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                key = market.get("key")
                if key not in ("h2h", "spreads"):
                    continue
                priced = _devig_market(key, market.get("outcomes") or [], home_name, away_name, odds_format)
                if priced is None:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "event_id": event.get("id"),
                        "commence_time": _parse_ts(event.get("commence_time")),
                        "home_team_name": home_name,
                        "away_team_name": away_name,
                        "home_team": team_map.get(home_name),
                        "away_team": team_map.get(away_name),
                        "bookmaker": book.get("key"),
                        "market": key,
                        "last_update": _parse_ts(market.get("last_update") or book.get("last_update")),
                        **priced,
                    }
                )
    if unknown_teams:
        logger.warning(
            "No nflverse abbreviation for team name(s) %s; add them to ODDS_TEAM_TO_NFLVERSE", sorted(unknown_teams)
        )
    if skipped:
        logger.debug("Skipped %d malformed or incomplete markets", skipped)
    return pl.DataFrame(rows, schema=BOOK_ODDS_SCHEMA)


def build_consensus(
    book_odds: pl.DataFrame,
    *,
    method: ConsensusMethod = "median",
    sigma: float = 13.45,
) -> pl.DataFrame:
    """Aggregate per-book fair probabilities into one row per event.

    ``market_home_wp`` comes from the moneyline consensus, renormalised so home
    and away sum to 1. Games with no moneyline fall back to the spread-implied
    win probability; ``wp_source`` records which one was used.
    """
    keys = ["event_id", "commence_time", "home_team_name", "away_team_name", "home_team", "away_team"]

    def agg(column: str) -> pl.Expr:
        return pl.col(column).median() if method == "median" else pl.col(column).mean()

    h2h = (
        book_odds.filter(pl.col("market") == "h2h")
        .group_by(keys)
        .agg(
            agg("home_fair_prob").alias("_h2h_home"),
            agg("away_fair_prob").alias("_h2h_away"),
            pl.len().alias("n_books_h2h"),
            pl.col("overround").mean().alias("h2h_overround"),
        )
    )

    spreads = book_odds.filter(pl.col("market") == "spreads")
    spreads = (
        spreads.with_columns(
            pl.Series(
                "_spread_wp",
                [
                    spread_implied_win_prob(point, cover, sigma)
                    for point, cover in zip(
                        spreads["home_point"].to_list(), spreads["home_fair_prob"].to_list(), strict=True
                    )
                ],
                dtype=pl.Float64,
            )
        )
        .group_by(keys)
        .agg(
            pl.col("home_point").median().alias("consensus_home_spread"),
            agg("home_fair_prob").alias("market_home_cover_prob"),
            agg("_spread_wp").alias("spread_implied_home_wp"),
            pl.len().alias("n_books_spread"),
        )
    )

    merged = h2h.join(spreads, on=keys, how="full", coalesce=True, nulls_equal=True)
    has_h2h = pl.col("_h2h_home").is_not_null() & pl.col("_h2h_away").is_not_null()
    merged = merged.with_columns(
        pl.when(has_h2h)
        .then(pl.col("_h2h_home") / (pl.col("_h2h_home") + pl.col("_h2h_away")))
        .otherwise(pl.col("spread_implied_home_wp"))
        .alias("market_home_wp"),
        pl.when(has_h2h)
        .then(pl.lit("h2h"))
        .when(pl.col("spread_implied_home_wp").is_not_null())
        .then(pl.lit("spread"))
        .alias("wp_source"),
        pl.col("n_books_h2h").fill_null(0).cast(pl.UInt32),
        pl.col("n_books_spread").fill_null(0).cast(pl.UInt32),
    ).with_columns((1.0 - pl.col("market_home_wp")).alias("market_away_wp"))
    return merged.drop("_h2h_home", "_h2h_away").sort("commence_time", "home_team_name")


def attach_schedule(consensus: pl.DataFrame, week_games: pl.DataFrame) -> pl.DataFrame:
    """Key consensus rows to nflverse ``game_id`` / ``season`` / ``week``.

    Matches on the unordered team pair, so a neutral-site game whose home/away
    designation differs between The Odds API and nflverse is still matched; those
    rows are flipped into nflverse orientation.
    """

    def pair_key(home: str, away: str) -> pl.Expr:
        return (
            pl.when(pl.col(home) < pl.col(away))
            .then(pl.concat_str([pl.col(home), pl.col(away)], separator="|"))
            .otherwise(pl.concat_str([pl.col(away), pl.col(home)], separator="|"))
        )

    if week_games.is_empty():
        return consensus.with_columns(
            pl.lit(None, pl.String).alias("game_id"),
            pl.lit(None, pl.Int32).alias("season"),
            pl.lit(None, pl.Int32).alias("week"),
        )

    schedule = week_games.select(
        pl.col("game_id"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("home_team").alias("_sched_home"),
        pl.col("away_team").alias("_sched_away"),
        pair_key("home_team", "away_team").alias("_pair"),
    )
    joined = consensus.with_columns(pair_key("home_team", "away_team").alias("_pair")).join(
        schedule, on="_pair", how="left"
    )

    flip = pl.col("_sched_home").is_not_null() & (pl.col("_sched_home") != pl.col("home_team"))
    n_flipped = joined.select(flip.sum()).item()
    if n_flipped:
        logger.info("Flipped %d game(s) whose home/away designation differs from nflverse", n_flipped)

    def flipped(column: str, flipped_value: pl.Expr) -> pl.Expr:
        return pl.when(flip).then(flipped_value).otherwise(pl.col(column)).alias(column)

    joined = joined.with_columns(
        flipped("home_team", pl.col("away_team")),
        flipped("away_team", pl.col("home_team")),
        flipped("home_team_name", pl.col("away_team_name")),
        flipped("away_team_name", pl.col("home_team_name")),
        flipped("market_home_wp", pl.col("market_away_wp")),
        flipped("market_away_wp", pl.col("market_home_wp")),
        flipped("consensus_home_spread", -pl.col("consensus_home_spread")),
        flipped("market_home_cover_prob", 1.0 - pl.col("market_home_cover_prob")),
        flipped("spread_implied_home_wp", 1.0 - pl.col("spread_implied_home_wp")),
    )
    unmatched = joined.filter(pl.col("game_id").is_null())
    if unmatched.height:
        logger.warning(
            "%d odds event(s) did not match a current-week nflverse game: %s",
            unmatched.height,
            unmatched.select(pl.concat_str(["away_team_name", "home_team_name"], separator=" @ "))
            .to_series()
            .to_list(),
        )
    return joined.drop("_pair", "_sched_home", "_sched_away")


def _save_odds_snapshot(book_odds: pl.DataFrame, consensus: pl.DataFrame, fetched_at: datetime, cfg: Settings) -> None:
    """Keep every pull: the history of de-vigged lines is what closing-line-value analysis needs."""
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    known = consensus.filter(pl.col("season").is_not_null())
    if known.height:
        season, week = int(known["season"][0]), int(known["week"][0])
        folder, stem = cfg.odds_dir / str(season), f"odds_{season}_wk{week:02d}_{stamp}"
    else:
        folder, stem = cfg.odds_dir / "unmatched", f"odds_upcoming_{stamp}"
    write_parquet_atomic(
        book_odds.with_columns(pl.lit(fetched_at).alias("fetched_at")), folder / f"{stem}_books.parquet"
    )
    write_parquet_atomic(consensus, folder / f"{stem}_consensus.parquet")
    logger.info("Saved odds snapshot %s/%s_*.parquet", folder, stem)


def fetch_market_win_probs(
    api_key: str | None = None,
    *,
    cfg: Settings = settings,
    cache: NflverseCache | None = None,
    client: OddsAPIClient | None = None,
    now: datetime | None = None,
    save_snapshot: bool = True,
) -> pl.DataFrame:
    """Consensus de-vigged market win probabilities for the current NFL week.

    Returns one row per game (see ``CONSENSUS_SCHEMA``) with ``market_home_wp``
    and ``market_away_wp`` keyed by nflverse ``game_id``. Games that have already
    kicked off are excluded so live in-game prices never leak in.

    Raises:
        MissingAPIKeyError: no key passed and ``ODDS_API_KEY`` is unset.
        OddsAPIAuthError, OddsAPIQuotaError, OddsAPIRateLimitError, OddsAPIError.
    """
    client = client or OddsAPIClient(api_key, cfg=cfg)  # fail fast on a missing key
    now = now or utc_now()
    cache = cache or (get_cache() if cfg is settings else NflverseCache(cfg))

    try:
        week_games = current_week_games(cache.sync_schedules(), now)
    except IngestError as exc:
        logger.warning("Schedules unavailable (%s); requesting every upcoming event without a week filter", exc)
        week_games = pl.DataFrame()

    window_end: datetime | None = None
    if week_games.is_empty():
        logger.warning("No upcoming games found in the current-season schedule; requesting every upcoming event")
    else:
        window_end = week_games["kickoff_utc"].max() + timedelta(hours=6)
        logger.info(
            "Fetching odds for %d week %d games (%s to %s)",
            week_games.filter(pl.col("kickoff_utc") > now).height,
            week_games["week"][0],
            _iso_z(now),
            _iso_z(window_end),
        )

    events = client.get_odds(commence_time_from=now, commence_time_to=window_end)
    book_odds = parse_odds_events(events, odds_format=cfg.odds_format)
    if book_odds.is_empty():
        logger.warning("The Odds API returned no usable h2h/spread prices for this window")
        return pl.DataFrame(schema=CONSENSUS_SCHEMA)

    consensus = build_consensus(book_odds, method=cfg.odds_consensus_method, sigma=cfg.spread_margin_sigma)
    consensus = attach_schedule(consensus, week_games).with_columns(
        pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("fetched_at")
    )
    consensus = consensus.select([pl.col(name).cast(dtype) for name, dtype in CONSENSUS_SCHEMA.items()])
    logger.info(
        "Market consensus for %d games from %d bookmakers (%s)",
        consensus.height,
        book_odds["bookmaker"].n_unique(),
        cfg.odds_consensus_method,
    )
    if save_snapshot:
        _save_odds_snapshot(book_odds, consensus, now, cfg)
    return consensus


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest", description="Sync nflverse data and market odds.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), help="datasets to sync (default: all)")
    parser.add_argument(
        "--seasons", nargs="+", type=int, help=f"seasons to sync (default: {settings.start_season}-current)"
    )
    parser.add_argument("--force", action="store_true", help="re-download the selected datasets/seasons")
    parser.add_argument("--odds", action="store_true", help="fetch current-week consensus odds from The Odds API")
    parser.add_argument("--skip-nflverse", action="store_true", help="skip the nflverse sync")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ...")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    settings.ensure_dirs()
    try:
        cache = get_cache()
        if not args.skip_nflverse:
            started = time.perf_counter()
            summary = cache.sync_all(args.datasets, args.seasons, force=args.force)
            for name, entries in summary.items():
                rows = sum(e.rows for e in entries)
                seasons = sorted(e.season for e in entries if e.season is not None)
                span = f"{seasons[0]}-{seasons[-1]}" if seasons else "all seasons"
                logger.info("%-15s %s rows cached (%s)", name, f"{rows:,}", span)
            logger.info("nflverse sync finished in %.1fs", time.perf_counter() - started)
        if args.odds:
            probs = fetch_market_win_probs(cache=cache)
            with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3):
                print(
                    probs.select(
                        "game_id",
                        "away_team",
                        "home_team",
                        "market_away_wp",
                        "market_home_wp",
                        "consensus_home_spread",
                        "n_books_h2h",
                        "wp_source",
                    )
                )
    except MissingAPIKeyError as exc:
        logger.error("%s", exc)
        return 2
    except IngestError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
