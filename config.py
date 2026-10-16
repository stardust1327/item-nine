"""Central configuration for the NFL Game Prediction Engine.

Every tunable lives here. Values can be overridden with environment variables or
a ``.env`` file at the project root (see ``.env.example``). Importing this module
performs no network I/O and creates no directories, so it is safe to import from
pipelines, tests and the Streamlit app alike.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

try:  # python-dotenv is optional; plain environment variables always work.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when python-dotenv is absent
    load_dotenv = None

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

ConsensusMethod = Literal["mean", "median"]
OddsFormat = Literal["american", "decimal"]


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env_str(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_path(name: str, default: Path) -> Path:
    raw = _env_str(name)
    return Path(raw).expanduser().resolve() if raw else default


# --------------------------------------------------------------------------- #
# nflverse datasets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """How one nflverse dataset is fetched and when its in-progress season goes stale.

    Attributes:
        name: Folder under ``data/raw/`` and key in the cache manifest.
        loader: Name of the ``nflreadpy`` function that downloads it.
        first_season: Earliest season nflverse publishes for this dataset.
        per_season: True if nflverse publishes one file per season. False for
            datasets shipped as a single all-seasons file (schedules).
        max_age_hours: Hard TTL for the in-progress season's cached copy. Every
            in-progress dataset additionally goes stale at the weekly refresh
            boundary (see ``Settings.weekly_refresh_weekday``).
    """

    name: str
    loader: str
    first_season: int
    per_season: bool
    max_age_hours: float


DATASETS: Final[dict[str, DatasetSpec]] = {
    # One small file covering every season: results, lines, rest, roof, QBs.
    "schedules": DatasetSpec("schedules", "load_schedules", 1999, per_season=False, max_age_hours=12),
    "pbp": DatasetSpec("pbp", "load_pbp", 1999, per_season=True, max_age_hours=24 * 7),
    "rosters_weekly": DatasetSpec("rosters_weekly", "load_rosters_weekly", 2002, per_season=True, max_age_hours=24 * 7),
    # Injury reports are published Wednesday-Friday for the upcoming game, so a
    # weekly refresh would always be one report behind. Refresh twice a day.
    "injuries": DatasetSpec("injuries", "load_injuries", 2009, per_season=True, max_age_hours=12),
}


# --------------------------------------------------------------------------- #
# The Odds API
# --------------------------------------------------------------------------- #
# Regulated US books. Every group of up to 10 bookmakers costs the same as one
# region, so this list costs len(markets) credits per call (2 for h2h + spreads).
# williamhill_us (Caesars) and fanatics are listed by The Odds API as paid-plan
# books; override ODDS_BOOKMAKERS if your plan excludes them.
DEFAULT_ODDS_BOOKMAKERS: Final[tuple[str, ...]] = (
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",
    "espnbet",  # theScore Bet (formerly ESPN BET)
    "betrivers",
    "fanatics",
    "hardrockbet",
)

# The Odds API full team names -> nflverse team abbreviations.
ODDS_TEAM_TO_NFLVERSE: Final[dict[str, str]] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings. Build with :meth:`Settings.from_env`."""

    # Paths
    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    logs_dir: Path

    # Seasons
    start_season: int

    # nflverse downloads and cache policy
    nflverse_timeout_s: int
    nflverse_max_retries: int
    nflverse_backoff_s: float
    weekly_refresh_weekday: int  # 0=Monday ... 6=Sunday
    weekly_refresh_hour_utc: int
    season_final_grace_days: int

    # The Odds API
    odds_api_key: str | None = field(repr=False)
    odds_api_base_url: str
    odds_sport_key: str
    odds_markets: tuple[str, ...]
    odds_bookmakers: tuple[str, ...]
    odds_regions: str
    odds_format: OddsFormat
    odds_timeout_s: float
    odds_max_retries: int
    odds_backoff_s: float
    odds_quota_warn_threshold: int
    odds_consensus_method: ConsensusMethod
    # Std. dev. of the final NFL scoring margin around the closing spread. Used to
    # turn a spread into a win probability when a game has no moneyline.
    spread_margin_sigma: float

    # Confidence tiers (Contract B). Stored in each payload, which is validated against its own thresholds.
    tier_lock_min_win_prob: float  # LOCK: pick's final win probability at least this (and the model favours it)
    tier_value_edge_min_pct: float  # VALUE EDGE: model at least this many points above the market on the pick
    tier_lean_min_win_prob: float  # LEAN: pick's final win probability at least this

    # LLM scouting reports (google-genai)
    gemini_api_key: str | None = field(repr=False)
    gemini_model: str
    llm_temperature: float
    llm_max_attempts: int  # per game, including repair attempts after a validation failure
    llm_timeout_s: float
    llm_max_concurrency: int
    llm_thinking_budget: int | None  # None = model default; 0 disables thinking on 2.5 Flash

    # Logging
    log_level: str

    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / "_manifest.json"

    @property
    def odds_dir(self) -> Path:
        return self.raw_dir / "odds"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def current_week_json(self) -> Path:
        return self.predictions_dir / "current_week.json"

    @property
    def enriched_report_json(self) -> Path:
        return self.predictions_dir / "weekly_report_enriched.json"

    def preds_archive_json(self, season: int, week: int) -> Path:
        """Versioned Contract B for one week: data/predictions/week_XX_YYYY_preds.json."""
        return self.predictions_dir / f"week_{week:02d}_{season}_preds.json"

    def enriched_archive_json(self, season: int, week: int) -> Path:
        """Versioned Contract C for one week: data/predictions/week_XX_YYYY_enriched.json."""
        return self.predictions_dir / f"week_{week:02d}_{season}_enriched.json"

    def newsletter_md(self, season: int, week: int) -> Path:
        """Weekly newsletter: reports/week_XX_YYYY.md (season in the name, so seasons never overwrite each other)."""
        return self.reports_dir / f"week_{week:02d}_{season}.md"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    def dataset_dir(self, name: str) -> Path:
        return self.raw_dir / name

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.models_dir,
            self.logs_dir,
            self.odds_dir,
            self.predictions_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or PROJECT_ROOT).resolve()
        data_dir = _env_path("NFL_DATA_DIR", root / "data")

        method = (_env_str("ODDS_CONSENSUS_METHOD", "median") or "median").lower()
        if method not in ("mean", "median"):
            raise ValueError(f"ODDS_CONSENSUS_METHOD must be 'mean' or 'median', got {method!r}")
        odds_format = (_env_str("ODDS_FORMAT", "american") or "american").lower()
        if odds_format not in ("american", "decimal"):
            raise ValueError(f"ODDS_FORMAT must be 'american' or 'decimal', got {odds_format!r}")

        start_season = _env_int("NFL_START_SEASON", 2016)
        if start_season < 1999:
            raise ValueError(f"NFL_START_SEASON must be >= 1999 (first nflverse pbp season), got {start_season}")
        lock = _env_float("NFL_TIER_LOCK_MIN_WIN_PROB", 0.75)
        value_edge = _env_float("NFL_VALUE_EDGE_MIN_PCT", 8.0)
        lean = _env_float("NFL_TIER_LEAN_MIN_WIN_PROB", 0.55)
        if not (0.5 <= lean <= lock <= 1.0) or not (0.0 < value_edge <= 100.0):
            raise ValueError(
                "Tier thresholds need 0.5 <= NFL_TIER_LEAN_MIN_WIN_PROB <= NFL_TIER_LOCK_MIN_WIN_PROB <= 1 "
                f"and 0 < NFL_VALUE_EDGE_MIN_PCT <= 100; got lean={lean}, lock={lock}, value_edge={value_edge}"
            )
        weekday = _env_int("NFL_WEEKLY_REFRESH_WEEKDAY", 1)
        hour = _env_int("NFL_WEEKLY_REFRESH_HOUR_UTC", 12)
        if not 0 <= weekday <= 6 or not 0 <= hour <= 23:
            raise ValueError("NFL_WEEKLY_REFRESH_WEEKDAY must be 0-6 and NFL_WEEKLY_REFRESH_HOUR_UTC 0-23")

        return cls(
            project_root=root,
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            interim_dir=data_dir / "interim",
            processed_dir=data_dir / "processed",
            models_dir=_env_path("NFL_MODELS_DIR", root / "models"),
            logs_dir=_env_path("NFL_LOGS_DIR", root / "logs"),
            start_season=start_season,
            nflverse_timeout_s=_env_int("NFLVERSE_TIMEOUT_S", 120),
            nflverse_max_retries=max(1, _env_int("NFLVERSE_MAX_RETRIES", 3)),
            nflverse_backoff_s=_env_float("NFLVERSE_BACKOFF_S", 2.0),
            # Tuesday 12:00 UTC: Monday Night Football has been processed upstream.
            weekly_refresh_weekday=weekday,
            weekly_refresh_hour_utc=hour,
            season_final_grace_days=_env_int("NFL_SEASON_FINAL_GRACE_DAYS", 7),
            odds_api_key=_env_str("ODDS_API_KEY"),
            odds_api_base_url=(_env_str("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4") or "").rstrip("/"),
            odds_sport_key=_env_str("ODDS_SPORT_KEY", "americanfootball_nfl") or "americanfootball_nfl",
            odds_markets=_env_tuple("ODDS_MARKETS", ("h2h", "spreads")),
            odds_bookmakers=_env_tuple("ODDS_BOOKMAKERS", DEFAULT_ODDS_BOOKMAKERS),
            odds_regions=_env_str("ODDS_REGIONS", "us") or "us",
            odds_format=odds_format,  # type: ignore[arg-type]
            odds_timeout_s=_env_float("ODDS_TIMEOUT_S", 20.0),
            odds_max_retries=max(0, _env_int("ODDS_MAX_RETRIES", 4)),
            odds_backoff_s=_env_float("ODDS_BACKOFF_S", 2.0),
            odds_quota_warn_threshold=_env_int("ODDS_QUOTA_WARN_THRESHOLD", 50),
            odds_consensus_method=method,  # type: ignore[arg-type]
            spread_margin_sigma=_env_float("NFL_SPREAD_MARGIN_SIGMA", 13.45),
            tier_lock_min_win_prob=lock,
            tier_value_edge_min_pct=value_edge,
            tier_lean_min_win_prob=lean,
            gemini_api_key=_env_str("GEMINI_API_KEY") or _env_str("GOOGLE_API_KEY"),
            gemini_model=_env_str("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.4),
            llm_max_attempts=max(1, _env_int("LLM_MAX_ATTEMPTS", 3)),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", 60.0),
            llm_max_concurrency=max(1, _env_int("LLM_MAX_CONCURRENCY", 4)),
            llm_thinking_budget=_env_int("GEMINI_THINKING_BUDGET", -1) if _env_str("GEMINI_THINKING_BUDGET") else None,
            log_level=(_env_str("LOG_LEVEL", "INFO") or "INFO").upper(),
        )


settings: Final[Settings] = Settings.from_env()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_logging_configured = False


def setup_logging(level: str | int | None = None, *, log_to_file: bool = True, cfg: Settings = settings) -> None:
    """Configure root logging once: console plus a rotating file in ``logs/``.

    Safe to call repeatedly (Streamlit re-runs scripts); later calls only change the level.
    """
    global _logging_configured
    resolved = level if level is not None else cfg.log_level
    if isinstance(resolved, str):
        resolved = resolved.upper()
    root = logging.getLogger()
    root.setLevel(resolved)
    if _logging_configured:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_to_file:
        try:
            cfg.logs_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                cfg.logs_dir / "nfl_engine.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # read-only filesystem, permissions, ...
            root.warning("File logging disabled (%s); logging to console only.", exc)

    for noisy in ("urllib3", "requests", "httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _logging_configured = True
