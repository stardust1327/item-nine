"""Leak-free game-level feature matrix.

Turns the raw nflverse Parquet cache into one row per scheduled game (played and
upcoming) at ``data/processed/features.parquet``. Every model input describes
what was knowable *before kickoff*.

Pipeline (pure functions, so tests can feed synthetic frames):

1. :func:`filter_garbage_time` drops 4th-quarter plays where the offense's win
   probability is below 5% or above 95%.
2. :func:`team_game_metrics` computes per team-game: offensive pass / rush EPA per
   play, defensive pass / rush EPA allowed per play, offensive early-down
   (1st & 2nd) success rate, and the starting QB's CPOE. These are *current-game
   box-score values*; they feed the rolling step and never reach the output.
3. :func:`pregame_team_features` takes an in-season EWMA over each team's games,
   grouped by team-season and shifted one game, so a Week N row only sees Weeks
   1..N-1. Weeks 1-4 blend that with the team's full Season Y-1 values: weight
   on the prior is 100% in Week 1, minus 25 points per week, 0% from Week 5.
4. :func:`build_game_features` joins home and away pre-game features onto the
   schedule and adds matchup / situational features: net pass and rush EPA
   mismatches, net rest days, divisional, neutral-site, outdoor and high-wind flags.

Column groups: :data:`ID_COLUMNS`, :data:`FEATURE_COLUMNS` (model inputs, all
pre-game), :data:`REFERENCE_COLUMNS` (closing market line, for benchmarking only,
never a model input) and :data:`LABEL_COLUMNS` (outcomes). Models must split
chronologically (walk-forward by season/week), never with a random split.

CLI::

    python -m src.features             # build from the cached raw data
    python -m src.features --refresh   # sync stale raw data first
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl

from config import Settings, settings, setup_logging
from src import ingest
from src.ingest import add_kickoff_utc, write_parquet_atomic

logger = logging.getLogger("src.features" if __name__ == "__main__" else __name__)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
GARBAGE_TIME_QTR: Final[int] = 4
GARBAGE_TIME_WP_LOW: Final[float] = 0.05
GARBAGE_TIME_WP_HIGH: Final[float] = 0.95

EWM_HALF_LIFE_GAMES: Final[float] = 4.0  # a game 4 games back carries half the weight of the latest
PRIOR_DECAY_PER_WEEK: Final[float] = 0.25  # prior weight: wk1 1.00, wk2 0.75, wk3 0.50, wk4 0.25, wk5+ 0
HIGH_WIND_MPH: Final[int] = 15

# pbp uses today's franchise codes for every season; schedules keep the historical ones.
FRANCHISE_CODES: Final[dict[str, str]] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

TEAM_METRICS: Final[tuple[str, ...]] = (
    "off_pass_epa",  # EPA per dropback (sacks and scrambles included)
    "off_rush_epa",  # EPA per designed run
    "def_pass_epa",  # EPA per dropback allowed (offense perspective: higher = worse defense)
    "def_rush_epa",  # EPA per designed run allowed
    "off_early_down_sr",  # success rate on 1st and 2nd down
    "qb_cpoe",  # starting QB completion % over expected
)
# Play count behind each metric; weights the play-level Season Y-1 prior.
METRIC_WEIGHTS: Final[dict[str, str]] = {
    "off_pass_epa": "n_off_pass",
    "off_rush_epa": "n_off_rush",
    "def_pass_epa": "n_def_pass",
    "def_rush_epa": "n_def_rush",
    "off_early_down_sr": "n_off_early_down",
    "qb_cpoe": "n_qb_cpoe",
}

PBP_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "qtr",
    "wp",
    "epa",
    "pass",
    "rush",
    "down",
    "success",
    "cpoe",
    "passer_player_id",
    "two_point_attempt",
)
SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "total",
    "home_rest",
    "away_rest",
    "div_game",
    "roof",
    "wind",
    "location",
    "home_qb_id",
    "away_qb_id",
    "spread_line",
)

ID_COLUMNS: Final[list[str]] = ["game_id", "season", "week", "game_type", "kickoff_utc", "home_team", "away_team"]
TEAM_FEATURE_COLUMNS: Final[list[str]] = [f"{side}_{m}_pre" for side in ("home", "away") for m in TEAM_METRICS]
MATCHUP_FEATURE_COLUMNS: Final[list[str]] = [
    "net_pass_epa_mismatch",
    "net_rush_epa_mismatch",
    "diff_early_down_sr",
    "diff_qb_cpoe",
    "net_rest_days",
    "is_divisional",
    "is_neutral_site",
    "is_outdoor",
    "is_high_wind_outdoor",
    "prior_weight",
]
FEATURE_COLUMNS: Final[list[str]] = TEAM_FEATURE_COLUMNS + MATCHUP_FEATURE_COLUMNS
REFERENCE_COLUMNS: Final[list[str]] = ["spread_line"]  # closing line: benchmark only, never a model input
LABEL_COLUMNS: Final[list[str]] = ["home_score", "away_score", "result", "total", "home_win"]
OUTPUT_COLUMNS: Final[list[str]] = ID_COLUMNS + FEATURE_COLUMNS + REFERENCE_COLUMNS + LABEL_COLUMNS


class FeatureBuildError(RuntimeError):
    """Input data cannot produce a valid feature matrix."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require(df: pl.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise FeatureBuildError(f"{name} is missing required columns: {missing}")


def normalize_team_codes(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Map relocated franchises (OAK, SD, STL) to their current codes."""
    return df.with_columns(pl.col(c).replace(FRANCHISE_CODES) for c in columns)


def prior_weight_expr(decay: float = PRIOR_DECAY_PER_WEEK) -> pl.Expr:
    """Weight on the Season Y-1 prior for a game in ``week``."""
    return (1.0 - decay * (pl.col("week") - 1)).clip(0.0, 1.0)


def _weighted_mean(value: str, weight: str) -> pl.Expr:
    has_value = pl.col(value).is_not_null()
    denominator = pl.col(weight).filter(has_value).sum()
    return pl.when(denominator > 0).then((pl.col(value) * pl.col(weight)).filter(has_value).sum() / denominator)


# --------------------------------------------------------------------------- #
# 1-2. Play-by-play -> current-game team metrics
# --------------------------------------------------------------------------- #
def filter_garbage_time(pbp: pl.DataFrame) -> pl.DataFrame:
    """Drop plays with ``qtr == 4`` and ``wp < 0.05`` or ``wp > 0.95``.

    Plays with a null ``wp`` are kept: an unknown win probability is not garbage time.
    """
    garbage = (
        (pl.col("qtr") == GARBAGE_TIME_QTR)
        & ((pl.col("wp") < GARBAGE_TIME_WP_LOW) | (pl.col("wp") > GARBAGE_TIME_WP_HIGH))
    ).fill_null(False)
    return pbp.filter(~garbage)


def team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Two rows per scheduled game (one per team) in chronological order within each team."""
    games = add_kickoff_utc(normalize_team_codes(schedules, ["home_team", "away_team"]))
    keys = ["game_id", "season", "game_type", "week", "kickoff_utc"]
    home = games.select(
        *keys,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("home_qb_id").alias("starter_qb_id"),
        pl.lit(True).alias("is_home"),
    )
    away = games.select(
        *keys,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("away_qb_id").alias("starter_qb_id"),
        pl.lit(False).alias("is_home"),
    )
    return pl.concat([home, away]).sort("team", "season", "kickoff_utc", "game_id")


def team_game_metrics(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Current-game efficiency per ``(game_id, team)``, after the garbage-time filter.

    These values describe the game itself, so they are only ever used as inputs
    to the shifted rolling step, never as features for the same game.
    """
    _require(pbp, PBP_COLUMNS, "pbp")
    base = filter_garbage_time(normalize_team_codes(pbp, ["posteam", "defteam"])).filter(
        pl.col("posteam").is_not_null() & (pl.col("two_point_attempt") != 1).fill_null(True)
    )
    # nflfastR convention: dropbacks (pass == 1, incl. sacks/scrambles) and designed runs (rush == 1).
    plays = base.filter(((pl.col("pass") == 1) | (pl.col("rush") == 1)) & pl.col("epa").is_not_null())
    is_pass = pl.col("pass") == 1
    is_rush = pl.col("rush") == 1
    early_down = ((pl.col("down") == 1) | (pl.col("down") == 2)).fill_null(False)

    offense = plays.group_by("game_id", pl.col("posteam").alias("team")).agg(
        pl.col("epa").filter(is_pass).mean().alias("off_pass_epa"),
        is_pass.sum().alias("n_off_pass"),
        pl.col("epa").filter(is_rush).mean().alias("off_rush_epa"),
        is_rush.sum().alias("n_off_rush"),
        pl.col("success").filter(early_down).mean().alias("off_early_down_sr"),
        early_down.sum().alias("n_off_early_down"),
    )
    defense = (
        plays.filter(pl.col("defteam").is_not_null())
        .group_by("game_id", pl.col("defteam").alias("team"))
        .agg(
            pl.col("epa").filter(is_pass).mean().alias("def_pass_epa"),
            is_pass.sum().alias("n_def_pass"),
            pl.col("epa").filter(is_rush).mean().alias("def_rush_epa"),
            is_rush.sum().alias("n_def_rush"),
        )
    )

    # Starting QB: the schedule's listed starter; if he has no charted attempts, the team's
    # passer with the most CPOE-charted attempts in the game.
    starters = team_games(schedules).select("game_id", "team", "starter_qb_id")
    order = ["_is_starter", "n_qb_cpoe", "passer_player_id"]
    qb = (
        base.filter(pl.col("cpoe").is_not_null() & pl.col("passer_player_id").is_not_null())
        .group_by("game_id", pl.col("posteam").alias("team"), "passer_player_id")
        .agg(pl.col("cpoe").mean().alias("qb_cpoe"), pl.len().alias("n_qb_cpoe"))
        .join(starters, on=["game_id", "team"], how="left")
        .with_columns((pl.col("passer_player_id") == pl.col("starter_qb_id")).fill_null(False).alias("_is_starter"))
        .group_by("game_id", "team")
        .agg(
            pl.col(c).sort_by(order, descending=[True, True, False]).first()
            for c in ("qb_cpoe", "n_qb_cpoe", "passer_player_id")
        )
        .rename({"passer_player_id": "qb_player_id"})
    )

    keys = ["game_id", "team"]
    metrics = offense.join(defense, on=keys, how="full", coalesce=True).join(qb, on=keys, how="left")
    return metrics.with_columns(pl.col(c).fill_null(0).cast(pl.UInt32) for c in METRIC_WEIGHTS.values()).sort(keys)


# --------------------------------------------------------------------------- #
# 3. Team-game metrics -> pre-game team features
# --------------------------------------------------------------------------- #
def season_priors(team_metrics: pl.DataFrame) -> pl.DataFrame:
    """Play-weighted full-season value of each metric per ``(team, season)``.

    Returned under ``season = Y + 1`` so it joins directly as the prior for the following season.
    """
    return (
        team_metrics.group_by("team", "season")
        .agg(_weighted_mean(m, w).alias(f"{m}_prior") for m, w in METRIC_WEIGHTS.items())
        .with_columns((pl.col("season") + 1).alias("season"))
    )


def pregame_team_features(
    metrics: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    half_life: float = EWM_HALF_LIFE_GAMES,
    decay: float = PRIOR_DECAY_PER_WEEK,
) -> pl.DataFrame:
    """Per team-game, the pre-game estimate of every metric (``{metric}_pre``).

    ``{metric}_ewma``: EWMA over the team's earlier games this season. Within each
    team-season, ordered by kickoff, the sequence is ``ewm_mean`` -> ``forward_fill``
    (carries the last value across unplayed or uncharted games) -> ``shift(1)``.
    So a game's own row never enters its own value.

    ``{metric}_prior``: the team's play-weighted Season Y-1 value.

    ``{metric}_pre``: ``w * prior + (1 - w) * ewma`` with ``w`` from :func:`prior_weight_expr`.
    When only one side exists (no prior for the first cached season or a new
    franchise, no in-season games yet) that side is used alone. ``w`` = 1 and
    ``w`` = 0 are honoured exactly.
    """
    games = team_games(schedules)
    with_metrics = games.join(metrics, on=["game_id", "team"], how="left").sort(
        "team", "season", "kickoff_utc", "game_id"
    )

    ewma = [
        pl.col(m)
        .ewm_mean(half_life=half_life, adjust=True, ignore_nulls=True)
        .forward_fill()
        .shift(1)
        .over(["team", "season"])
        .alias(f"{m}_ewma")
        for m in TEAM_METRICS
    ]
    played = with_metrics.filter(pl.col("n_off_pass").is_not_null() | pl.col("n_def_pass").is_not_null())
    features = with_metrics.with_columns(*ewma, prior_weight_expr(decay).alias("prior_weight")).join(
        season_priors(played), on=["team", "season"], how="left"
    )

    weight = pl.col("prior_weight")
    blended = []
    for m in TEAM_METRICS:
        prior, current = pl.col(f"{m}_prior"), pl.col(f"{m}_ewma")
        blended.append(
            pl.when(weight >= 1.0)
            .then(prior)
            .when(weight <= 0.0)
            .then(current)
            .when(prior.is_null())
            .then(current)
            .when(current.is_null())
            .then(prior)
            .otherwise(weight * prior + (1.0 - weight) * current)
            .alias(f"{m}_pre")
        )

    keep = ["game_id", "team", "opponent", "is_home", "season", "week", "game_type", "kickoff_utc", "prior_weight"]
    derived = [f"{m}_{kind}" for m in TEAM_METRICS for kind in ("ewma", "prior", "pre")]
    return features.with_columns(blended).select(*keep, *derived)


# --------------------------------------------------------------------------- #
# 4. Game-level matrix
# --------------------------------------------------------------------------- #
def build_game_features(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    weather: pl.DataFrame | None = None,
    half_life: float = EWM_HALF_LIFE_GAMES,
    decay: float = PRIOR_DECAY_PER_WEEK,
) -> pl.DataFrame:
    """One row per scheduled game with :data:`OUTPUT_COLUMNS`.

    Args:
        pbp: nflverse play-by-play with at least :data:`PBP_COLUMNS`.
        schedules: nflverse schedules with at least :data:`SCHEDULE_COLUMNS`.
        weather: optional ``game_id, wind`` (mph) forecasts. nflverse only fills
            ``wind`` after kickoff, so upcoming outdoor games get a null
            ``is_high_wind_outdoor`` unless a forecast is supplied here.
            Forecasts take precedence over the schedule's value.
    """
    _require(schedules, SCHEDULE_COLUMNS, "schedules")
    metrics = team_game_metrics(pbp, schedules)
    team_pre = pregame_team_features(metrics, schedules, half_life=half_life, decay=decay)

    games = add_kickoff_utc(normalize_team_codes(schedules, ["home_team", "away_team"]))
    if weather is not None:
        _require(weather, ("game_id", "wind"), "weather")
        games = (
            games.join(
                weather.select("game_id", pl.col("wind").cast(pl.Float64).alias("_wind_forecast")),
                on="game_id",
                how="left",
            )
            .with_columns(pl.coalesce("_wind_forecast", pl.col("wind").cast(pl.Float64)).alias("wind"))
            .drop("_wind_forecast")
        )

    def side(prefix: str) -> pl.DataFrame:
        return team_pre.select(
            "game_id",
            pl.col("team").alias(f"{prefix}_team"),
            *[pl.col(f"{m}_pre").alias(f"{prefix}_{m}_pre") for m in TEAM_METRICS],
        )

    matrix = games.join(side("home"), on=["game_id", "home_team"], how="left").join(
        side("away"), on=["game_id", "away_team"], how="left"
    )

    def h(m: str) -> pl.Expr:
        return pl.col(f"home_{m}_pre")

    def a(m: str) -> pl.Expr:
        return pl.col(f"away_{m}_pre")

    roof = pl.col("roof")
    is_outdoor = (
        pl.when(roof.is_in(["outdoors", "open"])).then(1).when(roof.is_in(["dome", "closed"])).then(0).cast(pl.Int8)
    )
    matrix = matrix.with_columns(
        # Offense vs. the defense it faces; defensive EPA is offense-perspective (higher = softer defense).
        ((h("off_pass_epa") + a("def_pass_epa")) - (a("off_pass_epa") + h("def_pass_epa"))).alias(
            "net_pass_epa_mismatch"
        ),
        ((h("off_rush_epa") + a("def_rush_epa")) - (a("off_rush_epa") + h("def_rush_epa"))).alias(
            "net_rush_epa_mismatch"
        ),
        (h("off_early_down_sr") - a("off_early_down_sr")).alias("diff_early_down_sr"),
        (h("qb_cpoe") - a("qb_cpoe")).alias("diff_qb_cpoe"),
        (pl.col("home_rest") - pl.col("away_rest")).cast(pl.Int32).alias("net_rest_days"),
        pl.col("div_game").cast(pl.Int8).alias("is_divisional"),
        (pl.col("location") == "Neutral").cast(pl.Int8).alias("is_neutral_site"),
        is_outdoor.alias("is_outdoor"),
        prior_weight_expr(decay).alias("prior_weight"),
        pl.when(pl.col("result") > 0).then(1).when(pl.col("result") < 0).then(0).cast(pl.Int8).alias("home_win"),
    ).with_columns(
        # Unknown roof state (retractable, pre-game) or unknown wind -> null, not False.
        pl.when(pl.col("is_outdoor") == 0)
        .then(0)
        .when((pl.col("is_outdoor") == 1) & pl.col("wind").is_not_null())
        .then((pl.col("wind") >= HIGH_WIND_MPH).cast(pl.Int8))
        .cast(pl.Int8)
        .alias("is_high_wind_outdoor")
    )
    return matrix.select(OUTPUT_COLUMNS).sort("kickoff_utc", "game_id")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_inputs(cfg: Settings = settings, *, refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read only the columns the feature build needs from the raw cache."""
    cache = ingest.get_cache() if cfg is settings else ingest.NflverseCache(cfg)
    schedules = cache.read("schedules", refresh=refresh)
    pbp = cache.read("pbp", columns=PBP_COLUMNS, refresh=refresh)
    return pbp, schedules


def build_and_save(
    cfg: Settings = settings, *, refresh: bool = False, weather: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Build the matrix from the raw cache and write it to ``cfg.features_path`` atomically."""
    pbp, schedules = load_inputs(cfg, refresh=refresh)
    features = build_game_features(pbp, schedules, weather=weather)
    write_parquet_atomic(features, cfg.features_path)
    _log_summary(features, cfg.features_path)
    return features


def load_features(cfg: Settings = settings) -> pl.DataFrame:
    if not cfg.features_path.exists():
        raise FileNotFoundError(f"{cfg.features_path} not found; run `python -m src.features` first")
    return pl.read_parquet(cfg.features_path)


def _log_summary(features: pl.DataFrame, path: Path) -> None:
    played = features.filter(pl.col("result").is_not_null())
    upcoming = features.filter(pl.col("result").is_null())
    logger.info(
        "Wrote %s: %s games (%d played, %d upcoming), seasons %d-%d, %d feature columns",
        path,
        f"{features.height:,}",
        played.height,
        upcoming.height,
        features["season"].min(),
        features["season"].max(),
        len(FEATURE_COLUMNS),
    )
    # The first cached season has no prior, so its early weeks are expected to be sparse.
    steady = played.filter((pl.col("season") > features["season"].min()) & (pl.col("week") >= 2))
    if steady.height:
        null_rates = {c: steady[c].null_count() / steady.height for c in TEAM_FEATURE_COLUMNS}
        worst = max(null_rates, key=null_rates.__getitem__)
        logger.info("Team-feature null rate after burn-in: max %.2f%% (%s)", 100 * null_rates[worst], worst)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.features", description="Build the game-level feature matrix.")
    parser.add_argument("--refresh", action="store_true", help="sync stale raw nflverse data before building")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ...")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    settings.ensure_dirs()
    try:
        build_and_save(refresh=args.refresh)
    except (FeatureBuildError, ingest.IngestError, FileNotFoundError) as exc:
        logger.error("Feature build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
