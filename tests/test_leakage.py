"""Temporal-leakage tests for src.features.

The central guarantee: a game's pre-game feature columns are a function of
information available before its kickoff only. These tests check that by:

* perturbing a game's own play-by-play and final score and asserting its
  feature row is bit-for-bit unchanged, while the same teams' *next* games do change;
* rebuilding every game's features from history truncated at its kickoff and
  asserting the row matches the full build;
* verifying the shift/EWMA/prior arithmetic and the garbage-time filter independently.
"""

from __future__ import annotations

import math
import random
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from config import settings
from src import features as F
from src.ingest import add_kickoff_utc

SEASONS = (2023, 2024)
WEEKS = 6
TEAMS = ("KC", "LV", "DEN", "LAC")
HISTORICAL_CODES = {2023: {"LV": "OAK", "LAC": "SD"}}  # schedules use relocated codes, pbp does not
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
STRENGTH = {"KC": 0.15, "LV": -0.05, "DEN": 0.0, "LAC": 0.05}
UPCOMING = (2024, WEEKS)  # last synthetic week has no results and no pbp yet
OUTCOME_COLUMNS = ("home_score", "away_score", "result", "total")
BOX_SCORE_COLUMNS = {*OUTCOME_COLUMNS, "overtime", "home_win", *F.TEAM_METRICS, *F.METRIC_WEIGHTS.values()}

PBP_SCHEMA = {
    "game_id": pl.String,
    "season": pl.Int32,
    "week": pl.Int32,
    "posteam": pl.String,
    "defteam": pl.String,
    "qtr": pl.Float64,
    "wp": pl.Float64,
    "epa": pl.Float64,
    "pass": pl.Float64,
    "rush": pl.Float64,
    "down": pl.Float64,
    "success": pl.Float64,
    "cpoe": pl.Float64,
    "passer_player_id": pl.String,
    "two_point_attempt": pl.Float64,
}


# --------------------------------------------------------------------------- #
# Synthetic league
# --------------------------------------------------------------------------- #
def _play(game_id: str, season: int, week: int, off: str, defense: str, **values: Any) -> dict[str, Any]:
    row = dict.fromkeys(PBP_SCHEMA)
    row.update(game_id=game_id, season=season, week=week, posteam=off, defteam=defense, two_point_attempt=0.0)
    row.update(values)
    return row


def _game_plays(rng: random.Random, game_id: str, season: int, week: int, home: str, away: str) -> list[dict]:
    plays = []
    for off, defense in ((home, away), (away, home)):
        args = (game_id, season, week, off, defense)
        for k in range(40):
            is_pass = k % 5 < 3
            epa = rng.gauss(STRENGTH[off] - 0.5 * STRENGTH[defense], 1.2)
            plays.append(
                _play(
                    *args,
                    qtr=float(1 + k * 4 // 40),
                    wp=rng.uniform(0.15, 0.85),
                    epa=epa,
                    success=float(epa > 0),
                    down=float(k % 3 + 1),
                    **{"pass": float(is_pass), "rush": float(not is_pass)},
                    cpoe=rng.gauss(2.0, 8.0) if is_pass and k % 7 else None,
                    passer_player_id=(f"QB2_{off}" if k == 37 else f"QB1_{off}") if is_pass else None,
                )
            )
        # Traps: everything below except the last two rows must be ignored by the metrics.
        dropback = {"pass": 1.0, "rush": 0.0, "down": 1.0}
        plays += [
            _play(*args, qtr=4.0, wp=0.99, epa=30.0, success=1.0, cpoe=60.0, passer_player_id=f"QB1_{off}", **dropback),
            _play(*args, qtr=4.0, wp=0.01, epa=-30.0, success=0.0, down=2.0, **{"pass": 0.0, "rush": 1.0}),
            _play(*args, qtr=2.0, wp=0.5, epa=40.0, success=1.0, two_point_attempt=1.0, **{**dropback, "down": None}),
            _play(*args, qtr=1.0, wp=0.5, epa=5.0, success=1.0, **{"pass": 0.0, "rush": 0.0}),  # punt
            _play(*args, qtr=4.0, wp=None, epa=rng.gauss(0, 1), success=0.0, **dropback),  # unknown wp: kept
            _play(*args, qtr=3.0, wp=0.99, epa=rng.gauss(0, 1), success=1.0, **dropback),  # not 4th qtr: kept
        ]
    return plays


def make_league(seed: int = 7) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = random.Random(seed)
    games, plays = [], []
    for season in SEASONS:
        code = {t: HISTORICAL_CODES.get(season, {}).get(t, t) for t in TEAMS}
        for week in range(1, WEEKS + 1):
            gameday = date(season, 9, 7) + timedelta(days=7 * (week - 1))
            for slot, (i, j) in enumerate(PAIRINGS[(week - 1) % len(PAIRINGS)]):
                home, away = (TEAMS[i], TEAMS[j]) if week % 2 else (TEAMS[j], TEAMS[i])
                game_id = f"{season}_{week:02d}_{code[away]}_{code[home]}"
                played = (season, week) != UPCOMING
                if played:
                    plays += _game_plays(rng, game_id, season, week, home, away)
                home_pts, away_pts = rng.randint(10, 35), rng.randint(10, 35)
                if home_pts == away_pts:
                    home_pts += 3
                games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "game_type": "REG",
                        "week": week,
                        "gameday": gameday.isoformat(),
                        "gametime": ("13:00", "16:25")[slot],
                        "home_team": code[home],
                        "away_team": code[away],
                        "home_score": home_pts if played else None,
                        "away_score": away_pts if played else None,
                        "result": home_pts - away_pts if played else None,
                        "total": home_pts + away_pts if played else None,
                        "home_rest": 7 if week > 1 else 0,
                        "away_rest": 6 if week > 1 else 0,
                        "div_game": int({i, j} == {0, 1}),
                        "roof": ("outdoors", "dome")[slot],
                        "wind": 18 if slot == 0 and week % 2 else (8 if slot == 0 else None),
                        "location": "Home",
                        "home_qb_id": f"QB1_{home}",
                        "away_qb_id": f"QB1_{away}",
                        "spread_line": 2.5,
                    }
                )
    return pl.DataFrame(plays, schema=PBP_SCHEMA), pl.DataFrame(games)


@pytest.fixture(scope="module")
def league() -> tuple[pl.DataFrame, pl.DataFrame]:
    return make_league()


@pytest.fixture(scope="module")
def built(league) -> pl.DataFrame:
    return F.build_game_features(*league)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def feature_row(features: pl.DataFrame, game_id: str) -> dict[str, Any]:
    rows = features.filter(pl.col("game_id") == game_id).select(F.FEATURE_COLUMNS).to_dicts()
    assert len(rows) == 1, f"expected one row for {game_id}"
    return rows[0]


def assert_rows_equal(left: dict[str, Any], right: dict[str, Any], tol: float = 1e-12) -> None:
    for column in F.FEATURE_COLUMNS:
        a, b = left[column], right[column]
        if a is None or b is None:
            assert a is None and b is None, f"{column}: {a!r} != {b!r}"
        else:
            assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{column}: {a!r} != {b!r}"


def as_of_kickoff(pbp: pl.DataFrame, schedules: pl.DataFrame, kickoff: datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
    """What the world looked like at ``kickoff``: no pbp and no final scores from that moment on."""
    kickoffs = add_kickoff_utc(schedules).select("game_id", "kickoff_utc")
    past = kickoffs.filter(pl.col("kickoff_utc") < kickoff)["game_id"].implode()
    is_past = pl.col("game_id").is_in(past)
    return (
        pbp.filter(is_past),
        schedules.with_columns(pl.when(is_past).then(pl.col(c)).alias(c) for c in OUTCOME_COLUMNS),
    )


def team_game_ids(features: pl.DataFrame, team: str, season: int) -> list[str]:
    games = features.filter(
        (pl.col("season") == season) & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    )
    return games.sort("kickoff_utc")["game_id"].to_list()


# metric -> (team column, play filter, value column) for the plain-Python oracle below.
ORACLE: dict[str, tuple[str, Any, str]] = {
    "off_pass_epa": ("posteam", lambda p: p["pass"] == 1, "epa"),
    "off_rush_epa": ("posteam", lambda p: p["rush"] == 1, "epa"),
    "def_pass_epa": ("defteam", lambda p: p["pass"] == 1, "epa"),
    "def_rush_epa": ("defteam", lambda p: p["rush"] == 1, "epa"),
    "off_early_down_sr": ("posteam", lambda p: p["down"] in (1.0, 2.0), "success"),
}


def reference_season_value(pbp: pl.DataFrame, team: str, season: int, metric: str) -> float:
    """Play-weighted season value of ``metric``, computed with plain Python as an independent oracle."""
    values = []
    for p in pbp.filter(pl.col("season") == season).to_dicts():
        garbage = p["qtr"] == 4 and p["wp"] is not None and (p["wp"] < 0.05 or p["wp"] > 0.95)
        if garbage or p["two_point_attempt"] == 1:
            continue
        if metric == "qb_cpoe":
            if p["posteam"] == team and p["passer_player_id"] == f"QB1_{team}" and p["cpoe"] is not None:
                values.append(p["cpoe"])
            continue
        team_column, keep, value = ORACLE[metric]
        scrimmage = (p["pass"] == 1 or p["rush"] == 1) and p["epa"] is not None
        if scrimmage and p[team_column] == team and keep(p):
            values.append(p[value])
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
# Column contract
# --------------------------------------------------------------------------- #
def test_feature_columns_exclude_box_score_and_outcomes(built):
    assert not set(F.FEATURE_COLUMNS) & BOX_SCORE_COLUMNS
    assert not set(F.FEATURE_COLUMNS) & set(F.LABEL_COLUMNS + F.REFERENCE_COLUMNS)
    assert all(c.endswith("_pre") for c in F.TEAM_FEATURE_COLUMNS)
    # No current-game team metric survives into the matrix under any name.
    for metric in F.TEAM_METRICS:
        for column in (metric, f"home_{metric}", f"away_{metric}"):
            assert column not in built.columns
    assert built.columns == F.OUTPUT_COLUMNS


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_perturbing_a_game_never_changes_its_own_features(league, built):
    pbp, schedules = league
    target = built.filter((pl.col("season") == 2024) & (pl.col("week") == 3)).row(0, named=True)
    gid = target["game_id"]
    this_game = pl.col("game_id") == gid

    pbp_hacked = pbp.with_columns(
        pl.when(this_game).then(pl.col("epa") * 10 + 7).otherwise(pl.col("epa")).alias("epa"),
        pl.when(this_game).then(1.0).otherwise(pl.col("success")).alias("success"),
        pl.when(this_game).then(pl.col("cpoe") + 40).otherwise(pl.col("cpoe")).alias("cpoe"),
    )
    schedules_hacked = schedules.with_columns(
        pl.when(this_game).then(pl.col("home_score") + 35).otherwise(pl.col("home_score")).alias("home_score"),
        pl.when(this_game).then(pl.col("result") + 35).otherwise(pl.col("result")).alias("result"),
        pl.when(this_game).then(pl.col("total") + 35).otherwise(pl.col("total")).alias("total"),
    )
    hacked = F.build_game_features(pbp_hacked, schedules_hacked)

    # The perturbation is real: the game's labels moved...
    assert hacked.filter(this_game)["result"][0] == target["result"] + 35
    # ...but none of its pre-game features did.
    assert_rows_equal(feature_row(built, gid), feature_row(hacked, gid))

    # Sensitivity check: the same teams' next games must see the change.
    for team in (target["home_team"], target["away_team"]):
        ids = team_game_ids(built, team, 2024)
        next_gid = ids[ids.index(gid) + 1]
        before, after = feature_row(built, next_gid), feature_row(hacked, next_gid)
        changed = [c for c in F.TEAM_FEATURE_COLUMNS if before[c] is not None and abs(before[c] - after[c]) > 1e-9]
        assert changed, f"{team}'s next game ignored the previous game's data"


def test_every_game_matches_an_as_of_kickoff_rebuild(league, built):
    pbp, schedules = league
    for game in built.iter_rows(named=True):
        pbp_then, schedules_then = as_of_kickoff(pbp, schedules, game["kickoff_utc"])
        rebuilt = F.build_game_features(pbp_then, schedules_then)
        assert_rows_equal(feature_row(built, game["game_id"]), feature_row(rebuilt, game["game_id"]))


def test_first_cached_season_week_one_has_no_team_features(built):
    week_one = built.filter((pl.col("season") == SEASONS[0]) & (pl.col("week") == 1))
    assert week_one.height == 2
    for column in F.TEAM_FEATURE_COLUMNS:
        assert week_one[column].null_count() == week_one.height, column


def test_upcoming_games_have_features_but_no_labels(built):
    upcoming = built.filter((pl.col("season") == UPCOMING[0]) & (pl.col("week") == UPCOMING[1]))
    assert upcoming.height == 2
    for column in F.LABEL_COLUMNS:
        assert upcoming[column].null_count() == upcoming.height, column
    for column in F.TEAM_FEATURE_COLUMNS:
        assert upcoming[column].null_count() == 0, column


# --------------------------------------------------------------------------- #
# Prior regression and rolling arithmetic
# --------------------------------------------------------------------------- #
def test_week_one_is_the_previous_season_exactly(league, built):
    pbp, _ = league
    week_one = built.filter((pl.col("season") == 2024) & (pl.col("week") == 1))
    assert week_one.height == 2
    for game in week_one.iter_rows(named=True):
        for side in ("home", "away"):
            for metric in F.TEAM_METRICS:
                expected = reference_season_value(pbp, game[f"{side}_team"], 2023, metric)
                assert game[f"{side}_{metric}_pre"] == pytest.approx(expected, rel=1e-12), (side, metric)


def test_prior_weight_decays_25_points_per_week(built):
    weights = built.filter(pl.col("season") == 2024).group_by("week").agg(pl.col("prior_weight").first()).sort("week")
    assert weights["prior_weight"].to_list() == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0, 0.0])


def adjusted_ewma(values: list[float], half_life: float) -> float:
    """Reference EWMA (``adjust=True``): weight (1 - alpha)^age, newest value has age 0."""
    decay = 0.5 ** (1 / half_life)  # = 1 - alpha
    weights = [decay ** (len(values) - 1 - i) for i in range(len(values))]
    return sum(w * v for w, v in zip(weights, values, strict=True)) / sum(weights)


def test_blend_uses_only_earlier_games(league):
    pbp, schedules = league
    metrics = F.team_game_metrics(pbp, schedules)
    team_pre = F.pregame_team_features(metrics, schedules)
    h = F.EWM_HALF_LIFE_GAMES

    for team in TEAMS:
        rows = team_pre.filter((pl.col("team") == team) & (pl.col("season") == 2024))
        by_week = {r["week"]: r for r in rows.iter_rows(named=True)}
        own = metrics.join(rows.select("game_id", "team", "week"), on=["game_id", "team"]).sort("week")
        assert own["week"].to_list() == [1, 2, 3, 4, 5]  # week 6 is upcoming: no current-game metrics
        for metric in F.TEAM_METRICS:
            x = own[metric].to_list()  # this team's own game values, weeks 1-5
            prior = by_week[1][f"{metric}_prior"]

            assert by_week[1][f"{metric}_ewma"] is None  # nothing before week 1
            assert by_week[1][f"{metric}_pre"] == pytest.approx(prior, rel=1e-12)
            assert by_week[2][f"{metric}_ewma"] == pytest.approx(x[0], rel=1e-12)  # week 2 sees week 1 only
            ewma_3 = adjusted_ewma(x[:2], h)
            assert by_week[3][f"{metric}_ewma"] == pytest.approx(ewma_3, rel=1e-12)
            assert by_week[3][f"{metric}_pre"] == pytest.approx(0.5 * prior + 0.5 * ewma_3, rel=1e-12)
            assert by_week[4][f"{metric}_pre"] == pytest.approx(
                0.25 * prior + 0.75 * adjusted_ewma(x[:3], h), rel=1e-12
            )
            assert by_week[5][f"{metric}_pre"] == by_week[5][f"{metric}_ewma"]  # week 5+: 0% prior
            # Upcoming week 6 sees weeks 1-5, and none of the current week.
            assert by_week[6][f"{metric}_pre"] == pytest.approx(adjusted_ewma(x[:5], h), rel=1e-12)


# --------------------------------------------------------------------------- #
# Garbage time and situational features
# --------------------------------------------------------------------------- #
def test_garbage_time_filter_rule():
    plays = pl.DataFrame(
        {
            "qtr": [4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 1.0],
            "wp": [0.04, 0.96, 0.05, 0.95, None, 0.99, 0.01],
        }
    )
    kept = F.filter_garbage_time(plays)
    assert kept.to_dicts() == [
        {"qtr": 4.0, "wp": 0.05},
        {"qtr": 4.0, "wp": 0.95},
        {"qtr": 4.0, "wp": None},
        {"qtr": 3.0, "wp": 0.99},
        {"qtr": 1.0, "wp": 0.01},
    ]


def test_garbage_and_non_scrimmage_plays_do_not_move_metrics(league):
    pbp, schedules = league
    clean = pbp.filter(
        ~((pl.col("qtr") == 4) & ((pl.col("wp") > 0.95) | (pl.col("wp") < 0.05))).fill_null(False)
        & (pl.col("two_point_attempt") != 1)
        & ((pl.col("pass") == 1) | (pl.col("rush") == 1))
    )
    assert clean.height < pbp.height
    with_traps = F.team_game_metrics(pbp, schedules)
    without = F.team_game_metrics(clean, schedules)
    for metric in F.TEAM_METRICS:
        diff = (with_traps[metric] - without[metric]).abs().max()
        assert diff is not None and diff < 1e-12, metric


def test_relocated_franchise_codes_link_across_seasons(built):
    # 2023 schedules say OAK/SD; pbp and 2024 say LV/LAC. The prior must still connect them.
    assert built.filter(pl.col("season") == 2023)["home_team"].is_in(["OAK", "SD"]).sum() == 0
    lv_week_one = built.filter((pl.col("season") == 2024) & (pl.col("week") == 1)).filter(
        (pl.col("home_team") == "LV") | (pl.col("away_team") == "LV")
    )
    side = "home" if lv_week_one["home_team"][0] == "LV" else "away"
    assert lv_week_one[f"{side}_off_pass_epa_pre"][0] is not None


def test_situational_flags(built):
    outdoor = built.filter(pl.col("is_outdoor") == 1)
    dome = built.filter(pl.col("is_outdoor") == 0)
    assert outdoor.height and dome.height
    assert dome["is_high_wind_outdoor"].to_list() == [0] * dome.height
    assert set(outdoor["is_high_wind_outdoor"].drop_nulls().to_list()) == {0, 1}
    assert built["net_rest_days"].drop_nulls().unique().sort().to_list() == [0, 1]
    assert set(built["is_divisional"].to_list()) == {0, 1}


def test_wind_forecast_override_fills_upcoming_games(league):
    pbp, schedules = league
    upcoming_ids = schedules.filter(pl.col("result").is_null())["game_id"].to_list()
    weather = pl.DataFrame({"game_id": upcoming_ids, "wind": [22] * len(upcoming_ids)})
    feats = F.build_game_features(pbp, schedules, weather=weather)
    upcoming = feats.filter(pl.col("game_id").is_in(upcoming_ids))
    assert upcoming.filter(pl.col("is_outdoor") == 1)["is_high_wind_outdoor"].to_list() == [1]


# --------------------------------------------------------------------------- #
# Project-wide guard: chronological splits only
# --------------------------------------------------------------------------- #
def test_no_random_train_test_split_anywhere():
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(path.relative_to(root))
        for folder in ("src", "app")
        for path in (root / folder).rglob("*.py")
        if re.search(r"\btrain_test_split\b", path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"random train_test_split is banned (use walk-forward splits): {offenders}"


# --------------------------------------------------------------------------- #
# Real data (skipped when the raw cache is absent)
# --------------------------------------------------------------------------- #
REAL_SEASONS = (2024, 2025)


@pytest.mark.skipif(
    not all((settings.raw_dir / "pbp" / f"pbp_{s}.parquet").exists() for s in REAL_SEASONS)
    or not (settings.raw_dir / "schedules" / "schedules.parquet").exists(),
    reason="raw nflverse cache not populated; run `python -m src.ingest`",
)
def test_real_data_as_of_kickoff_rebuild():
    from src.ingest import NflverseCache

    cache = NflverseCache(settings)
    pbp = cache.read("pbp", REAL_SEASONS, columns=F.PBP_COLUMNS, refresh=False)
    schedules = cache.read("schedules", REAL_SEASONS, refresh=False)
    full = F.build_game_features(pbp, schedules)

    season = full.filter(pl.col("season") == REAL_SEASONS[1])
    sample = [
        season.filter(pl.col("week") == week).sort("kickoff_utc")["game_id"][-1]
        for week in (1, 3, 9, int(season["week"].max()))
    ]
    for gid in sample:
        kickoff = full.filter(pl.col("game_id") == gid)["kickoff_utc"][0]
        rebuilt = F.build_game_features(*as_of_kickoff(pbp, schedules, kickoff))
        assert_rows_equal(feature_row(full, gid), feature_row(rebuilt, gid))
    # Week 1 of the second season leans entirely on the first season.
    assert season.filter(pl.col("week") == 1)["prior_weight"].unique().to_list() == [1.0]
