"""Tests for src.model: splits, calibration recipe, market blend, tiers and Contract B."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.calibration import CalibratedClassifierCV

from config import Settings
from src import features as F
from src import ingest
from src import model as M

NOW = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)
TEAMS = ["KC", "BUF", "PHI", "DAL", "SF", "DET", "BAL", "CIN", "GB", "LA", "MIA", "HOU", "PIT", "SEA", "MIN", "TB"]


# --------------------------------------------------------------------------- #
# Synthetic league with a known data-generating process
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _american(q: float) -> int:
    """American price for a vig-inclusive implied probability ``q``."""
    return round(-100 * q / (1 - q)) if q >= 0.5 else round(100 * (1 - q) / q)


def make_frame(
    seasons: range = range(2016, 2027),
    weeks: int = 18,
    upcoming: tuple[int, int] | None = (2026, 6),
    seed: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Feature matrix (``F.OUTPUT_COLUMNS``) and matching schedules with moneylines."""
    rng = np.random.default_rng(seed)
    rows, sched = [], []
    for season in seasons:
        last_week = upcoming[1] if upcoming and season == upcoming[0] else weeks
        for week in range(1, last_week + 1):
            kickoff = datetime(season, 9, 13, 17, tzinfo=UTC) + timedelta(days=7 * (week - 1))
            for g in range(8):
                home, away = TEAMS[(2 * g + week) % 16], TEAMS[(2 * g + 1 + 3 * week) % 16]
                if home == away:
                    away = TEAMS[(TEAMS.index(home) + 1) % 16]
                values = {c: float(rng.normal(0, 0.1)) for c in F.TEAM_FEATURE_COLUMNS}
                values |= {
                    "net_pass_epa_mismatch": float(rng.normal(0, 0.15)),
                    "net_rush_epa_mismatch": float(rng.normal(0, 0.1)),
                    "diff_early_down_sr": float(rng.normal(0, 0.05)),
                    "diff_qb_cpoe": float(rng.normal(0, 3)),
                    "net_rest_days": int(rng.choice([-3, 0, 0, 0, 3])),
                    "is_divisional": int(rng.random() < 0.35),
                    "is_neutral_site": int(rng.random() < 0.03),
                    "is_outdoor": int(rng.random() < 0.7),
                    "is_high_wind_outdoor": None if rng.random() < 0.1 else int(rng.random() < 0.1),
                    "prior_weight": max(0.0, 1 - 0.25 * (week - 1)),
                }
                logit = 4.0 * values["net_pass_epa_mismatch"] + 3.0 * values["net_rush_epa_mismatch"] + 0.15
                p_true = float(_sigmoid(np.array(logit)))
                p_market = float(_sigmoid(np.array(logit + rng.normal(0, 0.15))))
                played = not (upcoming and season == upcoming[0] and week == upcoming[1])
                home_win = int(rng.random() < p_true) if played else None
                gid = f"{season}_{week:02d}_{away}_{home}_{g}"
                rows.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "kickoff_utc": kickoff + timedelta(hours=3 * g),
                        "home_team": home,
                        "away_team": away,
                        **values,
                        "spread_line": round(10 * (p_market - 0.5), 1),
                        "home_score": (24 if home_win else 17) if played else None,
                        "away_score": (17 if home_win else 24) if played else None,
                        "result": (7 if home_win else -7) if played else None,
                        "total": 41 if played else None,
                        "home_win": home_win,
                    }
                )
                vig = 1.045
                sched.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "home_moneyline": _american(min(p_market * vig, 0.97)),
                        "away_moneyline": _american(min((1 - p_market) * vig, 0.97)),
                        "spread_line": round(10 * (p_market - 0.5), 1),
                    }
                )
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("kickoff_utc").dt.cast_time_unit("us"),
        pl.col(["season", "week"]).cast(pl.Int32),
        pl.col("home_win").cast(pl.Int8),
        *(pl.col(c).cast(pl.Int8) for c in ("is_divisional", "is_neutral_site", "is_outdoor", "is_high_wind_outdoor")),
    )
    return frame.select(F.OUTPUT_COLUMNS), pl.DataFrame(sched)


@pytest.fixture(scope="module")
def league() -> tuple[pl.DataFrame, pl.DataFrame]:
    return make_frame()


@pytest.fixture(scope="module")
def frame(league) -> pl.DataFrame:
    features, schedules = league
    market = M.nflverse_market_probs(schedules).select("game_id", "market_home_wp")
    return features.join(market, on="game_id", how="left")


@pytest.fixture(scope="module")
def payload(league) -> dict[str, Any]:
    features, schedules = league
    week_ids = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))["game_id"].to_list()
    # Live consensus for all but the last game, which has no line at all.
    nflverse = M.nflverse_market_probs(schedules).filter(pl.col("game_id").is_in(week_ids[:-1]))
    market = nflverse.with_columns(
        pl.lit("odds_api").alias("market_source"),
        pl.lit(8, pl.UInt32).alias("market_books"),
        pl.lit(NOW).alias("market_as_of"),
    )
    return M.predict_week(features, schedules, market, season=2026, week=6, now=NOW)


# --------------------------------------------------------------------------- #
# Spec conformance
# --------------------------------------------------------------------------- #
def test_hyperparameters_and_calibration_match_spec(frame):
    params = M.WinProbabilityModel().build_estimator().get_params()
    assert {k: params[k] for k in M.XGB_PARAMS} == {
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 250,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    data = M.labeled_rows(frame)
    fitted = M.WinProbabilityModel().fit(data.filter(pl.col("season") <= 2018), data.filter(pl.col("season") == 2019))
    assert isinstance(fitted.calibrated_, CalibratedClassifierCV)
    assert fitted.calibrated_.method == "sigmoid"
    p = fitted.predict_proba(data.filter(pl.col("season") == 2020))
    assert np.all((p > 0) & (p < 1))


def test_feature_labels_cover_every_feature():
    assert set(F.FEATURE_COLUMNS) <= set(M.FEATURE_LABELS)


# --------------------------------------------------------------------------- #
# Walk-forward splits
# --------------------------------------------------------------------------- #
def test_walk_forward_splits_are_expanding_and_chronological():
    splits = M.walk_forward_splits(range(2016, 2026))
    assert [s.test_season for s in splits] == list(range(2018, 2026))
    for s in splits:
        assert s.train_seasons[0] == 2016  # expanding window from the first season
        assert s.train_seasons == tuple(range(2016, s.test_season - 1))
        assert max(s.train_seasons) < s.calibration_season < s.test_season
        assert s.calibration_season == s.test_season - 1
    assert M.walk_forward_splits(range(2016, 2026), first_test_season=2023)[0].test_season == 2023
    assert M.walk_forward_splits([2016, 2017]) == []


def test_split_rejects_overlap_and_lookahead():
    with pytest.raises(ValueError):
        M.SeasonSplit((2016, 2017), 2017, 2018)  # trains on the calibration season
    with pytest.raises(ValueError):
        M.SeasonSplit((2016,), 2018, 2018)  # calibrates on the test season
    with pytest.raises(ValueError):
        M.SeasonSplit((2016, 2019), 2018, 2019)  # trains on the test season
    with pytest.raises(ValueError):
        M.SeasonSplit((), 2017, 2018)


def test_production_split():
    split = M.production_split(2026, range(2016, 2026))
    assert split == M.SeasonSplit(tuple(range(2016, 2025)), 2025, 2026)
    with pytest.raises(M.ModelError, match="calibration"):
        M.production_split(2026, range(2016, 2025))


# --------------------------------------------------------------------------- #
# Leakage at the model level
# --------------------------------------------------------------------------- #
def _flip_labels(frame: pl.DataFrame, season: int, weeks: tuple[int, ...] | None = None) -> pl.DataFrame:
    target = pl.col("season") == season
    if weeks is not None:
        target = target & pl.col("week").is_in(list(weeks))
    return frame.with_columns(
        pl.when(target).then(1 - pl.col("home_win")).otherwise(pl.col("home_win")).alias("home_win")
    )


def test_backtest_predictions_ignore_test_season_labels(frame):
    small = frame.filter(pl.col("season") <= 2021)
    base = M.walk_forward_backtest(small).predictions
    flipped_test = M.walk_forward_backtest(_flip_labels(small, 2020)).predictions
    flipped_cal = M.walk_forward_backtest(_flip_labels(small, 2019)).predictions

    def season_preds(df: pl.DataFrame, season: int) -> np.ndarray:
        return df.filter(pl.col("season") == season).sort("game_id")["final_home_wp"].to_numpy()

    # Flipping 2020 outcomes leaves the 2020 out-of-sample predictions untouched...
    np.testing.assert_array_equal(season_preds(base, 2020), season_preds(flipped_test, 2020))
    # ...and 2019 predictions too: they come from a fold that ends before 2020.
    np.testing.assert_array_equal(season_preds(base, 2019), season_preds(flipped_test, 2019))
    # Sensitivity: 2019 is the calibration season for 2020, so flipping it must move 2020.
    assert not np.allclose(season_preds(base, 2020), season_preds(flipped_cal, 2020))


def test_crossfit_scores_each_fold_with_the_other_folds_calibrator(frame):
    data = M.labeled_rows(frame)
    train, cal = data.filter(pl.col("season") <= 2018), data.filter(pl.col("season") == 2019)
    model = M.WinProbabilityModel().fit(train, cal)
    even = (cal["week"] % 2 == 0).to_numpy()
    base = model.crossfit_calibrated(cal)
    flipped = model.crossfit_calibrated(_flip_labels(cal, 2019, weeks=tuple(range(2, 19, 2))))
    np.testing.assert_array_equal(base[even], flipped[even])  # even weeks never calibrate themselves
    assert not np.allclose(base[~even], flipped[~even])  # ...but they do calibrate the odd weeks


def test_backtest_report_structure(frame):
    result = M.walk_forward_backtest(frame.filter(pl.col("season") <= 2020))
    assert result.report["test_seasons"] == [2018, 2019, 2020]
    assert set(result.predictions["season"].unique()) == {2018, 2019, 2020}
    for fold in result.report["folds"]:
        assert 0.0 <= fold["market_weight"] <= 1.0
        assert fold["test"]["final"]["n_games"] == 144
    json.dumps(result.report, allow_nan=False)


# --------------------------------------------------------------------------- #
# Market blend
# --------------------------------------------------------------------------- #
def test_fit_market_weight_recovers_true_mixture():
    rng = np.random.default_rng(11)
    n = 60_000
    market, model = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
    y = (rng.random(n) < 0.7 * market + 0.3 * model).astype(float)
    blend = M.fit_market_weight(model, market, y)
    assert blend.source == "optimized"
    assert blend.weight == pytest.approx(0.7, abs=0.03)
    assert blend.brier_final <= min(blend.brier_model, blend.brier_market)


def test_fit_market_weight_defaults_and_bounds():
    rng = np.random.default_rng(5)
    few = M.fit_market_weight(rng.random(20), rng.random(20), rng.integers(0, 2, 20))
    assert (few.weight, few.source) == (0.60, "default")
    none = M.fit_market_weight(rng.random(200), np.full(200, np.nan), rng.integers(0, 2, 200))
    assert (none.weight, none.source, none.n_games) == (0.60, "default", 0)
    # A perfect market drives the weight to the upper bound, never past it.
    y = rng.integers(0, 2, 500).astype(float)
    perfect = M.fit_market_weight(np.full(500, 0.5), np.where(y == 1, 0.99, 0.01), y)
    assert perfect.weight == pytest.approx(1.0, abs=1e-6)


def test_blend_probabilities_keeps_model_without_market():
    out = M.blend_probabilities(np.array([0.7, 0.4]), np.array([0.5, np.nan]), 0.6)
    np.testing.assert_allclose(out, [0.6 * 0.5 + 0.4 * 0.7, 0.4])


def test_nflverse_market_probs_matches_ingest_devig():
    schedules = pl.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "home_moneyline": [-150, 120, 50],
            "away_moneyline": [130, -140, -60],
            "spread_line": [3.0, -2.0, 1.0],
        }
    )
    out = M.nflverse_market_probs(schedules)
    expected_a = ingest.devig_multiplicative([ingest.implied_probability(-150), ingest.implied_probability(130)])[0]
    assert out["market_home_wp"][0] == pytest.approx(expected_a)
    assert out["market_home_wp"][1] == pytest.approx(
        ingest.devig_multiplicative([ingest.implied_probability(120), ingest.implied_probability(-140)])[0]
    )
    assert out["market_home_wp"][2] is None  # +50 is not a valid American price
    assert out["market_home_spread"].to_list() == [-3.0, 2.0, -1.0]  # sportsbook sign
    assert out["market_source"].to_list() == ["nflverse_close", "nflverse_close", None]


def test_metrics():
    y = np.array([1, 0, 1, 0])
    perfect = M.probability_metrics(y, y.astype(float))
    assert perfect["brier"] == 0.0 and perfect["accuracy"] == 1.0 and perfect["auc"] == 1.0
    coin = M.probability_metrics(y, np.full(4, 0.5))
    assert coin["brier"] == pytest.approx(0.25) and coin["log_loss"] == pytest.approx(math.log(2))
    assert M.probability_metrics(np.array([]), np.array([]))["brier"] is None


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("win_prob", "edge", "tier"),
    [
        (0.80, None, "LOCK"),
        (0.75, -9.0, "LOCK"),
        (0.62, 6.0, "VALUE EDGE"),
        (0.52, -5.0, "VALUE EDGE"),
        (0.62, 4.9, "LEAN"),
        (0.55, None, "LEAN"),
        (0.54, 1.0, "COIN FLIP"),
        (0.50, None, "COIN FLIP"),
    ],
)
def test_confidence_tier(win_prob, edge, tier):
    assert M.confidence_tier(win_prob, edge) == tier


# --------------------------------------------------------------------------- #
# Contract B payload
# --------------------------------------------------------------------------- #
def test_payload_is_valid_contract_b(payload):
    M.validate_contract_b(payload)
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text) == payload
    assert (payload["contract"], payload["season"], payload["week"]) == ("B", 2026, 6)
    assert payload["model"]["train_seasons"] == list(range(2016, 2025))
    assert payload["model"]["calibration_season"] == 2025
    assert len(payload["games"]) == 8
    for game in payload["games"]:
        assert game["status"] == "scheduled"
        assert game["confidence_tier"] in M.TIERS
        assert 1 <= len(game["top_drivers"]) <= M.TOP_DRIVERS
        side = game["pick"]["side"]
        assert game["final"][f"{side}_wp"] >= 0.5
        if game["market"] is not None:
            expected = 100 * (game["model"][f"{side}_wp"] - game["market"][f"{side}_wp"])
            assert game["model_edge_pct"] == pytest.approx(expected, abs=0.02)
            w = payload["model"]["market_weight"]
            blended = w * game["market"]["home_wp"] + (1 - w) * game["model"]["home_wp"]
            assert game["final"]["home_wp"] == pytest.approx(blended, abs=1e-3)


def test_game_without_market_line_uses_model_only(payload):
    no_line = [g for g in payload["games"] if g["market"] is None]
    assert len(no_line) == 1
    game = no_line[0]
    assert game["final"] == game["model"]
    assert game["model_edge_pct"] is None and game["value_team"] is None
    assert game["confidence_tier"] != "VALUE EDGE"


def test_drivers_are_aligned_with_the_true_signal(frame):
    # The synthetic outcome depends only on the two net EPA mismatches.
    data = M.labeled_rows(frame)
    fold = M.fit_fold(data, M.production_split(2025, data["season"].unique().to_list()))
    held_out = data.filter(pl.col("season") == 2025)
    contributions = fold.model.contributions(held_out)
    importance = np.abs(contributions).mean(axis=0)
    ranked = [F.FEATURE_COLUMNS[i] for i in np.argsort(-importance)[:2]]
    assert set(ranked) == {"net_pass_epa_mismatch", "net_rush_epa_mismatch"}

    j = F.FEATURE_COLUMNS.index("net_pass_epa_mismatch")
    assert np.corrcoef(held_out["net_pass_epa_mismatch"].to_numpy(), contributions[:, j])[0, 1] > 0.8

    # Where the signal is strong it is the top driver, and it favours the right side.
    values = M.to_matrix(held_out)
    checked = 0
    for i, row in enumerate(held_out.iter_rows(named=True)):
        if abs(row["net_pass_epa_mismatch"]) < 0.3:
            continue
        top = M.top_drivers(contributions[i], values[i], row["home_team"], row["away_team"])[0]
        assert top["feature"] == "net_pass_epa_mismatch"
        assert top["favors"] == (row["home_team"] if row["net_pass_epa_mismatch"] > 0 else row["away_team"])
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["games"][0].update(confidence_tier="SMASH"), "confidence_tier"),
        (lambda p: p["games"][0]["final"].update(home_wp=0.9, away_wp=0.9), "equal 1"),
        (lambda p: p["games"][0]["pick"].update(team="XXX"), "pick.team"),
        (lambda p: p["games"].append(copy.deepcopy(p["games"][0])), "unique"),
        (lambda p: p["games"][0].pop("top_drivers"), "missing"),
        (lambda p: p.update(contract="A"), "contract"),
        (lambda p: p["model"].update(market_weight=1.5), "market_weight"),
    ],
)
def test_validator_rejects_broken_payloads(payload, mutate, message):
    broken = copy.deepcopy(payload)
    mutate(broken)
    with pytest.raises(M.ContractError, match=message):
        M.validate_contract_b(broken)


# --------------------------------------------------------------------------- #
# Current-week market source priority
# --------------------------------------------------------------------------- #
def _consensus(game_ids: list[str], home_wp: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": game_ids,
            "market_home_wp": [home_wp] * len(game_ids),
            "consensus_home_spread": [-3.0] * len(game_ids),
            "n_books_h2h": pl.Series([7] * len(game_ids), dtype=pl.UInt32),
            "fetched_at": pl.Series([NOW] * len(game_ids), dtype=pl.Datetime("us", "UTC")),
        }
    )


def test_current_market_priority(tmp_path, monkeypatch, league):
    features, schedules = league
    cfg = dataclasses.replace(Settings.from_env(project_root=tmp_path), odds_api_key=None)
    week_games = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))
    ids = week_games["game_id"].to_list()

    snapshot_dir = cfg.odds_dir / "2026"
    snapshot_dir.mkdir(parents=True)
    _consensus(ids[:2], 0.61).write_parquet(snapshot_dir / "odds_2026_wk06_20261014T120000Z_consensus.parquet")
    _consensus(ids[:1], 0.66).write_parquet(snapshot_dir / "odds_2026_wk06_20261015T120000Z_consensus.parquet")

    offline = M.current_market_probs(week_games, schedules, cfg, use_odds_api=False, now=NOW).sort("game_id")
    by_id = {r["game_id"]: r for r in offline.iter_rows(named=True)}
    assert by_id[ids[0]]["market_source"] == "odds_api_snapshot"
    assert by_id[ids[0]]["market_home_wp"] == pytest.approx(0.66)  # newest snapshot wins
    assert by_id[ids[1]]["market_source"] == "nflverse_close"  # not in the newest snapshot
    assert {by_id[i]["market_source"] for i in ids[2:]} == {"nflverse_close"}

    monkeypatch.setattr(ingest, "fetch_market_win_probs", lambda **_: _consensus(ids, 0.55))
    live = M.current_market_probs(week_games, schedules, cfg, use_odds_api=True, now=NOW)
    assert set(live["market_source"]) == {"odds_api"}
    assert live["market_home_wp"].to_list() == pytest.approx([0.55] * len(ids))

    def broken(**_):
        raise ingest.OddsAPIQuotaError("out of credits")

    monkeypatch.setattr(ingest, "fetch_market_win_probs", broken)
    degraded = M.current_market_probs(week_games, schedules, cfg, use_odds_api=True, now=NOW)
    assert "odds_api" not in set(degraded["market_source"])
    assert degraded.height == len(ids)
