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

from config import Settings, settings
from src import features as F
from src import ingest
from src import model as M
from tests import fakes

NOW = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def league(synthetic_league) -> tuple[pl.DataFrame, pl.DataFrame]:
    return synthetic_league


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
    ("win_prob", "edge", "model_pick", "tier"),
    [
        (0.80, None, 0.78, "LOCK"),
        (0.75, -9.0, 0.66, "LOCK"),  # the model still favours the pick
        (0.78, -30.0, 0.45, "LEAN"),  # strong final, but the model picks the other side: not a LOCK
        (0.62, 6.0, 0.68, "VALUE EDGE"),
        (0.52, 5.0, 0.57, "VALUE EDGE"),  # value on the pick, even near a coin flip
        (0.52, -5.0, 0.47, "COIN FLIP"),  # value is on the OTHER side: never upgrades the pick
        (0.62, -12.0, 0.50, "LEAN"),
        (0.62, 4.99, 0.67, "LEAN"),
        (0.55, None, 0.55, "LEAN"),
        (0.54, 1.0, 0.55, "COIN FLIP"),
        (0.50, None, 0.50, "COIN FLIP"),
    ],
)
def test_confidence_tier(win_prob, edge, model_pick, tier):
    five_points = M.TierThresholds(lock_min_win_prob=0.75, value_edge_min_pct=5.0, lean_min_win_prob=0.55)
    assert M.confidence_tier(win_prob, edge, model_pick, five_points) == tier


@pytest.mark.parametrize(
    ("win_prob", "edge", "model_pick", "tier"),
    [
        (0.62, 6.0, 0.68, "LEAN"),  # a VALUE EDGE at 5 points is calibration noise at the 8-point default
        (0.52, 7.99, 0.60, "COIN FLIP"),
        (0.52, 8.0, 0.60, "VALUE EDGE"),
        (0.66, 12.5, 0.785, "VALUE EDGE"),
    ],
)
def test_default_value_edge_threshold_is_eight_points(win_prob, edge, model_pick, tier):
    assert settings.tier_value_edge_min_pct == 8.0
    assert M.TierThresholds.from_settings().value_edge_min_pct == 8.0
    assert M.confidence_tier(win_prob, edge, model_pick) == tier


def test_tier_thresholds_come_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_VALUE_EDGE_MIN_PCT", "6.5")
    monkeypatch.setenv("NFL_TIER_LOCK_MIN_WIN_PROB", "0.8")
    cfg = Settings.from_env(project_root=tmp_path)
    assert M.TierThresholds.from_settings(cfg) == M.TierThresholds(0.8, 6.5, 0.55)
    monkeypatch.setenv("NFL_TIER_LEAN_MIN_WIN_PROB", "0.9")  # lean above lock is rejected
    with pytest.raises(ValueError, match="Tier thresholds"):
        Settings.from_env(project_root=tmp_path)


def test_payload_records_and_is_validated_against_its_own_thresholds(league):
    features, schedules = league
    market = M.nflverse_market_probs(schedules)
    loose = M.TierThresholds(0.75, 1.0, 0.55)
    payload = M.predict_week(features, schedules, market, season=2026, week=6, now=NOW, thresholds=loose)
    assert payload["model"]["tier_thresholds"] == {
        "lock_min_win_prob": 0.75,
        "value_edge_min_pct": 1.0,
        "lean_min_win_prob": 0.55,
    }
    assert "1 percentage points" in payload["tier_rules"]["VALUE EDGE"]
    M.validate_contract_b(payload)  # built under 1.0, valid under 1.0 even though settings say 8.0
    tampered = copy.deepcopy(payload)
    tampered["model"]["tier_thresholds"]["value_edge_min_pct"] = 50.0  # tiers no longer follow the stated rule
    with pytest.raises(M.ContractError, match="tier_rules must describe|contradicts"):
        M.validate_contract_b(tampered)
    broken = copy.deepcopy(payload)
    broken["model"]["tier_thresholds"] = {"lock_min_win_prob": 0.5, "value_edge_min_pct": 8.0, "lean_min_win_prob": 0.7}
    with pytest.raises(M.ContractError, match="tier_thresholds"):
        M.validate_contract_b(broken)


def test_tiers_in_payload_are_directional(payload):
    for game in payload["games"]:
        side = game["pick"]["side"]
        if game["confidence_tier"] == "VALUE EDGE":
            assert game["value_team"] == game["pick"]["team"]
            assert game["model_edge_pct"] >= payload["model"]["tier_thresholds"]["value_edge_min_pct"]
        if game["confidence_tier"] == "LOCK":
            assert game["model"][f"{side}_wp"] >= 0.5
        if game["value_team"] is not None and game["value_team"] != game["pick"]["team"]:
            assert game["model_edge_pct"] < 0 and game["confidence_tier"] != "VALUE EDGE"


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
        (
            lambda p: p["games"][0].update(
                confidence_tier="LOCK" if p["games"][0]["confidence_tier"] != "LOCK" else "COIN FLIP"
            ),
            "contradicts",
        ),
        (lambda p: p["games"][0]["final"].update(home_wp=0.99, away_wp=0.01), "blend|pick"),
        (lambda p: p["games"][0].update(model_edge_pct=55.0), "model_edge_pct"),
        (
            lambda p: p["games"][0].update(
                value_team=p["games"][0]["home_team"]
                if p["games"][0]["value_team"] != p["games"][0]["home_team"]
                else p["games"][0]["away_team"]
            ),
            "value_team",
        ),
        (lambda p: p.update(season=True), "season"),
        (lambda p: p["games"][0].update(market="odds_api"), "market must be an object"),
        (lambda p: p["games"][0]["pick"].update(win_prob=None), "pick.win_prob"),
        (lambda p: p["model"].update(train_seasons=[*p["model"]["train_seasons"], p["season"]]), "walk-forward"),
        (lambda p: p["model"].update(calibration_season=p["season"]), "calibration_season"),
        (lambda p: p["games"][0].update(week=p["week"] + 1), "week must equal"),
        (lambda p: p["games"][0].update(kickoff_utc="2026-10-01T17:00:00Z"), "status inconsistent"),
        (lambda p: p["games"][0]["market"].update(as_of="2027-01-01T00:00:00Z"), "as_of"),
        (
            lambda p: p["games"][0]["top_drivers"][0].update(favors=p["games"][0]["top_drivers"][0]["favors"] + "X"),
            "driver",
        ),
        (lambda p: p["games"][0].update(imputed_inputs=["home_qb_cpoe_pre"]), "imputed_inputs"),
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
def _consensus(game_ids: list[str], home_wp: float, *, spread_only: tuple[str, ...] = ()) -> pl.DataFrame:
    n = len(game_ids)
    return pl.DataFrame(
        {
            "game_id": game_ids,
            "market_home_wp": [home_wp] * n,
            "wp_source": ["spread" if g in spread_only else "h2h" for g in game_ids],
            "consensus_home_spread": [-3.0] * n,
            "n_books_h2h": pl.Series([0 if g in spread_only else 7 for g in game_ids], dtype=pl.UInt32),
            "n_books_spread": pl.Series([3] * n, dtype=pl.UInt32),
            "fetched_at": pl.Series([NOW] * n, dtype=pl.Datetime("us", "UTC")),
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
    # Not in the newest snapshot -> nflverse. These games have not kicked off, so it is a pre-game line, not a close.
    assert by_id[ids[1]]["market_source"] == "nflverse_line"
    assert {by_id[i]["market_source"] for i in ids[2:]} == {"nflverse_line"}
    assert {by_id[i]["market_method"] for i in ids[1:]} == {"nflverse_moneyline"}
    # A line counts as a close only if it was observed after kickoff (schedules fetched after the game began).
    early_fetch, late_fetch = NOW - timedelta(days=1), NOW + timedelta(days=6)
    stale = M.current_market_probs(
        week_games,
        schedules,
        cfg,
        use_odds_api=False,
        now=NOW + timedelta(days=5),
        schedules_fetched_at_override=early_fetch,
    )
    nflverse = stale.filter(pl.col("market_source").str.starts_with("nflverse"))
    assert set(nflverse["market_source"]) == {"nflverse_line"}  # kicked off since, but the line predates kickoff
    assert set(nflverse["market_as_of"].to_list()) == {early_fetch}
    closes = M.current_market_probs(
        week_games, schedules, cfg, use_odds_api=False, now=late_fetch, schedules_fetched_at_override=late_fetch
    ).filter(pl.col("market_source").str.starts_with("nflverse"))
    assert set(closes["market_source"]) == {"nflverse_close"} and closes["market_as_of"].null_count() == closes.height
    future = M.current_market_probs(
        week_games, schedules, cfg, use_odds_api=False, now=NOW, schedules_fetched_at_override=NOW + timedelta(hours=3)
    ).filter(pl.col("market_source") == "nflverse_line")
    assert all(ts <= NOW for ts in future["market_as_of"].to_list())  # never dated after the prediction

    monkeypatch.setattr(ingest, "fetch_market_win_probs", lambda **_: _consensus(ids, 0.55, spread_only=(ids[0],)))
    live = M.current_market_probs(week_games, schedules, cfg, use_odds_api=True, now=NOW)
    assert set(live["market_source"]) == {"odds_api"}
    assert live["market_home_wp"].to_list() == pytest.approx([0.55] * len(ids))
    spread_row = live.filter(pl.col("game_id") == ids[0]).row(0, named=True)
    assert (spread_row["market_method"], spread_row["market_books"]) == ("spread_implied", 3)  # not 0 h2h books
    assert set(live.filter(pl.col("game_id") != ids[0])["market_method"]) == {"h2h"}

    def broken(**_):
        raise ingest.OddsAPIQuotaError("out of credits")

    monkeypatch.setattr(ingest, "fetch_market_win_probs", broken)
    degraded = M.current_market_probs(week_games, schedules, cfg, use_odds_api=True, now=NOW)
    assert "odds_api" not in set(degraded["market_source"])
    assert degraded.height == len(ids)


def test_validator_reports_non_object_payload_as_contract_error():
    with pytest.raises(M.ContractError, match="JSON object"):
        M.validate_contract_b([1, 2, 3])


# --------------------------------------------------------------------------- #
# Pre-game-unknown venue inputs (train/serve consistency)
# --------------------------------------------------------------------------- #
def test_unknown_venue_inputs_predict_like_their_default(frame):
    data = M.labeled_rows(frame)
    fitted = M.WinProbabilityModel().fit(data.filter(pl.col("season") <= 2018), data.filter(pl.col("season") == 2019))
    rows = data.filter(pl.col("season") == 2020).head(40)
    unknown = rows.with_columns(
        pl.lit(None, pl.Int8).alias("is_high_wind_outdoor"), pl.lit(None, pl.Int8).alias("is_outdoor")
    )
    defaulted = rows.with_columns(
        pl.lit(0, pl.Int8).alias("is_high_wind_outdoor"), pl.lit(0, pl.Int8).alias("is_outdoor")
    )
    np.testing.assert_array_equal(fitted.predict_proba(unknown), fitted.predict_proba(defaulted))
    assert np.isnan(M.to_matrix(unknown, raw=True)[:, F.FEATURE_COLUMNS.index("is_outdoor")]).all()


def test_drivers_skip_unobserved_inputs():
    columns = ["is_high_wind_outdoor", "net_pass_epa_mismatch", "net_rush_epa_mismatch"]
    drivers = M.top_drivers(
        np.array([-0.9, 0.4, -0.2]), np.array([np.nan, 0.12, -0.05]), "KC", "BUF", feature_columns=columns, k=3
    )
    assert [d["feature"] for d in drivers] == ["net_pass_epa_mismatch", "net_rush_epa_mismatch"]
    assert [d["favors"] for d in drivers] == ["KC", "BUF"]


def test_payload_reports_imputed_inputs(league):
    features, schedules = league
    week_ids = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))["game_id"].to_list()
    windless = features.with_columns(
        pl.when(pl.col("game_id") == week_ids[0])
        .then(None)
        .otherwise(pl.col("is_high_wind_outdoor"))
        .alias("is_high_wind_outdoor")
    )
    payload = M.predict_week(windless, schedules, M.nflverse_market_probs(schedules), season=2026, week=6, now=NOW)
    game = next(g for g in payload["games"] if g["game_id"] == week_ids[0])
    assert "is_high_wind_outdoor" in game["imputed_inputs"]
    assert all(d["feature"] != "is_high_wind_outdoor" for d in game["top_drivers"])


def test_failed_live_fetch_after_cache_refresh_does_not_abort(tmp_path, monkeypatch, league):
    """Regression: a schedules refresh during the live fetch must not stamp nflverse lines after `now`."""
    features, schedules = league
    cfg = dataclasses.replace(Settings.from_env(project_root=tmp_path), odds_api_key=None)
    week_games = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))
    read_at = NOW - timedelta(hours=20)

    def refresh_then_fail(**_):
        # what fetch_market_win_probs does when the schedules cache is stale: refresh it (stamped later), then fail
        manifest = ingest.CacheManifest(cfg.manifest_path)
        manifest.put(
            ingest.CacheEntry(
                "schedules", None, "schedules/schedules.parquet", NOW + timedelta(minutes=5), 1, None, False
            )
        )
        raise ingest.OddsAPIRateLimitError("rate limited")

    monkeypatch.setattr(ingest, "fetch_market_win_probs", refresh_then_fail)
    market = M.current_market_probs(
        week_games, schedules, cfg, use_odds_api=True, now=NOW, schedules_fetched_at_override=read_at
    )
    assert set(market["market_source"]) == {"nflverse_line"}
    assert set(market["market_as_of"].to_list()) == {read_at}
    M.predict_week(features, schedules, market, season=2026, week=6, now=NOW)  # validates; previously raised


@pytest.mark.skipif(
    not (settings.features_path.exists() and (settings.raw_dir / "schedules" / "schedules.parquet").exists()),
    reason="needs the real cache",
)
def test_run_current_week_accepts_naive_datetimes():
    payload = M.run_current_week(
        now=datetime(2026, 10, 16, 18), use_odds_api=False, rebuild_features=False, write=False
    )
    assert payload["generated_at"] == "2026-10-16T18:00:00Z"


def test_snapshots_from_after_now_are_ignored(tmp_path, league):
    """Replays/backfills must not see market prices saved after the prediction time."""
    features, schedules = league
    cfg = dataclasses.replace(Settings.from_env(project_root=tmp_path), odds_api_key=None)
    week_games = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))
    ids = week_games["game_id"].to_list()
    folder = cfg.odds_dir / "2026"
    folder.mkdir(parents=True)
    _consensus(ids, 0.61).with_columns(pl.lit(NOW - timedelta(hours=1)).alias("fetched_at")).write_parquet(
        folder / "odds_2026_wk06_20261016T170000Z_consensus.parquet"
    )
    _consensus(ids, 0.77).write_parquet(folder / "odds_2026_wk06_20261016T190000Z_consensus.parquet")  # after NOW
    assert M.latest_odds_snapshot(cfg, 2026, 6, NOW)["market_home_wp"][0] == pytest.approx(0.61)
    assert M.latest_odds_snapshot(cfg, 2026, 6)["market_home_wp"][0] == pytest.approx(0.77)
    market = M.current_market_probs(week_games, schedules, cfg, use_odds_api=False, now=NOW)
    assert market["market_home_wp"].to_list() == pytest.approx([0.61] * len(ids))


# --------------------------------------------------------------------------- #
# Week selection (current week, replays) and run_week outputs
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def nflverse_schedules(league) -> pl.DataFrame:
    features, moneylines = league
    return fakes.nflverse_schedules(features, moneylines)


def test_resolve_week_defaults_to_the_current_week(nflverse_schedules):
    target = M.resolve_week(nflverse_schedules, NOW)
    assert (target.season, target.week, target.is_current) == (2026, 6, True)
    assert target.games.height == 8 and target.games["kickoff_utc"].is_sorted()
    assert M.resolve_week(nflverse_schedules, NOW, season=2026) == target


def test_resolve_week_replays_past_weeks_and_rejects_future_ones(nflverse_schedules):
    past = M.resolve_week(nflverse_schedules, NOW, week=5)
    assert (past.season, past.week, past.is_current) == (2026, 5, False)
    old = M.resolve_week(nflverse_schedules, NOW, season=2024, week=18)
    assert (old.season, old.week, old.is_current) == (2024, 18, False)
    with pytest.raises(M.ModelError, match="week 7 is in the future; the current week is 6"):
        M.resolve_week(nflverse_schedules, NOW, week=7)
    with pytest.raises(M.ModelError, match="no 2031 games"):
        M.resolve_week(nflverse_schedules, NOW, season=2031)
    with pytest.raises(M.ModelError, match="2025 season has no upcoming games"):
        M.resolve_week(nflverse_schedules, NOW, season=2025)
    with pytest.raises(M.ModelError, match="no games scheduled in 2025 week 21"):
        M.resolve_week(nflverse_schedules, NOW, season=2025, week=21)


def test_resolve_week_season_follows_now_not_the_wall_clock(nflverse_schedules):
    """A pinned `now` picks its own season, so replays and tests don't depend on today's date."""
    with pytest.raises(M.ModelError, match="the 2025 season has no upcoming games"):
        M.resolve_week(nflverse_schedules, datetime(2026, 9, 1, tzinfo=UTC))  # before the 2026 season starts
