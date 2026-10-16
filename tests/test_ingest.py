"""Offline tests for src.ingest: odds math, cache policy, and API error handling."""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import nflreadpy as nfl
import polars as pl
import pytest
import requests

import src.ingest as ingest
from config import DATASETS, Settings
from src.ingest import (
    MissingAPIKeyError,
    NflverseCache,
    NflverseDownloadError,
    OddsAPIAuthError,
    OddsAPIClient,
    OddsAPIError,
    OddsAPIQuotaError,
    OddsAPIRateLimitError,
)

NOW = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)  # a Friday


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Settings:
    for var in ("NFL_DATA_DIR", "NFL_MODELS_DIR", "NFL_LOGS_DIR", "ODDS_API_KEY", "NFL_START_SEASON"):
        monkeypatch.delenv(var, raising=False)
    base = Settings.from_env(project_root=tmp_path)
    return dataclasses.replace(base, start_season=2024, odds_api_key="test-key", odds_backoff_s=0.0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda _s: None)


# --------------------------------------------------------------------------- #
# Odds math
# --------------------------------------------------------------------------- #
def test_american_odds_conversion():
    assert ingest.american_to_decimal(150) == pytest.approx(2.5)
    assert ingest.american_to_decimal(-200) == pytest.approx(1.5)
    assert ingest.implied_probability(-110) == pytest.approx(110 / 210)
    assert ingest.implied_probability(2.5, "decimal") == pytest.approx(0.4)
    with pytest.raises(ValueError):
        ingest.american_to_decimal(50)


def test_devig_multiplicative():
    raw = [ingest.implied_probability(-150), ingest.implied_probability(130)]  # 0.6, 0.4348
    fair = ingest.devig_multiplicative(raw)
    assert math.fsum(fair) == pytest.approx(1.0)
    assert fair[0] == pytest.approx(0.6 / (0.6 + 100 / 230))
    assert ingest.devig_multiplicative([0.5238, 0.5238]) == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        ingest.devig_multiplicative([0.5])
    with pytest.raises(ValueError):
        ingest.devig_multiplicative([0.5, 0.0])


def test_spread_implied_win_prob():
    assert ingest.spread_implied_win_prob(0.0, 0.5, 13.45) == pytest.approx(0.5)
    favorite = ingest.spread_implied_win_prob(-3.5, 0.5, 13.45)
    assert favorite == pytest.approx(ingest._STANDARD_NORMAL.cdf(3.5 / 13.45))
    # Mirror image from the away side sums to one.
    assert favorite + ingest.spread_implied_win_prob(3.5, 0.5, 13.45) == pytest.approx(1.0)
    # Paying more juice to cover means a stronger favourite.
    assert ingest.spread_implied_win_prob(-3.5, 0.55, 13.45) > favorite


# --------------------------------------------------------------------------- #
# Odds parsing / consensus / schedule matching
# --------------------------------------------------------------------------- #
def _book(key: str, h2h: tuple[int, int] | None, spread: tuple[float, int, int] | None, home: str, away: str):
    markets = []
    if h2h:
        markets.append({"key": "h2h", "outcomes": [{"name": home, "price": h2h[0]}, {"name": away, "price": h2h[1]}]})
    if spread:
        point, home_price, away_price = spread
        markets.append(
            {
                "key": "spreads",
                "outcomes": [
                    {"name": home, "price": home_price, "point": point},
                    {"name": away, "price": away_price, "point": -point},
                ],
            }
        )
    return {"key": key, "title": key, "markets": markets}


KC, BUF, DAL, PHI = "Kansas City Chiefs", "Buffalo Bills", "Dallas Cowboys", "Philadelphia Eagles"
EVENTS: list[dict[str, Any]] = [
    {
        "id": "evt1",
        "commence_time": "2026-10-18T20:25:00Z",
        "home_team": KC,
        "away_team": BUF,
        "bookmakers": [
            _book("draftkings", (-150, 130), (-3.0, -110, -110), KC, BUF),
            _book("fanduel", (-140, 120), (-2.5, -115, -105), KC, BUF),
            _book("betmgm", (-145, 125), None, KC, BUF),
        ],
    },
    {  # spreads only -> win probability falls back to the spread
        "id": "evt2",
        "commence_time": "2026-10-19T00:20:00Z",
        "home_team": PHI,
        "away_team": DAL,
        "bookmakers": [_book("draftkings", None, (-6.5, -110, -110), PHI, DAL)],
    },
]


def _fair_home(home_price: int, away_price: int) -> float:
    h, a = ingest.implied_probability(home_price), ingest.implied_probability(away_price)
    return h / (h + a)


def test_parse_and_consensus():
    books = ingest.parse_odds_events(EVENTS)
    assert books.height == 6  # 3 h2h + 2 spreads for evt1, 1 spread for evt2
    assert set(books["home_team"]) == {"KC", "PHI"}
    assert (books["home_fair_prob"] + books["away_fair_prob"]).to_list() == pytest.approx([1.0] * 6)

    consensus = ingest.build_consensus(books, method="median", sigma=13.45)
    kc = consensus.filter(pl.col("event_id") == "evt1").row(0, named=True)
    expected_home = sorted([_fair_home(-150, 130), _fair_home(-140, 120), _fair_home(-145, 125)])[1]
    assert kc["wp_source"] == "h2h"
    assert kc["n_books_h2h"] == 3 and kc["n_books_spread"] == 2
    assert kc["market_home_wp"] == pytest.approx(expected_home, abs=1e-12)
    assert kc["market_home_wp"] + kc["market_away_wp"] == pytest.approx(1.0)
    assert kc["consensus_home_spread"] == pytest.approx(-2.75)
    assert kc["h2h_overround"] > 0

    phi = consensus.filter(pl.col("event_id") == "evt2").row(0, named=True)
    assert phi["wp_source"] == "spread"
    assert phi["n_books_h2h"] == 0
    assert phi["market_home_wp"] == pytest.approx(ingest.spread_implied_win_prob(-6.5, 0.5, 13.45))

    mean = ingest.build_consensus(books, method="mean").filter(pl.col("event_id") == "evt1")
    expected_mean = sum([_fair_home(-150, 130), _fair_home(-140, 120), _fair_home(-145, 125)]) / 3
    assert mean["market_home_wp"][0] == pytest.approx(expected_mean)


def test_parse_skips_malformed_markets():
    events = [
        {
            "id": "bad",
            "commence_time": "2026-10-18T17:00:00Z",
            "home_team": KC,
            "away_team": BUF,
            "bookmakers": [
                {
                    "key": "x",
                    "markets": [{"key": "h2h", "outcomes": [{"name": KC, "price": 50}, {"name": BUF, "price": 120}]}],
                },
                {"key": "y", "markets": [{"key": "h2h", "outcomes": [{"name": KC, "price": -120}]}]},
                {"key": "z", "markets": [{"key": "totals", "outcomes": []}]},
            ],
        }
    ]
    assert ingest.parse_odds_events(events).is_empty()
    assert ingest.build_consensus(ingest.parse_odds_events(events)).is_empty()


def _week_games() -> pl.DataFrame:
    schedules = pl.DataFrame(
        {
            "game_id": ["2026_07_BUF_KC", "2026_07_DAL_PHI", "2026_07_NYJ_NE"],
            "season": [2026, 2026, 2026],
            "game_type": ["REG"] * 3,
            "week": [7, 7, 7],
            "gameday": ["2026-10-18", "2026-10-18", "2026-10-15"],
            "gametime": ["16:25", "20:20", "20:15"],
            "home_team": ["KC", "PHI", "NE"],
            "away_team": ["BUF", "DAL", "NYJ"],
            "result": [None, None, 3],
        }
    )
    return ingest.current_week_games(schedules, NOW, season=2026)


def test_current_week_games():
    games = _week_games()
    assert games["week"].unique().to_list() == [7]
    assert games.height == 3  # the already-played Thursday game stays in the week
    assert games["kickoff_utc"].dtype == pl.Datetime("us", "UTC")
    assert games.filter(pl.col("game_id") == "2026_07_BUF_KC")["kickoff_utc"][0] == datetime(
        2026, 10, 18, 20, 25, tzinfo=UTC
    )


def test_attach_schedule_matches_and_flips_orientation():
    consensus = ingest.build_consensus(ingest.parse_odds_events(EVENTS))
    # Pretend the book lists PHI-DAL with DAL at home (e.g. a neutral-site designation mismatch).
    swapped = consensus.with_columns(
        pl.when(pl.col("event_id") == "evt2").then(pl.lit("DAL")).otherwise(pl.col("home_team")).alias("home_team"),
        pl.when(pl.col("event_id") == "evt2").then(pl.lit("PHI")).otherwise(pl.col("away_team")).alias("away_team"),
        pl.when(pl.col("event_id") == "evt2")
        .then(1 - pl.col("market_home_wp"))
        .otherwise(pl.col("market_home_wp"))
        .alias("market_home_wp"),
        pl.when(pl.col("event_id") == "evt2")
        .then(1 - pl.col("market_away_wp"))
        .otherwise(pl.col("market_away_wp"))
        .alias("market_away_wp"),
        pl.when(pl.col("event_id") == "evt2")
        .then(-pl.col("consensus_home_spread"))
        .otherwise(pl.col("consensus_home_spread"))
        .alias("consensus_home_spread"),
    )
    attached = ingest.attach_schedule(swapped, _week_games())
    phi = attached.filter(pl.col("event_id") == "evt2").row(0, named=True)
    assert phi["game_id"] == "2026_07_DAL_PHI"
    assert (phi["home_team"], phi["away_team"]) == ("PHI", "DAL")
    assert phi["consensus_home_spread"] == pytest.approx(-6.5)
    assert phi["market_home_wp"] > 0.5
    kc = attached.filter(pl.col("event_id") == "evt1").row(0, named=True)
    assert (kc["game_id"], kc["season"], kc["week"]) == ("2026_07_BUF_KC", 2026, 7)


# --------------------------------------------------------------------------- #
# Cache policy
# --------------------------------------------------------------------------- #
def test_last_weekly_refresh(cfg):
    assert ingest.last_weekly_refresh(NOW, cfg) == datetime(2026, 10, 13, 12, tzinfo=UTC)
    tuesday_morning = datetime(2026, 10, 20, 9, tzinfo=UTC)
    assert ingest.last_weekly_refresh(tuesday_morning, cfg) == datetime(2026, 10, 13, 12, tzinfo=UTC)
    tuesday_noon = datetime(2026, 10, 20, 12, tzinfo=UTC)
    assert ingest.last_weekly_refresh(tuesday_noon, cfg) == tuesday_noon


def _schedules(super_bowl_2026_played: bool = False) -> pl.DataFrame:
    rows = []
    for season in (2024, 2025, 2026):
        rows.append((f"{season}_01_A_B", season, "REG", 1, f"{season}-09-08", 7))
        sb_result = 10 if season < 2026 or super_bowl_2026_played else None
        rows.append((f"{season}_22_C_D", season, "SB", 22, f"{season + 1}-02-08", sb_result))
    rows.append(("2026_07_E_F", 2026, "REG", 7, "2026-10-18", None))
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.String,
            "season": pl.Int32,
            "game_type": pl.String,
            "week": pl.Int32,
            "gameday": pl.String,
            "result": pl.Int32,
        },
        orient="row",
    ).with_columns(pl.lit("13:00").alias("gametime"), pl.lit("H").alias("home_team"), pl.lit("A").alias("away_team"))


def test_season_final_after(cfg):
    schedules = _schedules()
    assert ingest.season_final_after(schedules, 2024, 2026, cfg) == datetime(2025, 2, 16, tzinfo=UTC)
    assert ingest.season_final_after(schedules, 2026, 2026, cfg) is None
    assert ingest.season_final_after(_schedules(True), 2026, 2026, cfg) == datetime(2027, 2, 16, tzinfo=UTC)
    assert ingest.season_final_after(None, 2023, 2026, cfg) == datetime(2024, 3, 1, tzinfo=UTC)
    assert ingest.season_final_after(None, 2026, 2026, cfg) is None


def test_staleness_reason(cfg):
    spec = DATASETS["pbp"]

    def entry(fetched_at: datetime) -> ingest.CacheEntry:
        return ingest.CacheEntry("pbp", 2026, "pbp/pbp_2026.parquet", fetched_at, 10, 6, False)

    final_after = datetime(2025, 2, 16, tzinfo=UTC)
    assert ingest.staleness_reason(None, False, spec, None, NOW, cfg) == "not cached"
    assert ingest.staleness_reason(entry(NOW), False, spec, None, NOW, cfg) == "not cached"
    # Frozen: fetched after the season was final, however long ago.
    assert ingest.staleness_reason(entry(datetime(2025, 3, 1, tzinfo=UTC)), True, spec, final_after, NOW, cfg) is None
    # Cached mid-season, season is final now -> one more download.
    stale_mid_season = entry(datetime(2024, 12, 1, tzinfo=UTC))
    assert (
        ingest.staleness_reason(stale_mid_season, True, spec, final_after, NOW, cfg)
        == "cached before the season was final"
    )
    # In progress: fresh since Tuesday's boundary, stale before it.
    assert ingest.staleness_reason(entry(NOW - timedelta(days=1)), True, spec, None, NOW, cfg) is None
    assert (
        ingest.staleness_reason(entry(NOW - timedelta(days=4)), True, spec, None, NOW, cfg)
        == "weekly refresh boundary passed"
    )
    # Short-TTL dataset.
    injuries = DATASETS["injuries"]
    assert ingest.staleness_reason(entry(NOW - timedelta(hours=13)), True, injuries, None, NOW, cfg) == "older than 12h"


class FakeNflverse:
    """Stands in for nflreadpy loaders and counts downloads."""

    def __init__(self, schedules: pl.DataFrame, missing: set[tuple[str, int]] | None = None) -> None:
        self.schedules = schedules
        self.missing = missing or set()
        self.calls: list[tuple[str, Any]] = []

    def _not_found(self, name: str, season: int) -> ConnectionError:
        response = requests.Response()
        response.status_code = 404
        try:
            raise requests.HTTPError("404 Client Error", response=response)
        except requests.HTTPError as http_error:
            err = ConnectionError(f"Failed to download {name}_{season}: 404")
            err.__cause__ = http_error
            return err

    def load_schedules(self, seasons: Any) -> pl.DataFrame:
        self.calls.append(("schedules", seasons))
        return self.schedules

    def _per_season(self, name: str, season: int) -> pl.DataFrame:
        self.calls.append((name, season))
        if (name, season) in self.missing:
            raise self._not_found(name, season)
        data: dict[str, Any] = {"season": [season] * 3, "week": [1, 2, 3], "epa": [0.1, -0.2, 0.3]}
        if season >= 2025:
            data["new_col"] = ["a", "b", "c"]  # schema drift between seasons
        return pl.DataFrame(data)

    def load_pbp(self, season: int) -> pl.DataFrame:
        return self._per_season("pbp", season)

    def load_injuries(self, season: int) -> pl.DataFrame:
        return self._per_season("injuries", season)

    def load_rosters_weekly(self, season: int) -> pl.DataFrame:
        return self._per_season("rosters_weekly", season)

    def downloads(self, name: str) -> list[Any]:
        return [season for dataset, season in self.calls if dataset == name]


@pytest.fixture
def fake_nfl(monkeypatch) -> FakeNflverse:
    fake = FakeNflverse(_schedules())
    for loader in ("load_schedules", "load_pbp", "load_injuries", "load_rosters_weekly"):
        monkeypatch.setattr(nfl, loader, getattr(fake, loader))
    monkeypatch.setattr(nfl, "get_current_season", lambda roster=False: 2026)
    monkeypatch.setattr(ingest, "utc_now", lambda: NOW)
    return fake


def test_cache_downloads_completed_seasons_once(cfg, fake_nfl, monkeypatch):
    cache = NflverseCache(cfg)
    entries = cache.sync("pbp")
    assert fake_nfl.downloads("pbp") == [2024, 2025, 2026]
    assert {e.season: e.final for e in entries} == {2024: True, 2025: True, 2026: False}
    assert cfg.manifest_path.exists()
    assert (cfg.raw_dir / "pbp" / "pbp_2024.parquet").exists()
    assert not list((cfg.raw_dir / "pbp").glob("*.tmp"))

    cache.sync("pbp")  # nothing stale
    assert fake_nfl.downloads("pbp") == [2024, 2025, 2026]

    # A week later only the in-progress season is refreshed, from a fresh process.
    monkeypatch.setattr(ingest, "utc_now", lambda: NOW + timedelta(days=7))
    NflverseCache(cfg).sync("pbp")
    assert fake_nfl.downloads("pbp") == [2024, 2025, 2026, 2026]

    cache.sync("pbp", force=True, seasons=[2024])
    assert fake_nfl.downloads("pbp")[-1] == 2024


def test_cache_read_handles_schema_drift_and_columns(cfg, fake_nfl):
    cache = NflverseCache(cfg)
    df = cache.read("pbp")
    assert df.height == 9
    assert df.filter(pl.col("season") == 2024)["new_col"].null_count() == 3
    subset = cache.read("pbp", [2024, 2025], columns=["season", "new_col"], refresh=False)
    assert subset.columns == ["season", "new_col"]
    assert subset.height == 6
    schedules = cache.read("schedules", [2026], refresh=False)
    assert set(schedules["season"]) == {2026}


def test_cache_adopts_files_when_manifest_is_lost(cfg, fake_nfl):
    NflverseCache(cfg).sync("pbp")
    cfg.manifest_path.unlink()
    NflverseCache(cfg).sync("pbp")
    # Final seasons are adopted from disk; their mtime is after the season ended.
    assert fake_nfl.downloads("pbp") == [2024, 2025, 2026]


def test_missing_current_season_file_is_skipped(cfg, fake_nfl):
    fake_nfl.missing = {("injuries", 2026)}
    cache = NflverseCache(cfg)
    entries = cache.sync("injuries")
    assert [e.season for e in entries] == [2024, 2025]
    assert fake_nfl.downloads("injuries") == [2024, 2025, 2026]  # 404 is not retried
    assert cache.refresh_failures == []  # not published yet is expected (preseason), not a failed refresh


def test_failed_refresh_keeps_the_stale_copy_and_is_recorded(cfg, fake_nfl, monkeypatch):
    NflverseCache(cfg).sync_all(["schedules", "pbp"])
    monkeypatch.setattr(ingest, "utc_now", lambda: NOW + timedelta(days=7))  # the weekly refresh is due

    def offline(*_args):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(nfl, "load_pbp", offline)
    monkeypatch.setattr(nfl, "load_schedules", offline)
    cache = NflverseCache(cfg)
    summary = cache.sync_all(["schedules", "pbp"])
    assert [e.season for e in summary["pbp"]] == [2024, 2025, 2026]  # the old 2026 copy is still served
    assert [f.split(":")[0] for f in cache.refresh_failures] == ["schedules", "pbp 2026"]
    assert "network unreachable" in cache.refresh_failures[1]


@pytest.mark.parametrize(
    ("now", "season"),
    [
        (datetime(2026, 9, 9, 23, tzinfo=UTC), 2025),  # Labor Day 2026 is Sep 7; the season starts Thursday Sep 10
        (datetime(2026, 9, 10, tzinfo=UTC), 2026),
        (datetime(2027, 2, 10, tzinfo=UTC), 2026),  # Super Bowl month belongs to the previous season
        (datetime(2025, 9, 4, tzinfo=UTC), 2025),
    ],
)
def test_current_season_follows_now(now, season):
    assert ingest.current_season(now) == season


def test_missing_past_season_file_raises(cfg, fake_nfl):
    fake_nfl.missing = {("injuries", 2024)}
    with pytest.raises(NflverseDownloadError, match="404"):
        NflverseCache(cfg).sync("injuries")


def test_unknown_dataset_and_bad_seasons(cfg, fake_nfl):
    cache = NflverseCache(cfg)
    with pytest.raises(ValueError, match="Unknown dataset"):
        cache.sync("nope")
    with pytest.raises(ValueError, match="invalid seasons"):
        cache.sync("pbp", [2030])


# --------------------------------------------------------------------------- #
# The Odds API client
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = "" if body is None else str(body)
        self.reason = "reason"

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


QUOTA = {"x-requests-remaining": "480", "x-requests-used": "20", "x-requests-last": "2"}


def test_missing_api_key(cfg):
    with pytest.raises(MissingAPIKeyError, match="ODDS_API_KEY"):
        OddsAPIClient(cfg=dataclasses.replace(cfg, odds_api_key=None))
    with pytest.raises(MissingAPIKeyError):
        OddsAPIClient("   ", cfg=cfg)


def test_odds_request_params_and_quota(cfg):
    session = FakeSession(FakeResponse(200, EVENTS, QUOTA))
    client = OddsAPIClient(cfg=cfg, session=session)  # type: ignore[arg-type]
    events = client.get_odds(commence_time_from=NOW, commence_time_to=NOW + timedelta(days=4))
    assert events == EVENTS
    params = session.calls[0]["params"]
    assert session.calls[0]["url"] == "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    assert params["markets"] == "h2h,spreads"
    assert params["oddsFormat"] == "american"
    assert params["bookmakers"].startswith("draftkings,fanduel")
    assert "regions" not in params
    assert params["commenceTimeFrom"] == "2026-10-16T18:00:00Z"
    assert params["apiKey"] == "test-key"
    assert (client.requests_remaining, client.requests_used, client.requests_last) == (480, 20, 2)


def test_rate_limit_is_retried(cfg):
    session = FakeSession(
        FakeResponse(429, {"message": "slow down", "error_code": "EXCEEDED_FREQ_LIMIT"}, {"Retry-After": "1"}),
        FakeResponse(200, [], QUOTA),
    )
    assert OddsAPIClient(cfg=cfg, session=session).get_odds() == []  # type: ignore[arg-type]
    assert len(session.calls) == 2


def test_rate_limit_exhausts_retries(cfg):
    cfg = dataclasses.replace(cfg, odds_max_retries=2)
    session = FakeSession(*[FakeResponse(429, {"message": "slow down", "error_code": "EXCEEDED_FREQ_LIMIT"})] * 3)
    with pytest.raises(OddsAPIRateLimitError) as info:
        OddsAPIClient(cfg=cfg, session=session).get_odds()  # type: ignore[arg-type]
    assert info.value.status_code == 429
    assert len(session.calls) == 3


def test_quota_and_auth_errors(cfg):
    quota = FakeSession(
        FakeResponse(401, {"message": "Usage quota has been reached", "error_code": "OUT_OF_USAGE_CREDITS"})
    )
    with pytest.raises(OddsAPIQuotaError):
        OddsAPIClient(cfg=cfg, session=quota).get_odds()  # type: ignore[arg-type]
    assert len(quota.calls) == 1  # never retried

    bad_key = FakeSession(FakeResponse(401, {"message": "API key is not valid", "error_code": "INVALID_KEY"}))
    with pytest.raises(OddsAPIAuthError, match="rejected"):
        OddsAPIClient(cfg=cfg, session=bad_key).get_odds()  # type: ignore[arg-type]

    bad_params = FakeSession(FakeResponse(422, {"message": "Invalid market", "error_code": "INVALID_MARKET"}))
    with pytest.raises(OddsAPIError, match="422"):
        OddsAPIClient(cfg=cfg, session=bad_params).get_odds()  # type: ignore[arg-type]


def test_network_errors_never_leak_the_key(cfg):
    cfg = dataclasses.replace(cfg, odds_max_retries=1)
    leak = requests.ConnectionError("Max retries exceeded with url: /v4/sports/x/odds?apiKey=test-key&markets=h2h")
    session = FakeSession(leak, leak)
    with pytest.raises(OddsAPIError) as info:
        OddsAPIClient(cfg=cfg, session=session).get_odds()  # type: ignore[arg-type]
    assert "test-key" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_fetch_market_win_probs_end_to_end(cfg, fake_nfl, monkeypatch):
    fake_nfl.schedules = pl.concat([_schedules(), _week_games().drop("kickoff_utc")], how="diagonal_relaxed")
    session = FakeSession(FakeResponse(200, EVENTS, QUOTA))
    client = OddsAPIClient(cfg=cfg, session=session)  # type: ignore[arg-type]

    probs = ingest.fetch_market_win_probs(cfg=cfg, cache=NflverseCache(cfg), client=client, now=NOW)

    assert probs.schema == pl.Schema(ingest.CONSENSUS_SCHEMA)
    assert probs.sort("game_id")["game_id"].to_list() == ["2026_07_BUF_KC", "2026_07_DAL_PHI"]
    assert ((probs["market_home_wp"] + probs["market_away_wp"]) - 1).abs().max() < 1e-12
    params = session.calls[0]["params"]
    assert params["commenceTimeFrom"] == "2026-10-16T18:00:00Z"
    assert params["commenceTimeTo"] == "2026-10-19T06:20:00Z"  # last kickoff + 6h
    snapshots = sorted(p.name for p in (cfg.odds_dir / "2026").glob("*.parquet"))
    assert snapshots == [
        "odds_2026_wk07_20261016T180000Z_books.parquet",
        "odds_2026_wk07_20261016T180000Z_consensus.parquet",
    ]
