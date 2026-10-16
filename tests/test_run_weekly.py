"""Tests for run_weekly.py: stage order, flag handling, exit codes and failure short-circuiting.

Integration tests run the real Modules 3-4 on the synthetic league through a temporary raw cache (Module 2 is
replaced by the synthetic feature matrix, Module 1 by a no-op sync). Unit tests stub the stage functions.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import run_weekly as W
from config import Settings, settings
from src import features as F
from src import ingest
from src import model as M
from src import report_generator as R
from tests import fakes

NOW = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)  # Friday of 2026 week 6 in the synthetic league
STAGES = ("ingest", "features", "model", "report")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda _s: None)


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return dataclasses.replace(
        Settings.from_env(project_root=tmp_path), odds_api_key=None, gemini_api_key=None, llm_max_attempts=2
    )


@pytest.fixture
def odds_cfg(cfg) -> Settings:
    return dataclasses.replace(cfg, odds_api_key="test-odds-key")


class Calls:
    """Records which pipeline pieces ran (and how often the paid APIs were called)."""

    def __init__(self) -> None:
        self.syncs = 0
        self.odds_fetches = 0


@pytest.fixture
def league(cfg, synthetic_league, monkeypatch) -> Calls:
    """Temporary raw cache with the synthetic schedules; Module 2 writes the synthetic matrix; sync is a no-op."""
    features, moneylines = synthetic_league
    schedules = fakes.nflverse_schedules(features, moneylines)
    path = cfg.raw_dir / "schedules" / "schedules.parquet"
    path.parent.mkdir(parents=True)
    schedules.write_parquet(path)
    ingest.CacheManifest(cfg.manifest_path).put(
        ingest.CacheEntry(
            "schedules", None, "schedules/schedules.parquet", NOW - timedelta(hours=2), schedules.height, None, False
        )
    )
    calls = Calls()

    def sync_all(self, datasets=None, seasons=None, *, force=False):
        calls.syncs += 1
        return {"schedules": [], "pbp": []}

    def build_and_save(c: Settings, **_: Any) -> pl.DataFrame:
        ingest.write_parquet_atomic(features, c.features_path)
        return features

    def no_live_odds(**_: Any) -> pl.DataFrame:
        calls.odds_fetches += 1
        raise AssertionError("The Odds API must not be called in this test")

    monkeypatch.setattr(ingest.NflverseCache, "sync_all", sync_all)
    monkeypatch.setattr(F, "build_and_save", build_and_save)
    monkeypatch.setattr(ingest, "fetch_market_win_probs", no_live_odds)
    return calls


def _week_ids(synthetic_league, week: int = 6) -> list[str]:
    features, _ = synthetic_league
    return features.filter((pl.col("season") == 2026) & (pl.col("week") == week))["game_id"].to_list()


def _live_consensus(game_ids: list[str], home_wp: float = 0.58) -> pl.DataFrame:
    n = len(game_ids)
    return pl.DataFrame(
        {
            "game_id": game_ids,
            "market_home_wp": [home_wp] * n,
            "wp_source": ["h2h"] * n,
            "consensus_home_spread": [-2.5] * n,
            "n_books_h2h": pl.Series([6] * n, dtype=pl.UInt32),
            "n_books_spread": pl.Series([6] * n, dtype=pl.UInt32),
            "fetched_at": pl.Series([NOW] * n, dtype=pl.Datetime("us", "UTC")),
        }
    )


def _template_only(_cfg: Settings) -> R.ReportWriter:
    raise AssertionError("--skip-llm must not build a Gemini writer")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _statuses(result: W.RunResult) -> dict[str, str]:
    return {s.name: s.status for s in result.stages}


# --------------------------------------------------------------------------- #
# Integration: real Modules 3-4
# --------------------------------------------------------------------------- #
def test_offline_run_writes_every_contract(cfg, league):
    opts = W.Options(use_cached_odds=True, skip_llm=True)
    result = W.run(opts, cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_OK, result.error
    assert [s.name for s in result.stages] == list(STAGES)
    assert set(_statuses(result).values()) == {"ok"}
    assert (result.season, result.week) == (2026, 6)
    assert league.syncs == 1 and league.odds_fetches == 0
    expected = [
        cfg.features_path,
        cfg.current_week_json,
        cfg.preds_archive_json(2026, 6),
        cfg.enriched_report_json,
        cfg.enriched_archive_json(2026, 6),
        cfg.project_root / "reports" / "week_06_2026.md",
    ]
    assert result.outputs == expected
    assert all(p.exists() for p in expected)

    contract_b = _read(cfg.current_week_json)
    M.validate_contract_b(contract_b)
    assert contract_b == _read(cfg.preds_archive_json(2026, 6))
    assert {g["market"]["source"] for g in contract_b["games"]} == {"nflverse_line"}
    contract_c = _read(cfg.enriched_report_json)
    R.validate_contract_c(contract_c)
    assert contract_c == _read(cfg.enriched_archive_json(2026, 6))
    assert contract_c["source"]["generated_at"] == contract_b["generated_at"]
    assert contract_c["generator"]["mode"] == "template"
    assert "Weekly run: 2026 week 6" in W.format_summary(result, cfg)


def test_live_odds_and_gemini_write_ups(odds_cfg, league, synthetic_league, monkeypatch):
    ids = _week_ids(synthetic_league)

    def live(*, cfg: Settings, now: datetime) -> pl.DataFrame:
        league.odds_fetches += 1
        assert now == NOW
        return _live_consensus(ids)

    monkeypatch.setattr(ingest, "fetch_market_win_probs", live)
    writer = fakes.GroundedWriter()
    result = W.run(W.Options(), odds_cfg, now=NOW, writer_factory=lambda _c: writer)

    assert result.exit_code == W.EXIT_OK, (result.error, result.degraded)
    assert league.odds_fetches == 1  # one paid pull per run
    contract_b = _read(odds_cfg.current_week_json)
    assert {g["market"]["source"] for g in contract_b["games"]} == {"odds_api"}
    contract_c = _read(odds_cfg.enriched_report_json)
    assert contract_c["summary"]["n_llm"] == len(ids) and writer.calls == len(ids)
    assert contract_c["generator"] | {"temperature": None} == {
        "provider": "google-genai",
        "model": "fake-gemini",
        "mode": "llm",
        "temperature": None,
        "max_attempts": 2,
    }


def test_use_cached_odds_never_calls_the_odds_api(odds_cfg, league):
    result = W.run(W.Options(use_cached_odds=True, skip_llm=True), odds_cfg, now=NOW, writer_factory=_template_only)
    assert result.exit_code == W.EXIT_OK
    assert league.odds_fetches == 0


def test_failed_live_odds_are_degraded_but_written(odds_cfg, league, monkeypatch):
    def auth_error(**_: Any) -> pl.DataFrame:
        league.odds_fetches += 1
        raise ingest.OddsAPIAuthError("401: invalid key")

    monkeypatch.setattr(ingest, "fetch_market_win_probs", auth_error)
    result = W.run(W.Options(skip_llm=True), odds_cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_DEGRADED
    assert _statuses(result) == {"ingest": "ok", "features": "ok", "model": "degraded", "report": "ok"}
    assert result.degraded == ["live odds were requested but none of the 8 upcoming games got an Odds API line"]
    assert {g["market"]["source"] for g in _read(odds_cfg.current_week_json)["games"]} == {"nflverse_line"}
    assert odds_cfg.enriched_report_json.exists()  # degraded still runs the later stages
    assert "DEGRADED" in W.format_summary(result, odds_cfg)


def test_all_template_fallback_is_degraded(cfg, league):
    class Rambling:
        name = "fake-gemini"

        def complete(self, system_instruction: str, prompt: str) -> str:
            return "Not JSON at all."

    result = W.run(W.Options(use_cached_odds=True), cfg, now=NOW, writer_factory=lambda _c: Rambling())

    assert result.exit_code == W.EXIT_DEGRADED
    assert _statuses(result)["report"] == "degraded"
    assert result.degraded == ["no game got a Gemini write-up; every report is a template"]
    contract_c = _read(cfg.enriched_report_json)
    assert contract_c["generator"]["mode"] == "llm" and contract_c["summary"]["n_llm"] == 0


def test_skip_llm_is_not_degraded(cfg, league):
    result = W.run(W.Options(use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)
    assert result.exit_code == W.EXIT_OK and not result.degraded


def test_missing_gemini_key_fails_before_any_stage(cfg, league):
    result = W.run(W.Options(use_cached_odds=True), cfg, now=NOW)  # real writer factory, no key

    assert result.exit_code == W.EXIT_MISSING_KEY
    assert [(s.name, s.status) for s in result.stages] == [("preflight", "failed")]
    assert "GEMINI_API_KEY is not set" in result.error and "--skip-llm" in result.error
    assert league.syncs == 0
    assert not cfg.features_path.exists() and not cfg.current_week_json.exists()


def test_missing_odds_key_fails_before_predicting_the_current_week(cfg, league):
    result = W.run(W.Options(skip_llm=True), cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_MISSING_KEY
    assert _statuses(result) == {"ingest": "ok", "features": "ok", "model": "failed"}
    assert "ODDS_API_KEY is not set" in result.error and "--use-cached-odds" in result.error
    assert not cfg.current_week_json.exists() and league.odds_fetches == 0


def test_replay_writes_archives_only_and_needs_no_odds_key(cfg, league):
    """A past week never replaces the latest-run files, and live odds are not requested for it."""
    result = W.run(W.Options(week=5, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_OK, result.error
    assert (result.season, result.week) == (2026, 5)
    assert league.odds_fetches == 0
    assert not cfg.current_week_json.exists() and not cfg.enriched_report_json.exists()
    contract_b = _read(cfg.preds_archive_json(2026, 5))
    assert contract_b["week"] == 5 and {g["status"] for g in contract_b["games"]} == {"final"}
    assert {g["market"]["source"] for g in contract_b["games"]} == {"nflverse_close"}
    R.validate_contract_c(_read(cfg.enriched_archive_json(2026, 5)))
    assert (cfg.project_root / "reports" / "week_05_2026.md").exists()
    assert result.outputs[1:] == [
        cfg.preds_archive_json(2026, 5),
        cfg.enriched_archive_json(2026, 5),
        cfg.project_root / "reports" / "week_05_2026.md",
    ]


def test_replay_keeps_the_latest_current_week_files(cfg, league):
    W.run(W.Options(use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)
    latest_b, latest_c = cfg.current_week_json.read_bytes(), cfg.enriched_report_json.read_bytes()
    W.run(W.Options(week=4, skip_llm=True, skip_ingest=True), cfg, now=NOW, writer_factory=_template_only)
    assert cfg.current_week_json.read_bytes() == latest_b
    assert cfg.enriched_report_json.read_bytes() == latest_c


def test_future_week_is_rejected_without_writing(cfg, league):
    result = W.run(W.Options(week=8, use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_ERROR
    assert result.stages[-1].name == "model" and result.stages[-1].status == "failed"
    assert "week 8 is in the future; the current week is 6" in result.error
    assert sorted(p.name for p in cfg.predictions_dir.glob("*.json")) == ["last_run.json"]


def test_skip_ingest_does_not_sync(cfg, league):
    result = W.run(
        W.Options(skip_ingest=True, use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only
    )
    assert result.exit_code == W.EXIT_OK
    assert league.syncs == 0 and _statuses(result)["ingest"] == "skipped"


def test_failed_nflverse_refresh_is_degraded_but_continues(cfg, league, monkeypatch):
    def failing_sync(self, datasets=None, seasons=None, *, force=False):
        self.refresh_failures.append("pbp 2026: Failed to download pbp 2026 after 3 attempts: timeout")
        return {"schedules": [], "pbp": []}

    monkeypatch.setattr(ingest.NflverseCache, "sync_all", failing_sync)
    result = W.run(W.Options(use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_DEGRADED
    assert _statuses(result) == {"ingest": "degraded", "features": "ok", "model": "ok", "report": "ok"}
    assert result.degraded == [
        "nflverse refresh failed; used the older cached copy (pbp 2026: Failed to download pbp 2026 after 3 "
        "attempts: timeout)"
    ]


def test_ingest_syncs_only_the_datasets_the_pipeline_reads(cfg, monkeypatch):
    seen: list[Any] = []

    def sync_all(self, datasets=None, seasons=None, *, force=False):
        seen.append(tuple(datasets))
        return {"schedules": [], "pbp": [ingest.CacheEntry("pbp", s, "", NOW, 1, 1, False) for s in (2016, 2026)]}

    monkeypatch.setattr(ingest.NflverseCache, "sync_all", sync_all)
    outcome = W.stage_ingest(cfg)
    assert seen == [("schedules", "pbp")]
    assert outcome.detail == "schedules + pbp 2016-2026" and outcome.degraded == ()


def test_run_record_describes_a_degraded_run(odds_cfg, league, monkeypatch):
    empty = pl.DataFrame(schema=ingest.CONSENSUS_SCHEMA)  # what a live pull that finds no prices returns
    monkeypatch.setattr(ingest, "fetch_market_win_probs", lambda **_: empty)
    result = W.run(W.Options(skip_llm=True), odds_cfg, now=NOW, writer_factory=_template_only)
    record = _read(odds_cfg.last_run_json)

    assert result.exit_code == record["exit_code"] == W.EXIT_DEGRADED
    assert record["status"] == "degraded" and record["degraded"] == result.degraded
    assert (record["season"], record["week"], record["is_current"]) == (2026, 6, True)
    assert record["started_at"] == record["finished_at"] == "2026-10-16T18:00:00Z"
    assert record["options"] == dataclasses.asdict(W.Options(skip_llm=True))
    assert [s["name"] for s in record["stages"]] == list(STAGES)
    assert "data/predictions/current_week.json" in record["outputs"]
    assert record["error"] is None


def test_run_record_describes_a_failed_run(cfg, league):
    result = W.run(W.Options(week=7, use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)
    record = _read(cfg.last_run_json)
    assert record["exit_code"] == result.exit_code == W.EXIT_ERROR
    assert record["status"] == "error" and record["outputs"] == []
    assert record["error"].startswith("model: ModelError: 2026 week 7 is in the future")
    assert record["season"] is None and record["stages"][-1]["status"] == "failed"

    W.run(W.Options(use_cached_odds=True), cfg, now=NOW)  # no Gemini key
    assert _read(cfg.last_run_json)["status"] == "missing API key"


def test_an_unwritable_run_record_keeps_the_exit_code(cfg, league, monkeypatch):
    real_write = ingest.write_text_atomic

    def write(text: str, path: Path) -> None:
        if path.name == "last_run.json":
            raise PermissionError("read-only")
        real_write(text, path)

    monkeypatch.setattr(ingest, "write_text_atomic", write)
    result = W.run(W.Options(use_cached_odds=True, skip_llm=True), cfg, now=NOW, writer_factory=_template_only)
    assert result.exit_code == W.EXIT_OK and not cfg.last_run_json.exists()


# --------------------------------------------------------------------------- #
# Unit: orchestration with stubbed stages
# --------------------------------------------------------------------------- #
TARGET = M.WeekTarget(2026, 6, True, pl.DataFrame())


@pytest.fixture
def stubs(monkeypatch) -> list[str]:
    """Replace every stage with a stub that records its name; returns the call log."""
    log: list[str] = []

    def stub(name: str, value: Any = None):
        def fn(*_args: Any, **_kwargs: Any) -> W.Outcome:
            log.append(name)
            return W.Outcome(f"{name} done", value)

        return fn

    contract_c = {"summary": {"n_games": 1, "n_llm": 1, "n_template": 0}}
    monkeypatch.setattr(W, "stage_ingest", stub("ingest"))
    monkeypatch.setattr(W, "stage_features", stub("features", Path("features.parquet")))
    monkeypatch.setattr(W, "stage_backtest", stub("backtest", Path("backtest_report.json")))
    monkeypatch.setattr(W, "resolve_target", lambda *_a: log.append("resolve") or TARGET)
    monkeypatch.setattr(W, "stage_model", stub("model", {"games": []}))
    monkeypatch.setattr(W, "stage_report", stub("report", contract_c))
    return log


def test_stage_order_and_optional_stages(cfg, stubs):
    assert W.run(W.Options(skip_llm=True), cfg, now=NOW).exit_code == W.EXIT_OK
    assert stubs == ["ingest", "features", "resolve", "model", "report"]

    stubs.clear()
    W.run(W.Options(skip_llm=True, skip_ingest=True, backtest=True), cfg, now=NOW)
    assert stubs == ["features", "backtest", "resolve", "model", "report"]


@pytest.mark.parametrize("failing", STAGES)
@pytest.mark.parametrize(
    "error",
    [
        ingest.NflverseDownloadError("nflverse is down"),
        F.FeatureBuildError("pbp is missing required columns"),
        M.ModelError("no feature rows"),
        M.ContractError("Contract B failed validation"),
        R.LLMAuthError("Gemini rejected the request (HTTP 401)"),
        R.LLMQuotaError("daily quota exhausted"),
        FileNotFoundError("current_week.json not found"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_failed_stage_stops_the_run(cfg, stubs, monkeypatch, failing, error):
    def boom(*_args: Any, **_kwargs: Any) -> W.Outcome:
        stubs.append(failing)
        raise error

    monkeypatch.setattr(W, f"stage_{failing}", boom)
    result = W.run(W.Options(skip_llm=True), cfg, now=NOW)

    assert result.exit_code == W.EXIT_ERROR
    ran = [s for s in stubs if s != "resolve"]
    assert ran == list(STAGES[: STAGES.index(failing) + 1])  # nothing after the failure
    assert result.stages[-1].name == failing and result.stages[-1].status == "failed"
    assert result.error == f"{failing}: {type(error).__name__}: {error}"
    assert "Exit code 1 (error)" in W.format_summary(result, cfg)


def test_target_resolution_failure_stops_before_the_model(cfg, stubs, monkeypatch):
    def no_week(*_a: Any) -> M.WeekTarget:
        raise M.ModelError("the 2026 season has no upcoming games; pass a week to replay a past one")

    monkeypatch.setattr(W, "resolve_target", no_week)
    result = W.run(W.Options(skip_llm=True), cfg, now=NOW)
    assert result.exit_code == W.EXIT_ERROR
    assert stubs == ["ingest", "features"]
    assert result.stages[-1].name == "model" and result.season is None


def test_unexpected_errors_exit_1_with_a_traceback(cfg, stubs, monkeypatch, caplog):
    def bug(*_args: Any, **_kwargs: Any) -> W.Outcome:
        raise KeyError("home_wp")

    monkeypatch.setattr(W, "stage_model", bug)
    with caplog.at_level(logging.ERROR, logger="run_weekly"):
        result = W.run(W.Options(skip_llm=True), cfg, now=NOW)
    assert result.exit_code == W.EXIT_ERROR
    assert "report" not in stubs
    assert any(r.exc_info for r in caplog.records)


def test_degraded_stages_still_run_later_stages(cfg, stubs, monkeypatch):
    monkeypatch.setattr(W, "stage_ingest", lambda _c: W.Outcome("stale", degraded=("pbp refresh failed",)))
    result = W.run(W.Options(skip_llm=True), cfg, now=NOW)
    assert result.exit_code == W.EXIT_DEGRADED
    assert result.degraded == ["pbp refresh failed"]
    assert stubs[-1] == "report"


def test_each_stage_gets_a_fresh_clock_unless_pinned(cfg, stubs, monkeypatch):
    times = iter([datetime(2026, 10, 16, 18, m, tzinfo=UTC) for m in range(10)])
    monkeypatch.setattr(ingest, "utc_now", lambda: next(times))
    seen: dict[str, datetime] = {}
    monkeypatch.setattr(W, "resolve_target", lambda _c, _o, now: seen.setdefault("resolve", now) and TARGET)
    monkeypatch.setattr(W, "stage_model", lambda _c, _o, _t, now: seen.setdefault("model", now) and W.Outcome(""))
    monkeypatch.setattr(
        W,
        "stage_report",
        lambda _c, _o, _t, now, _w: seen.setdefault("report", now) and W.Outcome("", {"summary": {"n_llm": 1}}),
    )
    W.run(W.Options(skip_llm=True), cfg)
    assert seen["resolve"] == seen["model"] < seen["report"]  # the model stage resolves and predicts at one instant

    seen.clear()
    W.run(W.Options(skip_llm=True), cfg, now=NOW)
    assert set(seen.values()) == {NOW}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_parse_args_maps_every_flag():
    opts, level = W.parse_args(
        ["--season", "2025", "--week", "7", "--use-cached-odds", "--skip-llm", "--skip-ingest", "--backtest"]
    )
    assert opts == W.Options(2025, 7, True, True, True, True)
    assert level is None
    assert W.parse_args([]) == (W.Options(), None)
    assert W.parse_args(["--log-level", "DEBUG"])[1] == "DEBUG"


@pytest.mark.parametrize("argv", [["--week", "0"], ["--week", "23"], ["--week", "six"], ["--season", "1998"]])
def test_parse_args_rejects_bad_values(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        W.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("code", [W.EXIT_OK, W.EXIT_ERROR, W.EXIT_MISSING_KEY, W.EXIT_DEGRADED])
def test_main_returns_the_run_exit_code_and_prints_a_summary(monkeypatch, capsys, code):
    captured: dict[str, Any] = {}

    def fake_run(opts: W.Options, cfg: Settings) -> W.RunResult:
        captured.update(opts=opts, cfg=cfg)
        return W.RunResult(exit_code=code, season=2026, week=6)

    monkeypatch.setattr(W, "run", fake_run)
    monkeypatch.setattr(W, "setup_logging", lambda _level: None)
    assert W.main(["--use-cached-odds", "--skip-llm"]) == code
    assert captured["opts"] == W.Options(use_cached_odds=True, skip_llm=True)
    assert captured["cfg"] is settings
    assert f"Exit code {code} " in capsys.readouterr().out


def test_main_reports_interrupts_as_130(monkeypatch):
    def interrupted(*_a: Any) -> W.RunResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(W, "run", interrupted)
    monkeypatch.setattr(W, "setup_logging", lambda _level: None)
    assert W.main([]) == W.EXIT_INTERRUPTED


def test_exit_codes_match_the_report_generator():
    assert (W.EXIT_OK, W.EXIT_ERROR, W.EXIT_MISSING_KEY, W.EXIT_DEGRADED) == (0, 1, 2, 3)
    assert (R.EXIT_OK, R.EXIT_ERROR, R.EXIT_MISSING_KEY, R.EXIT_DEGRADED) == (0, 1, 2, 3)


# --------------------------------------------------------------------------- #
# Real cache (read-only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (settings.raw_dir / "schedules" / "schedules.parquet").exists() or not settings.manifest_path.exists(),
    reason="needs the real cache",
)
def test_real_cache_offline_run_is_read_only(tmp_path):
    """The full pipeline on the real nflverse cache, writing only under tmp_path."""
    cfg = dataclasses.replace(
        Settings.from_env(project_root=tmp_path), raw_dir=settings.raw_dir, odds_api_key=None, gemini_api_key=None
    )
    raw_files = {p: p.stat().st_mtime_ns for p in settings.raw_dir.rglob("*") if p.is_file()}
    opts = W.Options(season=2026, skip_ingest=True, use_cached_odds=True, skip_llm=True)
    result = W.run(opts, cfg, now=NOW, writer_factory=_template_only)

    assert result.exit_code == W.EXIT_OK, result.error
    assert (result.season, result.week) == (2026, 6)
    assert all(p.exists() and p.is_relative_to(tmp_path) for p in result.outputs)
    R.validate_contract_c(_read(cfg.enriched_report_json))
    assert {p: p.stat().st_mtime_ns for p in settings.raw_dir.rglob("*") if p.is_file()} == raw_files
