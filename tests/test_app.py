"""Tests for app.py: the data layer (discovery, loading, banners, tables, chart) and AppTest smoke runs."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

import app as A
from config import Settings
from src import model as M
from src import report_generator as R
from tests import fakes

NOW = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def contract_b(synthetic_league) -> dict[str, Any]:
    """A real Contract B for 2026 week 6 (one game without a market line)."""
    features, schedules = synthetic_league
    ids = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))["game_id"].to_list()
    market = M.nflverse_market_probs(schedules).filter(pl.col("game_id").is_in(ids[:-1]))
    return M.predict_week(features, schedules, market, season=2026, week=6, now=NOW)


@pytest.fixture(scope="module")
def contract_b_week5(synthetic_league) -> dict[str, Any]:
    features, schedules = synthetic_league
    return M.predict_week(features, schedules, M.nflverse_market_probs(schedules), season=2026, week=5, now=NOW)


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return dataclasses.replace(Settings.from_env(project_root=tmp_path), gemini_api_key=None, llm_max_attempts=2)


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def _template_c(cfg: Settings, b: dict[str, Any], now: datetime = NOW) -> dict[str, Any]:
    return R.enrich_payload(b, None, cfg=cfg, now=now)


def _llm_c(cfg: Settings, b: dict[str, Any]) -> dict[str, Any]:
    return R.enrich_payload(b, fakes.GroundedWriter(), cfg=cfg, now=NOW)


class Rambling:
    """A Gemini stand-in whose replies never validate, so every game falls back to the template."""

    name = "fake-gemini"

    def complete(self, system_instruction: str, prompt: str) -> str:
        return "Not JSON."


def _all_template_llm_c(cfg: Settings, b: dict[str, Any]) -> dict[str, Any]:
    return R.enrich_payload(b, Rambling(), cfg=cfg, now=NOW)


def _publish(cfg: Settings, b: dict[str, Any], c: dict[str, Any] | None, *, latest: bool = True) -> None:
    """Write a week the way the pipeline does: archives always, latest-run files for the current week."""
    _write(cfg.preds_archive_json(b["season"], b["week"]), b)
    if latest:
        _write(cfg.current_week_json, b)
    if c is not None:
        _write(cfg.enriched_archive_json(b["season"], b["week"]), c)
        if latest:
            _write(cfg.enriched_report_json, c)


def _record(b: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    record = {
        "record": "run",
        "schema_version": "1.0.0",
        "started_at": "2026-10-16T17:59:00Z",
        "finished_at": "2026-10-16T18:01:00Z",
        "season": b["season"],
        "week": b["week"],
        "is_current": True,
        "exit_code": 0,
        "status": "ok",
        "degraded": [],
        "error": None,
        "stages": [],
        "outputs": [],
    }
    return record | overrides


# --------------------------------------------------------------------------- #
# Discovery and loading
# --------------------------------------------------------------------------- #
def test_discover_weeks_lists_archives_newest_first(cfg, contract_b, contract_b_week5):
    _publish(cfg, contract_b, _template_c(cfg, contract_b))
    _publish(cfg, contract_b_week5, None, latest=False)
    _write(cfg.predictions_dir / "week_03_2025_preds.json", {**contract_b_week5, "season": 2025, "week": 3})
    _write(cfg.predictions_dir / "week_3_2025_preds.json", "{}")  # not an archive name
    _write(cfg.predictions_dir / "backtest_report.json", {"pooled": {}})

    weeks = A.discover_weeks(cfg)
    assert [(w.season, w.week) for w in weeks] == [(2026, 6), (2026, 5), (2025, 3)]
    assert [w.label for w in weeks] == ["2026 · Week 6", "2026 · Week 5", "2025 · Week 3"]
    assert weeks[0].preds == cfg.preds_archive_json(2026, 6)  # the archive, not current_week.json
    assert weeks[0].enriched == cfg.enriched_archive_json(2026, 6)
    assert weeks[1].enriched is None
    assert A.default_week_index(cfg, weeks) == 0


def test_latest_files_fill_in_a_week_without_archives(cfg, contract_b, contract_b_week5):
    """Files written before archives existed (or whose archive is missing) still show up."""
    _write(cfg.current_week_json, contract_b)
    _write(cfg.enriched_report_json, _template_c(cfg, contract_b))
    _publish(cfg, contract_b_week5, None, latest=False)

    weeks = A.discover_weeks(cfg)
    assert [(w.season, w.week) for w in weeks] == [(2026, 6), (2026, 5)]
    assert (weeks[0].preds, weeks[0].enriched) == (cfg.current_week_json, cfg.enriched_report_json)


def test_default_week_is_the_latest_run_not_the_newest_archive(cfg, contract_b, contract_b_week5):
    _publish(cfg, contract_b, None, latest=False)
    _publish(cfg, contract_b_week5, None)  # current_week.json is week 5
    weeks = A.discover_weeks(cfg)
    assert weeks[A.default_week_index(cfg, weeks)].week == 5


def test_discover_weeks_handles_an_empty_or_missing_directory(cfg):
    assert A.discover_weeks(cfg) == []
    assert A.default_week_index(cfg, []) == 0
    _write(cfg.current_week_json, "not json")
    assert A.discover_weeks(cfg) == []


def test_load_week_validates_both_contracts(cfg, contract_b):
    _publish(cfg, contract_b, _template_c(cfg, contract_b))
    data = A.load_week(A.discover_weeks(cfg)[0])
    assert data.problems == []
    assert data.contract_b == contract_b
    assert data.contract_c["contract"] == "C"


@pytest.mark.parametrize("broken", ["preds", "enriched"])
def test_load_week_reports_bad_files_instead_of_trusting_them(cfg, contract_b, broken):
    contract_c = _template_c(cfg, contract_b)
    _publish(cfg, contract_b, contract_c)
    if broken == "preds":
        tampered = copy.deepcopy(contract_b)
        tampered["games"][0]["final"]["home_wp"] = 1.7
        _write(cfg.preds_archive_json(2026, 6), tampered)
    else:
        _write(cfg.enriched_archive_json(2026, 6), '{"contract": "C", "games": ')  # truncated write

    data = A.load_week(A.discover_weeks(cfg)[0])
    assert len(data.problems) == 1 and data.problems[0].startswith(f"week_06_2026_{broken}.json: ")
    if broken == "preds":
        assert data.contract_b == R.contract_b_view(contract_c)  # Contract C embeds a validated Contract B
    else:
        assert data.contract_c is None and data.contract_b == contract_b


def test_load_run_record_ignores_other_files(cfg, contract_b):
    assert A.load_run_record(cfg) is None
    _write(cfg.last_run_json, "[1, 2]")
    assert A.load_run_record(cfg) is None
    _write(cfg.last_run_json, _record(contract_b))
    assert A.load_run_record(cfg)["exit_code"] == 0


# --------------------------------------------------------------------------- #
# Banners
# --------------------------------------------------------------------------- #
def _titles(items: list[A.Banner]) -> list[tuple[str, str]]:
    return [(b.level, b.title) for b in items]


def test_template_mode_banner(cfg, contract_b):
    data = A.WeekData(A.WeekRef(2026, 6), contract_b, _template_c(cfg, contract_b))
    assert _titles(A.banners(data, None)) == [
        ("warning", "Template mode: write-ups are rule-based, not Gemini"),
        ("info", "1 game(s) have no market line"),
    ]


def test_all_template_llm_run_is_a_degraded_banner(cfg, contract_b):
    contract_c = _all_template_llm_c(cfg, contract_b)
    items = A.banners(A.WeekData(A.WeekRef(2026, 6), contract_b, contract_c), None)
    assert items[0] == A.Banner(
        "error",
        "Degraded run: a live input was not used",
        ("No game got a Gemini write-up; every report is a template",),
    )


def test_degraded_run_record_adds_its_reasons_once(cfg, contract_b):
    contract_c = _all_template_llm_c(cfg, contract_b)
    record = _record(
        contract_b,
        exit_code=3,
        status="degraded",
        degraded=[
            "live odds were requested but none of the 8 upcoming games got an Odds API line",
            A.DEGRADED_NO_LLM,
        ],
    )
    degraded = A.banners(A.WeekData(A.WeekRef(2026, 6), contract_b, contract_c), record)[0]
    assert degraded.level == "error"
    assert degraded.lines == (
        "Live odds were requested but none of the 8 upcoming games got an Odds API line",
        "No game got a Gemini write-up; every report is a template",
    )


def test_run_record_from_another_run_is_ignored(cfg, contract_b):
    contract_c = _llm_c(cfg, contract_b)
    data = A.WeekData(A.WeekRef(2026, 6), contract_b, contract_c)
    other_week = _record(contract_b, week=5, exit_code=3, degraded=["live odds were requested but none ..."])
    earlier_run = _record(
        contract_b,
        exit_code=3,
        degraded=["live odds were requested but none ..."],
        started_at="2026-10-15T10:00:00Z",
        finished_at="2026-10-15T10:01:00Z",
    )
    for record in (other_week, earlier_run):
        assert all(b.level != "error" for b in A.banners(data, record))
    assert A.run_covers(_record(contract_b), contract_b)


def test_failed_last_run_is_an_error_banner_on_every_week(cfg, contract_b):
    record = _record(contract_b, exit_code=1, status="error", error="model: ModelError: no feature rows", week=None)
    first = A.banners(A.WeekData(A.WeekRef(2026, 6), contract_b, _llm_c(cfg, contract_b)), record)[0]
    assert first.level == "error" and first.title == "The last pipeline run failed (error, exit code 1)"
    assert first.lines[0] == "model: ModelError: no feature rows"


def test_missing_and_stale_reports_are_flagged(cfg, contract_b):
    no_report = A.banners(A.WeekData(A.WeekRef(2026, 6), contract_b, None), None)
    assert ("warning", "No scouting report for this week") in _titles(no_report)

    newer_b = {**contract_b, "generated_at": "2026-10-17T12:00:00Z"}
    stale = A.banners(A.WeekData(A.WeekRef(2026, 6), newer_b, _llm_c(cfg, contract_b)), None)
    assert ("warning", "The write-ups are older than the predictions") in _titles(stale)


def test_partial_fallback_banner(cfg, contract_b):
    contract_c = _llm_c(cfg, contract_b)
    contract_c["summary"] = {**contract_c["summary"], "n_llm": 6, "n_template": 2}
    items = A.banners(A.WeekData(A.WeekRef(2026, 6), contract_b, contract_c), None)
    assert ("warning", "2 of 8 write-ups fell back to the template") in _titles(items)
    assert all(level != "error" for level, _ in _titles(items))


# --------------------------------------------------------------------------- #
# Tables and chart
# --------------------------------------------------------------------------- #
def test_market_summary_counts_sources_and_latest_as_of(contract_b):
    b = copy.deepcopy(contract_b)
    b["games"][0]["market"] |= {"source": "odds_api_snapshot", "as_of": "2026-10-16T17:00:00Z"}
    b["games"][1]["market"] |= {"source": "odds_api_snapshot", "as_of": "2026-10-16T17:30:00Z"}
    assert A.market_summary(b) == [
        ("nflverse closing line", 5, None),  # closes carry no as-of time
        ("Saved Odds API snapshot", 2, "Oct 16, 2026 17:30 UTC"),  # the latest snapshot time
        ("No market line", 1, None),
    ]


def test_game_rows_use_pick_side_values_truncated_like_the_newsletter(contract_b):
    rows = A.game_rows(contract_b)
    assert len(rows) == len(contract_b["games"])
    assert [r["Kickoff"] for r in rows] == [
        R.local_kickoff(g["kickoff_utc"]) for g in sorted(contract_b["games"], key=lambda g: g["kickoff_utc"])
    ]
    by_matchup = {r["Matchup"]: r for r in rows}
    for g in contract_b["games"]:
        row = by_matchup[A.matchup(g)]
        assert row["Pick"] == g["pick"]["team"] and row["Tier"] == g["confidence_tier"]
        assert row["Win prob"] == R.pct_down(g["pick"]["win_prob"]) <= 100 * g["pick"]["win_prob"]
        if g["market"] is None:
            assert row["Market"] is None and row["Edge (pts)"] is None and row["Market source"] == "No line"
        else:
            assert row["Edge (pts)"] == R.edge_down(g["model_edge_pct"])


def test_chart_data_puts_every_estimate_on_the_pick_side(contract_b):
    points, connectors, order = A.chart_data(contract_b)
    n_games = len(contract_b["games"])
    assert len(order) == n_games and len(points) == 3 * n_games - 1  # one game has no market dot
    assert len(connectors) == n_games - 1
    finals = [p["prob"] for p in points if p["series"] == "Final (blend)"]
    assert finals == sorted(finals, reverse=True) and min(finals) >= 0.5  # strongest pick on top
    for g in contract_b["games"]:
        label = next(o for o in order if o.startswith(f"{g['pick']['team']} over "))
        model = next(p for p in points if p["game"] == label and p["series"] == "Model")
        assert model["prob"] == A.pick_side(g, g["model"])


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_probability_chart_spec(contract_b, mode):
    spec = A.probability_chart(contract_b, mode).to_dict()
    assert len(spec["layer"]) == 3
    dots = spec["layer"][2]
    colors = A.CHART_COLORS[mode]
    assert dots["encoding"]["color"]["scale"]["range"] == [colors[s] for s in A.SERIES]
    assert dots["mark"]["stroke"] == colors["surface"]  # the 2px surface ring
    domain = dots["encoding"]["x"]["scale"]["domain"]
    assert 0.0 < domain[0] <= 0.5 and domain[1] == 1.0
    assert all(layer["encoding"]["x"]["scale"]["zero"] is False for layer in spec["layer"])
    assert spec["height"] == 30 * len(contract_b["games"]) + 40


def test_write_up_meta(cfg, contract_b):
    llm_game = _llm_c(cfg, contract_b)["games"][0]
    assert A.write_up_meta(llm_game) == "Written by fake-gemini (1 attempt)"
    template_game = _template_c(cfg, contract_b)["games"][0]
    assert A.write_up_meta(template_game) == "Rule-based template"
    fallback = {
        **template_game,
        "report_meta": {**template_game["report_meta"], "fallback_reason": "3 invalid replies"},
    }
    assert A.write_up_meta(fallback) == "Rule-based template (fallback: 3 invalid replies)"


def test_fmt_time_accepts_strings_and_datetimes():
    assert A.fmt_time("2026-10-16T18:00:00Z") == "Oct 16, 2026 18:00 UTC"
    assert A.fmt_time(NOW, year=False) == "Oct 16, 18:00 UTC"
    assert A.fmt_time(None) == A.fmt_time("garbage") == "unknown time"


# --------------------------------------------------------------------------- #
# AppTest smoke runs
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_env(cfg, monkeypatch) -> Settings:
    monkeypatch.setenv("NFL_DATA_DIR", str(cfg.data_dir))
    monkeypatch.setenv("NFL_REPORTS_DIR", str(cfg.reports_dir))
    return cfg


def _run_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_without_data_explains_how_to_get_some(app_env):
    at = _run_app()
    assert at.title[0].value == "NFL Weekly Scouting Report"
    assert "No predictions found" in at.info[0].value
    assert not at.selectbox


def test_app_renders_a_week_and_switches_archives(app_env, contract_b, contract_b_week5):
    cfg = app_env
    _publish(cfg, contract_b_week5, _template_c(cfg, contract_b_week5), latest=False)
    _publish(cfg, contract_b, _llm_c(cfg, contract_b))
    _write(cfg.newsletter_md(2026, 6), "# newsletter\n")

    at = _run_app()
    assert at.title[0].value == "NFL Week 6 Scouting Report — 2026 Season"
    assert at.selectbox[0].options == ["2026 · Week 6", "2026 · Week 5"]
    assert [h.value for h in at.subheader] == ["Model vs market", "Scouting reports"]
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == len(contract_b["games"])
    assert not at.error and not at.warning  # an all-Gemini run with live data raises no banner
    headlines = [m.value for m in at.markdown if m.value.startswith("##### ")]
    assert len(headlines) == len(contract_b["games"])

    at.selectbox[0].select(1).run()
    assert not at.exception
    assert at.title[0].value == "NFL Week 5 Scouting Report — 2026 Season"
    assert any("Template mode" in w.value for w in at.warning)


def test_app_shows_the_degraded_banner_first(app_env, contract_b):
    cfg = app_env
    _publish(cfg, contract_b, _template_c(cfg, contract_b))
    _write(
        cfg.last_run_json,
        _record(
            contract_b,
            exit_code=3,
            status="degraded",
            degraded=["live odds were requested but none of the 8 upcoming games got an Odds API line"],
        ),
    )
    at = _run_app()
    assert len(at.error) == 1
    assert "Degraded run" in at.error[0].value and "Live odds were requested" in at.error[0].value


def test_app_survives_a_corrupt_file(app_env, contract_b):
    cfg = app_env
    _publish(cfg, contract_b, None)
    _write(cfg.enriched_archive_json(2026, 6), "{ truncated")
    at = _run_app()
    assert any("week_06_2026_enriched.json" in e.value for e in at.error)
    assert at.title[0].value.startswith("NFL Week 6")
    assert any("No scouting report" in w.value for w in at.warning)


def test_app_with_only_unreadable_predictions(app_env, contract_b):
    cfg = app_env
    tampered = copy.deepcopy(contract_b)
    tampered["games"][0]["final"]["home_wp"] = 1.7
    _write(cfg.preds_archive_json(2026, 6), tampered)
    at = _run_app()
    assert at.title[0].value == "NFL 2026 · Week 6"
    assert any("week_06_2026_preds.json" in e.value for e in at.error)
    assert not at.dataframe


def test_app_does_not_read_the_real_data_dir_when_redirected(app_env):
    """The smoke tests above rely on NFL_DATA_DIR being honoured at run time."""
    assert Settings.from_env().data_dir == app_env.data_dir
    assert Settings.from_env().reports_dir == app_env.reports_dir


def test_importing_the_app_runs_nothing_and_loads_no_streamlit():
    """The data layer is importable on its own: no page is rendered and Streamlit is only imported by main()."""
    code = "import sys, app; assert 'streamlit' not in sys.modules, 'streamlit imported'; print('ok')"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_run_record_reason_text_is_shared_with_the_orchestrator():
    import run_weekly

    assert A.DEGRADED_NO_LLM is run_weekly.DEGRADED_NO_LLM
