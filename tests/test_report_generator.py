"""Tests for src.report_generator: Pydantic grounding, repair loop, Gemini client handling, Contract C, newsletter."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

from config import Settings, settings
from src import model as M
from src import report_generator as R
from tests.fakes import GroundedWriter

NOW = datetime(2026, 10, 16, 18, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def payload(synthetic_league) -> dict[str, Any]:
    """A real Contract B payload built by src.model from the synthetic league (one game has no market line)."""
    import polars as pl

    features, schedules = synthetic_league
    week_ids = features.filter((pl.col("season") == 2026) & (pl.col("week") == 6))["game_id"].to_list()
    market = M.nflverse_market_probs(schedules).filter(pl.col("game_id").is_in(week_ids[:-1]))
    return M.predict_week(features, schedules, market, season=2026, week=6, now=NOW)


@pytest.fixture(scope="module")
def facts(payload) -> list[R.GameFacts]:
    return [R.game_facts(g, payload) for g in payload["games"]]


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return dataclasses.replace(
        Settings.from_env(project_root=tmp_path), gemini_api_key=None, llm_max_attempts=3, llm_max_concurrency=4
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda _s: None)


def _grounded_json(f: R.GameFacts, **overrides: str) -> str:
    """A valid report written from the fact sheet (what a well-behaved LLM would return)."""
    pick = f.sheet["pick"]
    report = {
        "headline": f"{R.team_nickname(f.pick_abbr)} look like the right side this week",
        "the_pick_summary": (
            f"The {pick['team']} are the pick at {pick['final_win_probability_pct']:.0f}%, "
            f"graded {f.sheet['confidence_tier']} by the blended forecast."
        ),
        "key_mismatch": "The biggest gap is in efficiency per play, which tilts this matchup toward the pick.",
        "risk_factor": "Small samples early in the season make every efficiency estimate noisy.",
    }
    report.update(overrides)
    return json.dumps(report)


class ScriptedWriter:
    """Returns queued responses in order (strings, callables of the prompt, or exceptions)."""

    name = "fake-gemini"

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system_instruction: str, prompt: str) -> str:
        self.prompts.append(prompt)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item(prompt) if callable(item) else item


# --------------------------------------------------------------------------- #
# Pydantic model: format and grounding
# --------------------------------------------------------------------------- #
def test_report_requires_all_four_fields(facts):
    data = json.loads(_grounded_json(facts[0]))
    R.validate_report(data, facts[0])
    for field in ("headline", "the_pick_summary", "key_mismatch", "risk_factor"):
        broken = {k: v for k, v in data.items() if k != field}
        with pytest.raises(ValidationError, match=field):
            R.validate_report(broken, facts[0])
    with pytest.raises(ValidationError, match="Extra inputs"):
        R.validate_report({**data, "confidence": "high"}, facts[0])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("headline", "x" * 91, "at most 90"),
        ("headline", "Short", "at least 10"),
        ("headline", "A perfectly fine headline that ends.", "period"),
        ("headline", "**Bold** headline is not allowed here", "markdown"),
        ("the_pick_summary", "First line of the summary here.\nSecond line of the summary.", "single paragraph"),
        ("key_mismatch", "# Heading pretending to be a mismatch statement", "markdown"),
        ("risk_factor", "Watch out for <script>alert(1)</script> in this game", "HTML"),
    ],
)
def test_report_format_rules(facts, field, value, message):
    data = json.loads(_grounded_json(facts[0]))
    data[field] = value
    with pytest.raises(ValidationError, match=message):
        R.validate_report(data, facts[0])


def test_grounding_rejects_invented_percentages_and_accepts_rounding(facts):
    f = facts[0]
    pick_pct = f.sheet["pick"]["final_win_probability_pct"]
    rounded = _grounded_json(f, risk_factor=f"Even at {round(pick_pct)}% this pick can lose on a single turnover.")
    R.validate_report(rounded, f)  # whole-number rounding of a fact-sheet value is fine
    invented = _grounded_json(f, risk_factor="Teams in this spot have covered 71.3% of the time since 2010.")
    with pytest.raises(ValidationError, match="percentages not in the fact sheet"):
        R.validate_report(invented, f)
    points = _grounded_json(f, key_mismatch="The model is 33.3 percentage points off the market on this pick.")
    with pytest.raises(ValidationError, match="percentages not in the fact sheet"):
        R.validate_report(points, f)


def test_grounding_requires_the_pick_and_bans_hype(facts):
    f = facts[0]
    other = R.team_name(f.other_abbr)
    with pytest.raises(ValidationError, match="must name the pick"):
        R.validate_report(
            _grounded_json(f, the_pick_summary=f"The {other} are on the wrong side of this one, per the model."), f
        )
    for hype in ("This one is guaranteed to cash for the pick", "A sure thing if there ever was one this week"):
        with pytest.raises(ValidationError, match="hype or betting"):
            R.validate_report(_grounded_json(f, risk_factor=hype), f)


def test_pick_can_be_named_by_abbreviation_or_nickname(facts):
    f = facts[0]
    tail = f"at {f.pick_pct:g}% ({f.tier}), per the blended forecast."
    for alias in R.team_aliases(f.pick_abbr):
        R.validate_report(_grounded_json(f, the_pick_summary=f"{alias} is the pick here {tail}"), f)


def test_format_rules_apply_without_context_but_grounding_does_not(facts):
    data = json.loads(_grounded_json(facts[0], risk_factor="Teams like this cover 71.3% of the time historically."))
    R.ScoutingReport.model_validate(data)  # no context -> no grounding
    with pytest.raises(ValidationError):
        R.ScoutingReport.model_validate({**data, "headline": "bad\nheadline here"})


# --------------------------------------------------------------------------- #
# Fact sheet
# --------------------------------------------------------------------------- #
def test_fact_sheet_numbers_come_from_the_payload(payload, facts):
    for game, f in zip(payload["games"], facts, strict=True):
        side = game["pick"]["side"]
        assert f.pick_abbr == game["pick"]["team"]
        assert R.pct_down(game["pick"]["win_prob"]) in f.allowed_percentages
        assert R.pct_down(game["model"][f"{side}_wp"]) in f.allowed_percentages
        if game["market"] is not None:
            assert R.pct_down(game["market"][f"{side}_wp"]) in f.allowed_percentages
            assert round(abs(game["model_edge_pct"]), 1) in f.allowed_percentages
        else:
            assert f.sheet["market"] == "no market line available"
        assert len(f.sheet["top_drivers"]) == len(game["top_drivers"])


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("diff_early_down_sr", -0.117, "11.7 percentage points in favor of the Buffalo Bills"),
        ("diff_early_down_sr", 0.05, "5.0 percentage points in favor of the Kansas City Chiefs"),
        ("net_pass_epa_mismatch", -0.2, "0.200 EPA per play in favor of the Buffalo Bills"),
        ("net_rest_days", -3, "3 more days of rest for the Buffalo Bills"),
        ("diff_qb_cpoe", 4.2, "4.2 percentage points of CPOE in favor of the Kansas City Chiefs"),
    ],
)
def test_matchup_drivers_are_phrased_from_the_favored_side(feature, value, expected):
    stat, text = R.describe_driver(feature, value, "KC", "BUF")
    assert text == expected
    assert "-" not in text  # no home-perspective signs


def test_team_drivers_name_the_team():
    stat, text = R.describe_driver("away_def_rush_epa_pre", -0.301, "JAX", "HOU")
    assert stat == "Houston Texans run defense EPA allowed per rush"
    assert text == "-0.301 EPA per play"
    assert "Home" not in R.describe_driver("home_qb_cpoe_pre", 3.0, "JAX", "HOU")[0]


# --------------------------------------------------------------------------- #
# Template reports
# --------------------------------------------------------------------------- #
def test_template_report_is_grounded_for_every_game(facts):
    for f in facts:
        report = R.template_report(f)  # raises if it fails its own validator
        assert f.sheet["pick"]["team"] in report.the_pick_summary


def test_template_handles_no_market_and_coin_flips(payload, facts):
    no_line = next(f for f, g in zip(facts, payload["games"], strict=True) if g["market"] is None)
    assert "No market line" in R.template_report(no_line).the_pick_summary
    flip_game = copy.deepcopy(next(g for g in payload["games"] if g["market"] is not None))
    flip_game.update(confidence_tier="COIN FLIP", value_team=flip_game["pick"]["team"], imputed_inputs=[])
    flip_game["pick"]["win_prob"] = 0.5123
    report = R.template_report(R.game_facts(flip_game, payload))
    assert "coin flip" in report.headline.lower() and "51.2%" in report.risk_factor


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #
def test_repair_loop_feeds_validation_errors_back(facts):
    f = facts[0]
    bad = _grounded_json(f, risk_factor="Teams like this cover 71.3% of the time historically.")
    writer = ScriptedWriter(bad, _grounded_json(f))
    result = R.generate_report(writer, f, max_attempts=3)
    assert (result.generator, result.attempts, result.fallback_reason) == ("fake-gemini", 2, None)
    assert "rejected by the validator" in writer.prompts[1] and "71.3" in writer.prompts[1]
    assert result.validation_errors and "attempt 1" in result.validation_errors[0]


def test_malformed_json_and_code_fences(facts):
    f = facts[0]
    fenced = "```json\n" + _grounded_json(f) + "\n```"
    assert R.generate_report(ScriptedWriter(fenced), f, max_attempts=1).attempts == 1
    result = R.generate_report(ScriptedWriter("not json at all", _grounded_json(f)), f, max_attempts=2)
    assert result.attempts == 2 and result.generator == "fake-gemini"


def test_falls_back_to_template_after_max_attempts(facts):
    f = facts[0]
    writer = ScriptedWriter(_grounded_json(f, headline="x"))
    result = R.generate_report(writer, f, max_attempts=3)
    assert result.generator == R.TEMPLATE_GENERATOR
    assert result.attempts == 3 and len(writer.prompts) == 3
    assert "failed validation" in result.fallback_reason
    with pytest.raises(R.ReportError, match="failed validation"):
        R.generate_report(ScriptedWriter(_grounded_json(f, headline="x")), f, max_attempts=2, allow_fallback=False)


def test_request_errors_fall_back_but_auth_errors_abort(facts):
    f = facts[0]
    result = R.generate_report(ScriptedWriter(R.LLMRequestError("HTTP 503")), f, max_attempts=3)
    assert result.generator == R.TEMPLATE_GENERATOR and "503" in result.fallback_reason
    with pytest.raises(R.LLMAuthError):
        R.generate_report(ScriptedWriter(R.LLMAuthError("bad key")), f, max_attempts=3)


# --------------------------------------------------------------------------- #
# Gemini client wrapper (fake SDK client, real google.genai error types)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self._text = text
        self.candidates = []

    @property
    def text(self) -> str | None:
        return self._text


class _FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeClient:
    def __init__(self, *outcomes: Any) -> None:
        self.models = _FakeModels(list(outcomes))


def _api_error(cls: type, code: int, status: str, message: str, **extra: Any) -> Exception:
    return cls(code, {"error": {"code": code, "status": status, "message": message, **extra}})


def test_gemini_writer_requests_json_with_schema(cfg):
    client = _FakeClient('{"headline": "ok"}')
    writer = R.GeminiReportWriter(cfg=cfg, client=client)
    assert writer.name == "gemini-2.5-flash"
    assert writer.complete(R.SYSTEM_INSTRUCTION, "prompt") == '{"headline": "ok"}'
    call = client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is R._ResponseSchema
    assert call["config"].system_instruction == R.SYSTEM_INSTRUCTION
    assert call["config"].temperature == cfg.llm_temperature


def test_gemini_writer_missing_key(cfg):
    with pytest.raises(R.MissingLLMKeyError, match="GEMINI_API_KEY"):
        R.GeminiReportWriter(cfg=cfg)


def test_gemini_writer_error_handling(cfg):
    invalid_key = _api_error(
        genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."
    )
    with pytest.raises(R.LLMAuthError, match="400"):
        R.GeminiReportWriter(cfg=cfg, client=_FakeClient(invalid_key)).complete(R.SYSTEM_INSTRUCTION, "p")

    retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]
    rate_limited = _api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota", details=retry_info)
    client = _FakeClient(rate_limited, "{}")
    assert R.GeminiReportWriter(cfg=cfg, client=client).complete(R.SYSTEM_INSTRUCTION, "p") == "{}"
    assert len(client.models.calls) == 2
    assert R._retry_delay(rate_limited, 1) == 7.0

    overloaded = _api_error(genai_errors.ServerError, 503, "UNAVAILABLE", "overloaded")
    writer = R.GeminiReportWriter(cfg=cfg, client=_FakeClient(*[overloaded] * 3), max_retries=2)
    with pytest.raises(R.LLMRequestError, match="503"):
        writer.complete(R.SYSTEM_INSTRUCTION, "p")

    bad_request = _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "Invalid JSON schema")
    with pytest.raises(R.LLMConfigError, match="400"):
        R.GeminiReportWriter(cfg=cfg, client=_FakeClient(bad_request)).complete(R.SYSTEM_INSTRUCTION, "p")

    with pytest.raises(R.LLMEmptyResponseError, match="no text"):
        R.GeminiReportWriter(cfg=cfg, client=_FakeClient(None)).complete(R.SYSTEM_INSTRUCTION, "p")


# --------------------------------------------------------------------------- #
# Contract C
# --------------------------------------------------------------------------- #
def test_enrich_produces_valid_contract_c(payload, cfg):
    writer = GroundedWriter()
    contract_c = R.enrich_payload(payload, writer, cfg=cfg, now=NOW, newsletter_path="reports/week_06_2026.md")
    R.validate_contract_c(contract_c)
    assert writer.calls == len(payload["games"])
    assert (contract_c["contract"], contract_c["season"], contract_c["week"]) == ("C", 2026, 6)
    assert contract_c["summary"]["n_llm"] == len(payload["games"]) and contract_c["summary"]["n_template"] == 0
    assert [g["game_id"] for g in contract_c["games"]] == [g["game_id"] for g in payload["games"]]  # order kept
    for enriched, original in zip(contract_c["games"], payload["games"], strict=True):
        assert set(enriched["report"]) == {"headline", "the_pick_summary", "key_mismatch", "risk_factor"}
        assert enriched["report_meta"]["generator"] == "fake-gemini"
        assert {k: v for k, v in enriched.items() if k not in ("report", "report_meta")} == original
    json.dumps(contract_c, allow_nan=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda c: c["games"][0]["report"].update(risk_factor="History says 71.3% of teams cover in this spot."),
            "fact sheet",
        ),
        (lambda c: c["games"][0]["report"].pop("key_mismatch"), "key_mismatch"),
        (lambda c: c["summary"].update(n_template=5), "summary must be"),
        (lambda c: c["summary"]["tiers"].update(LOCK=99), "summary must be"),
        (lambda c: c["games"][0].update(report=json.dumps(c["games"][0]["report"])), "report must be an object"),
        (lambda c: c["games"][0]["report_meta"].pop("fallback_reason"), "report_meta must hold"),
        (lambda c: c["generator"].pop("model"), "generator must have keys"),
        (lambda c: c.pop("tier_rules"), "tier_rules"),
        (lambda c: c.update(source=None), "source must describe"),
        (lambda c: c["games"][0]["report_meta"].update(generator="gpt-9"), "generator"),
        (
            lambda c: c["games"][0].update(
                confidence_tier="LOCK" if c["games"][0]["confidence_tier"] != "LOCK" else "LEAN"
            ),
            "embedded Contract B",
        ),
        (lambda c: c.update(contract="B"), "contract must be"),
    ],
)
def test_contract_c_validator_rejects_tampering(payload, cfg, mutate, message):
    contract_c = R.enrich_payload(payload, None, cfg=cfg, now=NOW)
    broken = copy.deepcopy(contract_c)
    mutate(broken)
    with pytest.raises(R.ContractCError, match=message):
        R.validate_contract_c(broken)


def test_contract_c_validator_never_crashes_on_garbage():
    for garbage in ([1, 2], {"contract": "C", "games": "nope"}, {"contract": "C", "games": [None]}):
        with pytest.raises(R.ContractCError):
            R.validate_contract_c(garbage)


# --------------------------------------------------------------------------- #
# End to end: files on disk and the newsletter
# --------------------------------------------------------------------------- #
def test_run_report_writes_contract_c_and_newsletter(payload, cfg):
    cfg.predictions_dir.mkdir(parents=True)
    cfg.current_week_json.write_text(json.dumps(payload))
    backtest = {
        "test_seasons": [2018, 2025],
        "pooled": {
            k: {"brier": b, "n_games": 2298} for k, b in (("model", 0.2329), ("market", 0.2114), ("final", 0.2117))
        },
    }
    (cfg.predictions_dir / "backtest_report.json").write_text(json.dumps(backtest))

    contract_c = R.run_report(cfg, mode="llm", writer=GroundedWriter(), now=NOW)

    saved = json.loads(cfg.enriched_report_json.read_text())
    R.validate_contract_c(saved)
    assert saved == contract_c
    md_path = cfg.reports_dir / "week_06_2026.md"  # season in the name: seasons never overwrite each other
    assert contract_c["newsletter_path"] == "reports/week_06_2026.md" and md_path.exists()
    archived = json.loads((cfg.predictions_dir / "week_06_2026_enriched.json").read_text())
    assert archived == saved  # versioned Contract C copy
    md = md_path.read_text()
    assert md.startswith("# NFL Week 6 Scouting Report — 2026 Season")
    for game in contract_c["games"]:
        assert f"{game['away_team']} @ {game['home_team']} — {game['report']['headline']}" in md
        assert game["report"]["the_pick_summary"] in md
    for tier in {g["confidence_tier"] for g in contract_c["games"]}:
        assert re.search(rf"^## {tier.title()}s$", md, re.MULTILINE)
    assert "Brier score model 0.2329" in md and "Not betting advice" in md


def test_template_mode_needs_no_key_and_llm_mode_does(payload, cfg):
    cfg.predictions_dir.mkdir(parents=True)
    cfg.current_week_json.write_text(json.dumps(payload))
    contract_c = R.run_report(cfg, mode="template", now=NOW)
    assert contract_c["generator"]["mode"] == "template" and contract_c["summary"]["n_template"] == len(
        payload["games"]
    )
    with pytest.raises(R.MissingLLMKeyError):
        R.run_report(cfg, mode="llm", now=NOW)


def test_invalid_contract_b_input_is_rejected(payload, cfg):
    cfg.predictions_dir.mkdir(parents=True)
    broken = copy.deepcopy(payload)
    broken["games"][0]["confidence_tier"] = "SMASH"
    cfg.current_week_json.write_text(json.dumps(broken))
    with pytest.raises(M.ContractError):
        R.run_report(cfg, mode="template", now=NOW)
    with pytest.raises(FileNotFoundError):
        R.run_report(dataclasses.replace(cfg, data_dir=cfg.data_dir / "missing"), mode="template", now=NOW)


def test_newsletter_marks_finished_games(payload, cfg):
    done = copy.deepcopy(payload)
    game = done["games"][0]
    game.update(status="final", result={"home_score": 10, "away_score": 27}, kickoff_utc="2026-10-15T00:15:00Z")
    contract_c = R.enrich_payload(done, None, cfg=cfg, now=NOW)
    md = R.render_newsletter(contract_c)
    winner_is_pick = game["pick"]["team"] == game["away_team"]
    assert f"{game['away_team']} 27, {game['home_team']} 10 (pick {'won' if winner_is_pick else 'lost'})" in md
    assert "## Already played" in md


@pytest.mark.skipif(
    not settings.current_week_json.exists(), reason="no real current_week.json; run `python -m src.model`"
)
def test_real_payload_template_report(cfg):
    payload = R.load_contract_b(settings.current_week_json)
    contract_c = R.enrich_payload(payload, None, cfg=cfg, now=datetime.now(UTC))
    R.validate_contract_c(contract_c)
    assert R.render_newsletter(contract_c).count("### ") == len(payload["games"])


# --------------------------------------------------------------------------- #
# Regressions from the adversarial review
# --------------------------------------------------------------------------- #
def _summary_with(f: R.GameFacts, body: str) -> str:
    return _grounded_json(f, the_pick_summary=body)


def test_pick_summary_must_state_tier_and_final_probability(facts):  # RG-1
    f = facts[0]
    name = R.team_name(f.pick_abbr)
    with pytest.raises(ValidationError, match="confidence tier"):
        R.validate_report(_summary_with(f, f"The {name} are the pick at {f.pick_pct:g}% in a close game."), f)
    with pytest.raises(ValidationError, match="final win probability"):
        R.validate_report(_summary_with(f, f"The {name} are the pick here, graded {f.tier} by the forecast."), f)
    wrong = next(t for t in M.TIERS if t != f.tier)
    with pytest.raises(ValidationError, match="the tier for this game"):
        R.validate_report(_summary_with(f, f"The {name} are a {wrong} at {f.pick_pct:g}% ({f.tier})."), f)
    with pytest.raises(ValidationError, match="headline must name"):
        R.validate_report(_grounded_json(f, headline="A big game with plenty on the line this week"), f)


def test_teams_outside_the_game_are_rejected(facts):  # C1
    f = facts[0]
    outsider = next(abbr for abbr in R.TEAM_NAMES if abbr not in f.team_aliases)
    with pytest.raises(ValidationError, match="not in this game"):
        R.validate_report(
            _grounded_json(f, risk_factor=f"This looks like last week's {R.team_name(outsider)} upset."), f
        )
    # lowercase common words that happen to be nicknames are not team mentions
    R.validate_report(
        _grounded_json(f, risk_factor="Paying the bills on third down has been an issue for both teams."), f
    )


def test_percent_tolerance_is_one_decimal_or_whole_number_rounding(facts):  # C1
    f = facts[0]
    fact = max(f.allowed_percentages - R.ALWAYS_ALLOWED_PERCENTAGES)
    near_miss = round(fact + 0.4, 1)
    with pytest.raises(ValidationError, match="not in the fact sheet"):
        R.validate_report(
            _grounded_json(f, risk_factor=f"Teams in this spot win {near_miss}% of the time, roughly."), f
        )
    R.validate_report(_grounded_json(f, risk_factor=f"Teams in this spot win {fact:g}% of the time in this model."), f)


@pytest.mark.parametrize(
    "text",
    [
        "```json the pick is solid this week",
        "~~~ fenced block opener in the risk factor",
        "<!-- hidden comment that never closes",
        "See [the chart](https://evil.example/x) for details",
        "Tracking pixel ![x](https://evil.example/p.gif) here",
        "Read more at https://example.com about this matchup",
        "This is **really** the key matchup of the week",
        "Use `code` formatting inside the risk factor",
        "1. Starts like a numbered list in the risk factor",
    ],
)
def test_markup_links_and_code_are_rejected(facts, text):  # RG-5 / C7
    with pytest.raises(ValidationError, match="plain prose"):
        R.validate_report(_grounded_json(facts[0], risk_factor=text), facts[0])


def test_newsletter_escapes_report_text():  # RG-5 defence in depth
    assert R.md_text("a [b](c) <x> *y* `z` ~~w~~ | q") == r"a \[b\](c) \<x\> \*y\* \`z\` \~\~w\~\~ \| q"
    assert R.md_text("1. list").startswith(r"1\.")


@pytest.mark.parametrize(
    "text",
    [
        "The Bills simply can’t lose this one at home",  # curly apostrophe
        "This one is guaran​tee​d to go the pick's way",  # zero-width spaces
        "This is our best bet of the entire week of football",
        "A sure-thing spot for the favorite in this matchup",
        "Bet on the favorite before the line moves this week",
    ],
)
def test_banned_language_survives_unicode_tricks(facts, text):  # RG-6
    with pytest.raises(ValidationError, match="hype or betting"):
        R.validate_report(_grounded_json(facts[0], risk_factor=text), facts[0])


def test_reasonable_wording_is_not_rejected(facts):  # RG-7
    f = facts[0]
    R.validate_report(_grounded_json(f, risk_factor="There is no guarantee here: one turnover could swing it."), f)
    R.validate_report(_grounded_json(f, risk_factor="Nothing is guaranteed in a divisional game like this one."), f)
    assert "Kansas City" in R.team_aliases("KC") and "Niners" in R.team_aliases("SF")
    assert R.team_city("LA") is None and R.team_city("NYJ") is None  # shared markets are ambiguous
    stat, text = R.describe_driver("home_qb_cpoe_pre", 3.1, "KC", "BUF")
    assert "3.1 percentage points" in text  # CPOE is groundable as percentage points


def test_display_values_never_cross_tier_thresholds():  # C2
    assert R.pct_down(0.7496) == 74.9 and R.pct_down(0.755) == 75.5 and R.pct_down(0.5) == 50.0
    assert R.edge_down(4.95) == 4.9 and R.edge_down(-12.94) == -12.9 and R.edge_down(5.0) == 5.0
    assert R._pair_pct(0.7496) == (74.9, 25.1)


def test_writer_label_reflects_actual_generation(payload, cfg):  # C5
    all_failed = R.enrich_payload(payload, ScriptedWriter(R.LLMRequestError("HTTP 503")), cfg=cfg, now=NOW)
    assert R._writer_label(all_failed) == "rule-based templates (Gemini unavailable)"
    assert "write-ups: rule-based templates (Gemini unavailable)" in R.render_newsletter(all_failed)
    mixed = copy.deepcopy(all_failed)
    mixed["summary"].update(n_llm=5, n_template=len(payload["games"]) - 5)
    assert R._writer_label(mixed).startswith("fake-gemini for 5 of")


def test_run_wide_errors_abort_without_writing(payload, cfg):  # runtime F1
    cfg.predictions_dir.mkdir(parents=True)
    cfg.current_week_json.write_text(json.dumps(payload))
    for error in (
        _api_error(genai_errors.ClientError, 404, "NOT_FOUND", "models/gemini-2.5-flsh is not found"),
        _api_error(genai_errors.ClientError, 400, "FAILED_PRECONDITION", "User location is not supported"),
    ):
        writer = R.GeminiReportWriter(cfg=cfg, client=_FakeClient(*[error] * 20))
        with pytest.raises(R.LLMConfigError):
            R.run_report(cfg, writer=writer, now=NOW)
        assert not cfg.enriched_report_json.exists() and not list(cfg.reports_dir.glob("*.md"))


def test_expired_key_and_daily_quota_are_fatal(cfg):  # runtime F2, F3
    expired = _api_error(
        genai_errors.ClientError,
        400,
        "INVALID_ARGUMENT",
        "API key expired. Please renew the API key.",
        details=[{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_EXPIRED"}],
    )
    assert R.classify_api_error(expired) is R.LLMAuthError
    daily = _api_error(
        genai_errors.ClientError,
        429,
        "RESOURCE_EXHAUSTED",
        "quota exceeded",
        details=[
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
            }
        ],
    )
    client = _FakeClient(daily, "{}")
    with pytest.raises(R.LLMQuotaError):
        R.GeminiReportWriter(cfg=cfg, client=client).complete(R.SYSTEM_INSTRUCTION, "p")
    assert len(client.models.calls) == 1  # no futile retries
    per_minute = _api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "rate limited")
    assert R.classify_api_error(per_minute) is None  # retried


def test_empty_reply_uses_an_attempt_not_the_whole_budget(facts):  # runtime F4
    f = facts[0]
    writer = ScriptedWriter(R.LLMEmptyResponseError("no text"), _grounded_json(f))
    result = R.generate_report(writer, f, max_attempts=3)
    assert (result.generator, result.attempts) == ("fake-gemini", 2)
    assert "no text" in result.validation_errors[0]


def test_missing_tier_rules_is_a_contract_b_error(payload):  # runtime F5
    broken = copy.deepcopy(payload)
    del broken["tier_rules"]["LEAN"]
    with pytest.raises(M.ContractError, match="tier_rules"):
        M.validate_contract_b(broken)


def test_stale_contract_b_is_flagged(payload):  # critic: stale input
    later = datetime(2026, 10, 27, 12, tzinfo=UTC)
    warnings = R.stale_contract_b_warnings(payload, later)
    assert any("days ago" in w for w in warnings) and any("already kicked off" in w for w in warnings)
    assert R.stale_contract_b_warnings(payload, NOW) == []


def test_playoff_titles_and_track_record_follow_the_data(payload, cfg):  # contract C6, C9
    playoff = copy.deepcopy(payload)
    for g in playoff["games"]:
        g["game_type"] = "SB"
    contract_c = R.enrich_payload(playoff, None, cfg=cfg, now=NOW)
    assert R.render_newsletter(contract_c).startswith("# NFL Super Bowl Scouting Report — 2026 Season")
    model_wins = {
        "test_seasons": [2018, 2025],
        "pooled": {
            "model": {"brier": 0.20, "n_games": 9},
            "market": {"brier": 0.21, "n_games": 9},
            "final": {"brier": 0.205, "n_games": 9},
        },
    }
    assert "The model has been the most accurate" in R._track_record(model_wins, 0.3)


def test_main_exit_codes(payload, cfg, monkeypatch):  # degraded runs are visible to orchestration
    cfg.predictions_dir.mkdir(parents=True)
    cfg.current_week_json.write_text(json.dumps(payload))
    monkeypatch.setattr(R, "settings", cfg)
    monkeypatch.setattr(R, "GeminiReportWriter", lambda cfg, model=None: ScriptedWriter(R.LLMRequestError("HTTP 503")))
    assert R.main([]) == R.EXIT_DEGRADED and cfg.enriched_report_json.exists()
    assert R.main(["--template"]) == R.EXIT_OK

    def fatal(cfg, model=None):
        return ScriptedWriter(R.LLMConfigError("model not found"))

    cfg.enriched_report_json.unlink()
    monkeypatch.setattr(R, "GeminiReportWriter", fatal)
    assert R.main([]) == R.EXIT_ERROR and not cfg.enriched_report_json.exists()
