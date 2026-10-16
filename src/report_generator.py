"""Weekly scouting reports: Contract B -> LLM prose -> Contract C + Markdown newsletter.

Pipeline
    1. Read and validate ``data/predictions/current_week.json`` (Contract B).
    2. For each game, build a compact *fact sheet* from the payload and ask Gemini
       (``google-genai``, default ``gemini-2.5-flash``) for a JSON scouting report.
    3. Validate the reply with :class:`ScoutingReport` (Pydantic v2). That checks
       ``headline``, ``the_pick_summary``, ``key_mismatch`` and ``risk_factor`` for
       length and formatting, and checks grounding against the fact sheet: every
       percentage must come from it, the pick must be named, and no hype or
       betting language is allowed. A rejected reply is sent back with the
       validation errors for repair, up to ``LLM_MAX_ATTEMPTS`` attempts. After
       that the game gets a deterministic template report, validated by the same
       model and flagged in ``report_meta``.
    4. Write ``data/predictions/weekly_report_enriched.json`` (Contract C =
       Contract B plus ``report`` / ``report_meta`` per game, re-validated) and
       ``reports/week_XX_report.md``.

CLI::

    python -m src.report_generator              # Gemini (needs GEMINI_API_KEY)
    python -m src.report_generator --template   # no LLM: deterministic reports
    python -m src.report_generator --strict     # fail instead of falling back to a template
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from config import ODDS_TEAM_TO_NFLVERSE, Settings, settings, setup_logging
from src import ingest
from src import model as M

logger = logging.getLogger("src.report_generator" if __name__ == "__main__" else __name__)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
CONTRACT_NAME: Final[str] = "C"
CONTRACT_VERSION: Final[str] = "1.0.0"
PROVIDER: Final[str] = "google-genai"
TEMPLATE_GENERATOR: Final[str] = "template"
NEWSLETTER_NAME: Final[str] = "week_{week:02d}_report.md"
TIER_ORDER: Final[tuple[str, ...]] = M.TIERS  # LOCK, VALUE EDGE, LEAN, COIN FLIP
PERCENT_TOLERANCE: Final[float] = 0.55  # a quoted percentage may round a fact-sheet value to the nearest integer
BANNED_PHRASES: Final[tuple[str, ...]] = (
    "guarantee",
    "sure thing",
    "can't lose",
    "cannot lose",
    "can not lose",
    "free money",
    "lock of the",
    "mortal lock",
    "bet the house",
    "bet your",
    "max bet",
    "hammer the",
)
_PERCENT_RE: Final = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|percentage[- ]points?\b|pct\b)", re.IGNORECASE)
_HTML_RE: Final = re.compile(r"<\s*/?\s*[a-zA-Z]")

Mode = Literal["llm", "template"]


class ReportError(RuntimeError):
    """The report could not be generated."""


class MissingLLMKeyError(ReportError):
    """No Gemini API key configured."""


class LLMAuthError(ReportError):
    """The Gemini API rejected the key or project; retrying will not help."""


class LLMRequestError(ReportError):
    """A Gemini request failed (network, quota, server, empty or blocked response)."""


class ContractCError(ValueError):
    """A payload violates Contract C."""


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #
def _team_directory() -> dict[str, str]:
    names: dict[str, str] = {}
    for name, abbr in ODDS_TEAM_TO_NFLVERSE.items():
        names.setdefault(abbr, name)  # first listing is the current name (e.g. Commanders before Football Team)
    return names


TEAM_NAMES: Final[dict[str, str]] = _team_directory()


def team_name(abbr: str) -> str:
    return TEAM_NAMES.get(abbr, abbr)


def team_nickname(abbr: str) -> str:
    return team_name(abbr).split()[-1] if abbr in TEAM_NAMES else abbr


def team_aliases(abbr: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((abbr, team_name(abbr), team_nickname(abbr))))


# --------------------------------------------------------------------------- #
# Fact sheet (the only information the LLM may use)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class GameFacts:
    """Everything a report may say about one game, plus what grounding validation needs."""

    game_id: str
    sheet: dict[str, Any]  # sent to the LLM as JSON
    pick_abbr: str
    other_abbr: str
    allowed_percentages: frozenset[float]
    team_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def pick_aliases(self) -> tuple[str, ...]:
        return self.team_aliases[self.pick_abbr]


def _pct(p: float) -> float:
    return round(100.0 * p, 1)


def _strength(impact_logit: float) -> str:
    size = abs(impact_logit)
    return "strong" if size >= 0.4 else "moderate" if size >= 0.15 else "slight"


_TEAM_METRIC_TEXT: Final[dict[str, str]] = {
    "off_pass_epa": "offense EPA per dropback",
    "off_rush_epa": "offense EPA per rush",
    "def_pass_epa": "pass defense EPA allowed per dropback",
    "def_rush_epa": "run defense EPA allowed per rush",
    "off_early_down_sr": "early-down success rate",
    "qb_cpoe": "starting QB CPOE",
}
_MATCHUP_TEXT: Final[dict[str, str]] = {
    "net_pass_epa_mismatch": "Net passing EPA mismatch",
    "net_rush_epa_mismatch": "Net rushing EPA mismatch",
    "diff_early_down_sr": "Early-down success rate gap",
    "diff_qb_cpoe": "Quarterback CPOE gap",
    "net_rest_days": "Rest advantage",
    "is_divisional": "Divisional game",
    "is_neutral_site": "Neutral site",
    "is_outdoor": "Outdoor venue",
    "is_high_wind_outdoor": "High wind (15+ mph)",
    "prior_weight": "Early-season reliance on last season",
}


def describe_driver(feature: str, value: float, home: str, away: str) -> tuple[str, str]:
    """``(stat, value_text)`` for a driver, phrased from the team's side, never the home team's sign.

    Matchup features are home-minus-away differences, so a negative value is an
    edge for the away team. They are described by magnitude and the team they
    favour. Team-level features name their team instead of "Home"/"Away".
    """
    for prefix, team in (("home_", home), ("away_", away)):
        if feature.startswith(prefix) and feature.endswith("_pre"):
            metric = feature[len(prefix) : -len("_pre")]
            stat = f"{team_name(team)} {_TEAM_METRIC_TEXT.get(metric, metric)}"
            if metric == "off_early_down_sr":
                return stat, f"{_pct(value):.1f}%"
            if metric == "qb_cpoe":
                return stat, f"{value:+.1f} points"
            return stat, f"{value:+.3f} EPA per play"
    stat = _MATCHUP_TEXT.get(feature, feature)
    ahead = team_name(home if value > 0 else away)
    if feature.startswith("is_"):
        return stat, "yes" if value >= 0.5 else "no"
    if feature == "prior_weight":
        return stat, f"{_pct(value):g}% of the estimate still comes from last season"
    if value == 0:
        return stat, "even"
    if feature == "net_rest_days":
        return stat, f"{abs(value):.0f} more days of rest for the {ahead}"
    if feature == "diff_early_down_sr":
        return stat, f"{abs(_pct(value)):.1f} percentage points in favor of the {ahead}"
    if feature == "diff_qb_cpoe":
        return stat, f"{abs(value):.1f} CPOE points in favor of the {ahead}"
    return stat, f"{abs(value):.3f} EPA per play in favor of the {ahead}"


_SOURCE_TEXT: Final[dict[str, str]] = {
    "odds_api": "de-vigged live sportsbook consensus",
    "odds_api_snapshot": "de-vigged saved sportsbook consensus",
    "nflverse_line": "de-vigged pre-game moneyline",
    "nflverse_close": "de-vigged closing moneyline",
}
_UNKNOWN_INPUT_TEXT: Final[dict[str, str]] = {
    "is_high_wind_outdoor": "wind forecast",
    "is_outdoor": "retractable roof open/closed",
}


def game_facts(game: dict[str, Any], payload: dict[str, Any]) -> GameFacts:
    """Fact sheet for one Contract B game. Deterministic, so Contract C validation can rebuild it."""
    home, away = game["home_team"], game["away_team"]
    side = game["pick"]["side"]
    pick, other = (home, away) if side == "home" else (away, home)
    market = game["market"]
    weight = payload["model"]["market_weight"]
    venue = game["venue"]

    def flag(value: bool | None, unknown: str = "unknown before kickoff") -> str:
        return unknown if value is None else ("yes" if value else "no")

    rest = game["net_rest_days"]
    if rest is None or rest == 0:
        rest_text = "even rest"
    else:
        rest_text = f"{team_name(home if rest > 0 else away)} have {abs(rest)} more days of rest"

    sheet: dict[str, Any] = {
        "matchup": f"{team_name(away)} at {team_name(home)}",
        "game_type": game["game_type"],
        "status": game["status"],
        "kickoff_utc": game["kickoff_utc"],
        "home_team": {"abbr": home, "name": team_name(home)},
        "away_team": {"abbr": away, "name": team_name(away)},
        "venue": {
            "neutral_site": flag(venue["is_neutral_site"]),
            "outdoor": flag(venue["is_outdoor"]),
            "high_wind": flag(venue["is_high_wind"], "unknown (no forecast)"),
            "divisional_game": flag(venue["is_divisional"]),
        },
        "rest": rest_text,
        "pick": {
            "team": team_name(pick),
            "abbr": pick,
            "final_win_probability_pct": _pct(game["pick"]["win_prob"]),
            "opponent_win_probability_pct": _pct(1.0 - game["pick"]["win_prob"]),
        },
        "confidence_tier": game["confidence_tier"],
        "tier_meaning": payload["tier_rules"][game["confidence_tier"]],
        "model_win_probability_pct": {
            pick: _pct(game["model"][f"{side}_wp"]),
            other: _pct(1.0 - game["model"][f"{side}_wp"]),
        },
        "market": (
            {
                "win_probability_pct": {pick: _pct(market[f"{side}_wp"]), other: _pct(1.0 - market[f"{side}_wp"])},
                "home_spread": market["home_spread"],
                "source": _SOURCE_TEXT.get(market["source"], market["source"]),
            }
            if market is not None
            else "no market line available"
        ),
        "model_minus_market_on_pick_pct_points": game["model_edge_pct"],
        "model_value_side": (
            {"team": team_name(game["value_team"]), "edge_pct_points": game["value_edge_pct"]}
            if game["value_team"]
            else None
        ),
        "blend": f"final = {_pct(weight):g}% market + {_pct(1 - weight):g}% model",
        "top_drivers": [
            {
                "stat": stat,
                "value": value_text,
                "favors": team_name(d["favors"]),
                "strength": _strength(d["impact_logit"]),
            }
            for d in game["top_drivers"]
            for stat, value_text in [describe_driver(d["feature"], d["value"], home, away)]
        ],
        "unknown_before_kickoff": [_UNKNOWN_INPUT_TEXT.get(c, c) for c in game["imputed_inputs"]],
    }
    if game["result"] is not None:
        sheet["final_score"] = {home: game["result"]["home_score"], away: game["result"]["away_score"]}
    return GameFacts(
        game_id=game["game_id"],
        sheet=sheet,
        pick_abbr=pick,
        other_abbr=other,
        allowed_percentages=frozenset(_collect_percentages(sheet)),
        team_aliases={home: team_aliases(home), away: team_aliases(away)},
    )


def _collect_percentages(node: Any, key: str = "") -> set[float]:
    """Every percentage the fact sheet states: numeric ``*_pct*`` fields and ``N%`` inside strings."""
    found: set[float] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            found |= _collect_percentages(v, k if "pct" in k else key)
    elif isinstance(node, list):
        for v in node:
            found |= _collect_percentages(v, key)
    elif isinstance(node, (int, float)) and not isinstance(node, bool) and "pct" in key and math.isfinite(node):
        found.add(round(abs(float(node)), 1))
    elif isinstance(node, str):
        found |= {round(float(m), 1) for m in _PERCENT_RE.findall(node)}
    return found


# --------------------------------------------------------------------------- #
# Pydantic contract for the LLM output
# --------------------------------------------------------------------------- #
class ScoutingReport(BaseModel):
    """Validated scouting report for one game.

    Format checks always run. Grounding checks run when validated with
    ``context={"facts": GameFacts}``: allowed percentages, pick named,
    no hype or betting language, no HTML.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    headline: str = Field(
        min_length=10, max_length=90, description="One-line headline, no markdown, no trailing period."
    )
    the_pick_summary: str = Field(
        min_length=40,
        max_length=420,
        description=(
            "1-2 sentences: name the pick, its final win probability and confidence tier, "
            "and how the model compares with the market."
        ),
    )
    key_mismatch: str = Field(
        min_length=40,
        max_length=420,
        description="1-2 sentences on the top statistical driver(s) in plain English and which team they favor.",
    )
    risk_factor: str = Field(
        min_length=30,
        max_length=420,
        description="1-2 sentences on the main reason the pick could be wrong.",
    )

    @field_validator("headline", "the_pick_summary", "key_mismatch", "risk_factor")
    @classmethod
    def _single_paragraph(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("must be a single paragraph (no line breaks)")
        if value.lstrip().startswith(("#", ">", "-", "*", "|")):
            raise ValueError("must be plain prose, not markdown")
        if _HTML_RE.search(value):
            raise ValueError("must not contain HTML")
        return re.sub(r"\s{2,}", " ", value)

    @field_validator("headline")
    @classmethod
    def _headline_style(cls, value: str) -> str:
        if value.endswith("."):
            raise ValueError("headline must not end with a period")
        if "**" in value or "__" in value:
            raise ValueError("headline must not use markdown emphasis")
        return value

    @model_validator(mode="after")
    def _grounded(self, info: ValidationInfo) -> ScoutingReport:
        facts = (info.context or {}).get("facts") if isinstance(info.context, dict) else None
        if facts is None:
            return self
        text = " ".join((self.headline, self.the_pick_summary, self.key_mismatch, self.risk_factor))
        lowered = text.lower()
        banned = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
        if banned:
            raise ValueError(f"hype or betting language is not allowed: {banned}")
        if not any(_mentions(self.the_pick_summary, alias) for alias in facts.pick_aliases):
            raise ValueError(f"the_pick_summary must name the pick ({' / '.join(facts.pick_aliases)})")
        unsupported = sorted(
            {
                float(m)
                for m in _PERCENT_RE.findall(text)
                if not any(abs(float(m) - a) <= PERCENT_TOLERANCE for a in facts.allowed_percentages)
            }
        )
        if unsupported:
            raise ValueError(
                f"percentages not in the fact sheet: {unsupported} (allowed: {sorted(facts.allowed_percentages)})"
            )
        return self


def _mentions(text: str, alias: str) -> bool:
    if alias.isupper() and len(alias) <= 3:  # abbreviations: exact case, word boundaries
        return re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", text) is not None
    return re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE) is not None


class _ResponseSchema(BaseModel):
    """Plain schema sent to Gemini as ``response_schema`` (validation happens in :class:`ScoutingReport`)."""

    headline: str = Field(max_length=90, description=ScoutingReport.model_fields["headline"].description)
    the_pick_summary: str = Field(
        max_length=420, description=ScoutingReport.model_fields["the_pick_summary"].description
    )
    key_mismatch: str = Field(max_length=420, description=ScoutingReport.model_fields["key_mismatch"].description)
    risk_factor: str = Field(max_length=420, description=ScoutingReport.model_fields["risk_factor"].description)


def validate_report(data: dict[str, Any] | str, facts: GameFacts) -> ScoutingReport:
    """Parse (JSON text or dict) and validate a report against its fact sheet."""
    if isinstance(data, str):
        return ScoutingReport.model_validate_json(_strip_code_fence(data), context={"facts": facts})
    return ScoutingReport.model_validate(data, context={"facts": facts})


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return match.group(1) if match else stripped


def _format_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc']) or 'report'}: {err['msg']}" for err in exc.errors()]


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION: Final[str] = """\
You are the lead analyst of a weekly NFL scouting newsletter powered by a quantitative prediction engine.
You turn one game's model output into clear, accurate prose for informed fans.

Rules:
1. Use ONLY the facts in the JSON fact sheet. Never invent players, injuries, weather, records, trends,
   scores or numbers.
2. Any percentage you write must appear in the fact sheet (you may round it to a whole number). Do not compute new ones.
3. Refer to teams by the names given in the fact sheet.
4. Never describe a pick as guaranteed or certain, and never tell readers to bet.
5. Explain stats in plain English: EPA per play measures how much each play adds to expected points;
   success rate is the share of plays that keep an offense on schedule; CPOE is completion percentage
   above expectation. "Favors" names the team the stat helps.
6. Respond with JSON only, matching the schema:
   - headline: one line, at most 90 characters, no markdown, no trailing period.
   - the_pick_summary: 1-2 sentences naming the pick, its final win probability and its confidence tier,
     and how the model compares with the market.
   - key_mismatch: 1-2 sentences on the top statistical driver(s) and which team they favor.
   - risk_factor: 1-2 sentences on the main way this pick could go wrong (for example the model and market
     disagreeing, a near coin-flip probability, inputs unknown before kickoff, or a rest disadvantage).
"""


def build_prompt(facts: GameFacts, *, feedback: Sequence[str] = (), previous: str | None = None) -> str:
    prompt = "Write the scouting report for this game.\n\nFACT SHEET (JSON):\n" + json.dumps(facts.sheet, indent=2)
    if feedback:
        prompt += (
            "\n\nYour previous response was rejected by the validator:\n"
            + "\n".join(f"- {line}" for line in feedback)
            + (f"\n\nPrevious response:\n{previous}" if previous else "")
            + "\n\nReturn a corrected JSON report that fixes every issue."
        )
    return prompt


# --------------------------------------------------------------------------- #
# LLM writers
# --------------------------------------------------------------------------- #
class ReportWriter(Protocol):
    name: str

    def complete(self, system_instruction: str, prompt: str) -> str:
        """Return the raw JSON text of a report."""
        ...


_AUTH_HINTS: Final[tuple[str, ...]] = ("api key not valid", "api_key_invalid", "permission", "unauthenticated")


class GeminiReportWriter:
    """``google-genai`` client wrapper: JSON-mode request with schema, timeouts and transport retries."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cfg: Settings = settings,
        model: str | None = None,
        client: Any | None = None,
        max_retries: int = 4,
    ) -> None:
        from google import genai
        from google.genai import types

        key = api_key if api_key is not None else cfg.gemini_api_key
        if client is None and (not key or not key.strip()):
            raise MissingLLMKeyError(
                "Gemini API key is missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env, "
                "or run with --template. Keys: https://aistudio.google.com/apikey"
            )
        self._types = types
        self.name = model or cfg.gemini_model
        self.cfg = cfg
        self.max_retries = max_retries
        self._client = client or genai.Client(
            api_key=key.strip(),
            http_options=types.HttpOptions(timeout=int(cfg.llm_timeout_s * 1000)),
        )
        config: dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": cfg.llm_temperature,
            "response_mime_type": "application/json",
            "response_schema": _ResponseSchema,
        }
        if cfg.llm_thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=cfg.llm_thinking_budget)
        self._config = types.GenerateContentConfig(**config)

    def complete(self, system_instruction: str, prompt: str) -> str:
        from google.genai import errors

        config = self._config
        if system_instruction != SYSTEM_INSTRUCTION:
            config = config.model_copy(update={"system_instruction": system_instruction})
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._client.models.generate_content(model=self.name, contents=prompt, config=config)
            except errors.APIError as exc:
                message = (exc.message or str(exc)).lower()
                if exc.code in (401, 403) or (exc.code == 400 and any(h in message for h in _AUTH_HINTS)):
                    raise LLMAuthError(f"Gemini rejected the request (HTTP {exc.code}): {exc.message}") from None
                retryable = exc.code == 429 or (exc.code is not None and exc.code >= 500)
                if not retryable or attempt > self.max_retries:
                    raise LLMRequestError(f"Gemini request failed (HTTP {exc.code}): {exc.message}") from None
                delay = _retry_delay(exc, attempt)
                logger.warning("Gemini HTTP %s (%s); retrying in %.1fs", exc.code, exc.status, delay)
                time.sleep(delay)
                continue
            except Exception as exc:  # network errors from httpx and similar transport failures
                if attempt > self.max_retries:
                    raise LLMRequestError(f"Gemini request failed: {type(exc).__name__}: {exc}") from None
                delay = _retry_delay(None, attempt)
                logger.warning("Gemini transport error (%s); retrying in %.1fs", type(exc).__name__, delay)
                time.sleep(delay)
                continue
            text = _response_text(response)
            if not text:
                raise LLMRequestError(f"Gemini returned no text (finish reason: {_finish_reason(response)})")
            return text
        raise AssertionError("unreachable")  # pragma: no cover


def _retry_delay(exc: Any, attempt: int) -> float:
    """Server-suggested RetryInfo delay when present, else exponential backoff with jitter (capped at 60s)."""
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in (details.get("error") or {}).get("details") or []:
            delay = item.get("retryDelay") if isinstance(item, dict) else None
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return min(float(delay[:-1]), 60.0)
                except ValueError:
                    pass
    base = min(2.0 ** (attempt - 1), 30.0)
    return base + random.uniform(0, base / 4)


def _response_text(response: Any) -> str | None:
    try:
        return response.text
    except (ValueError, AttributeError):  # blocked or empty candidates
        return None


def _finish_reason(response: Any) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except (AttributeError, IndexError, TypeError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Deterministic template (no LLM; also the per-game fallback)
# --------------------------------------------------------------------------- #
def _article(pct: float) -> str:
    """'an' before numbers spoken with a vowel sound (8, 11, 18, 80-89...), else 'a'."""
    whole = int(pct)
    return "an" if str(whole).startswith("8") or whole in (11, 18) else "a"


def template_report(facts: GameFacts) -> ScoutingReport:
    """Rule-based report built only from fact-sheet values; always passes grounding validation."""
    sheet = facts.sheet
    pick, other = facts.pick_abbr, facts.other_abbr
    pick_pct = sheet["pick"]["final_win_probability_pct"]
    tier = sheet["confidence_tier"]
    model_pick = sheet["model_win_probability_pct"][pick]
    market = sheet["market"] if isinstance(sheet["market"], dict) else None

    headline = {
        "LOCK": f"{team_nickname(pick)} are clear favorites over the {team_nickname(other)}",
        "VALUE EDGE": f"Model sees value in the {team_nickname(pick)} against the {team_nickname(other)}",
        "LEAN": f"Slight lean to the {team_nickname(pick)} over the {team_nickname(other)}",
        "COIN FLIP": f"{team_nickname(pick)} and {team_nickname(other)} are close to a coin flip",
    }[tier]

    summary = (
        f"The blended forecast makes the {team_name(pick)} {_article(pick_pct)} {pick_pct:.1f}% favorite ({tier})."
    )
    if market is not None:
        summary += (
            f" The model has them at {model_pick:.1f}% against the market's {market['win_probability_pct'][pick]:.1f}%."
        )
    else:
        summary += " No market line was available, so this is the model's number alone."

    drivers = sheet["top_drivers"]
    if drivers:
        first = drivers[0]
        mismatch = f"{first['stat']} is the biggest factor ({first['value']}), a {first['strength']} edge"
        mismatch += "." if first["favors"] in first["value"] else f" for the {first['favors']}."
        if len(drivers) > 1:
            second = drivers[1]
            mismatch += f" Next: {second['stat']} ({second['value']})"
            mismatch += "." if second["favors"] in second["value"] else f", favoring the {second['favors']}."
    else:
        mismatch = (
            "No single statistic dominates this matchup; the forecast reflects a broad mix of team efficiency inputs."
        )

    value = sheet["model_value_side"]
    rest = sheet["rest"]
    if value and value["team"] != team_name(pick):
        risk = (
            f"The model rates the {value['team']} {value['edge_pct_points']:.1f} percentage points above the market, "
            "so it is less sure of this pick than the line suggests."
        )
    elif market is None:
        risk = "There is no market line to check the model against, so this pick rests on the model alone."
    elif pick_pct < 55.0:
        risk = f"At {pick_pct:.1f}%, this is close to a coin flip, and one turnover could swing it."
    elif sheet["unknown_before_kickoff"]:
        unknown = sheet["unknown_before_kickoff"]
        verb = "was" if len(unknown) == 1 else "were"
        risk = (
            f"The {' and '.join(unknown)} {verb} not known when this forecast was made, so conditions could shift it."
        )
    elif rest != "even rest" and not rest.startswith(team_name(pick)):
        risk = f"Rest works against the pick: the {rest}."
    else:
        risk = f"Even as favorites, the {team_name(pick)} lose this matchup about {100 - pick_pct:.0f}% of the time."
    return validate_report(
        {"headline": headline, "the_pick_summary": summary, "key_mismatch": mismatch, "risk_factor": risk}, facts
    )


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GeneratedReport:
    report: ScoutingReport
    generator: str
    attempts: int
    fallback_reason: str | None = None
    validation_errors: list[str] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "attempts": self.attempts,
            "fallback_reason": self.fallback_reason,
            "validation_errors": self.validation_errors,
        }


def generate_report(
    writer: ReportWriter | None,
    facts: GameFacts,
    *,
    max_attempts: int,
    allow_fallback: bool = True,
) -> GeneratedReport:
    """LLM report with validate-and-repair retries; template fallback (or error when ``allow_fallback`` is False)."""
    if writer is None:
        return GeneratedReport(template_report(facts), TEMPLATE_GENERATOR, 0)

    feedback: list[str] = []
    previous: str | None = None
    history: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw = writer.complete(SYSTEM_INSTRUCTION, build_prompt(facts, feedback=feedback, previous=previous))
        except LLMAuthError:
            raise
        except LLMRequestError as exc:
            reason = str(exc)
            break
        try:
            return GeneratedReport(validate_report(raw, facts), writer.name, attempt, validation_errors=history)
        except ValidationError as exc:
            feedback = _format_errors(exc)
            previous = raw[:2000]
            history += [f"attempt {attempt}: {line}" for line in feedback]
            reason = f"failed validation after {attempt} attempt(s)"
            logger.info(
                "%s: report rejected (attempt %d/%d): %s", facts.game_id, attempt, max_attempts, "; ".join(feedback)
            )
    else:
        attempt = max_attempts

    if not allow_fallback:
        raise ReportError(f"{facts.game_id}: {reason}; last errors: {history[-3:]}")
    logger.warning("%s: using template report (%s)", facts.game_id, reason)
    return GeneratedReport(template_report(facts), TEMPLATE_GENERATOR, attempt, reason, history)


# --------------------------------------------------------------------------- #
# Contract C
# --------------------------------------------------------------------------- #
def load_contract_b(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.model predict` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    M.validate_contract_b(payload)
    return payload


def enrich_payload(
    payload: dict[str, Any],
    writer: ReportWriter | None,
    *,
    cfg: Settings = settings,
    now: datetime | None = None,
    allow_fallback: bool = True,
    newsletter_path: str | None = None,
) -> dict[str, Any]:
    """Contract B payload -> Contract C payload (validated)."""
    now = now or datetime.now(UTC)
    games = payload["games"]
    facts = [game_facts(g, payload) for g in games]

    def run(f: GameFacts) -> GeneratedReport:
        return generate_report(writer, f, max_attempts=cfg.llm_max_attempts, allow_fallback=allow_fallback)

    workers = 1 if writer is None else min(cfg.llm_max_concurrency, max(len(facts), 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        generated = list(pool.map(run, facts))

    enriched_games = [
        {**game, "report": result.report.model_dump(), "report_meta": result.meta()}
        for game, result in zip(games, generated, strict=True)
    ]
    n_template = sum(r.generator == TEMPLATE_GENERATOR for r in generated)
    contract_c = {
        "contract": CONTRACT_NAME,
        "schema_version": CONTRACT_VERSION,
        "generated_at": M._iso(now),
        "season": payload["season"],
        "week": payload["week"],
        "source": {
            "contract": payload["contract"],
            "schema_version": payload["schema_version"],
            "generated_at": payload["generated_at"],
        },
        "generator": {
            "provider": PROVIDER if writer is not None else None,
            "model": writer.name if writer is not None else None,
            "mode": "llm" if writer is not None else "template",
            "temperature": cfg.llm_temperature if writer is not None else None,
            "max_attempts": cfg.llm_max_attempts if writer is not None else 0,
        },
        "summary": {
            "n_games": len(games),
            "n_llm": len(games) - n_template,
            "n_template": n_template,
            "tiers": {tier: sum(g["confidence_tier"] == tier for g in games) for tier in TIER_ORDER},
        },
        "newsletter_path": newsletter_path,
        "model": payload["model"],
        "tier_rules": payload["tier_rules"],
        "games": enriched_games,
    }
    validate_contract_c(contract_c)
    return contract_c


def contract_b_view(payload: dict[str, Any]) -> dict[str, Any]:
    """The Contract B payload embedded in a Contract C payload."""
    source = payload.get("source") or {}
    return {
        "contract": source.get("contract"),
        "schema_version": source.get("schema_version"),
        "generated_at": source.get("generated_at"),
        "season": payload.get("season"),
        "week": payload.get("week"),
        "model": payload.get("model"),
        "tier_rules": payload.get("tier_rules"),
        "games": [
            {k: v for k, v in g.items() if k not in ("report", "report_meta")} if isinstance(g, dict) else g
            for g in payload.get("games") or []
        ],
    }


def validate_contract_c(payload: Any) -> None:
    """Raise :class:`ContractCError` listing every violation.

    The embedded Contract B must validate, and every report must re-validate
    against the fact sheet rebuilt from its own game, including grounding.
    """
    if not isinstance(payload, dict):
        raise ContractCError("Contract C violations:\n  - payload must be a JSON object")
    problems: list[str] = []
    try:
        _check_contract_c(payload, problems)
    except Exception as exc:  # noqa: BLE001 - malformed input must surface as a ContractCError
        problems.append(f"malformed payload ({type(exc).__name__}: {exc})")
    if problems:
        raise ContractCError("Contract C violations:\n  - " + "\n  - ".join(problems))


def _check_contract_c(payload: dict[str, Any], problems: list[str]) -> None:
    if payload.get("contract") != CONTRACT_NAME:
        problems.append(f"contract must be {CONTRACT_NAME!r}")
    if not isinstance(payload.get("schema_version"), str):
        problems.append("schema_version must be a string")
    if M._parse_iso(payload.get("generated_at")) is None:
        problems.append("generated_at must be ISO-8601")
    view = contract_b_view(payload)
    try:
        M.validate_contract_b(view)
    except M.ContractError as exc:
        problems.append(f"embedded Contract B is invalid: {exc}")
        return

    generator = payload.get("generator")
    if not isinstance(generator, dict) or generator.get("mode") not in ("llm", "template"):
        problems.append("generator.mode must be 'llm' or 'template'")
        return
    allowed_generators = {TEMPLATE_GENERATOR} | ({generator.get("model")} if generator["mode"] == "llm" else set())
    counts: Counter[str] = Counter()
    for n, game in enumerate(payload["games"]):
        where = f"games[{n}] ({game.get('game_id')})"
        meta = game.get("report_meta")
        if not isinstance(meta, dict) or meta.get("generator") not in allowed_generators:
            problems.append(f"{where}: report_meta.generator must be one of {sorted(map(str, allowed_generators))}")
            continue
        if not M._is_int(meta.get("attempts")) or meta["attempts"] < 0:
            problems.append(f"{where}: report_meta.attempts must be a non-negative int")
        counts[meta["generator"]] += 1
        try:
            validate_report(game.get("report"), game_facts(game, view))
        except ValidationError as exc:
            problems.append(f"{where}: report invalid: {'; '.join(_format_errors(exc))}")

    summary = payload.get("summary")
    n_games = len(payload["games"])
    expected = {
        "n_games": n_games,
        "n_template": counts[TEMPLATE_GENERATOR],
        "n_llm": n_games - counts[TEMPLATE_GENERATOR],
    }
    if not isinstance(summary, dict) or any(summary.get(k) != v for k, v in expected.items()):
        problems.append(f"summary counts must be {expected}")
    path = payload.get("newsletter_path")
    if path is not None and not isinstance(path, str):
        problems.append("newsletter_path must be a string or null")


# --------------------------------------------------------------------------- #
# Markdown newsletter
# --------------------------------------------------------------------------- #
def _local_kickoff(iso: str) -> str:
    kickoff = M._parse_iso(iso)
    if kickoff is None:
        return iso
    try:
        from zoneinfo import ZoneInfo

        local = kickoff.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 - no tz database on this platform
        return kickoff.strftime("%a %b %d, %H:%M UTC")
    return local.strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")


def _edge_text(game: dict[str, Any]) -> str:
    edge = game["model_edge_pct"]
    return "n/a" if edge is None else f"{edge:+.1f} pts"


def _result_text(game: dict[str, Any]) -> str:
    if game["status"] != "final" or not game["result"]:
        return "in progress" if game["status"] == "in_progress" else "—"
    home, away = game["result"]["home_score"], game["result"]["away_score"]
    winner = game["home_team"] if home > away else game["away_team"] if away > home else None
    score = f"{game['away_team']} {away}, {game['home_team']} {home}"
    if winner is None:
        return f"{score} (tie)"
    return f"{score} (pick {'won' if winner == game['pick']['team'] else 'lost'})"


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_newsletter(contract_c: dict[str, Any], backtest_report: dict[str, Any] | None = None) -> str:
    """Standalone Markdown newsletter for the week."""
    season, week = contract_c["season"], contract_c["week"]
    model = contract_c["model"]
    gen = contract_c["generator"]
    generated = M._parse_iso(contract_c["generated_at"])
    writer = gen["model"] if gen["mode"] == "llm" else "rule-based templates"
    games = contract_c["games"]
    lines = [
        f"# NFL Week {week} Scouting Report — {season} Season",
        "",
        f"_Generated {generated:%b %d, %Y %H:%M} UTC · win probabilities: calibrated XGBoost blended with the "
        f"de-vigged market ({model['market_weight']:.0%} market weight) · write-ups: {writer}_",
        "",
        "## At a glance",
        "",
        "| Matchup | Kickoff | Pick | Win prob | Tier | Model vs market | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in games:
        lines.append(
            f"| {g['away_team']} @ {g['home_team']} | {_md_cell(_local_kickoff(g['kickoff_utc']))} "
            f"| {g['pick']['team']} | "
            f"{g['pick']['win_prob']:.1%} | {g['confidence_tier']} | {_edge_text(g)} | {_result_text(g)} |"
        )

    upcoming = [g for g in games if g["status"] != "final"]
    finished = [g for g in games if g["status"] == "final"]
    for tier in TIER_ORDER:
        group = [g for g in upcoming if g["confidence_tier"] == tier]
        if not group:
            continue
        lines += ["", f"## {tier.title()}{'s' if not tier.endswith('S') else ''}", ""]
        for g in sorted(group, key=lambda x: -x["pick"]["win_prob"]):
            lines += _game_section(g)
    if finished:
        lines += ["", "## Already played", ""]
        for g in finished:
            lines += _game_section(g)

    lines += ["", "## How to read this", ""]
    lines += [f"- **{tier}**: {contract_c['tier_rules'][tier]}." for tier in TIER_ORDER]
    lines.append(
        "- **Model vs market** is the model's win probability for the pick minus the de-vigged market's, in "
        "percentage points. The final number blends the two."
    )
    if backtest_report and isinstance(backtest_report.get("pooled"), dict):
        pooled = backtest_report["pooled"]
        seasons = backtest_report.get("test_seasons") or []
        try:
            lines.append(
                f"- **Track record** (walk-forward, out of sample, {seasons[0]}-{seasons[-1]}, "
                f"{pooled['final']['n_games']:,} games): Brier score model {pooled['model']['brier']:.4f}, market "
                f"{pooled['market']['brier']:.4f}, blend {pooled['final']['brier']:.4f} (lower is better). "
                "The market has "
                "been the stronger forecaster, which is why the blend leans on it."
            )
        except (KeyError, TypeError, IndexError):
            logger.warning("backtest report is missing pooled metrics; track record omitted")
    lines += [
        "",
        "_For information and entertainment only. Not betting advice: probabilities are estimates and upsets happen._",
        "",
    ]
    return "\n".join(lines)


def _game_section(g: dict[str, Any]) -> list[str]:
    report = g["report"]
    side = g["pick"]["side"]
    market = g["market"]
    market_text = (
        f"{g['pick']['team']} {market[f'{side}_wp']:.1%} ({_SOURCE_TEXT.get(market['source'], market['source'])})"
        if market
        else "no line"
    )
    lines = [
        f"### {g['away_team']} @ {g['home_team']} — {report['headline']}",
        "",
        f"**Pick:** {g['pick']['team']} {g['pick']['win_prob']:.1%} · **Tier:** {g['confidence_tier']} · "
        f"**Model:** {g['pick']['team']} {g['model'][f'{side}_wp']:.1%} · **Market:** {market_text} · "
        f"**Kickoff:** {_local_kickoff(g['kickoff_utc'])}",
        "",
        report["the_pick_summary"],
        "",
        f"**Key mismatch.** {report['key_mismatch']}",
        "",
        f"**Risk factor.** {report['risk_factor']}",
    ]
    if g["status"] == "final" and g["result"]:
        lines += [
            "",
            f"**Final:** {g['away_team']} {g['result']['away_score']}, {g['home_team']} {g['result']['home_score']}",
        ]
    if g["report_meta"]["generator"] == TEMPLATE_GENERATOR and g["report_meta"]["fallback_reason"]:
        lines += ["", "_Write-up generated from the rule-based template._"]
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def newsletter_path(cfg: Settings, week: int) -> Path:
    return cfg.reports_dir / NEWSLETTER_NAME.format(week=week)


def _load_backtest(cfg: Settings) -> dict[str, Any] | None:
    path = cfg.predictions_dir / "backtest_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s); track record omitted", path, exc)
        return None


def run_report(
    cfg: Settings = settings,
    *,
    mode: Mode = "llm",
    writer: ReportWriter | None = None,
    input_path: Path | None = None,
    allow_fallback: bool = True,
    now: datetime | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Read Contract B, generate reports, write Contract C and the newsletter. Returns the Contract C payload."""
    payload = load_contract_b(input_path or cfg.current_week_json)
    if mode == "llm" and writer is None:
        writer = GeminiReportWriter(cfg=cfg)
    if mode == "template":
        writer = None
    md_path = newsletter_path(cfg, payload["week"])
    rel_md = str(md_path.relative_to(cfg.project_root)) if md_path.is_relative_to(cfg.project_root) else str(md_path)

    started = time.perf_counter()
    contract_c = enrich_payload(
        payload, writer, cfg=cfg, now=now, allow_fallback=allow_fallback, newsletter_path=rel_md
    )
    summary = contract_c["summary"]
    logger.info(
        "Generated %d reports in %.1fs (%d %s, %d template)",
        summary["n_games"],
        time.perf_counter() - started,
        summary["n_llm"],
        contract_c["generator"]["model"] or "llm",
        summary["n_template"],
    )
    if write:
        ingest.write_text_atomic(json.dumps(contract_c, indent=2, allow_nan=False) + "\n", cfg.enriched_report_json)
        ingest.write_text_atomic(render_newsletter(contract_c, _load_backtest(cfg)), md_path)
        logger.info("Wrote %s and %s", cfg.enriched_report_json, md_path)
    return contract_c


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.report_generator", description="Generate the weekly scouting report."
    )
    parser.add_argument("--template", action="store_true", help="skip the LLM; write deterministic template reports")
    parser.add_argument("--strict", action="store_true", help="fail instead of falling back to a template report")
    parser.add_argument("--model", default=None, help=f"Gemini model (default {settings.gemini_model})")
    parser.add_argument(
        "--input", type=Path, default=None, help="Contract B JSON (default data/predictions/current_week.json)"
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    settings.ensure_dirs()
    try:
        writer = None if args.template else GeminiReportWriter(cfg=settings, model=args.model)
        contract_c = run_report(
            mode="template" if args.template else "llm",
            writer=writer,
            input_path=args.input,
            allow_fallback=not args.strict,
        )
    except MissingLLMKeyError as exc:
        logger.error("%s", exc)
        return 2
    except (ReportError, ContractCError, M.ContractError, FileNotFoundError, ValueError) as exc:
        logger.error("Report generation failed: %s", exc)
        return 1
    print(f"Wrote {settings.enriched_report_json} and {contract_c['newsletter_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
