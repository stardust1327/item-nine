"""Weekly scouting reports: Contract B -> LLM prose -> Contract C + Markdown newsletter.

Pipeline
    1. Read and validate ``data/predictions/current_week.json`` (Contract B).
    2. For each game, build a compact *fact sheet* from the payload and ask Gemini
       (``google-genai``, default ``gemini-3.8-flash``) for a JSON scouting report.
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
       ``reports/week_XX_YYYY.md``.

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
import unicodedata
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
CONTRACT_VERSION: Final[str] = "1.1.0"
PROVIDER: Final[str] = "google-genai"
TEMPLATE_GENERATOR: Final[str] = "template"
TIER_ORDER: Final[tuple[str, ...]] = M.TIERS  # LOCK, STAT MISMATCH, LEAN, COIN FLIP
SECTION_TITLES: Final[dict[str, str]] = {
    "LOCK": "Locks",
    "STAT MISMATCH": "Stat mismatches (model vs. market)",
    "LEAN": "Leans",
    "COIN FLIP": "Coin flips",
}
ALWAYS_ALLOWED_PERCENTAGES: Final[frozenset[float]] = frozenset({50.0})  # "a 50/50 game", "50% coin flip"
# Hype and betting calls, matched on normalised text (NFKC, ASCII quotes and dashes, no zero-width characters).
BANNED_PATTERNS: Final[tuple[str, ...]] = (
    r"(?<!no )(?<!not )(?<!nothing is )(?<!isn't )(?<!never )guarantee",
    r"sure[ -]thing",
    r"can'?t lose|cannot lose|can not lose|can'?t[ -]miss",
    r"free money",
    r"lock of the",
    r"mortal lock",
    r"slam[ -]dunk",
    r"bet the house|bet your|max bet|best bet|bet on|place a bet|\bwager",
    r"\bparlay",
    r"hammer the",
    r"against the spread|\bATS\b",
)
_BANNED_RE: Final = re.compile("|".join(f"(?:{p})" for p in BANNED_PATTERNS), re.IGNORECASE)
_PERCENT_RE: Final = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b|percentage[- ]points?\b|pct\b)", re.IGNORECASE
)
# Anything that is markup rather than prose: HTML tags/comments, code, links, images, URLs, emphasis, list starts.
_MARKUP_RE: Final = re.compile(
    r"<\s*[!/a-zA-Z]|```|~~|`|\]\(|!\[|https?://|\bwww\.|\*|__|^\s*(?:#|>|[-+]\s|\d+[.)]\s|\|)",
)
_TIER_TOKEN_RE: Final = re.compile(r"\b(" + "|".join(re.escape(t) for t in M.TIERS) + r")\b")

Mode = Literal["llm", "template"]


class ReportError(RuntimeError):
    """The report could not be generated."""


class MissingLLMKeyError(ReportError):
    """No Gemini API key configured."""


class LLMAuthError(ReportError):
    """The Gemini API rejected the key or project; retrying will not help."""


class LLMConfigError(ReportError):
    """A request error that every game would hit (unknown model, unsupported location, bad request).

    The run is aborted rather than falling back game by game.
    """


class LLMQuotaError(ReportError):
    """The daily request quota is exhausted; retrying today will not help."""


class LLMRequestError(ReportError):
    """A Gemini request failed for this game (network, rate limit, server error)."""


class LLMEmptyResponseError(LLMRequestError):
    """Gemini returned no text (empty or blocked candidate). Counts as one failed attempt."""


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


_EXTRA_NICKNAMES: Final[dict[str, tuple[str, ...]]] = {"SF": ("Niners",), "NYG": ("G-Men",), "TB": ("Bucs",)}


def team_city(abbr: str) -> str | None:
    """Market name ('Kansas City', 'Green Bay'), or None when two franchises share it (Los Angeles, New York)."""
    if abbr not in TEAM_NAMES:
        return None
    city = team_name(abbr).rsplit(" ", 1)[0]
    shared = sum(1 for other in TEAM_NAMES if team_name(other).rsplit(" ", 1)[0] == city)
    return city if shared == 1 else None


def team_aliases(abbr: str) -> tuple[str, ...]:
    """Ways prose may name a team: abbreviation, full name, nickname, unambiguous city, common nicknames."""
    city = team_city(abbr)
    names = (abbr, team_name(abbr), team_nickname(abbr), *([city] if city else []), *_EXTRA_NICKNAMES.get(abbr, ()))
    return tuple(dict.fromkeys(names))


def normalize_text(text: str) -> str:
    """NFKC, ASCII quotes/dashes, no zero-width or other format characters, single spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.translate(
        str.maketrans(
            {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2212": "-"}
        )
    )
    return re.sub(r"[ \t\u00a0]{2,}|\u00a0", " ", text).strip()


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
    tier: str
    pick_pct: float  # final win probability of the pick, as displayed
    allowed_percentages: frozenset[float]
    team_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    foreign_team_names: tuple[str, ...] = ()  # full names and nicknames of teams not in this game

    @property
    def pick_aliases(self) -> tuple[str, ...]:
        return self.team_aliases[self.pick_abbr]

    @property
    def game_aliases(self) -> tuple[str, ...]:
        return self.team_aliases[self.pick_abbr] + self.team_aliases[self.other_abbr]


def _pct(p: float) -> float:
    return round(100.0 * p, 1)


def pct_down(p: float) -> float:
    """Probability as a percentage, truncated to 1 decimal so a displayed value never reaches a tier
    threshold the unrounded value missed (74.96% shows as 74.9%, not 75.0%)."""
    return math.floor(round(1000.0 * p, 6)) / 10.0


def edge_down(edge_pct: float) -> float:
    """Edge in percentage points truncated toward zero to 1 decimal (4.95 shows as 4.9, not 5.0)."""
    return math.trunc(round(10.0 * edge_pct, 6)) / 10.0


def _pair_pct(pick_side_wp: float) -> tuple[float, float]:
    """(pick side, other side) percentages that add up to exactly 100."""
    pick = pct_down(pick_side_wp)
    return pick, round(100.0 - pick, 1)


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
                return stat, f"{value:+.1f} percentage points"
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
        return stat, f"{abs(value):.1f} percentage points of CPOE in favor of the {ahead}"
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

    final_pick, final_other = _pair_pct(game["pick"]["win_prob"])
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
            "final_win_probability_pct": final_pick,
            "opponent_win_probability_pct": final_other,
        },
        "confidence_tier": game["confidence_tier"],
        "tier_meaning": payload["tier_rules"][game["confidence_tier"]],
        "model_win_probability_pct": dict(zip((pick, other), _pair_pct(game["model"][f"{side}_wp"]), strict=True)),
        "market": (
            {
                "win_probability_pct": dict(zip((pick, other), _pair_pct(market[f"{side}_wp"]), strict=True)),
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
        tier=game["confidence_tier"],
        pick_pct=final_pick,
        allowed_percentages=frozenset(_collect_percentages(sheet) | ALWAYS_ALLOWED_PERCENTAGES),
        team_aliases={home: team_aliases(home), away: team_aliases(away)},
        foreign_team_names=tuple(
            name
            for abbr in TEAM_NAMES
            if abbr not in (home, away)
            for name in dict.fromkeys((team_name(abbr), team_nickname(abbr), *_EXTRA_NICKNAMES.get(abbr, ())))
            if name not in team_aliases(home) + team_aliases(away)
        ),
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

    Format checks always run: lengths, a single paragraph, plain prose with no
    markup, links or code. Text is Unicode-normalised first. Grounding checks
    run when validated with ``context={"facts": GameFacts}``:

    * the_pick_summary names the pick and states its tier and final win probability;
    * no other tier is named, and no team outside this game is mentioned;
    * the headline names one of the two teams;
    * every percentage matches a fact-sheet value (1 decimal, or rounded to a whole number);
    * no hype or betting language.
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

    @field_validator("headline", "the_pick_summary", "key_mismatch", "risk_factor", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        return normalize_text(value) if isinstance(value, str) else value

    @field_validator("headline", "the_pick_summary", "key_mismatch", "risk_factor")
    @classmethod
    def _plain_prose(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("must be a single paragraph (no line breaks)")
        if _MARKUP_RE.search(value):
            raise ValueError("must be plain prose: no markdown, HTML, code, links, URLs or list markers")
        return value

    @field_validator("headline")
    @classmethod
    def _headline_style(cls, value: str) -> str:
        if value.endswith("."):
            raise ValueError("headline must not end with a period")
        return value

    @model_validator(mode="after")
    def _grounded(self, info: ValidationInfo) -> ScoutingReport:
        facts = (info.context or {}).get("facts") if isinstance(info.context, dict) else None
        if facts is None:
            return self
        fields = {
            "headline": self.headline,
            "the_pick_summary": self.the_pick_summary,
            "key_mismatch": self.key_mismatch,
            "risk_factor": self.risk_factor,
        }
        text = " ".join(fields.values())
        problems: list[str] = []

        banned = sorted({m.group(0) for m in _BANNED_RE.finditer(text)})
        if banned:
            problems.append(f"hype or betting language is not allowed: {banned}")
        if not any(_mentions(self.the_pick_summary, alias) for alias in facts.pick_aliases):
            problems.append(f"the_pick_summary must name the pick ({' / '.join(facts.pick_aliases)})")
        if not any(_mentions(self.headline, alias) for alias in facts.game_aliases):
            problems.append("headline must name one of the two teams in this game")
        if not re.search(rf"\b{re.escape(facts.tier)}\b", self.the_pick_summary, re.IGNORECASE):
            problems.append(f"the_pick_summary must state the confidence tier ({facts.tier})")
        wrong_tiers = sorted({t for t in _TIER_TOKEN_RE.findall(text) if t != facts.tier})
        if wrong_tiers:
            problems.append(f"the tier for this game is {facts.tier}, not {wrong_tiers}")
        quoted = [float(m) for m in _PERCENT_RE.findall(self.the_pick_summary)]
        if not any(_percent_matches(v, facts.pick_pct) for v in quoted):
            problems.append(f"the_pick_summary must give the pick's final win probability ({facts.pick_pct:g}%)")
        quoted_all = {float(m) for m in _PERCENT_RE.findall(text)}
        unsupported = sorted(
            v for v in quoted_all if not any(_percent_matches(v, a) for a in facts.allowed_percentages)
        )
        if unsupported:
            problems.append(
                f"percentages not in the fact sheet: {unsupported} (allowed: {sorted(facts.allowed_percentages)})"
            )
        foreign = sorted({name for name in facts.foreign_team_names if _mentions_name(text, name)})
        if foreign:
            problems.append(f"mentions teams that are not in this game: {foreign}")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def _percent_matches(value: float, fact: float) -> bool:
    """A quoted percentage matches a fact if it equals it at 1-decimal precision or is its whole-number rounding."""
    return abs(value - fact) <= 0.051 or (value.is_integer() and value == math.floor(fact + 0.5))


def _mentions_name(text: str, name: str) -> bool:
    """Case-sensitive whole-word match, so lowercase words ('bills', 'giants') do not count as teams."""
    return re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text) is not None


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
7. A STAT MISMATCH tier means the play-by-play model and the betting market disagree by a wide margin on this game,
   in either direction. Describe it as the numbers and the line disagreeing, never as an edge, value or opportunity.
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


# Machine-readable google.rpc.ErrorInfo reasons and message hints that mean "this key/project cannot call Gemini".
_AUTH_REASONS: Final[frozenset[str]] = frozenset(
    {
        "API_KEY_INVALID",
        "API_KEY_EXPIRED",
        "API_KEY_SERVICE_BLOCKED",
        "API_KEY_HTTP_REFERRER_BLOCKED",
        "API_KEY_IP_ADDRESS_BLOCKED",
        "API_KEY_ANDROID_APP_BLOCKED",
        "API_KEY_IOS_APP_BLOCKED",
        "SERVICE_DISABLED",
        "CONSUMER_INVALID",
        "ACCESS_TOKEN_EXPIRED",
    }
)
_AUTH_HINTS: Final[tuple[str, ...]] = (
    "api key not valid",
    "api key expired",
    "api key was reported as leaked",
    "permission denied",
    "unauthenticated",
)


def _error_details(exc: Any) -> list[dict[str, Any]]:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return []
    items = (details.get("error") or {}).get("details") or []
    return [item for item in items if isinstance(item, dict)]


def classify_api_error(exc: Any) -> type[ReportError] | None:
    """Exception class to raise for a google.genai APIError, or None if it is worth retrying.

    Fatal for the whole run: auth problems (401/403, ErrorInfo API_KEY_*, ...), daily
    quota exhaustion, unknown models (404), and other 400s (unsupported location,
    request-shape problems). These would be the same for every game. Retried:
    per-minute rate limits (429) and server errors (5xx).
    """
    code, status = getattr(exc, "code", None), (getattr(exc, "status", None) or "")
    message = (getattr(exc, "message", None) or str(exc)).lower()
    details = _error_details(exc)
    reasons = {str(d.get("reason", "")) for d in details}
    if (
        code in (401, 403)
        or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or reasons & _AUTH_REASONS
        or any(h in message for h in _AUTH_HINTS)
    ):
        return LLMAuthError
    if code == 429:
        quota_ids = [
            str(v.get("quotaId", "")) for d in details for v in (d.get("violations") or []) if isinstance(v, dict)
        ]
        return LLMQuotaError if any("perday" in q.replace("_", "").lower() for q in quota_ids) else None
    if code is not None and code >= 500:
        return None
    if code in (400, 404):
        return LLMConfigError
    return LLMRequestError


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
                detail = f"HTTP {exc.code} {exc.status or ''}: {exc.message}".replace("  ", " ")
                error_class = classify_api_error(exc)
                if error_class is not None:
                    raise error_class(f"Gemini rejected the request ({detail})") from None
                if attempt > self.max_retries:
                    raise LLMRequestError(f"Gemini request failed after {attempt} attempts ({detail})") from None
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
                raise LLMEmptyResponseError(
                    f"Gemini returned no text (finish reason: {_finish_reason(response)}, "
                    f"block reason: {_block_reason(response)})"
                )
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


def _block_reason(response: Any) -> str:
    try:
        return str(response.prompt_feedback.block_reason or "none")
    except AttributeError:
        return "none"


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
        "STAT MISMATCH": f"The numbers and the line split on the {team_nickname(pick)} and the {team_nickname(other)}",
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
    if tier == "STAT MISMATCH" and value and value["team"] == team_name(pick):
        risk = (
            f"The model rates the {value['team']} {value['edge_pct_points']:.1f} percentage points above the market, "
            "and a gap that large often means the line knows something the play-by-play numbers miss."
        )
    elif value and value["team"] != team_name(pick):
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
        except LLMEmptyResponseError as exc:  # a failed attempt; ask again (nothing to repair)
            history.append(f"attempt {attempt}: {exc}")
            reason = f"empty or blocked reply after {attempt} attempt(s)"
            feedback, previous = [], None
            continue
        except LLMRequestError as exc:  # this game only (network, rate limit, server); fatal errors propagate
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
    """The Contract B payload embedded in a Contract C payload (non-list ``games`` pass through for validation)."""
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    games = payload.get("games")
    return {
        "contract": source.get("contract"),
        "schema_version": source.get("schema_version"),
        "generated_at": source.get("generated_at"),
        "season": payload.get("season"),
        "week": payload.get("week"),
        "model": payload.get("model"),
        "tier_rules": payload.get("tier_rules"),
        "games": (
            [
                {k: v for k, v in g.items() if k not in ("report", "report_meta")} if isinstance(g, dict) else g
                for g in games
            ]
            if isinstance(games, list)
            else games
        ),
    }


def validate_contract_c(payload: Any) -> None:
    """Raise :class:`ContractCError` listing every violation.

    The embedded Contract B must validate, and every report must re-validate
    against the fact sheet rebuilt from its own game, including grounding.
    Every key consumers index (``report``, ``report_meta``, ``generator``,
    ``summary``) is checked for presence and type.
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


_REPORT_META_KEYS: Final[frozenset[str]] = frozenset({"generator", "attempts", "fallback_reason", "validation_errors"})
_GENERATOR_KEYS: Final[frozenset[str]] = frozenset({"provider", "model", "mode", "temperature", "max_attempts"})


def _check_contract_c(payload: dict[str, Any], problems: list[str]) -> None:
    if payload.get("contract") != CONTRACT_NAME:
        problems.append(f"contract must be {CONTRACT_NAME!r}")
    if not isinstance(payload.get("schema_version"), str):
        problems.append("schema_version must be a string")
    if M._parse_iso(payload.get("generated_at")) is None:
        problems.append("generated_at must be ISO-8601")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("contract") != M.CONTRACT_NAME:
        problems.append(f"source must describe the Contract {M.CONTRACT_NAME} payload it was built from")
    view = contract_b_view(payload)
    try:
        M.validate_contract_b(view)
    except M.ContractError as exc:
        problems.append(f"embedded Contract B is invalid: {exc}")
        return

    generator = payload.get("generator")
    if (
        not isinstance(generator, dict)
        or set(generator) != _GENERATOR_KEYS
        or generator.get("mode") not in ("llm", "template")
    ):
        problems.append(f"generator must have keys {sorted(_GENERATOR_KEYS)} and mode 'llm' or 'template'")
        return
    if generator["mode"] == "llm":
        if not (isinstance(generator["model"], str) and generator["model"]) or generator["provider"] != PROVIDER:
            problems.append(f"llm mode needs generator.provider {PROVIDER!r} and a model name")
            return
        allowed_generators = {TEMPLATE_GENERATOR, generator["model"]}
    else:
        allowed_generators = {TEMPLATE_GENERATOR}
    counts: Counter[str] = Counter()
    for n, game in enumerate(payload["games"]):
        where = f"games[{n}] ({game.get('game_id')})"
        meta, report = game.get("report_meta"), game.get("report")
        meta_ok = (
            isinstance(meta, dict)
            and set(meta) == _REPORT_META_KEYS
            and meta["generator"] in allowed_generators
            and M._is_int(meta["attempts"])
            and meta["attempts"] >= 0
            and (meta["fallback_reason"] is None or isinstance(meta["fallback_reason"], str))
            and isinstance(meta["validation_errors"], list)
            and all(isinstance(e, str) for e in meta["validation_errors"])
        )
        if not meta_ok:
            problems.append(
                f"{where}: report_meta must hold {sorted(_REPORT_META_KEYS)} "
                f"with a generator in {sorted(allowed_generators)}"
            )
            continue
        counts[meta["generator"]] += 1
        if not isinstance(report, dict):
            problems.append(f"{where}: report must be an object")
            continue
        try:
            validate_report(report, game_facts(game, view))
        except ValidationError as exc:
            problems.append(f"{where}: report invalid: {'; '.join(_format_errors(exc))}")

    summary = payload.get("summary")
    n_games = len(payload["games"])
    expected = {
        "n_games": n_games,
        "n_template": counts[TEMPLATE_GENERATOR],
        "n_llm": n_games - counts[TEMPLATE_GENERATOR],
        "tiers": {tier: sum(g["confidence_tier"] == tier for g in payload["games"]) for tier in TIER_ORDER},
    }
    if not isinstance(summary, dict) or any(summary.get(k) != v for k, v in expected.items()):
        problems.append(f"summary must be {expected}")
    path = payload.get("newsletter_path")
    if path is not None and not isinstance(path, str):
        problems.append("newsletter_path must be a string or null")


# --------------------------------------------------------------------------- #
# Markdown newsletter
# --------------------------------------------------------------------------- #
_ROUND_NAMES: Final[dict[str, str]] = {
    "WC": "Wild Card Round",
    "DIV": "Divisional Round",
    "CON": "Conference Championships",
    "SB": "Super Bowl",
}
_MD_SPECIAL_RE: Final = re.compile(r"([\\`*_\[\]<>~|#])")


def md_text(text: str) -> str:
    """Escape Markdown/HTML metacharacters in prose before it goes into the newsletter (defence in depth)."""
    escaped = _MD_SPECIAL_RE.sub(r"\\\1", text)
    return re.sub(r"^(\s*\d+)([.)])", r"\1\\\2", escaped)


def local_kickoff(iso: str) -> str:
    kickoff = M._parse_iso(iso)
    if kickoff is None:
        return iso
    try:
        from zoneinfo import ZoneInfo

        local = kickoff.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 - no tz database on this platform
        return kickoff.strftime("%a %b %d, %H:%M UTC")
    return local.strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")


def pct_text(p: float) -> str:
    return f"{pct_down(p):.1f}%"


def edge_text(game: dict[str, Any]) -> str:
    edge = game["model_edge_pct"]
    return "n/a" if edge is None else f"{edge_down(edge):+.1f} pts"


def result_text(game: dict[str, Any]) -> str:
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


def week_label(payload: dict[str, Any]) -> str:
    """The week as a reader says it: "Week 6", or the playoff round ("Wild Card Round") if all games share one."""
    types = {g["game_type"] for g in payload["games"]}
    label = _ROUND_NAMES[next(iter(types))] if len(types) == 1 and next(iter(types)) in _ROUND_NAMES else None
    return label or f"Week {payload['week']}"


def _title(contract_c: dict[str, Any]) -> str:
    return f"# NFL {week_label(contract_c)} Scouting Report — {contract_c['season']} Season"


def writer_label(contract_c: dict[str, Any]) -> str:
    gen, summary = contract_c["generator"], contract_c["summary"]
    if gen["mode"] == "template" or summary["n_llm"] == 0:
        return "rule-based templates" + (" (Gemini unavailable)" if gen["mode"] == "llm" else "")
    if summary["n_template"]:
        return f"{gen['model']} for {summary['n_llm']} of {summary['n_games']} games, rule-based templates for the rest"
    return gen["model"]


def track_record(backtest_report: dict[str, Any] | None, market_weight: float) -> str | None:
    if not backtest_report or not isinstance(backtest_report.get("pooled"), dict):
        return None
    pooled = backtest_report["pooled"]
    seasons = backtest_report.get("test_seasons") or []
    try:
        model, market, final = (float(pooled[k]["brier"]) for k in ("model", "market", "final"))
        n_games, first, last = int(pooled["final"]["n_games"]), seasons[0], seasons[-1]
    except (KeyError, TypeError, ValueError, IndexError):
        logger.warning("backtest report is missing pooled metrics; track record omitted")
        return None
    best = min((("the market", market), ("the model", model), ("the blend", final)), key=lambda kv: kv[1])[0]
    return (
        f"- **Track record** (walk-forward, out of sample, {first}-{last}, {n_games:,} games): Brier score model "
        f"{model:.4f}, market {market:.4f}, blend {final:.4f} (lower is better). {best[0].upper() + best[1:]} has "
        f"been the most accurate; this week's blend puts {market_weight:.0%} of its weight on the market."
    )


def render_newsletter(contract_c: dict[str, Any], backtest_report: dict[str, Any] | None = None) -> str:
    """Standalone Markdown newsletter for the week."""
    model = contract_c["model"]
    generated = M._parse_iso(contract_c["generated_at"])
    as_of = M._parse_iso(contract_c["source"]["generated_at"])
    games = contract_c["games"]
    lines = [
        _title(contract_c),
        "",
        f"_Generated {generated:%b %d, %Y %H:%M} UTC from predictions made {as_of:%b %d, %Y %H:%M} UTC · win "
        f"probabilities: calibrated XGBoost blended with the de-vigged market ({model['market_weight']:.0%} market "
        f"weight) · write-ups: {writer_label(contract_c)}_",
        "",
        "## At a glance",
        "",
        "| Matchup | Kickoff | Pick | Win prob | Tier | Model vs market | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in games:
        lines.append(
            f"| {g['away_team']} @ {g['home_team']} | {_md_cell(local_kickoff(g['kickoff_utc']))} "
            f"| {g['pick']['team']} | {pct_text(g['pick']['win_prob'])} | {g['confidence_tier']} | {edge_text(g)} "
            f"| {result_text(g)} |"
        )

    upcoming = [g for g in games if g["status"] != "final"]
    finished = [g for g in games if g["status"] == "final"]
    for tier in TIER_ORDER:
        group = [g for g in upcoming if g["confidence_tier"] == tier]
        if not group:
            continue
        lines += ["", f"## {SECTION_TITLES[tier]}", ""]
        for g in sorted(group, key=lambda x: -x["pick"]["win_prob"]):
            lines += _game_section(g)
    if finished:
        lines += ["", "## Already played", ""]
        for g in finished:
            lines += _game_section(g)

    lines += ["", "## How to read this", ""]
    lines += [f"- **{M.TIER_LABELS[tier]}**: {contract_c['tier_rules'][tier]}." for tier in TIER_ORDER]
    lines.append(
        "- **Model vs market** is the model's win probability for the pick minus the de-vigged market's, in "
        "percentage points. The final number blends the two."
    )
    record = track_record(backtest_report, model["market_weight"])
    if record:
        lines.append(record)
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
        f"{g['pick']['team']} {pct_text(market[f'{side}_wp'])} ({_SOURCE_TEXT.get(market['source'], market['source'])})"
        if market
        else "no line"
    )
    lines = [
        f"### {g['away_team']} @ {g['home_team']} — {md_text(report['headline'])}",
        "",
        f"**Pick:** {g['pick']['team']} {pct_text(g['pick']['win_prob'])} · "
        f"**Tier:** {M.TIER_LABELS[g['confidence_tier']]} · "
        f"**Model:** {g['pick']['team']} {pct_text(g['model'][f'{side}_wp'])} · **Market:** {market_text} · "
        f"**Kickoff:** {local_kickoff(g['kickoff_utc'])}",
        "",
        md_text(report["the_pick_summary"]),
        "",
        f"**Key mismatch.** {md_text(report['key_mismatch'])}",
        "",
        f"**Risk factor.** {md_text(report['risk_factor'])}",
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
STALE_AFTER: Final = timedelta(days=3)


def newsletter_path(cfg: Settings, season: int, week: int) -> Path:
    """reports/week_XX_YYYY.md (CLAUDE.md section 2)."""
    return cfg.newsletter_md(season, week)


def _load_backtest(cfg: Settings) -> dict[str, Any] | None:
    path = cfg.predictions_dir / "backtest_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s); track record omitted", path, exc)
        return None


def stale_contract_b_warnings(payload: dict[str, Any], now: datetime) -> list[str]:
    """Reasons the Contract B payload may no longer describe the week (logged, not fatal)."""
    warnings = []
    generated = M._parse_iso(payload["generated_at"])
    if generated is not None and now - generated > STALE_AFTER:
        warnings.append(f"predictions were generated {(now - generated).days} days ago ({payload['generated_at']})")
    kicked_off = [
        g["game_id"]
        for g in payload["games"]
        if g["status"] == "scheduled" and (M._parse_iso(g["kickoff_utc"]) or now) <= now
    ]
    if kicked_off:
        warnings.append(
            f"{len(kicked_off)} game(s) marked scheduled have already kicked off ({', '.join(kicked_off[:3])})"
        )
    return warnings


def run_report(
    cfg: Settings = settings,
    *,
    mode: Mode = "llm",
    writer: ReportWriter | None = None,
    input_path: Path | None = None,
    allow_fallback: bool = True,
    now: datetime | None = None,
    write: bool = True,
    update_latest: bool = True,
) -> dict[str, Any]:
    """Read Contract B, generate reports, write Contract C and the newsletter. Returns the Contract C payload.

    Contract C goes to its ``week_XX_YYYY_enriched.json`` archive and, with ``update_latest``, to
    ``weekly_report_enriched.json``. Nothing is written if a run-wide error occurs (auth, unknown model, daily quota).
    """
    now = M._as_utc(now or datetime.now(UTC))
    payload = load_contract_b(input_path or cfg.current_week_json)
    for warning in stale_contract_b_warnings(payload, now):
        logger.warning("Contract B may be stale: %s. Re-run `python -m src.model predict`.", warning)
    if mode == "llm" and writer is None:
        writer = GeminiReportWriter(cfg=cfg)
    if mode == "template":
        writer = None
    md_path = newsletter_path(cfg, payload["season"], payload["week"])
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
        text = json.dumps(contract_c, indent=2, allow_nan=False) + "\n"
        archive = cfg.enriched_archive_json(payload["season"], payload["week"])
        if update_latest:
            ingest.write_text_atomic(text, cfg.enriched_report_json)
        ingest.write_text_atomic(text, archive)
        ingest.write_text_atomic(render_newsletter(contract_c, _load_backtest(cfg)), md_path)
        written = [cfg.enriched_report_json.name] if update_latest else []
        logger.info("Wrote %s and %s", ", ".join([*written, archive.name]), md_path)
    return contract_c


EXIT_OK, EXIT_ERROR, EXIT_MISSING_KEY, EXIT_DEGRADED = 0, 1, 2, 3


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Exit codes: 0 ok, 1 error (nothing written), 2 missing API key, 3 written but no Gemini write-ups."""
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
            settings,  # module global looked up at call time (tests can redirect it)
            mode="template" if args.template else "llm",
            writer=writer,
            input_path=args.input,
            allow_fallback=not args.strict,
        )
    except MissingLLMKeyError as exc:
        logger.error("%s", exc)
        return EXIT_MISSING_KEY
    except (ReportError, ContractCError, M.ContractError, FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        logger.error("Report generation failed (nothing written): %s: %s", type(exc).__name__, exc)
        return EXIT_ERROR
    print(f"Wrote {settings.enriched_report_json} and {contract_c['newsletter_path']}")
    if not args.template and contract_c["summary"]["n_llm"] == 0:
        logger.warning("No game got a Gemini write-up; every report fell back to the template")
        return EXIT_DEGRADED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
