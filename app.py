"""Streamlit dashboard: the week's predictions (Contract B) and scouting reports (Contract C).

    streamlit run app.py

The dashboard only reads what the pipeline wrote: ``data/predictions/`` (the week archives, the latest-run files
and ``last_run.json``) and ``reports/`` (newsletter downloads). It never calls an API or retrains. To read other
directories, set ``NFL_DATA_DIR`` / ``NFL_REPORTS_DIR``.

The data layer (everything above ``main``) makes no Streamlit calls, so it can be unit-tested.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from config import Settings
from run_weekly import DEGRADED_NO_LLM
from src import model as M
from src import report_generator as R

ARCHIVE_RE: Final = re.compile(r"^week_(\d{2})_(\d{4})_(preds|enriched)\.json$")

SOURCE_LABELS: Final[dict[str, str]] = {
    "odds_api": "Live Odds API consensus",
    "odds_api_snapshot": "Saved Odds API snapshot",
    "nflverse_line": "nflverse pre-game line",
    "nflverse_close": "nflverse closing line",
}
TIER_BADGE_COLORS: Final[dict[str, str]] = {
    "LOCK": "blue",
    "VALUE EDGE": "violet",
    "LEAN": "green",
    "COIN FLIP": "gray",
}

# Chart palette: categorical slots 1-2 of the reference palette (validated light and dark), ink for the blend.
CHART_COLORS: Final[dict[str, dict[str, str]]] = {
    "light": {
        "Market": "#2a78d6",
        "Model": "#eb6834",
        "Final (blend)": "#0b0b0b",
        "surface": "#ffffff",
        "rule": "#c3c2b7",
        "grid": "#e1e0d9",
    },
    "dark": {
        "Market": "#3987e5",
        "Model": "#d95926",
        "Final (blend)": "#f0efec",
        "surface": "#0e1117",
        "rule": "#383835",
        "grid": "#2c2c2a",
    },
}
SERIES: Final[tuple[str, ...]] = ("Market", "Model", "Final (blend)")


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class WeekRef:
    """The files for one week. Archives are preferred; latest-run files fill in a week that has no archive."""

    season: int
    week: int
    preds: Path | None = None
    enriched: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.season} · Week {self.week}"


@dataclass(slots=True)
class WeekData:
    ref: WeekRef
    contract_b: dict[str, Any] | None = None
    contract_c: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)  # files that could not be read or failed validation


@dataclass(frozen=True, slots=True)
class Banner:
    level: Literal["error", "warning", "info"]
    title: str
    lines: tuple[str, ...] = ()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _season_week(path: Path) -> tuple[int, int] | None:
    try:
        data = _read_json(path)
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("season"), int) and isinstance(data.get("week"), int):
        return data["season"], data["week"]
    return None


def discover_weeks(cfg: Settings) -> list[WeekRef]:
    """Every week with predictions or a report on disk, newest first."""
    found: dict[tuple[int, int], dict[str, Path]] = {}
    if cfg.predictions_dir.is_dir():
        for path in cfg.predictions_dir.iterdir():
            if match := ARCHIVE_RE.match(path.name):
                week, season, kind = int(match[1]), int(match[2]), match[3]
                found.setdefault((season, week), {})[kind] = path
    for kind, path in (("preds", cfg.current_week_json), ("enriched", cfg.enriched_report_json)):
        key = _season_week(path) if path.exists() else None
        if key is not None and kind not in found.get(key, {}):  # e.g. written before archives existed
            found.setdefault(key, {})[kind] = path
    return [
        WeekRef(season, week, files.get("preds"), files.get("enriched"))
        for (season, week), files in sorted(found.items(), reverse=True)
    ]


def default_week_index(cfg: Settings, weeks: list[WeekRef]) -> int:
    """The latest run's week (``current_week.json``) if it is on the list, else the newest week."""
    key = _season_week(cfg.current_week_json) if cfg.current_week_json.exists() else None
    return next((i for i, ref in enumerate(weeks) if (ref.season, ref.week) == key), 0)


def load_week(ref: WeekRef) -> WeekData:
    """Read and validate the week's Contract B and C. A file that fails is reported, never trusted."""
    data = WeekData(ref)
    if ref.preds is not None:
        try:
            payload = _read_json(ref.preds)
            M.validate_contract_b(payload)
            data.contract_b = payload
        except (OSError, ValueError) as exc:  # includes JSON and contract errors
            data.problems.append(f"{ref.preds.name}: {exc}")
    if ref.enriched is not None:
        try:
            payload = _read_json(ref.enriched)
            R.validate_contract_c(payload)
            data.contract_c = payload
        except (OSError, ValueError) as exc:
            data.problems.append(f"{ref.enriched.name}: {exc}")
    if data.contract_b is None and data.contract_c is not None:
        data.contract_b = R.contract_b_view(data.contract_c)  # Contract C embeds a validated Contract B
    return data


def load_run_record(cfg: Settings) -> dict[str, Any] | None:
    try:
        record = _read_json(cfg.last_run_json)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("record") == "run" else None


def run_covers(record: dict[str, Any], contract_b: dict[str, Any]) -> bool:
    """Whether ``record`` is the run that produced these predictions (same week, generated during the run)."""
    generated = M._parse_iso(contract_b.get("generated_at"))
    started, finished = M._parse_iso(record.get("started_at")), M._parse_iso(record.get("finished_at"))
    same_week = (record.get("season"), record.get("week")) == (contract_b["season"], contract_b["week"])
    return same_week and None not in (generated, started, finished) and started <= generated <= finished


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def banners(data: WeekData, record: dict[str, Any] | None) -> list[Banner]:
    """Warnings for the selected week, most severe first."""
    out = [Banner("error", "Could not load a file for this week", (p,)) for p in data.problems]
    if record is not None and record.get("exit_code") in (1, 2):
        out.append(
            Banner(
                "error",
                f"The last pipeline run failed ({record.get('status')}, exit code {record.get('exit_code')})",
                (
                    f"{record.get('error')}",
                    f"Finished {fmt_time(record.get('finished_at'))}. This page shows the latest files that were "
                    "written; nothing after the failed stage was updated.",
                ),
            )
        )
    b, c = data.contract_b, data.contract_c
    if b is None:
        return out

    degraded: list[str] = []
    if record is not None and record.get("exit_code") == 3 and run_covers(record, b):
        degraded += [str(reason) for reason in record.get("degraded") or []]
    all_template = c is not None and c["generator"]["mode"] == "llm" and c["summary"]["n_llm"] == 0
    if all_template and DEGRADED_NO_LLM not in degraded:
        degraded.append(DEGRADED_NO_LLM)
    if degraded:
        out.append(Banner("error", "Degraded run: a live input was not used", tuple(_sentence(d) for d in degraded)))

    if c is None:
        out.append(
            Banner(
                "warning",
                "No scouting report for this week",
                ("Predictions only. Run `python run_weekly.py` (or `python -m src.report_generator`) for write-ups.",),
            )
        )
    else:
        if c["source"]["generated_at"] != b["generated_at"]:
            out.append(
                Banner(
                    "warning",
                    "The write-ups are older than the predictions",
                    (
                        f"Write-ups were built from predictions made {fmt_time(c['source']['generated_at'])}; the "
                        f"probabilities shown were made {fmt_time(b['generated_at'])}. Re-run the report stage.",
                    ),
                )
            )
        summary = c["summary"]
        if c["generator"]["mode"] == "template":
            out.append(
                Banner(
                    "warning",
                    "Template mode: write-ups are rule-based, not Gemini",
                    ("The report stage ran with --skip-llm / --template.",),
                )
            )
        elif summary["n_llm"] and summary["n_template"]:
            out.append(
                Banner(
                    "warning",
                    f"{summary['n_template']} of {summary['n_games']} write-ups fell back to the template",
                    ("Gemini's replies for those games failed validation; see each card for the reason.",),
                )
            )
    no_line = [g for g in b["games"] if g["market"] is None]
    if no_line:
        out.append(
            Banner(
                "info",
                f"{len(no_line)} game(s) have no market line",
                ("Their probabilities are model-only: " + ", ".join(matchup(g) for g in no_line) + ".",),
            )
        )
    return out


def fmt_time(value: Any, *, year: bool = True) -> str:
    """An ISO-8601 string or datetime as "Oct 16, 2026 18:00 UTC"."""
    ts = M._as_utc(value) if isinstance(value, datetime) else M._parse_iso(value)
    if ts is None:
        return "unknown time"
    return ts.strftime("%b %d, %Y %H:%M UTC" if year else "%b %d, %H:%M UTC")


def matchup(game: dict[str, Any]) -> str:
    return f"{game['away_team']} @ {game['home_team']}"


def pick_side(game: dict[str, Any], block: dict[str, Any] | None) -> float | None:
    """A {home_wp, away_wp} block's probability for the pick side."""
    if block is None:
        return None
    return block["home_wp"] if game["pick"]["side"] == "home" else block["away_wp"]


def market_summary(contract_b: dict[str, Any]) -> list[tuple[str, int, str | None]]:
    """(source label, games, latest as-of) per market source, most common first."""
    counts: Counter[str] = Counter()
    as_of: dict[str, datetime] = {}
    for game in contract_b["games"]:
        market = game["market"]
        source = market["source"] if market else "none"
        counts[source] += 1
        stamp = M._parse_iso(market["as_of"]) if market else None
        if stamp is not None and (source not in as_of or stamp > as_of[source]):
            as_of[source] = stamp
    summary = []
    for source, n in counts.most_common():
        label = "No market line" if source == "none" else SOURCE_LABELS.get(source, source)
        summary.append((label, n, fmt_time(as_of[source]) if source in as_of else None))
    return summary


def writer_summary(contract_c: dict[str, Any] | None) -> str:
    if contract_c is None:
        return "None yet"
    gen, summary = contract_c["generator"], contract_c["summary"]
    if gen["mode"] == "template":
        return "Template"
    return f"{gen['model']} ({summary['n_llm']}/{summary['n_games']})"


def game_rows(contract_b: dict[str, Any]) -> list[dict[str, Any]]:
    """The overview table: one row per game, percentages truncated like the newsletter (never across a tier)."""
    rows = []
    for g in sorted(contract_b["games"], key=lambda g: g["kickoff_utc"]):
        market = pick_side(g, g["market"])
        rows.append(
            {
                "Kickoff": R.local_kickoff(g["kickoff_utc"]),
                "Matchup": matchup(g),
                "Pick": g["pick"]["team"],
                "Win prob": R.pct_down(g["pick"]["win_prob"]),
                "Model": R.pct_down(pick_side(g, g["model"])),
                "Market": None if market is None else R.pct_down(market),
                "Edge (pts)": None if g["model_edge_pct"] is None else R.edge_down(g["model_edge_pct"]),
                "Tier": g["confidence_tier"],
                "Market source": SOURCE_LABELS.get(g["market"]["source"], "") if g["market"] else "No line",
                "Result": R.result_text(g),
            }
        )
    return rows


def chart_data(contract_b: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Points (Market/Model/Final on the pick's side), market-to-model connectors, and the row order."""
    games = sorted(contract_b["games"], key=lambda g: (-g["pick"]["win_prob"], g["kickoff_utc"]))
    points, connectors, order = [], [], []
    for g in games:
        opponent = g["away_team"] if g["pick"]["side"] == "home" else g["home_team"]
        label = f"{g['pick']['team']} over {opponent}"
        order.append(label)
        values = {
            "Market": pick_side(g, g["market"]),
            "Model": pick_side(g, g["model"]),
            "Final (blend)": g["pick"]["win_prob"],
        }
        edge = "n/a" if g["model_edge_pct"] is None else f"{R.edge_down(g['model_edge_pct']):+.1f} pts"
        for series, value in values.items():
            if value is not None:
                points.append(
                    {
                        "game": label,
                        "series": series,
                        "prob": value,
                        "pct": f"{R.pct_down(value):.1f}%",
                        "matchup": matchup(g),
                        "tier": g["confidence_tier"],
                        "edge": edge,
                    }
                )
        if values["Market"] is not None:
            low, high = sorted((values["Market"], values["Model"]))
            connectors.append({"game": label, "low": low, "high": high})
    return points, connectors, order


def probability_chart(contract_b: dict[str, Any], mode: str = "light") -> Any:
    """Dot plot: market vs model vs final win probability for each pick, strongest pick on top."""
    import altair as alt

    colors = CHART_COLORS["dark" if mode == "dark" else "light"]
    points, connectors, order = chart_data(contract_b)
    low = min([0.5, *(p["prob"] for p in points)])
    x_scale = alt.Scale(domain=[max(0.0, int(low * 20) / 20), 1.0], nice=False, zero=False)
    y = alt.Y("game:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False))
    axis = alt.Axis(format="%", tickCount=6, grid=True, gridColor=colors["grid"], domain=False)

    fifty = (
        alt.Chart(alt.Data(values=[{"x": 0.5}]))
        .mark_rule(color=colors["rule"], strokeWidth=1)
        .encode(x=alt.X("x:Q", scale=x_scale, axis=axis, title="Pick's win probability"))
    )
    gap = (
        alt.Chart(alt.Data(values=connectors))
        .mark_rule(color=colors["rule"], strokeWidth=2, strokeCap="round")
        .encode(x=alt.X("low:Q", scale=x_scale, axis=axis, title="Pick's win probability"), x2="high:Q", y=y)
    )
    dots = (
        alt.Chart(alt.Data(values=points))
        .mark_point(filled=True, size=110, opacity=1, stroke=colors["surface"], strokeWidth=2)
        .encode(
            x=alt.X("prob:Q", scale=x_scale, axis=axis, title="Pick's win probability"),
            y=y,
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(SERIES), range=[colors[s] for s in SERIES]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            shape=alt.Shape(
                "series:N", scale=alt.Scale(domain=list(SERIES), range=["circle", "circle", "diamond"]), legend=None
            ),
            tooltip=[
                alt.Tooltip("matchup:N", title="Game"),
                alt.Tooltip("series:N", title="Estimate"),
                alt.Tooltip("pct:N", title="Pick's win prob"),
                alt.Tooltip("edge:N", title="Model vs market"),
                alt.Tooltip("tier:N", title="Tier"),
            ],
        )
    )
    return alt.layer(fifty, gap, dots).properties(height=30 * len(order) + 40)


def write_up_meta(game: dict[str, Any]) -> str:
    meta = game["report_meta"]
    if meta["generator"] != R.TEMPLATE_GENERATOR:
        attempts = meta["attempts"]
        return f"Written by {meta['generator']} ({attempts} attempt{'s' if attempts != 1 else ''})"
    if meta["fallback_reason"]:
        return f"Rule-based template (fallback: {meta['fallback_reason']})"
    return "Rule-based template"


def load_backtest(cfg: Settings) -> dict[str, Any] | None:
    try:
        report = _read_json(cfg.predictions_dir / "backtest_report.json")
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) and isinstance(report.get("pooled"), dict) else None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _render_banners(st: Any, items: list[Banner]) -> None:
    for banner in items:
        body = f"**{banner.title}**" + "".join(f"\n\n{line}" for line in banner.lines)
        icon = {"error": ":material/error:", "warning": ":material/warning:", "info": ":material/info:"}[banner.level]
        getattr(st, banner.level)(body, icon=icon)


def _render_header(st: Any, data: WeekData, warnings: list[Banner]) -> None:
    """Title, then the warning banners (so a degraded or template run is the first thing a reader sees), then
    the run metadata."""
    b, c = data.contract_b, data.contract_c
    assert b is not None
    st.title(f"NFL {R.week_label(b)} Scouting Report — {b['season']} Season")
    _render_banners(st, warnings)
    tiers = Counter(g["confidence_tier"] for g in b["games"])
    st.caption(
        f"{len(b['games'])} games · "
        + " · ".join(f"{tiers[t]} {t}" for t in M.TIERS if tiers[t])
        + " · win probabilities are a calibrated XGBoost model blended with the de-vigged market. Not betting advice."
    )
    markets = market_summary(b)
    tiles = [
        ("Predictions made", fmt_time(b["generated_at"], year=False)),
        ("Odds", "; ".join(f"{label} ({n})" + (f", as of {as_of}" if as_of else "") for label, n, as_of in markets)),
        ("Market weight", f"{b['model']['market_weight']:.0%} market · {1 - b['model']['market_weight']:.0%} model"),
        ("Write-ups", writer_summary(c) + ("" if c is None else f", {fmt_time(c['generated_at'], year=False)}")),
    ]
    for col, (label, value) in zip(st.columns([1, 2, 1, 1.4]), tiles, strict=True):
        col.caption(label)
        col.markdown(f"**{R.md_text(value)}**")


def _render_overview(st: Any, b: dict[str, Any]) -> None:
    st.subheader("Model vs market")
    mode = getattr(getattr(st.context, "theme", None), "type", None) or "light"
    st.altair_chart(probability_chart(b, mode), width="stretch")
    st.caption(
        "Each row is a pick. Dots show the market's and the model's win probability for the picked team; the "
        "diamond is the blend the pick and tier come from. The gap between the dots is the model's edge."
    )
    pct = st.column_config.NumberColumn(format="%.1f%%")
    rows = game_rows(b)
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=35 * (len(rows) + 1) + 3,  # every game visible without an inner scroll
        column_config={
            "Win prob": pct,
            "Model": pct,
            "Market": pct,
            "Edge (pts)": st.column_config.NumberColumn(format="%+.1f", help="Model minus market, on the pick"),
        },
    )


def _render_cards(st: Any, c: dict[str, Any]) -> None:
    st.subheader("Scouting reports")
    present = [t for t in M.TIERS if any(g["confidence_tier"] == t for g in c["games"])]
    chosen = st.pills("Tiers", present, selection_mode="multi", default=present, key="tiers") or []
    games = sorted(
        (g for g in c["games"] if g["confidence_tier"] in chosen),
        key=lambda g: (M.TIERS.index(g["confidence_tier"]), g["kickoff_utc"]),
    )
    if not games:
        st.info("No games match the selected tiers.")
    for game in games:
        report = game["report"]
        with st.container(border=True):
            top = st.columns([4, 1])
            top[0].markdown(f"##### {R.md_text(report['headline'])}")
            with top[1]:
                st.badge(game["confidence_tier"], color=TIER_BADGE_COLORS[game["confidence_tier"]])
            market = pick_side(game, game["market"])
            st.caption(
                f"{matchup(game)} · {R.local_kickoff(game['kickoff_utc'])} · pick {game['pick']['team']} "
                f"{R.pct_text(game['pick']['win_prob'])} · model {R.pct_text(pick_side(game, game['model']))} · "
                f"market {'n/a' if market is None else R.pct_text(market)} · edge {R.edge_text(game)}"
                + (f" · {R.result_text(game)}" if game["status"] == "final" else "")
            )
            st.markdown(R.md_text(report["the_pick_summary"]))
            st.markdown(f"**Key mismatch.** {R.md_text(report['key_mismatch'])}")
            st.markdown(f"**Risk factor.** {R.md_text(report['risk_factor'])}")
            st.caption(write_up_meta(game))


def _render_details(st: Any, cfg: Settings, b: dict[str, Any], c: dict[str, Any] | None) -> None:
    with st.expander("Model, tiers and track record"):
        model = b["model"]
        st.markdown(
            f"- **Training:** {model['train_seasons'][0]}–{model['train_seasons'][-1]}, calibrated on "
            f"{model['calibration_season']} (walk-forward; never a random split).\n"
            f"- **Blend:** {model['market_weight']:.0%} market weight ({model['market_weight_source']})."
        )
        st.markdown(
            "**Tier rules** (the first match wins):\n\n"
            + "\n".join(f"- **{tier}:** {R.md_text(rule)}" for tier, rule in b["tier_rules"].items())
        )
        backtest = load_backtest(cfg)
        record = R.track_record(backtest, model["market_weight"]) if backtest else None
        st.markdown(record or "_No backtest report yet: run `python run_weekly.py --backtest`._")
        newsletter = cfg.newsletter_md(b["season"], b["week"])
        if c is not None and newsletter.exists():
            st.download_button(
                "Download the newsletter (Markdown)",
                newsletter.read_text(encoding="utf-8"),
                file_name=newsletter.name,
                mime="text/markdown",
            )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="NFL Weekly Scouting Report", page_icon=":material/sports_football:", layout="wide")
    cfg = Settings.from_env()  # read per run, so NFL_DATA_DIR / NFL_REPORTS_DIR take effect without a restart
    weeks = discover_weeks(cfg)
    if not weeks:
        st.title("NFL Weekly Scouting Report")
        st.info(
            f"No predictions found in `{cfg.predictions_dir}`. Run `python run_weekly.py` "
            "(or `python run_weekly.py --use-cached-odds --skip-llm` to avoid API calls) and reload."
        )
        return

    with st.sidebar:
        st.header("Week")
        index = st.selectbox(
            "Week",
            range(len(weeks)),
            index=default_week_index(cfg, weeks),
            format_func=lambda i: weeks[i].label,
            key="week",
            label_visibility="collapsed",
        )
        st.caption(f"Reading `{cfg.predictions_dir}`")
    ref = weeks[index]
    data = load_week(ref)
    record = load_run_record(cfg)
    if data.contract_b is None:
        st.title(f"NFL {ref.label}")
        _render_banners(st, banners(data, record))
        return

    _render_header(st, data, banners(data, record))
    _render_overview(st, data.contract_b)
    if data.contract_c is not None:
        _render_cards(st, data.contract_c)
    _render_details(st, cfg, data.contract_b, data.contract_c)


if __name__ == "__main__":
    main()
