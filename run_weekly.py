"""Weekly pipeline: ingest -> features -> model -> scouting report (Modules 1-4).

Usage::

    python run_weekly.py                                # current week: refresh data, live odds, Gemini write-ups
    python run_weekly.py --use-cached-odds --skip-llm   # no API calls: saved odds snapshot, template reports
    python run_weekly.py --season 2026 --week 5         # replay a past week (writes that week's archives only)

Stages run in order, and each reads what the previous one wrote: the raw cache, then Contract A, then Contract B.
A failed stage stops the run, so later stages never run on missing or older inputs.

The latest-run files (``current_week.json``, ``weekly_report_enriched.json``) are written only for the current
week. Every run writes the versioned archives and the newsletter for its week.

Exit codes (shared with ``python -m src.report_generator``):
    0    OK
    1    a stage failed; later stages did not run
    2    missing API key: GEMINI_API_KEY (unless --skip-llm), or ODDS_API_KEY for the current week (unless
         --use-cached-odds)
    3    degraded: outputs written, but a requested live input was not used (no Gemini write-up for any game, no
         live odds for any upcoming game, or an nflverse refresh failed and an older cached copy was used)
    130  interrupted
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from config import Settings, settings, setup_logging
from src import features as F
from src import ingest
from src import model as M
from src import report_generator as R

logger = logging.getLogger("run_weekly")

EXIT_OK, EXIT_ERROR, EXIT_MISSING_KEY, EXIT_DEGRADED = R.EXIT_OK, R.EXIT_ERROR, R.EXIT_MISSING_KEY, R.EXIT_DEGRADED
EXIT_INTERRUPTED: Final[int] = 130

PIPELINE_DATASETS: Final[tuple[str, ...]] = ("schedules", "pbp")  # the raw inputs Modules 2-3 read

# Errors a stage is expected to raise; anything else is logged with a traceback (it is a bug).
EXPECTED_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ingest.IngestError,
    F.FeatureBuildError,
    M.ModelError,
    M.ContractError,
    R.ReportError,
    R.ContractCError,
    FileNotFoundError,
    ValueError,
)

StageStatus = Literal["ok", "degraded", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class Options:
    season: int | None = None
    week: int | None = None
    use_cached_odds: bool = False
    skip_llm: bool = False
    skip_ingest: bool = False
    backtest: bool = False


@dataclass(slots=True)
class StageResult:
    name: str
    status: StageStatus
    seconds: float = 0.0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a stage returns: a one-line summary, its product, and why it is degraded (empty = ok)."""

    detail: str
    value: Any = None
    degraded: tuple[str, ...] = ()


@dataclass(slots=True)
class RunResult:
    exit_code: int = EXIT_OK
    stages: list[StageResult] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)  # why the run is degraded (exit 3)
    outputs: list[Path] = field(default_factory=list)
    season: int | None = None
    week: int | None = None
    error: str | None = None


class MissingKeyError(RuntimeError):
    """A required API key is not configured (exit code 2)."""


WriterFactory = Callable[[Settings], R.ReportWriter]


def _cache(cfg: Settings) -> ingest.NflverseCache:
    return ingest.get_cache() if cfg is settings else ingest.NflverseCache(cfg)


# --------------------------------------------------------------------------- #
# Stages. Each returns an Outcome; a raised exception fails the stage and stops the run.
# --------------------------------------------------------------------------- #
def stage_ingest(cfg: Settings) -> Outcome:
    """Module 1: refresh the stale parts of the nflverse cache the pipeline reads."""
    cache = _cache(cfg)
    summary = cache.sync_all(PIPELINE_DATASETS)
    seasons = sorted(e.season for e in summary.get("pbp", []) if e.season is not None)
    detail = "schedules + " + (f"pbp {seasons[0]}-{seasons[-1]}" if seasons else "no pbp")
    failures = tuple(f"nflverse refresh failed; used the older cached copy ({f})" for f in cache.refresh_failures)
    return Outcome(detail, degraded=failures)


def stage_features(cfg: Settings) -> Outcome:
    """Module 2: rebuild Contract A from the raw cache."""
    features = F.build_and_save(cfg)
    unplayed = features.filter(features["result"].is_null()).height
    return Outcome(f"{features.height:,} games ({unplayed} without a result)", cfg.features_path)


def stage_backtest(cfg: Settings) -> Outcome:
    """Optional: walk-forward backtest (feeds the newsletter's and the dashboard's track record)."""
    pooled = M.run_backtest(cfg, rebuild_features=False).report["pooled"]
    detail = "pooled Brier model {:.4f} | market {:.4f} | final {:.4f}".format(
        pooled["model"]["brier"], pooled["market"]["brier"], pooled["final"]["brier"]
    )
    return Outcome(detail, cfg.predictions_dir / "backtest_report.json")


def resolve_target(cfg: Settings, opts: Options, now: datetime) -> M.WeekTarget:
    schedules = _cache(cfg).read("schedules", refresh=False)
    return M.resolve_week(schedules, now, season=opts.season, week=opts.week)


def stage_model(cfg: Settings, opts: Options, target: M.WeekTarget, now: datetime) -> Outcome:
    """Module 3: predict the week (live odds for the current week unless --use-cached-odds) and write Contract B."""
    live = not opts.use_cached_odds and target.is_current
    if live and not cfg.odds_api_key:
        raise MissingKeyError("ODDS_API_KEY is not set. Add it to .env, or pass --use-cached-odds.")
    payload = M.run_week(cfg, target=target, now=now, use_odds_api=live, rebuild_features=False)
    sources = Counter(g["market"]["source"] if g["market"] else "no line" for g in payload["games"])
    tiers = Counter(g["confidence_tier"] for g in payload["games"])
    detail = "{} games | market {} | tiers {}".format(
        len(payload["games"]),
        ", ".join(f"{n} {s}" for s, n in sources.most_common()),
        ", ".join(f"{tiers[t]} {t}" for t in M.TIERS if tiers[t]),
    )
    shortfall = live_odds_shortfall(payload) if live else None
    return Outcome(detail, payload, (shortfall,) if shortfall else ())


def live_odds_shortfall(payload: dict[str, Any]) -> str | None:
    """Why the run is degraded on market data: live odds were requested but no upcoming game got them."""
    upcoming = [g for g in payload["games"] if g["status"] == "scheduled"]
    live = [g for g in upcoming if g["market"] and g["market"]["source"] == "odds_api"]
    if len(live) < len(upcoming):
        logger.warning("%d of %d upcoming games have no live Odds API line", len(upcoming) - len(live), len(upcoming))
    if upcoming and not live:
        return f"live odds were requested but none of the {len(upcoming)} upcoming games got an Odds API line"
    return None


def stage_report(
    cfg: Settings, opts: Options, target: M.WeekTarget, now: datetime, writer: R.ReportWriter | None
) -> Outcome:
    """Module 4: scouting reports (Gemini, or templates with --skip-llm) -> Contract C and the newsletter."""
    contract_c = R.run_report(
        cfg,
        mode="template" if opts.skip_llm else "llm",
        writer=writer,
        input_path=cfg.preds_archive_json(target.season, target.week),
        now=now,
        update_latest=target.is_current,
    )
    summary = contract_c["summary"]
    if opts.skip_llm:
        return Outcome(f"{summary['n_games']} template reports (--skip-llm)", contract_c)
    if summary["n_template"]:
        logger.warning("%d of %d games fell back to the template report", summary["n_template"], summary["n_games"])
    detail = f"{summary['n_llm']} {contract_c['generator']['model']} / {summary['n_template']} template"
    degraded = ("no game got a Gemini write-up; every report is a template",) if summary["n_llm"] == 0 else ()
    return Outcome(detail, contract_c, degraded)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def default_writer_factory(cfg: Settings) -> R.ReportWriter:
    return R.GeminiReportWriter(cfg=cfg)  # raises MissingLLMKeyError without a key; makes no network call


def _make_writer(factory: WriterFactory, cfg: Settings) -> R.ReportWriter:
    try:
        return factory(cfg)
    except R.MissingLLMKeyError:
        raise MissingKeyError(
            "GEMINI_API_KEY is not set. Add it to .env (https://aistudio.google.com/apikey), "
            "or pass --skip-llm for template reports."
        ) from None


def _clock(now: datetime | None) -> Callable[[], datetime]:
    return (lambda: M._as_utc(now)) if now is not None else ingest.utc_now


def run(
    opts: Options,
    cfg: Settings = settings,
    *,
    now: datetime | None = None,
    writer_factory: WriterFactory = default_writer_factory,
) -> RunResult:
    """Run the pipeline. Returns the per-stage record and the exit code; never raises for a stage failure.

    ``now`` pins the clock for every stage (tests, replays). By default each stage reads the clock when it
    starts, so live odds are requested only for games that have not kicked off by then.
    """
    clock = _clock(now)
    result = RunResult()
    running = "preflight"

    def stage(name: str, fn: Callable[[], Outcome]) -> Any:
        nonlocal running
        running = name
        logger.info("== %s ==", name)
        started = time.perf_counter()
        outcome = fn()
        status: StageStatus = "degraded" if outcome.degraded else "ok"
        result.stages.append(StageResult(name, status, time.perf_counter() - started, outcome.detail))
        result.degraded += outcome.degraded
        return outcome.value

    def skip(name: str, why: str) -> None:
        logger.info("== %s == skipped (%s)", name, why)
        result.stages.append(StageResult(name, "skipped", 0.0, why))

    try:
        cfg.ensure_dirs()
        writer = None if opts.skip_llm else _make_writer(writer_factory, cfg)  # fails before any work

        if opts.skip_ingest:
            skip("ingest", "--skip-ingest: cached nflverse data used as-is")
        else:
            stage("ingest", lambda: stage_ingest(cfg))
        result.outputs.append(stage("features", lambda: stage_features(cfg)))
        if opts.backtest:
            result.outputs.append(stage("backtest", lambda: stage_backtest(cfg)))

        running = "model"
        model_now = clock()
        target = resolve_target(cfg, opts, model_now)
        result.season, result.week = target.season, target.week
        logger.info(
            "Target: %d week %d (%s)", target.season, target.week, "current week" if target.is_current else "replay"
        )
        stage("model", lambda: stage_model(cfg, opts, target, model_now))
        if target.is_current:
            result.outputs.append(cfg.current_week_json)
        result.outputs.append(cfg.preds_archive_json(target.season, target.week))

        stage("report", lambda: stage_report(cfg, opts, target, clock(), writer))
        if target.is_current:
            result.outputs.append(cfg.enriched_report_json)
        result.outputs += [
            cfg.enriched_archive_json(target.season, target.week),
            R.newsletter_path(cfg, target.season, target.week),
        ]
    except MissingKeyError as exc:
        return _fail(result, running, EXIT_MISSING_KEY, exc)
    except EXPECTED_ERRORS as exc:
        return _fail(result, running, EXIT_ERROR, exc)
    except Exception as exc:  # a bug: still a failed run with a clean exit code, but keep the traceback
        logger.exception("Unexpected error in %s", running)
        return _fail(result, running, EXIT_ERROR, exc)

    result.exit_code = EXIT_DEGRADED if result.degraded else EXIT_OK
    return result


def _fail(result: RunResult, stage_name: str, code: int, exc: BaseException) -> RunResult:
    result.error = f"{stage_name}: {type(exc).__name__}: {exc}"
    result.stages.append(StageResult(stage_name, "failed", 0.0, f"{type(exc).__name__}: {exc}"))
    result.exit_code = code
    logger.error("%s failed; later stages did not run. %s: %s", stage_name, type(exc).__name__, exc)
    return result


def format_summary(result: RunResult, cfg: Settings = settings) -> str:
    def rel(path: Path) -> str:
        return str(path.relative_to(cfg.project_root)) if path.is_relative_to(cfg.project_root) else str(path)

    target = f"{result.season} week {result.week}" if result.season is not None else "no target week"
    lines = [f"Weekly run: {target}"]
    lines += [f"  {s.name:<9} {s.status:<9} {s.seconds:6.1f}s  {s.detail}" for s in result.stages]
    if result.outputs and result.exit_code in (EXIT_OK, EXIT_DEGRADED):
        lines.append("Outputs:")
        lines += [f"  {rel(p)}" for p in result.outputs]
    if result.degraded:
        lines.append("DEGRADED:")
        lines += [f"  - {reason}" for reason in result.degraded]
    meaning = {
        EXIT_OK: "ok",
        EXIT_ERROR: "error",
        EXIT_MISSING_KEY: "missing API key",
        EXIT_DEGRADED: "degraded",
        EXIT_INTERRUPTED: "interrupted",
    }
    lines.append(f"Exit code {result.exit_code} ({meaning.get(result.exit_code, '?')})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _week(value: str) -> int:
    week = int(value)
    if not 1 <= week <= 22:
        raise argparse.ArgumentTypeError(f"week must be 1-22, got {week}")
    return week


def _season(value: str) -> int:
    season = int(value)
    if season < 1999:
        raise argparse.ArgumentTypeError(f"season must be 1999 or later, got {season}")
    return season


def parse_args(argv: Sequence[str] | None = None) -> tuple[Options, str | None]:
    parser = argparse.ArgumentParser(
        prog="python run_weekly.py",
        description="Run the weekly pipeline: ingest -> features -> model -> scouting report.",
        epilog="Exit codes: 0 ok, 1 error, 2 missing API key, 3 degraded (outputs written), 130 interrupted.",
    )
    parser.add_argument("--season", type=_season, help="NFL season (default: the season in progress)")
    parser.add_argument("--week", type=_week, help="week to predict (default: the current week); past weeks replay")
    parser.add_argument(
        "--use-cached-odds",
        action="store_true",
        help="do not call The Odds API (saves credits); use the latest saved snapshot, then nflverse lines",
    )
    parser.add_argument("--skip-llm", action="store_true", help="skip Gemini; write deterministic template reports")
    parser.add_argument("--skip-ingest", action="store_true", help="use the cached nflverse data without refreshing")
    parser.add_argument("--backtest", action="store_true", help="also re-run the walk-forward backtest")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ...")
    args = parser.parse_args(argv)
    opts = Options(
        season=args.season,
        week=args.week,
        use_cached_odds=args.use_cached_odds,
        skip_llm=args.skip_llm,
        skip_ingest=args.skip_ingest,
        backtest=args.backtest,
    )
    return opts, args.log_level


def main(argv: Sequence[str] | None = None) -> int:
    opts, log_level = parse_args(argv)
    setup_logging(log_level)
    cfg = settings  # module global looked up at call time (tests can redirect it)
    try:
        result = run(opts, cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED
    print(format_summary(result, cfg))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
