"""Shared fixtures: a synthetic league with a known data-generating process."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src import features as F

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


@pytest.fixture(scope="session")
def synthetic_league() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Feature matrix and schedules, 2016-2026 (2026 week 6 upcoming).

    Outcomes depend only on the two net EPA mismatches.
    """
    return make_frame()
