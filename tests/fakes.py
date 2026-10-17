"""Test doubles shared by several test modules."""

from __future__ import annotations

import json
import threading

import polars as pl

from src import report_generator as R


def nflverse_schedules(features: pl.DataFrame, moneylines: pl.DataFrame) -> pl.DataFrame:
    """The synthetic league as an nflverse schedules file: ``gameday``/``gametime`` in US Eastern, results, lines."""
    eastern = pl.col("kickoff_utc").dt.convert_time_zone("America/New_York")
    return (
        features.select(
            "game_id",
            "season",
            "week",
            "game_type",
            "kickoff_utc",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "result",
            "total",
        )
        .join(moneylines.drop("season"), on="game_id", how="left")
        .with_columns(eastern.dt.strftime("%Y-%m-%d").alias("gameday"), eastern.dt.strftime("%H:%M").alias("gametime"))
        .drop("kickoff_utc")
    )


class GroundedWriter:
    """Thread-safe fake LLM that writes a grounded report from the fact sheet in the prompt."""

    name = "fake-gemini"

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system_instruction: str, prompt: str) -> str:
        with self._lock:
            self.calls += 1
        assert system_instruction == R.SYSTEM_INSTRUCTION
        sheet = json.loads(prompt.split("FACT SHEET (JSON):\n", 1)[1].split("\n\nYour previous", 1)[0])
        pick = sheet["pick"]
        return json.dumps(
            {
                "headline": f"{pick['team']} ({pick['abbr']}) carry the edge into kickoff",
                "the_pick_summary": (
                    f"The blended forecast picks the {pick['team']} at {pick['final_win_probability_pct']}% "
                    f"({sheet['confidence_tier']})."
                ),
                "key_mismatch": (
                    f"{sheet['top_drivers'][0]['stat']} favors the {sheet['top_drivers'][0]['favors']}."
                    if sheet["top_drivers"]
                    else "No single statistic separates these teams this week."
                ),
                "risk_factor": f"The opponent still wins {pick['opponent_win_probability_pct']}% of the time.",
            }
        )
