# item-nine

NFL Game Prediction Engine and Weekly Scouting Report web app (Python 3.11+).

## Project layout

The layout follows CLAUDE.md §3. Implemented so far: `config.py`, `src/ingest.py`, `src/features.py`, `src/model.py`, `src/report_generator.py` and their tests. `run_weekly.py` and `app.py` are planned.

```
item-nine/
├── CLAUDE.md                  # project rules (sections 1-3 checked in so far)
├── config.py                  # all settings: paths, seasons, cache policy, API keys, logging
├── requirements.txt
├── pyproject.toml             # pytest + ruff config
├── .env.example               # ODDS_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY + optional overrides
├── run_weekly.py              # (planned) end-to-end CLI orchestrator
├── app.py                     # (planned) Streamlit dashboard
├── data/                      # git-ignored
│   ├── raw/                   # pbp/, schedules/, rosters_weekly/, injuries/, odds/, _manifest.json
│   ├── processed/             # features.parquet
│   └── predictions/           # current_week.json (Contract B), weekly_report_enriched.json (Contract C), backtests
├── reports/                   # weekly Markdown newsletters
├── src/
│   ├── ingest.py              # nflreadpy + The Odds API -> data/raw (Parquet cache, de-vig)
│   ├── features.py            # leak-free game-level matrix -> data/processed/features.parquet
│   ├── model.py               # walk-forward XGBoost + sigmoid calibration + market blend -> Contract B
│   └── report_generator.py    # Gemini scouting reports + Pydantic validation -> Contract C + newsletter
└── tests/                     # test_leakage.py (leakage + probability bounds), test_ingest.py, test_model.py, ...
```

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add ODDS_API_KEY

python -m src.ingest               # first run downloads 2016-current (~30s, ~135 MB)
python -m src.ingest               # later runs only refresh what is stale
python -m src.ingest --odds        # also pull current-week consensus market odds
python -m src.features             # build data/processed/features.parquet (<1s)
python -m src.model                # walk-forward backtest + data/predictions/current_week.json (~10s)
python -m pytest
```

From Python:

```python
from src.ingest import get_pbp, get_schedules, get_injuries, get_weekly_rosters, fetch_market_win_probs

pbp = get_pbp(columns=["game_id", "season", "week", "posteam", "defteam", "epa", "success"])
market = fetch_market_win_probs()  # game_id, market_home_wp, market_away_wp, consensus_home_spread, ...
```

## Ingestion behavior

**Cache.** Each dataset is stored as one Parquet file per season under `data/raw/<dataset>/`. `data/raw/_manifest.json` records when each file was fetched.

- A season counts as final once its Super Bowl has a result and a 7-day stat-correction window has passed. Final seasons are downloaded once and never again.
- The in-progress season is refreshed once a week, after Tuesday 12:00 UTC, when Monday Night Football has been processed upstream.
- Injuries and schedules also expire after 12 hours, because injury reports come out Wednesday through Friday.
- Writes are atomic. If a refresh fails, the existing cached copy is kept.

**Market odds.** The pipeline fetches h2h and spreads from 8 major US books for games in the current week that have not kicked off yet. That costs 2 credits per call.

- Each book's vig is removed with the multiplicative method: p_fair = p / Σp.
- Books are combined with the median (configurable), and the result is renormalized so home + away = 1.
- Games without a moneyline fall back to a win probability implied by the spread, using Normal(μ, σ = 13.45).
- Every pull is saved to `data/raw/odds/` for closing-line-value analysis.

## Feature matrix

`src/features.py` writes one row per scheduled game, both played and upcoming, to `data/processed/features.parquet`. Columns are grouped as `ID_COLUMNS`, `FEATURE_COLUMNS` (model inputs, all known before kickoff), `REFERENCE_COLUMNS` (the closing spread, used only for benchmarking) and `LABEL_COLUMNS`.

- **Garbage time.** 4th-quarter plays with offense win probability below 5% or above 95% are dropped before any metric is computed.
- **Team-game metrics.** Offensive pass and rush EPA per play, defensive pass and rush EPA allowed per play, offensive early-down success rate, and the starting QB's CPOE.
- **No leakage.** Each team's EWMA (half-life 4 games) is computed within its season, then forward-filled and shifted by one game. A week N game only sees weeks 1 through N-1.
- **Prior regression.** In weeks 1-4, each metric blends the team's play-weighted season Y-1 value with the in-season EWMA. The prior gets 100% of the weight in week 1, 25 points less each week, and none from week 5. The first cached season (2016) has no prior, so start training at 2017.
- **Matchups.** Net pass and rush EPA mismatch (offense vs. the opposing defense), differences in early-down success rate and QB CPOE, net rest days, and divisional, neutral-site, outdoor and high-wind (15+ mph) flags.
- **Unknowns stay null.** nflverse fills in wind only after kickoff, and retractable-roof status isn't known before game day. Upcoming games keep those flags null unless you pass a wind forecast through `build_game_features(..., weather=...)`.

`tests/test_leakage.py` changes a game's own play-by-play and final score and asserts its features don't move. It also rebuilds every game from data cut off at that game's kickoff and asserts the result matches the full build.
