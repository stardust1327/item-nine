# item-nine

NFL Game Prediction Engine and Weekly Scouting Report web app (Python 3.11+).

## Project layout

Implemented so far: `config.py`, `src/ingest.py` and `tests/test_ingest.py`. Everything else in the tree below is the planned layout.

```
item-nine/
├── config.py                  # all settings: paths, seasons, cache policy, Odds API, logging
├── requirements.txt
├── pyproject.toml             # pytest + ruff config
├── .env.example               # ODDS_API_KEY and optional overrides
├── src/
│   ├── ingest.py              # nflreadpy + The Odds API -> data/raw (Parquet cache)
│   ├── features/
│   │   ├── team_efficiency.py # EPA/play, success rate, early-down pass rate; decayed rolling windows
│   │   ├── qb.py              # QB-adjusted EPA, starter changes
│   │   ├── injuries.py        # weighted availability from injury reports + weekly rosters
│   │   ├── situational.py     # rest, travel, roof/surface, divisional, primetime
│   │   └── build.py           # game-level matrix, strictly as-of kickoff (no leakage)
│   ├── models/
│   │   ├── train.py           # XGBoost, walk-forward CV by season/week
│   │   ├── calibrate.py       # isotonic / Platt on out-of-fold predictions
│   │   ├── predict.py         # current-week win probabilities
│   │   └── evaluate.py        # log loss, Brier, ECE, model vs. market baseline
│   ├── market/
│   │   └── edges.py           # model vs. market_home_wp, fractional Kelly sizing
│   └── reports/
│       └── scouting.py        # weekly matchup report payloads
├── app/
│   ├── streamlit_app.py       # `streamlit run app/streamlit_app.py`
│   └── pages/                 # Weekly Picks, Scouting Report, Model Diagnostics, Data Health
├── data/                      # git-ignored
│   ├── raw/                   # pbp/, schedules/, rosters_weekly/, injuries/, odds/, _manifest.json
│   ├── interim/
│   └── processed/             # feature matrices
├── models/                    # trained artifacts (git-ignored)
├── logs/
├── notebooks/
└── tests/
```

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add ODDS_API_KEY

python -m src.ingest               # first run downloads 2016-current (~30s, ~135 MB)
python -m src.ingest               # later runs only refresh what is stale
python -m src.ingest --odds        # also pull current-week consensus market odds
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
