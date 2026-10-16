# item-nine

NFL Game Prediction Engine and Weekly Scouting Report web app (Python 3.11+).

## Project layout

CLAUDE.md is the authoritative spec (invariants, Contracts A/B/C, the run record, configuration and exit codes). Every module in the layout below is implemented and tested.

```
item-nine/
├── CLAUDE.md                  # authoritative spec: invariants, Contracts A/B/C, env, exit codes
├── config.py                  # all settings: paths, seasons, cache policy, API keys, logging
├── requirements.txt
├── pyproject.toml             # pytest + ruff config
├── .env.example               # ODDS_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY + optional overrides
├── run_weekly.py              # weekly orchestrator: ingest -> features -> model -> report
├── app.py                     # Streamlit dashboard (read-only)
├── data/                      # git-ignored
│   ├── raw/                   # pbp/, schedules/, rosters_weekly/, injuries/, odds/, _manifest.json
│   ├── processed/             # features.parquet
│   └── predictions/           # current_week.json (Contract B), weekly_report_enriched.json (Contract C),
│                              # week_XX_YYYY_preds.json / week_XX_YYYY_enriched.json archives,
│                              # last_run.json (run record), backtests
├── reports/                   # weekly Markdown newsletters (week_XX_YYYY.md)
├── src/
│   ├── ingest.py              # nflreadpy + The Odds API -> data/raw (Parquet cache, de-vig)
│   ├── features.py            # leak-free game-level matrix -> data/processed/features.parquet
│   ├── model.py               # walk-forward XGBoost + sigmoid calibration + market blend -> Contract B
│   └── report_generator.py    # Gemini scouting reports + Pydantic validation -> Contract C + newsletter
└── tests/                     # test_leakage.py (leakage + probability bounds), one test file per module
```

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add ODDS_API_KEY and GEMINI_API_KEY (or use the no-API flags below)

python run_weekly.py --backtest    # first run: downloads 2016-current (~30s, ~135 MB), backtest, week predictions
python run_weekly.py               # each week: refresh stale data, live odds (2 credits), Gemini write-ups
python run_weekly.py --use-cached-odds --skip-llm   # no API calls at all
streamlit run app.py               # dashboard at http://localhost:8501
python -m pytest
```

Each module also has its own CLI: `python -m src.ingest [--odds]`, `python -m src.features`, `python -m src.model [backtest|predict]` and `python -m src.report_generator [--template]`.

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

## Model and weekly predictions

`python -m src.model` runs the walk-forward backtest and then predicts the current week.

- **Walk-forward validation.** To score season S, the model trains on 2016 through S-2, calibrates on S-1 and is scored on S. There is never a random split.
- **Model.** `XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=250, subsample=0.8, colsample_bytree=0.8)`, frozen after training and wrapped in `CalibratedClassifierCV(method="sigmoid")`.
- **Blend.** `final = w·market + (1-w)·model`. The market weight `w` is chosen by `scipy.optimize` to minimize Brier score on cross-fitted probabilities from the calibration season. It starts from 0.60, and 0.60 is the fallback.
- **Market data.** The current week uses de-vigged Odds API consensus lines, then a saved snapshot, then nflverse lines. History uses nflverse closing lines.
- **Output.** `data/predictions/current_week.json` (Contract B, schema 1.3.0) holds, per game: the pick, final/model/market win probabilities, `model_edge_pct`, `value_team`, a tier, and the top 3 SHAP drivers. A copy is archived as `data/predictions/week_XX_YYYY_preds.json` so later weeks and seasons never overwrite it.
- **Tiers.** LOCK means the pick is at least 75% and the model also favors it. **STAT MISMATCH (Model vs. Market)** means the play-by-play model and the market differ by at least 8 points on the pick, in either direction. Only the week's 2 largest gaps get it; the rest fall through to LEAN or COIN FLIP. It flags a disagreement between the numbers and the line, not a betting edge. LEAN means at least 55%. Everything else is a COIN FLIP. The settings are `NFL_TIER_LOCK_MIN_WIN_PROB`, `NFL_STAT_MISMATCH_MIN_PCT`, `NFL_STAT_MISMATCH_MAX_PER_WEEK` and `NFL_TIER_LEAN_MIN_WIN_PROB`. Each payload records the thresholds it was built with, and the validator checks tiers against those, so archived weeks stay valid when the settings change.
- **Backtest output.** `data/predictions/backtest_report.json` and `backtest_predictions.parquet`. Pooled out-of-sample Brier for 2018–2026: model 0.233, market 0.211, blend 0.212.

## Scouting reports

`python -m src.report_generator` turns Contract B into prose with Gemini (`google-genai`, default `gemini-2.5-flash`, key `GEMINI_API_KEY`). `--template` skips the LLM.

- **Validation.** Each game's report is checked by a Pydantic model with four required fields: `headline`, `the_pick_summary`, `key_mismatch` and `risk_factor`. It must be plain prose, and it is grounded against that game's fact sheet:
  - it names the pick, its tier and its final win probability
  - every percentage comes from the fact sheet
  - no other team or tier is mentioned
  - there is no hype or betting language
- **Retries.** A rejected reply is sent back with the errors, up to 3 attempts. After that the game gets a deterministic template report, flagged in `report_meta`.
- **Output.** `data/predictions/weekly_report_enriched.json` (Contract C: Contract B plus `report` and `report_meta` per game, re-validated), its archive `data/predictions/week_XX_YYYY_enriched.json`, and the newsletter `reports/week_XX_YYYY.md`.
- **Exit codes.** 0 ok; 1 error, with nothing written (auth, unknown model, daily quota, bad input); 2 missing API key; 3 written, but no game got a Gemini write-up.

## Feature matrix

`src/features.py` writes one row per scheduled game, both played and upcoming, to `data/processed/features.parquet`. Columns are grouped as `ID_COLUMNS`, `FEATURE_COLUMNS` (model inputs, all known before kickoff), `REFERENCE_COLUMNS` (the closing spread, used only for benchmarking) and `LABEL_COLUMNS`.

- **Garbage time.** 4th-quarter plays with offense win probability below 5% or above 95% are dropped before any metric is computed.
- **Team-game metrics.** Offensive pass and rush EPA per play, defensive pass and rush EPA allowed per play, offensive early-down success rate, and the starting QB's CPOE.
- **No leakage.** Each team's EWMA (half-life 4 games) is computed within its season, then forward-filled and shifted by one game. A week N game only sees weeks 1 through N-1.
- **Prior regression.** In weeks 1-4, each metric blends the team's play-weighted season Y-1 value with the in-season EWMA. The prior gets 100% of the weight in week 1, 25 points less each week, and none from week 5. The first cached season (2016) has no prior, so start training at 2017.
- **Matchups.** Net pass and rush EPA mismatch (offense vs. the opposing defense), differences in early-down success rate and QB CPOE, net rest days, and divisional, neutral-site, outdoor and high-wind (15+ mph) flags.
- **Unknowns stay null.** nflverse fills in wind only after kickoff, and retractable-roof status isn't known before game day. Upcoming games keep those flags null unless you pass a wind forecast through `build_game_features(..., weather=...)`.

`tests/test_leakage.py` changes a game's own play-by-play and final score and asserts its features don't move. It also rebuilds every game from data cut off at that game's kickoff and asserts the result matches the full build.

## Weekly run

`python run_weekly.py` runs Modules 1-4 in order:
1. **Ingest:** refresh the stale nflverse schedules and play-by-play.
2. **Features:** rebuild the feature matrix.
3. **Model:** predict the week, with live odds for the current week.
4. **Report:** write the scouting reports and the newsletter.

If a stage fails, the run stops and later stages don't run. It then prints a stage-by-stage summary and writes `data/predictions/last_run.json`.

| Flag | Effect |
|---|---|
| `--use-cached-odds` | No Odds API call. Uses the latest saved snapshot from or before the prediction time, then nflverse lines. |
| `--skip-llm` | Deterministic template write-ups; no Gemini key needed. |
| `--season S --week W` | Replay a past week. Only that week's archives and newsletter are written; `current_week.json` is left alone. Weeks after the current one are rejected. |
| `--skip-ingest` | Use the cached nflverse data as-is (offline). |
| `--backtest` | Also re-run the walk-forward backtest, which feeds the track record. |

**Exit codes:**
- 0 ok
- 1 a stage failed
- 2 a missing API key. Gemini is checked before any work. The Odds API key is checked before predicting the current week.
- 3 degraded. Outputs are written, but one of these happened:
  - no game got a Gemini write-up
  - live odds were requested but no upcoming game got one
  - an nflverse refresh failed and an older cached copy was used
- 130 interrupted

## Dashboard

`streamlit run app.py` reads the files the pipeline wrote and nothing else. It makes no API calls and doesn't retrain.

- **Week picker:** the sidebar lists every archived week, newest first. It opens on the latest run's week.
- **Banners**, at the top under the title:
  - Errors: unreadable files, a failed last run, a degraded run.
  - Warnings: no report yet, write-ups older than the predictions, template mode, Gemini fallbacks.
  - Info: games without a market line.
- **Metadata row:** prediction time, odds source with its snapshot time, market weight, and who wrote the write-ups.
- **Chart:** model vs market, as a dot plot of each pick's win probability. Market and model are dots and the blend is a diamond.
- **Table:** the same values per game, with edge, tier and result.
- **Scouting reports:** one card per game with a tier badge and the four write-up fields. The cards can be filtered by tier.
- **Model, tiers and track record:** an expander with the tier rules, the walk-forward track record and a newsletter download.

`NFL_DATA_DIR` and `NFL_REPORTS_DIR` point it at other directories.
