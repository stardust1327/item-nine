# CLAUDE.md — item-nine: NFL Prediction Engine & Weekly Scouting Report

This document is the authoritative architecture, data-contract and engineering specification for `item-nine`.
Contributors, agents and reviewers must follow the schemas, invariants and rules below.

The code is the executable form of this document. Every contract has a validator, named below. Producers run it
before writing, and consumers may run it after reading. Any change to a contract, path or invariant must update this
file in the same PR.

---

## 1. System overview

The pipeline is six sequential components. Each writes the input of the next.

| # | Component | File | Output |
|---|---|---|---|
| 1 | Ingestion | `src/ingest.py` | `data/raw/`: nflverse Parquet cache plus `_manifest.json`; Odds API snapshots |
| 2 | Feature engineering | `src/features.py` | **Contract A**: `data/processed/features.parquet` |
| 3 | Model and live odds pricing | `src/model.py` (the Odds API client and de-vig live in `src/ingest.py`) | **Contract B** |
| 4 | Report generator | `src/report_generator.py` | **Contract C** and the Markdown newsletter |
| 5 | Weekly orchestrator | `run_weekly.py` | Runs 1→4 in order, stops on a failed stage; **run record** `last_run.json` |
| 6 | Dashboard | `app.py` | Streamlit view of Contracts B and C for any archived week, with degraded/template banners |

## 2. Directory structure

```text
item-nine/
├── CLAUDE.md
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example                          # ODDS_API_KEY, GEMINI_API_KEY (+ optional overrides)
├── config.py                             # Settings: paths, seasons, cache policy, API/LLM settings, tier thresholds
├── run_weekly.py                         # Module 5
├── app.py                                # Module 6
├── .streamlit/config.toml                # dashboard chrome (viewer toolbar, no usage stats)
├── .github/workflows/ci_and_weekly.yml   # CI (ruff + pytest) and the scheduled weekly run (Section 9)
├── src/
│   ├── __init__.py
│   ├── ingest.py                         # Module 1 (+ Odds API client, de-vig, snapshot writer)
│   ├── features.py                       # Module 2
│   ├── model.py                          # Module 3
│   └── report_generator.py               # Module 4
├── data/                                 # git-ignored, except data/predictions/*.json (committed by the weekly job)
│   ├── raw/
│   │   ├── _manifest.json                # fetch time and finality of every cached file
│   │   ├── pbp/ schedules/ rosters_weekly/ injuries/
│   │   └── odds/YYYY/odds_YYYY_wkXX_<UTC stamp>_{books,consensus}.parquet   # Odds API snapshots
│   ├── processed/
│   │   └── features.parquet              # Contract A
│   └── predictions/
│       ├── current_week.json             # Contract B (latest run)
│       ├── week_XX_YYYY_preds.json       # Contract B (versioned archive)
│       ├── weekly_report_enriched.json   # Contract C (latest run)
│       ├── week_XX_YYYY_enriched.json    # Contract C (versioned archive)
│       ├── last_run.json                 # run record of the last run_weekly.py run
│       ├── backtest_report.json          # walk-forward metrics
│       └── backtest_predictions.parquet  # out-of-sample predictions
├── reports/
│   └── week_XX_YYYY.md                   # Markdown newsletter, e.g. week_06_2026.md (committed by the weekly job)
└── tests/
    ├── conftest.py                       # shared synthetic league
    ├── fakes.py                          # shared test doubles (fake Gemini writer, nflverse-style schedules)
    ├── test_ingest.py                    # Module 1
    ├── test_leakage.py                   # Module 2 leakage, probability bounds, project-wide guards
    ├── test_model.py                     # Module 3
    ├── test_report_generator.py          # Module 4
    ├── test_run_weekly.py                # Module 5
    └── test_app.py                       # Module 6
```

`XX` is the zero-padded week, so week 6 is `06`. `YYYY` is the NFL season, which is the year the season starts. Paths
are built only by `config.Settings` (`preds_archive_json`, `enriched_archive_json`, `newsletter_md`). Never hard-code
them elsewhere.

## 3. Tech stack and library rules

- **Python 3.11+**. Lint and format with `ruff`, line length 120.
- **NFL data: `nflreadpy` only.** **NEVER** import or use `nfl_data_py`; it was deprecated in 2025. Enforced by
  `tests/test_leakage.py::test_nfl_data_py_is_never_used`.
- **Data:** `polars`, with Parquet via `pyarrow`.
- **ML:** `xgboost`, `scikit-learn` (`CalibratedClassifierCV`, `brier_score_loss`, `log_loss`) and `scipy.optimize`.
  Drivers are SHAP values from XGBoost's native TreeSHAP (`pred_contribs`).
- **LLM:** `google-genai`, default model `gemini-2.5-flash`. All output is validated with `pydantic>=2`.
- **Frontend:** `streamlit`. **Tests:** `pytest`.
- **Secrets:** `.env`, which is git-ignored, holds `ODDS_API_KEY` and `GEMINI_API_KEY`. Never commit, print or log keys;
  the Odds API client redacts them from errors.

## 4. Modeling invariants

Each invariant has tests; breaking one is a blocking defect.

1. **Garbage time.** Drop play-by-play rows with `qtr == 4` and `wp < 0.05` or `wp > 0.95` before computing any metric.
   Rows with a null `wp` are kept.
2. **Team-game metrics.** Offensive pass and rush EPA per play, defensive pass and rush EPA allowed per play, offensive
   early-down (1st/2nd) success rate, and the starting QB's CPOE. Plays follow the nflfastR convention
   (`pass == 1 | rush == 1`), with 2-point attempts excluded.
3. **Zero temporal leakage.** Within each team-season, ordered by kickoff, apply EWMA (half-life 4 games), then
   forward-fill, then `shift(1)`. A week N row sees only weeks 1..N-1. A game's own box score never appears in its
   features.
4. **Weeks 1–4 prior.** Blend the team's play-weighted Season Y-1 value with the in-season EWMA. The prior gets 100% in
   week 1, 75% in week 2, 50% in week 3, 25% in week 4, and 0% from week 5.
5. **Team codes.** `OAK→LV`, `SD→LAC` and `STL→LA` everywhere, so joins and priors survive relocations.
6. **Walk-forward validation.** To score season S, train on all seasons from 2016 through S-2, calibrate on S-1 and
   evaluate on S. Production uses the same recipe. **Never** use a random split; `train_test_split` is banned by a test.
7. **Model.** `XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=250, subsample=0.8, colsample_bytree=0.8)`,
   frozen and wrapped in `CalibratedClassifierCV(method="sigmoid")`, which is fit on S-1.
8. **Train/serve consistency.** Venue inputs unknown before kickoff (wind, retractable-roof state) take
   `model.MODEL_INPUT_DEFAULTS` identically in training and serving. Contract A keeps them null. Drivers describe only
   observed inputs.
9. **Market blend.** `final = w·market + (1-w)·model`. `w ∈ [0, 1]` is chosen by `scipy.optimize.minimize` to minimize
   Brier score on cross-fitted calibration-season probabilities. It starts at 0.60, and 0.60 is also the fallback.
10. **Market data.** De-vig is multiplicative (`p / Σp`).
    - History uses nflverse closing moneylines.
    - The current week uses, in priority order: the live Odds API consensus (median of books), then the latest snapshot
      saved **at or before** prediction time, then nflverse lines.
    - A line is `nflverse_close` only if it was observed after kickoff. Otherwise it is `nflverse_line`, stamped with its
      fetch time.
11. **Confidence tiers.** These use configurable thresholds (Section 7). For each game the first match wins; then the STAT MISMATCH cap is applied to the week.

    | Tier | Rule (defaults) |
    |---|---|
    | `LOCK` | The pick's final win probability is ≥ `lock_min_win_prob` (0.75), **and** the model also favors the pick. |
    | `STAT MISMATCH` | The model and the market differ by ≥ `mismatch_min_pct` (8.0) percentage points on the pick, **in either direction**: \|`model_edge_pct`\| ≥ 8.0, i.e. 0.08 in probability. **Capped** at `mismatch_max_per_week` (2) games per week, largest \|gap\| first, ties to the earlier kickoff, then the lower `game_id`. Games over the cap fall through to `LEAN` or `COIN FLIP`. |
    | `LEAN` | The pick's final win probability is ≥ `lean_min_win_prob` (0.55). |
    | `COIN FLIP` | Everything else. |

    `STAT MISMATCH` flags a large disagreement between the play-by-play model and the betting line. It is **not** a
    betting edge, and the newsletter, dashboard and LLM prompt say so: the display label is "STAT MISMATCH (Model vs.
    Market)" (`model.TIER_LABELS`). Because of the cap, tiers are derived for the whole week (`model.week_tiers`) from
    the published, rounded values, so anyone can recompute them from the JSON.

## 5. Data contracts and pipeline behavior

### Contract A — `data/processed/features.parquet`

One row per scheduled game, played or upcoming. The column groups are defined in `src/features.py`:

| Group | Columns |
|---|---|
| `ID_COLUMNS` | `game_id, season, week, game_type, kickoff_utc, home_team, away_team` |
| `FEATURE_COLUMNS` (22, model inputs, pre-game only) | `{home,away}_{off_pass_epa, off_rush_epa, def_pass_epa, def_rush_epa, off_early_down_sr, qb_cpoe}_pre`, `net_pass_epa_mismatch, net_rush_epa_mismatch, diff_early_down_sr, diff_qb_cpoe, net_rest_days, is_divisional, is_neutral_site, is_outdoor, is_high_wind_outdoor, prior_weight` |
| `REFERENCE_COLUMNS` | `spread_line`: the closing line, used for benchmarking only and **never** as a model input |
| `LABEL_COLUMNS` | `home_score, away_score, result, total, home_win` (null for unplayed games and ties) |

Validated by `tests/test_leakage.py`. Perturbing a game's own play-by-play or score must leave its features
bit-identical, and rebuilding every game from history cut at its kickoff must match the full build.

### Contract B — weekly predictions (schema `1.3.0`)

Written to `data/predictions/current_week.json` (latest run) and `week_XX_YYYY_preds.json` (archive). Validated by
`src.model.validate_contract_b`.

| Field | Type / meaning |
|---|---|
| `contract`, `schema_version` | `"B"`, `"1.3.0"` |
| `generated_at` | ISO-8601 UTC prediction time |
| `season`, `week` | int |
| `model` | Fields:<br>`algorithm` and `hyperparameters`<br>`train_seasons` (all ≤ season-2)<br>`calibration_season` (= season-1)<br>`market_weight` ∈ [0,1] and `market_weight_source` (`optimized` or `default`)<br>`weight_fit`, an in-sample Brier summary<br>`input_defaults`<br>`tier_thresholds` {`lock_min_win_prob`, `mismatch_min_pct`, `mismatch_max_per_week`, `lean_min_win_prob`} |
| `tier_rules` | {tier → human-readable rule}; must match `model.tier_thresholds` |
| `games[]` | One object per game of the week (fields below) |

Per game:

| Field | Type / meaning |
|---|---|
| `game_id`, `season`, `week`, `game_type` | nflverse identifiers. `game_type` is `REG`, `WC`, `DIV`, `CON` or `SB`. |
| `kickoff_utc`, `status` | ISO-8601. `status` is `scheduled` (kickoff after `generated_at`), `in_progress` or `final`. |
| `home_team`, `away_team` | nflverse abbreviations |
| `venue` | {`is_neutral_site`, `is_outdoor`, `is_high_wind`, `is_divisional`}: bool or null (null = unknown before kickoff) |
| `net_rest_days` | int or null (home minus away) |
| `market` | null when there is no line. Otherwise:<br>`home_wp`, `away_wp` (de-vigged)<br>`home_spread` (sportsbook sign: negative = home favored)<br>`source` (`odds_api`, `odds_api_snapshot`, `nflverse_line` or `nflverse_close`)<br>`method` (`h2h`, `spread_implied` or `nflverse_moneyline`)<br>`books`<br>`as_of` |
| `model`, `final` | {`home_wp`, `away_wp`}, summing to 1. `final` = blend at `market_weight`, or `model` when `market` is null. |
| `pick` | {`team`, `side`, `win_prob`}: the side `final` favors |
| `model_edge_pct` | `100·(model − market)` on the pick side, in percentage points; null without a market line |
| `value_team`, `value_edge_pct` | The side the model rates above the market, and by how much |
| `confidence_tier` | `LOCK`, `STAT MISMATCH`, `LEAN` or `COIN FLIP` (Section 4.11; validated for the whole week) |
| `top_drivers[]` | ≤ 3 objects {`feature`, `label`, `value`, `impact_logit`, `favors`}, ordered by abs(`impact_logit`); observed inputs only |
| `imputed_inputs[]` | Venue inputs that took `MODEL_INPUT_DEFAULTS` |
| `result` | {`home_score`, `away_score`} when `final`, otherwise null |

### Contract C — enriched weekly report (schema `1.1.0`)

Written to `data/predictions/weekly_report_enriched.json` (latest run) and `week_XX_YYYY_enriched.json` (archive).
Validated by `src.report_generator.validate_contract_c`, which also re-validates the embedded Contract B and
re-grounds every report against its own game.

| Field | Type / meaning |
|---|---|
| `contract`, `schema_version`, `generated_at` | `"C"`, `"1.0.0"`, ISO-8601 |
| `season`, `week`, `model`, `tier_rules` | Copied from Contract B |
| `source` | {`contract`: `"B"`, `schema_version`, `generated_at`}: the predictions this report was built from |
| `generator` | {`provider`, `model`, `mode` (`llm` or `template`), `temperature`, `max_attempts`} |
| `summary` | {`n_games`, `n_llm`, `n_template`, `tiers`: {tier → count}} |
| `newsletter_path` | Repository-relative path of the newsletter |
| `games[]` | Each Contract B game, plus the two objects below |
| `games[].report` | {`headline`, `the_pick_summary`, `key_mismatch`, `risk_factor`}: validated prose |
| `games[].report_meta` | {`generator` (model name or `template`), `attempts`, `fallback_reason`, `validation_errors[]`} |

A run is **degraded** when `generator.mode == "llm"` and `summary.n_llm == 0`. The orchestrator signals this with
exit code 3 and the dashboard shows a banner.

### Newsletter — `reports/week_XX_YYYY.md`

The newsletter contains:
- a glance table (kickoff in ET, pick, win probability, tier, model vs market, result)
- one section per tier, with each game's four report fields
- an "already played" section
- the tier rules
- the walk-forward track record
- a not-betting-advice disclaimer

Displayed numbers are truncated, so they never round across a tier threshold.

### Module 5 — `run_weekly.py`

- **Stages:** preflight (Gemini key) → ingest (`schedules`, `pbp` only) → features → optional backtest → model → report.
  Each stage reads what the previous one wrote. A failed stage stops the run; a degraded stage does not.
- **Target week:** `model.resolve_week`. The default is the current week: the earliest week, in the season of the next
  game to kick off (by the run's clock), that still has a game yet to kick off. In the days before the opener this is
  already week 1 of the new season. `--week` may name a past week (a replay); a later week is an error.
- **Latest-run files** (`current_week.json`, `weekly_report_enriched.json`) are written only for the current week. Every
  run writes the week's archives and newsletter, so a replay never replaces the latest predictions.
- **Live odds** are requested only for the current week and only without `--use-cached-odds`, at most one pull (2
  credits) per run. Each stage reads the clock when it starts, so games that kicked off during earlier stages get no
  live price.
- **Degraded** (exit 3): an nflverse refresh failed and an older cached copy was used; live odds were requested but no
  upcoming game got an `odds_api` line; or the LLM ran and no game got a write-up (`DEGRADED_NO_LLM`).

### Run record — `data/predictions/last_run.json` (schema `1.0.0`)

Written by `run_weekly.py` after every run, including failed ones. A failure to write it is logged and never changes
the exit code.

| Field | Type / meaning |
|---|---|
| `record`, `schema_version` | `"run"`, `"1.0.0"` |
| `started_at`, `finished_at` | ISO-8601 UTC |
| `options` | The CLI options (`season`, `week`, `use_cached_odds`, `skip_llm`, `skip_ingest`, `backtest`) |
| `season`, `week`, `is_current` | The target week, or null if the run failed before choosing one |
| `exit_code`, `status` | 0/1/2/3 and `ok`, `error`, `missing API key` or `degraded` |
| `degraded[]` | Why the run is degraded (empty unless exit code 3) |
| `error` | `"<stage>: <ErrorType>: <message>"`, or null |
| `stages[]` | {`name`, `status` (`ok`, `degraded`, `skipped`, `failed`), `seconds`, `detail`} |
| `outputs[]` | Repository-relative paths written (empty unless exit code 0 or 3) |

### Module 6 — `app.py`

- **Read-only.** The dashboard reads only `data/predictions/` and `reports/`. It never calls an API, retrains, or
  writes. `Settings.from_env()` is read on every rerun.
- **Weeks.** Weeks come from the archives. A latest-run file fills in a week that has no archive. The default week is
  that of `current_week.json`.
- **Validation.** Contract B and C are validated on load. A file that fails is reported in an error banner and never
  displayed. A valid Contract C stands in for an invalid Contract B.
- **Banners**, in this order:
  - **Errors:** unreadable files; a failed last run (exit 1/2); a degraded run. The run record's reasons apply only
    when the record is the run that produced the shown predictions: same week, and `started_at ≤ generated_at ≤
    finished_at`. An all-template LLM report is also degraded.
  - **Warnings:** no report yet; write-ups older than the predictions; template mode; partial template fallback.
  - **Info:** games without a market line.
- **Display.** The chart and table show pick-side probabilities, truncated like the newsletter. Chart colors are
  categorical slots 1–2 of the dataviz reference palette, validated in light and dark mode. Every value in the chart is
  also in the table.

## 6. LLM rules (Module 4)

- The LLM sees only the per-game fact sheet built by `game_facts`. It is never given raw logits or numbers the
  validator can't check.
- `ScoutingReport` (Pydantic) enforces two kinds of rules.
  - **Format:** length limits; one paragraph of plain prose with no markdown, HTML, code, links or URLs; text
    Unicode-normalized first.
  - **Grounding**, against the fact sheet:
    - the pick, its tier and its final win probability are stated
    - no other tier or outside team is named
    - the headline names a team in the game
    - every percentage matches a fact, at 1 decimal or rounded to a whole number
    - no hype or betting language
- **Retries:** a rejected reply is re-prompted with the errors, up to `LLM_MAX_ATTEMPTS`. After that the game falls back
  to the deterministic template, which is flagged in `report_meta`.
- **Errors:**
  - Auth failures, unknown models, location errors, 400 request errors and the daily quota abort the run before
    anything is written.
  - Per-minute 429s and 5xx errors are retried.

## 7. Configuration

All settings live in `config.Settings`, populated from the environment or `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `ODDS_API_KEY` | — | Required for live odds |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | Required for LLM reports |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `NFL_STAT_MISMATCH_MIN_PCT` | `8.0` | STAT MISMATCH threshold: \|model − market\| on the pick, in percentage points |
| `NFL_STAT_MISMATCH_MAX_PER_WEEK` | `2` | STAT MISMATCH games per week, largest gaps first (0 turns the tier off) |
| `NFL_TIER_LOCK_MIN_WIN_PROB` | `0.75` | LOCK threshold |
| `NFL_TIER_LEAN_MIN_WIN_PROB` | `0.55` | LEAN threshold |
| `NFL_START_SEASON` | `2016` | First cached season |
| `ODDS_BOOKMAKERS` | 8 major US books | Books used for the consensus (≤ 10 books counts as 1 region of credits) |
| `LLM_MAX_ATTEMPTS`, `LLM_TEMPERATURE`, `LLM_MAX_CONCURRENCY`, `GEMINI_THINKING_BUDGET` | 3, 0.4, 4, model default | Generation controls |
| `NFL_DATA_DIR`, `NFL_REPORTS_DIR`, `NFL_MODELS_DIR`, `NFL_LOGS_DIR` | Under the repo | Path overrides (the dashboard reads them on every rerun) |

## 8. Commands and exit codes

| Command | What it does |
|---|---|
| `python -m src.ingest [--odds]` | Sync the nflverse cache. `--odds` also pulls live odds, at 2 credits per call. |
| `python -m src.features` | Build Contract A |
| `python -m src.model [backtest\|predict\|all] [--no-odds]` | Walk-forward backtest and/or Contract B. `--no-odds` uses a snapshot or nflverse lines instead of a live pull. |
| `python -m src.report_generator [--template] [--strict]` | Contract C and the newsletter |
| `python run_weekly.py [--season S] [--week W] [--use-cached-odds] [--skip-llm] [--skip-ingest] [--backtest]` | The full weekly pipeline (Section 5, Module 5) |
| `streamlit run app.py` | The dashboard (read-only) |

The report generator and orchestrator share these exit codes:

| Code | Report generator | Orchestrator |
|---|---|---|
| 0 | OK | OK |
| 1 | Error; nothing written | A stage failed; later stages did not run |
| 2 | Missing Gemini API key | Missing API key: `GEMINI_API_KEY` (unless `--skip-llm`), checked before any stage; `ODDS_API_KEY` for the current week (unless `--use-cached-odds`), checked before predicting |
| 3 | Written, but no game got an LLM write-up | Written, but degraded (Section 5, run record) |
| 130 | — | Interrupted |

## 9. Testing and review rules

- `pytest` and `ruff check` must pass before every push.
- Every bug fix lands with a regression test.
- Tests never call live APIs; they use fake clients that raise the SDKs' real error types. Tests that need the real
  cache skip when it is absent.
- Routine runs use cached odds (`--use-cached-odds` / `--no-odds`). The Odds API bills credits per call.
- Contract changes bump `schema_version`, update the validator and this file together, and add validator mutation tests.
- **CI** (`.github/workflows/ci_and_weekly.yml`, job `test`) runs `ruff check`, `ruff format --check` and `pytest` on
  every push to `main` and every pull request.
- **Weekly job** (`weekly_report`) runs `python run_weekly.py --backtest` on Tuesdays 13:00 UTC and Fridays 21:00 UTC,
  or on demand (`workflow_dispatch`, with optional `season`, `week`, `use_cached_odds` and `skip_llm` inputs). It uses
  the `ODDS_API_KEY` and `GEMINI_API_KEY` repository secrets and caches `data/raw` between runs. It commits
  `data/predictions/*.json` and `reports/*.md`, and also commits `last_run.json` after a failed run so the dashboard
  can show it. Exit 3 is a job warning; exit 1 or 2 fails the job.
