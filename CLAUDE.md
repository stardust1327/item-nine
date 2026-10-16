# Claude.md > nfl-prediction-engine  
  
# CLAUDE.md — NFL Prediction Engine & Weekly Scouting Report App  
  
## 1. Project Overview  
A production-grade NFL game prediction engine and automated weekly scouting report web application. The system ingests play-by-play data via `nflreadpy` and live sportsbook lines via The Odds API, engineers leak-free rolling efficiency metrics (EPA, Success Rate, CPOE), trains a calibrated walk-forward XGBoost model, blends model probabilities with de-vigged market baselines, translates predictions into plain-English scouting reports using an LLM with strict Pydantic validation, and serves the output in a Streamlit dashboard.  
  
## 2. Tech Stack & Library Rules  
- **Language:** Python 3.11+  
- **NFL Data Source:** `nflreadpy` (CRITICAL: **NEVER** import or use `nfl_data_py`, which was deprecated in late 2025).  
- **Data Manipulation:** `pandas` (with `pyarrow` backend for Parquet storage) or `polars`.  
- **Machine Learning:** `xgboost`, `scikit-learn` (`CalibratedClassifierCV`, `brier_score_loss`, `log_loss`), `scipy.optimize`, `shap`.  
- **LLM & Validation:** `pydantic>=2.0`, `google-genai` (default: `gemini-2.5-flash`) or `anthropic` (`claude-sonnet-4-20250514` / `claude-opus-4-20250514`).  
- **Frontend:** `streamlit`  
- **Testing:** `pytest`  
  
## 3. Repository Structure  
```text  
nfl-predictor/  
├── CLAUDE.md  
├── config.py                        # Paths, constants, hyperparams, team mappings  
├── requirements.txt  
├── .env.example                     # ODDS_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY  
├── run_weekly.py                    # End-to-end CLI orchestrator  
├── app.py                           # Streamlit frontend dashboard  
├── data/  
│   ├── raw/                         # Cached Parquet files by season & live odds  
│   ├── processed/                   # Leak-free engineered feature matrices  
│   └── predictions/                 # JSON outputs for current week & historical backtests  
├── reports/                         # Exported weekly Markdown reports (week_XX_YYYY.md)  
├── src/  
│   ├── __init__.py  
│   ├── ingest.py                    # nflreadpy & The Odds API fetchers + vig stripping  
│   ├── features.py                  # Garbage-time filter, EWMA, prior regression, differentials  
│   ├── model.py                     # Walk-forward XGBoost, calibration, market ensemble, SHAP  
│   └── report_generator.py          # Pydantic LLM translator & Markdown compiler  
└── tests/  
    └── test_leakage.py              # Automated temporal leakage & probability bounds tests  
