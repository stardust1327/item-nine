"""Win-probability model: walk-forward validation, calibration, market blend, weekly export.

Walk-forward by season (expanding window)
    To evaluate season S, XGBoost is trained on every season from the first one
    through S-2. The sigmoid calibrator and the market weight are fit on S-1, and
    S is scored strictly out of sample. Production predictions for the current
    season use the identical recipe. Rows are never shuffled or randomly split.

Model
    ``XGBClassifier(max_depth=4, learning_rate=0.03, n_estimators=250,
    subsample=0.8, colsample_bytree=0.8)`` is frozen after training and wrapped in
    ``CalibratedClassifierCV(method="sigmoid")``, which is fit on the calibration
    season so probabilities track empirical win rates.

Market-regressed ensemble
    ``final = w * market + (1 - w) * model``. ``w`` in [0, 1] is found with
    ``scipy.optimize.minimize`` (start and fallback 0.60) to minimise Brier score
    on the calibration (validation) season. Calibration and blending share that
    season, so the weight is fit on *cross-fitted* calibrated probabilities:
    calibrate on odd weeks to score even weeks and vice versa. That way it never
    sees probabilities from a calibrator that was fit on the same games.
    Historical market probabilities are de-vigged nflverse closing moneylines.
    The current week uses The Odds API consensus (``src.ingest``), then the latest
    saved snapshot, then nflverse lines.

Contract B
    ``data/predictions/current_week.json``. Its structure is documented in
    :func:`build_payload` and enforced by :func:`validate_contract_b`.

CLI::

    python -m src.model backtest            # walk-forward evaluation report
    python -m src.model predict             # current-week Contract B JSON
    python -m src.model all                 # both (default)
    python -m src.model predict --no-odds   # skip The Odds API, use cached/nflverse lines
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from config import Settings, settings, setup_logging
from src import features as F
from src import ingest

try:  # scikit-learn >= 1.6; `cv="prefit"` was removed in 1.8
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - exercised only on scikit-learn < 1.6
    FrozenEstimator = None

logger = logging.getLogger("src.model" if __name__ == "__main__" else __name__)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
XGB_PARAMS: Final[dict[str, Any]] = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 250,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
XGB_RUNTIME_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}
CALIBRATION_METHOD: Final = "sigmoid"
RANDOM_STATE: Final[int] = 42

DEFAULT_MARKET_WEIGHT: Final[float] = 0.60
MIN_BLEND_GAMES: Final[int] = 100  # fewer validation games with a market line -> keep the default weight

Tier = Literal["LOCK", "VALUE EDGE", "LEAN", "COIN FLIP"]
TIERS: Final[tuple[str, ...]] = ("LOCK", "VALUE EDGE", "LEAN", "COIN FLIP")
LOCK_MIN_WIN_PROB: Final[float] = 0.75  # final win probability of the pick (and the model must favour it too)
VALUE_EDGE_MIN_PCT: Final[float] = 5.0  # model above market on the pick side, in percentage points
LEAN_MIN_WIN_PROB: Final[float] = 0.55
TOP_DRIVERS: Final[int] = 3

# Model-input values for venue flags that are unknown before kickoff. nflverse only
# fills `wind` after the game, and a retractable roof's state is decided on game day,
# so every upcoming game would otherwise reach XGBoost as "missing". That branch was
# learned from historical data gaps (91 of 174 null-wind outdoor games are 2022), not
# from a pre-game state. The same values are applied in training and serving so both
# see one consistent definition: unknown wind -> no high wind (11% base rate), unknown
# roof -> closed (~85% of historical retractable games). Raw nulls remain in
# features.parquet; drivers are reported only for observed values.
MODEL_INPUT_DEFAULTS: Final[dict[str, float]] = {"is_outdoor": 0.0, "is_high_wind_outdoor": 0.0}

CONTRACT_NAME: Final[str] = "B"
CONTRACT_VERSION: Final[str] = "1.1.0"
GAME_STATUSES: Final[tuple[str, ...]] = ("scheduled", "in_progress", "final")
# nflverse_close: line for a game that has kicked off; nflverse_line: pre-game line as last fetched.
MARKET_SOURCES: Final[tuple[str, ...]] = ("odds_api", "odds_api_snapshot", "nflverse_close", "nflverse_line")
MARKET_METHODS: Final[tuple[str, ...]] = ("h2h", "spread_implied", "nflverse_moneyline")

_METRIC_LABELS: Final[dict[str, str]] = {
    "off_pass_epa": "offense EPA per dropback",
    "off_rush_epa": "offense EPA per rush",
    "def_pass_epa": "pass defense EPA allowed",
    "def_rush_epa": "run defense EPA allowed",
    "off_early_down_sr": "early-down success rate",
    "qb_cpoe": "starting QB CPOE",
}
FEATURE_LABELS: Final[dict[str, str]] = {
    **{
        f"{side}_{m}_pre": f"{side.title()} {label}" for side in ("home", "away") for m, label in _METRIC_LABELS.items()
    },
    "net_pass_epa_mismatch": "Net passing EPA mismatch",
    "net_rush_epa_mismatch": "Net rushing EPA mismatch",
    "diff_early_down_sr": "Early-down success rate edge",
    "diff_qb_cpoe": "QB CPOE edge",
    "net_rest_days": "Rest advantage (days)",
    "is_divisional": "Divisional game",
    "is_neutral_site": "Neutral site",
    "is_outdoor": "Outdoor venue",
    "is_high_wind_outdoor": "High wind (15+ mph)",
    "prior_weight": "Early-season prior weight",
}


class ModelError(RuntimeError):
    """The data cannot support the requested training or prediction."""


class ContractError(ValueError):
    """A payload violates Contract B."""


# --------------------------------------------------------------------------- #
# Market probabilities
# --------------------------------------------------------------------------- #
def _american_implied(column: str) -> pl.Expr:
    """Vig-inclusive implied probability of an American price (null if the price is invalid)."""
    price = pl.col(column).cast(pl.Float64)
    return pl.when(price >= 100).then(100.0 / (price + 100.0)).when(price <= -100).then(-price / (-price + 100.0))


def nflverse_market_probs(schedules: pl.DataFrame) -> pl.DataFrame:
    """Multiplicatively de-vigged home win probability from nflverse closing moneylines.

    Same method as ``src.ingest.devig_multiplicative``: ``p_fair = p / (p_home + p_away)``.
    ``market_home_spread`` uses the sportsbook sign (negative = home favoured);
    nflverse's ``spread_line`` is positive when the home team is favoured.
    """
    home, away = _american_implied("home_moneyline"), _american_implied("away_moneyline")
    return schedules.select(
        "game_id",
        (home / (home + away)).alias("market_home_wp"),
        (-pl.col("spread_line").cast(pl.Float64)).alias("market_home_spread"),
    ).with_columns(
        pl.when(pl.col("market_home_wp").is_not_null()).then(pl.lit("nflverse_close")).alias("market_source"),
        pl.when(pl.col("market_home_wp").is_not_null()).then(pl.lit("nflverse_moneyline")).alias("market_method"),
        pl.lit(None, pl.UInt32).alias("market_books"),
        pl.lit(None, pl.Datetime("us", "UTC")).alias("market_as_of"),
    )


# --------------------------------------------------------------------------- #
# Frames and splits
# --------------------------------------------------------------------------- #
def labeled_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Games with a decisive result and at least one team feature (drops ties, unplayed games and empty rows)."""
    has_team_signal = pl.any_horizontal(pl.col(F.TEAM_FEATURE_COLUMNS).is_not_null())
    return frame.filter(pl.col("home_win").is_not_null() & has_team_signal)


def to_matrix(frame: pl.DataFrame, columns: Sequence[str] = F.FEATURE_COLUMNS, *, raw: bool = False) -> np.ndarray:
    """Float64 design matrix for the model.

    Pre-game-unknown venue flags take :data:`MODEL_INPUT_DEFAULTS`, the same in
    training and serving. Any other null becomes NaN, which XGBoost treats as
    missing. ``raw=True`` skips the defaults (used to tell observed values from
    imputed ones).
    """
    exprs = []
    for c in columns:
        expr = pl.col(c).cast(pl.Float64)
        if not raw and c in MODEL_INPUT_DEFAULTS:
            expr = expr.fill_null(MODEL_INPUT_DEFAULTS[c])
        exprs.append(expr)
    return frame.select(exprs).to_numpy()


def _column(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Float64).to_numpy()


@dataclass(frozen=True, slots=True)
class SeasonSplit:
    """Expanding-window fold: fit on ``train_seasons``, calibrate on S-1, evaluate on S."""

    train_seasons: tuple[int, ...]
    calibration_season: int
    test_season: int

    def __post_init__(self) -> None:
        if not self.train_seasons:
            raise ValueError("a split needs at least one training season")
        if self.calibration_season != self.test_season - 1:
            raise ValueError(f"calibration season must be {self.test_season - 1}, got {self.calibration_season}")
        if max(self.train_seasons) > self.test_season - 2:
            raise ValueError(f"training seasons must end by {self.test_season - 2}: {self.train_seasons}")

    def describe(self) -> str:
        first, last = self.train_seasons[0], self.train_seasons[-1]
        train = f"{first}" if first == last else f"{first}-{last}"
        return f"train {train} | calibrate {self.calibration_season} | test {self.test_season}"


def walk_forward_splits(
    seasons: Iterable[int],
    *,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
) -> list[SeasonSplit]:
    """Every season S with a preceding calibration season S-1 and at least one season <= S-2 to train on."""
    available = sorted(set(int(s) for s in seasons))
    splits = []
    for test in available:
        train = tuple(s for s in available if s <= test - 2)
        if not train or (test - 1) not in available:
            continue
        if first_test_season is not None and test < first_test_season:
            continue
        if last_test_season is not None and test > last_test_season:
            continue
        splits.append(SeasonSplit(train, test - 1, test))
    return splits


def production_split(target_season: int, labeled_seasons: Iterable[int]) -> SeasonSplit:
    """The split used to predict ``target_season``: train <= S-2, calibrate on S-1."""
    available = set(int(s) for s in labeled_seasons)
    train = tuple(sorted(s for s in available if s <= target_season - 2))
    if (target_season - 1) not in available or not train:
        raise ModelError(
            f"Predicting {target_season} needs labeled games for {target_season - 1} (calibration) and at least "
            f"one season <= {target_season - 2} (training); have {sorted(available)}"
        )
    return SeasonSplit(train, target_season - 1, target_season)


def _season_rows(frame: pl.DataFrame, seasons: Iterable[int]) -> pl.DataFrame:
    return frame.filter(pl.col("season").is_in(list(seasons)))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _sigmoid_calibrator(estimator: XGBClassifier, X: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    """Fit only the sigmoid map on (X, y); the already-trained estimator stays frozen."""
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(estimator), method=CALIBRATION_METHOD)
    else:  # pragma: no cover - scikit-learn < 1.6
        calibrator = CalibratedClassifierCV(estimator, method=CALIBRATION_METHOD, cv="prefit")
    return calibrator.fit(X, y)


def _require_both_classes(y: np.ndarray, what: str) -> None:
    if np.unique(y).size < 2:
        raise ModelError(f"{what} contains a single class; cannot fit")


class WinProbabilityModel:
    """XGBoost trained on past seasons, sigmoid-calibrated on the following season."""

    def __init__(self, feature_columns: Sequence[str] = F.FEATURE_COLUMNS, *, random_state: int = RANDOM_STATE) -> None:
        self.feature_columns = list(feature_columns)
        self.random_state = random_state
        self.estimator_: XGBClassifier | None = None
        self.calibrated_: CalibratedClassifierCV | None = None

    def build_estimator(self) -> XGBClassifier:
        return XGBClassifier(**XGB_PARAMS, **XGB_RUNTIME_PARAMS, random_state=self.random_state)

    def fit(self, train: pl.DataFrame, calibration: pl.DataFrame) -> WinProbabilityModel:
        X_train, y_train = to_matrix(train, self.feature_columns), _labels(train)
        X_cal, y_cal = to_matrix(calibration, self.feature_columns), _labels(calibration)
        _require_both_classes(y_train, "training data")
        _require_both_classes(y_cal, "calibration data")
        self.estimator_ = self.build_estimator().fit(X_train, y_train)
        self.calibrated_ = _sigmoid_calibrator(self.estimator_, X_cal, y_cal)
        if list(self.calibrated_.classes_) != [0, 1]:
            raise ModelError(f"unexpected classes {self.calibrated_.classes_}")
        return self

    def _check_fitted(self) -> None:
        if self.estimator_ is None or self.calibrated_ is None:
            raise ModelError("model is not fitted")

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        """Calibrated P(home win)."""
        self._check_fitted()
        return self.calibrated_.predict_proba(to_matrix(frame, self.feature_columns))[:, 1]

    def predict_raw(self, frame: pl.DataFrame) -> np.ndarray:
        """Uncalibrated XGBoost P(home win)."""
        self._check_fitted()
        return self.estimator_.predict_proba(to_matrix(frame, self.feature_columns))[:, 1]

    def crossfit_calibrated(self, calibration: pl.DataFrame) -> np.ndarray:
        """Calibrated probabilities for the calibration season, each scored by a calibrator that never saw it.

        Two folds by week parity. Used only to fit the market weight honestly; if a
        fold lacks both classes, falls back to the in-sample calibrated values.
        """
        self._check_fitted()
        X, y = to_matrix(calibration, self.feature_columns), _labels(calibration)
        odd = (calibration["week"].to_numpy() % 2) == 1
        out = np.empty(len(y), dtype=float)
        for score_mask in (odd, ~odd):
            fit_mask = ~score_mask
            if not score_mask.any():
                continue
            if np.unique(y[fit_mask]).size < 2:
                logger.debug("Cross-fit fold lacks both classes; using in-sample calibration")
                return self.calibrated_.predict_proba(X)[:, 1]
            fold = _sigmoid_calibrator(self.estimator_, X[fit_mask], y[fit_mask])
            out[score_mask] = fold.predict_proba(X[score_mask])[:, 1]
        return out

    def contributions(self, frame: pl.DataFrame) -> np.ndarray:
        """Per-feature SHAP contributions to the raw log-odds of a home win (bias column dropped).

        Calibration is a monotone map applied on top, so signs and rankings carry over.
        """
        self._check_fitted()
        dmatrix = xgb.DMatrix(to_matrix(frame, self.feature_columns))
        return self.estimator_.get_booster().predict(dmatrix, pred_contribs=True)[:, :-1]


def _labels(frame: pl.DataFrame) -> np.ndarray:
    labels = frame["home_win"]
    if labels.null_count():
        raise ModelError("labels contain nulls; pass labeled_rows(frame)")
    return labels.cast(pl.Int64).to_numpy()


# --------------------------------------------------------------------------- #
# Metrics and blending
# --------------------------------------------------------------------------- #
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean |predicted - observed| over equal-width probability bins."""
    bins = np.clip(np.digitize(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        total += in_bin.sum() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(total / len(y))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    """Brier, log loss, accuracy, AUC and ECE for P(home win)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"n_games": 0, "brier": None, "log_loss": None, "accuracy": None, "auc": None, "ece": None}
    return {
        "n_games": len(y),
        "brier": float(brier_score_loss(y, p, pos_label=1)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "ece": expected_calibration_error(y, p),
    }


def blend_probabilities(model_p: np.ndarray, market_p: np.ndarray, market_weight: float) -> np.ndarray:
    """``w * market + (1 - w) * model``; games without a market line keep the model probability."""
    model_p = np.asarray(model_p, dtype=float)
    market_p = np.asarray(market_p, dtype=float)
    blended = market_weight * market_p + (1.0 - market_weight) * model_p
    return np.where(np.isfinite(market_p), blended, model_p)


def compare_probabilities(
    y: np.ndarray, model_p: np.ndarray, market_p: np.ndarray, final_p: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    """Model, market and final metrics on the same games (those with a market line), plus all-games variants."""
    has_market = np.isfinite(market_p)
    return {
        "model": probability_metrics(y[has_market], model_p[has_market]),
        "market": probability_metrics(y[has_market], market_p[has_market]),
        "final": probability_metrics(y[has_market], final_p[has_market]),
        "model_all_games": probability_metrics(y, model_p),
        "final_all_games": probability_metrics(y, final_p),
    }


@dataclass(frozen=True, slots=True)
class MarketBlend:
    weight: float
    source: Literal["optimized", "default"]
    n_games: int
    brier_model: float | None
    brier_market: float | None
    brier_final: float | None


def fit_market_weight(
    model_p: np.ndarray,
    market_p: np.ndarray,
    y: np.ndarray,
    *,
    default: float = DEFAULT_MARKET_WEIGHT,
    min_games: int = MIN_BLEND_GAMES,
) -> MarketBlend:
    """Market weight in [0, 1] minimising Brier score of the blend on validation games.

    Uses ``scipy.optimize.minimize`` (L-BFGS-B, bounded, started at ``default``).
    Falls back to ``default`` when fewer than ``min_games`` games have a market
    line or the optimiser fails.
    """
    model_p, market_p, y = (np.asarray(a, dtype=float) for a in (model_p, market_p, y))
    usable = np.isfinite(model_p) & np.isfinite(market_p) & np.isfinite(y)
    n = int(usable.sum())
    if n == 0:
        logger.warning("No validation games with a market line; market weight defaults to %.2f", default)
        return MarketBlend(default, "default", 0, None, None, None)
    q, m, t = model_p[usable], market_p[usable], y[usable]

    def brier(weight: float) -> float:
        return float(np.mean((weight * m + (1.0 - weight) * q - t) ** 2))

    if n < min_games:
        logger.warning(
            "Only %d validation games with a market line (< %d); market weight defaults to %.2f", n, min_games, default
        )
        return MarketBlend(default, "default", n, brier(0.0), brier(1.0), brier(default))

    result = minimize(lambda w: brier(float(w[0])), x0=np.array([default]), bounds=[(0.0, 1.0)], method="L-BFGS-B")
    weight = float(np.clip(result.x[0], 0.0, 1.0))
    if not result.success or not math.isfinite(weight):
        logger.warning("Market-weight optimisation failed (%s); using %.2f", result.message, default)
        weight, source = default, "default"
    else:
        source = "optimized"
    return MarketBlend(weight, source, n, brier(0.0), brier(1.0), brier(weight))


# --------------------------------------------------------------------------- #
# Training recipe shared by backtest and production
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FittedFold:
    split: SeasonSplit
    model: WinProbabilityModel
    blend: MarketBlend
    validation: dict[str, dict[str, float | int | None]]


def fit_fold(data: pl.DataFrame, split: SeasonSplit, *, random_state: int = RANDOM_STATE) -> FittedFold:
    """Train on ``split.train_seasons``, calibrate and fit the market weight on ``split.calibration_season``."""
    train = _season_rows(data, split.train_seasons)
    calibration = _season_rows(data, [split.calibration_season])
    if calibration.is_empty():
        raise ModelError(f"no labeled games in calibration season {split.calibration_season}")
    model = WinProbabilityModel(random_state=random_state).fit(train, calibration)

    crossfit = model.crossfit_calibrated(calibration)
    market = _column(calibration, "market_home_wp")
    y = _labels(calibration)
    blend = fit_market_weight(crossfit, market, y)
    # In-sample for the market weight (fit on these games); the calibration step is cross-fitted.
    validation = compare_probabilities(y, crossfit, market, blend_probabilities(crossfit, market, blend.weight))
    logger.info(
        "%s: %d train / %d calibration games, market weight %.3f (%s)",
        split.describe(),
        train.height,
        calibration.height,
        blend.weight,
        blend.source,
    )
    return FittedFold(split, model, blend, validation)


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BacktestResult:
    predictions: pl.DataFrame
    report: dict[str, Any]


def walk_forward_backtest(
    frame: pl.DataFrame,
    *,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    random_state: int = RANDOM_STATE,
) -> BacktestResult:
    """Score every eligible season out of sample with the production recipe.

    ``frame`` is the feature matrix joined with ``market_home_wp``. ``model``,
    ``market`` and ``final`` are scored on the same games (those with a market
    line); ``*_all_games`` also cover games without one.
    """
    data = labeled_rows(frame)
    splits = walk_forward_splits(
        data["season"].unique().to_list(), first_test_season=first_test_season, last_test_season=last_test_season
    )
    if not splits:
        raise ModelError(f"not enough seasons for a walk-forward split: {sorted(data['season'].unique().to_list())}")

    predictions: list[pl.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for split in splits:
        fold = fit_fold(data, split, random_state=random_state)
        test = _season_rows(data, [split.test_season])
        model_p = fold.model.predict_proba(test)
        market_p = _column(test, "market_home_wp")
        final_p = blend_probabilities(model_p, market_p, fold.blend.weight)
        y = _labels(test)
        metrics = compare_probabilities(y, model_p, market_p, final_p)
        folds.append(
            {
                "test_season": split.test_season,
                "calibration_season": split.calibration_season,
                "train_seasons": list(split.train_seasons),
                "market_weight": fold.blend.weight,
                "market_weight_source": fold.blend.source,
                "validation": fold.validation,
                "test": metrics,
            }
        )
        logger.info(
            "Season %d out of sample: Brier model %.4f | market %.4f | final %.4f (n=%d)",
            split.test_season,
            metrics["model"]["brier"] or float("nan"),
            metrics["market"]["brier"] or float("nan"),
            metrics["final"]["brier"] or float("nan"),
            len(y),
        )
        predictions.append(
            test.select("game_id", "season", "week", "game_type", "home_team", "away_team", "home_win").with_columns(
                pl.Series("model_home_wp", model_p),
                pl.Series("market_home_wp", market_p).fill_nan(None),
                pl.Series("final_home_wp", final_p),
                pl.lit(fold.blend.weight).alias("market_weight"),
            )
        )

    oos = pl.concat(predictions)
    y = oos["home_win"].cast(pl.Int64).to_numpy()
    model_p, final_p = oos["model_home_wp"].to_numpy(), oos["final_home_wp"].to_numpy()
    market_p = oos["market_home_wp"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    report = {
        "generated_at": _iso(datetime.now(UTC)),
        "algorithm": _algorithm_description(),
        "hyperparameters": dict(XGB_PARAMS),
        "test_seasons": [s.test_season for s in splits],
        "folds": folds,
        "pooled": compare_probabilities(y, model_p, market_p, final_p),
    }
    return BacktestResult(oos, report)


# --------------------------------------------------------------------------- #
# Current-week market lines
# --------------------------------------------------------------------------- #
_MARKET_COLUMNS: Final[dict[str, Any]] = {
    "game_id": pl.String,
    "market_home_wp": pl.Float64,
    "market_home_spread": pl.Float64,
    "market_source": pl.String,
    "market_method": pl.String,
    "market_books": pl.UInt32,
    "market_as_of": pl.Datetime("us", "UTC"),
}


def _normalize_market(df: pl.DataFrame) -> pl.DataFrame:
    """Select/cast :data:`_MARKET_COLUMNS`; absent optional columns become null."""
    return df.select(
        (pl.col(name) if name in df.columns else pl.lit(None)).cast(dtype).alias(name)
        for name, dtype in _MARKET_COLUMNS.items()
    )


def _odds_consensus_to_market(consensus: pl.DataFrame, source: str) -> pl.DataFrame:
    """Odds API consensus rows -> market columns. Spread-only games report the spread books and method."""
    if "wp_source" not in consensus.columns:  # frames written before spread-only pricing was tracked
        consensus = consensus.with_columns(
            pl.lit("h2h").alias("wp_source"), pl.lit(None, pl.UInt32).alias("n_books_spread")
        )
    spread_only = pl.col("wp_source") == "spread"
    return _normalize_market(
        consensus.filter(pl.col("game_id").is_not_null() & pl.col("market_home_wp").is_not_null()).select(
            "game_id",
            "market_home_wp",
            pl.col("consensus_home_spread").alias("market_home_spread"),
            pl.lit(source).alias("market_source"),
            pl.when(spread_only).then(pl.lit("spread_implied")).otherwise(pl.lit("h2h")).alias("market_method"),
            pl.when(spread_only).then(pl.col("n_books_spread")).otherwise(pl.col("n_books_h2h")).alias("market_books"),
            pl.col("fetched_at").alias("market_as_of"),
        )
    )


def _nflverse_current_lines(
    schedules: pl.DataFrame, week_games: pl.DataFrame, fetched_at: datetime | None, now: datetime
) -> pl.DataFrame:
    """nflverse moneylines for the week, labelled by when the cached line was *observed*.

    A line counts as a close (``nflverse_close``) only if the schedules cache was
    fetched after kickoff. Otherwise it is a pre-game line (``nflverse_line``),
    stamped with that fetch time, even if the game has kicked off since.
    ``fetched_at`` must describe the ``schedules`` frame passed in; it is clamped
    to ``now`` so a line is never dated after the prediction.
    """
    observed = None if fetched_at is None else min(_as_utc(fetched_at), now)
    kickoffs = week_games.select("game_id", "kickoff_utc")
    closed = pl.lit(False) if observed is None else pl.col("kickoff_utc") <= pl.lit(observed, pl.Datetime("us", "UTC"))
    return (
        _normalize_market(nflverse_market_probs(schedules))
        .join(kickoffs, on="game_id", how="inner")
        .with_columns(
            pl.when(pl.col("market_home_wp").is_null())
            .then(None)
            .when(closed)
            .then(pl.lit("nflverse_close"))
            .otherwise(pl.lit("nflverse_line"))
            .alias("market_source"),
            pl.when(closed).then(None).otherwise(pl.lit(observed, pl.Datetime("us", "UTC"))).alias("market_as_of"),
        )
        .drop("kickoff_utc")
    )


def schedules_fetched_at(cfg: Settings) -> datetime | None:
    """When the cached schedules (and their moneylines) were downloaded, per the cache manifest."""
    entry = ingest.CacheManifest(cfg.manifest_path).get("schedules", None)
    return entry.fetched_at if entry is not None else None


def add_kickoff_if_missing(games: pl.DataFrame) -> pl.DataFrame:
    return games if "kickoff_utc" in games.columns else ingest.add_kickoff_utc(games)


def latest_odds_snapshot(cfg: Settings, season: int, week: int, now: datetime | None = None) -> pl.DataFrame | None:
    """Most recent consensus snapshot saved by ``src.ingest`` for this season/week, as of ``now``.

    Snapshots stamped after ``now`` are ignored, so a replayed or backfilled
    prediction never sees market prices from its future.
    """
    cutoff = _as_utc(now) if now is not None else None
    usable: list[tuple[datetime, Path]] = []
    for path in (cfg.odds_dir / str(season)).glob(f"odds_{season}_wk{week:02d}_*_consensus.parquet"):
        stamp = path.stem.removesuffix("_consensus").rsplit("_", 1)[-1]
        try:
            taken = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            logger.warning("Ignoring odds snapshot with an unrecognised name: %s", path.name)
            continue
        if cutoff is None or taken <= cutoff:
            usable.append((taken, path))
    if not usable:
        return None
    taken, path = max(usable)
    logger.debug("Latest saved odds snapshot as of %s: %s", _iso(cutoff), path.name)
    return pl.read_parquet(path)


def current_market_probs(
    week_games: pl.DataFrame,
    schedules: pl.DataFrame,
    cfg: Settings = settings,
    *,
    use_odds_api: bool = True,
    now: datetime | None = None,
    schedules_fetched_at_override: datetime | None = None,
) -> pl.DataFrame:
    """Best available de-vigged market probability per current-week game.

    Priority: live Odds API consensus, then the latest saved Odds API snapshot for
    the week, then nflverse moneylines (``nflverse_line`` before kickoff,
    ``nflverse_close`` after). Games that already kicked off are absent from the
    Odds API, so they fall back to nflverse closing lines.
    """
    now = _as_utc(now or datetime.now(UTC))
    # Read before the live fetch: fetch_market_win_probs may refresh the schedules cache, and the
    # moneylines used below come from the `schedules` frame passed in, not from that refresh.
    lines_fetched_at = (
        schedules_fetched_at(cfg) if schedules_fetched_at_override is None else schedules_fetched_at_override
    )
    season, week = int(week_games["season"][0]), int(week_games["week"][0])
    ids = week_games["game_id"].implode()
    candidates: list[pl.DataFrame] = []
    if use_odds_api:
        try:
            candidates.append(_odds_consensus_to_market(ingest.fetch_market_win_probs(cfg=cfg, now=now), "odds_api"))
        except ingest.MissingAPIKeyError:
            logger.info("ODDS_API_KEY is not set; using saved snapshots / nflverse lines")
        except ingest.IngestError as exc:
            logger.warning("Live odds unavailable (%s); using saved snapshots / nflverse lines", exc)
    snapshot = latest_odds_snapshot(cfg, season, week, now)
    if snapshot is not None:
        candidates.append(_odds_consensus_to_market(snapshot, "odds_api_snapshot"))
    candidates.append(_nflverse_current_lines(schedules, add_kickoff_if_missing(week_games), lines_fetched_at, now))

    ranked = pl.concat([c.with_columns(pl.lit(rank).alias("_rank")) for rank, c in enumerate(candidates)])
    return (
        ranked.filter(pl.col("game_id").is_in(ids) & pl.col("market_home_wp").is_not_null())
        .sort("_rank", "game_id")
        .group_by("game_id", maintain_order=True)
        .first()
        .drop("_rank")
    )


# --------------------------------------------------------------------------- #
# Contract B payload
# --------------------------------------------------------------------------- #
def _as_utc(ts: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons with tz-aware kickoffs never fail."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _probability_pair(home_wp: float) -> dict[str, float]:
    home = round(float(home_wp), 4)
    return {"home_wp": home, "away_wp": round(1.0 - home, 4)}


def _algorithm_description() -> str:
    return f'XGBClassifier + CalibratedClassifierCV(method="{CALIBRATION_METHOD}")'


def confidence_tier(pick_win_prob: float, model_edge_pct: float | None, model_pick_win_prob: float) -> Tier:
    """Map a pick to a tier (first match wins).

    LOCK        the pick's final win probability is >= 75% and the model also favours the pick
    VALUE EDGE  the model rates the pick >= 5 percentage points above the market (``model_edge_pct >= +5``)
    LEAN        the pick's final win probability is >= 55%
    COIN FLIP   everything else

    The edge is directional. When the model rates the *other* side above the
    market, ``value_team`` names that side, and the disagreement never upgrades
    the pick's tier.
    """
    if pick_win_prob >= LOCK_MIN_WIN_PROB and model_pick_win_prob >= 0.5:
        return "LOCK"
    if model_edge_pct is not None and model_edge_pct >= VALUE_EDGE_MIN_PCT:
        return "VALUE EDGE"
    if pick_win_prob >= LEAN_MIN_WIN_PROB:
        return "LEAN"
    return "COIN FLIP"


def top_drivers(
    contributions: np.ndarray,
    raw_values: np.ndarray,
    home_team: str,
    away_team: str,
    *,
    feature_columns: Sequence[str] = F.FEATURE_COLUMNS,
    k: int = TOP_DRIVERS,
) -> list[dict[str, Any]]:
    """The ``k`` observed features with the largest absolute contribution to this game's raw log-odds.

    Features whose input was not observed (``raw_values`` NaN, e.g. wind before
    kickoff) are skipped: an imputed or missing input is not a statistical driver.
    """
    drivers: list[dict[str, Any]] = []
    for i in np.argsort(-np.abs(contributions), kind="stable"):
        if len(drivers) == k:
            break
        impact = float(contributions[i])
        if impact == 0.0:
            break
        if not math.isfinite(float(raw_values[i])):
            continue
        column = feature_columns[i]
        drivers.append(
            {
                "feature": column,
                "label": FEATURE_LABELS.get(column, column),
                "value": round(float(raw_values[i]), 4),
                "impact_logit": round(impact, 4),
                "favors": home_team if impact > 0 else away_team,
            }
        )
    return drivers


def _game_status(row: dict[str, Any], now: datetime) -> str:
    if row["result"] is not None:
        return "final"
    return "in_progress" if row["kickoff_utc"] <= now else "scheduled"


def _flag(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _decision_fields(
    model_pair: dict[str, float], market_pair: dict[str, float] | None, final_pair: dict[str, float]
) -> dict[str, Any]:
    """Pick, edge, value side and tier, all derived from the *published* (rounded) probabilities.

    :func:`validate_contract_b` re-derives these from the JSON with this same function.
    """
    side = "home" if final_pair["home_wp"] >= 0.5 else "away"
    pick_wp = final_pair[f"{side}_wp"]
    if market_pair is None:
        edge_pick = edge_home = None
    else:
        edge_home = round(100.0 * (model_pair["home_wp"] - market_pair["home_wp"]), 2)
        edge_pick = round(100.0 * (model_pair[f"{side}_wp"] - market_pair[f"{side}_wp"]), 2)
    value_side = None if not edge_home else ("home" if edge_home > 0 else "away")
    return {
        "side": side,
        "pick_wp": pick_wp,
        "model_edge_pct": edge_pick,
        "value_side": value_side,
        "value_edge_pct": None if value_side is None else abs(edge_home),
        "tier": confidence_tier(pick_wp, edge_pick, model_pair[f"{side}_wp"]),
    }


def build_payload(
    week_rows: pl.DataFrame,
    fold: FittedFold,
    *,
    season: int,
    week: int,
    now: datetime,
) -> dict[str, Any]:
    """Assemble the Contract B payload (``schema_version`` :data:`CONTRACT_VERSION`).

    ``week_rows`` is the feature matrix for the week's games, left-joined with the
    current market columns (:data:`_MARKET_COLUMNS`). Per game:

    * ``market`` / ``model`` / ``final``: home and away win probabilities (``market`` is null without a line;
      ``market.method`` says how it was priced);
    * ``pick``: the side ``final`` favours, with its final win probability;
    * ``model_edge_pct``: ``100 * (model - market)`` on the pick side, in percentage points (negative means
      the market rates the pick higher than the model does);
    * ``value_team`` / ``value_edge_pct``: the side the model rates above the market, and by how much;
    * ``confidence_tier``: :func:`confidence_tier` on the published values;
    * ``top_drivers``: :func:`top_drivers` (observed inputs only), explained against the uncalibrated log-odds;
    * ``imputed_inputs``: venue inputs unknown before kickoff that took :data:`MODEL_INPUT_DEFAULTS`.
    """
    now = _as_utc(now)
    rows = week_rows.sort("kickoff_utc", "game_id")
    model_p = fold.model.predict_proba(rows)
    market_p = _column(rows, "market_home_wp")
    final_p = blend_probabilities(model_p, market_p, fold.blend.weight)
    contributions = fold.model.contributions(rows)
    raw_values = to_matrix(rows, raw=True)
    column_index = {c: j for j, c in enumerate(F.FEATURE_COLUMNS)}

    games = []
    for i, row in enumerate(rows.iter_rows(named=True)):
        home, away = row["home_team"], row["away_team"]
        has_market = bool(np.isfinite(market_p[i]))
        model_pair = _probability_pair(model_p[i])
        market_pair = _probability_pair(market_p[i]) if has_market else None
        final_pair = _probability_pair(final_p[i])
        decision = _decision_fields(model_pair, market_pair, final_pair)
        side = decision["side"]
        status = _game_status(row, now)
        games.append(
            {
                "game_id": row["game_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "game_type": row["game_type"],
                "kickoff_utc": _iso(row["kickoff_utc"]),
                "status": status,
                "home_team": home,
                "away_team": away,
                "venue": {
                    "is_neutral_site": _flag(row["is_neutral_site"]),
                    "is_outdoor": _flag(row["is_outdoor"]),
                    "is_high_wind": _flag(row["is_high_wind_outdoor"]),
                    "is_divisional": _flag(row["is_divisional"]),
                },
                "net_rest_days": row["net_rest_days"],
                "market": (
                    {
                        **market_pair,
                        "home_spread": _round(row["market_home_spread"], 2),
                        "source": row["market_source"],
                        "method": row["market_method"],
                        "books": row["market_books"],
                        "as_of": _iso(row["market_as_of"]),
                    }
                    if market_pair is not None
                    else None
                ),
                "model": model_pair,
                "final": final_pair,
                "pick": {"team": home if side == "home" else away, "side": side, "win_prob": decision["pick_wp"]},
                "model_edge_pct": decision["model_edge_pct"],
                "value_team": None if decision["value_side"] is None else row[f"{decision['value_side']}_team"],
                "value_edge_pct": decision["value_edge_pct"],
                "confidence_tier": decision["tier"],
                "top_drivers": top_drivers(contributions[i], raw_values[i], home, away),
                "imputed_inputs": [
                    c for c in MODEL_INPUT_DEFAULTS if not math.isfinite(raw_values[i][column_index[c]])
                ],
                "result": (
                    {"home_score": row["home_score"], "away_score": row["away_score"]} if status == "final" else None
                ),
            }
        )

    split, blend = fold.split, fold.blend
    return {
        "contract": CONTRACT_NAME,
        "schema_version": CONTRACT_VERSION,
        "generated_at": _iso(now),
        "season": season,
        "week": week,
        "model": {
            "algorithm": _algorithm_description(),
            "hyperparameters": dict(XGB_PARAMS),
            "train_seasons": list(split.train_seasons),
            "calibration_season": split.calibration_season,
            "market_weight": round(blend.weight, 4),
            "market_weight_source": blend.source,
            "weight_fit": {
                "season": split.calibration_season,
                "in_sample": True,
                "note": "market weight fit on these games; model probabilities cross-fitted (odd/even weeks)",
                "brier_model": _round(blend.brier_model),
                "brier_market": _round(blend.brier_market),
                "brier_final": _round(blend.brier_final),
                "n_games": blend.n_games,
            },
            "input_defaults": dict(MODEL_INPUT_DEFAULTS),
        },
        "tier_rules": {
            "LOCK": f"pick win probability >= {LOCK_MIN_WIN_PROB:.0%} and the model also favours the pick",
            "VALUE EDGE": f"model rates the pick >= {VALUE_EDGE_MIN_PCT:g} percentage points above the market",
            "LEAN": f"pick win probability >= {LEAN_MIN_WIN_PROB:.0%}",
            "COIN FLIP": "otherwise",
        },
        "games": games,
    }


_GAME_KEYS: Final[tuple[str, ...]] = (
    "game_id", "season", "week", "game_type", "kickoff_utc", "status", "home_team", "away_team", "venue",
    "net_rest_days", "market", "model", "final", "pick", "model_edge_pct", "value_team", "value_edge_pct",
    "confidence_tier", "top_drivers", "imputed_inputs", "result",
)  # fmt: skip
_VENUE_KEYS: Final[tuple[str, ...]] = ("is_neutral_site", "is_outdoor", "is_high_wind", "is_divisional")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_prob(x: Any) -> bool:
    return _is_num(x) and 0.0 <= x <= 1.0


def _parse_iso(x: Any) -> datetime | None:
    if not isinstance(x, str):
        return None
    try:
        ts = datetime.fromisoformat(x.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _close(a: Any, b: Any, tol: float) -> bool:
    return _is_num(a) and _is_num(b) and abs(a - b) <= tol


def validate_contract_b(payload: Any) -> None:
    """Raise :class:`ContractError` listing every violation of Contract B.

    Checks types (booleans are not numbers), ranges, and cross-field consistency:
    final = blend(model, market) at ``model.market_weight``; pick, edge,
    value_team and tier re-derived from the published probabilities; driver signs.
    It also checks temporal consistency: walk-forward metadata, game season/week,
    status vs kickoff, and market timestamps vs ``generated_at``. Malformed
    structures are reported, never raised as other exception types.
    """
    if not isinstance(payload, dict):
        raise ContractError("Contract B violations:\n  - payload must be a JSON object")
    problems: list[str] = []
    try:
        _check_contract_b(payload, problems)
    except Exception as exc:  # noqa: BLE001 - a malformed payload must surface as a ContractError
        problems.append(f"malformed payload ({type(exc).__name__}: {exc})")
    if problems:
        raise ContractError("Contract B violations:\n  - " + "\n  - ".join(problems))


def _check_contract_b(payload: dict[str, Any], problems: list[str]) -> None:
    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def check_pair(pair: Any, where: str) -> bool:
        if not isinstance(pair, dict) or not _is_prob(pair.get("home_wp")) or not _is_prob(pair.get("away_wp")):
            problems.append(f"{where}: home_wp/away_wp must be probabilities")
            return False
        if abs(pair["home_wp"] + pair["away_wp"] - 1.0) > 1e-3:
            problems.append(f"{where}: home_wp + away_wp must equal 1")
            return False
        return True

    check(payload.get("contract") == CONTRACT_NAME, f"contract must be {CONTRACT_NAME!r}")
    check(isinstance(payload.get("schema_version"), str), "schema_version must be a string")
    generated_at = _parse_iso(payload.get("generated_at"))
    check(generated_at is not None, "generated_at must be ISO-8601")
    season, week = payload.get("season"), payload.get("week")
    check(_is_int(season), "season must be an int")
    check(_is_int(week), "week must be an int")

    weight = None
    model = payload.get("model")
    if isinstance(model, dict):
        weight = model.get("market_weight")
        check(_is_prob(weight), "model.market_weight must be in [0, 1]")
        check(model.get("market_weight_source") in ("optimized", "default"), "model.market_weight_source invalid")
        train = model.get("train_seasons")
        calibration = model.get("calibration_season")
        if isinstance(train, list) and train and all(_is_int(t) for t in train) and _is_int(season):
            check(max(train) <= season - 2, "model.train_seasons must end by season - 2 (walk-forward)")
        else:
            problems.append("model.train_seasons must be a non-empty list of ints")
        check(
            _is_int(calibration) and _is_int(season) and calibration == season - 1,
            "model.calibration_season must be season - 1",
        )
    else:
        problems.append("model must be an object")

    tier_rules = payload.get("tier_rules")
    check(
        isinstance(tier_rules, dict)
        and set(tier_rules) == set(TIERS)
        and all(isinstance(v, str) and v for v in tier_rules.values()),
        f"tier_rules must map each of {TIERS} to a description",
    )

    games = payload.get("games")
    if not isinstance(games, list):
        problems.append("games must be a list")
        return
    ids = [g.get("game_id") for g in games if isinstance(g, dict)]
    check(len(ids) == len(set(ids)), "game_id values must be unique")

    for n, game in enumerate(games):
        where = f"games[{n}]"
        if not isinstance(game, dict):
            problems.append(f"{where} must be an object")
            continue
        missing = [key for key in _GAME_KEYS if key not in game]
        if missing:
            problems.append(f"{where} missing {missing}")
            continue
        where = f"games[{n}] ({game['game_id']})"
        home, away = game["home_team"], game["away_team"]
        check(
            isinstance(home, str) and isinstance(away, str) and home and away and home != away,
            f"{where}: teams invalid",
        )
        check(game["season"] == season and _is_int(game["season"]), f"{where}: season must equal the payload season")
        check(game["week"] == week and _is_int(game["week"]), f"{where}: week must equal the payload week")
        kickoff = _parse_iso(game["kickoff_utc"])
        check(kickoff is not None, f"{where}: kickoff_utc must be ISO-8601")
        status = game["status"]
        check(status in GAME_STATUSES, f"{where}: status must be one of {GAME_STATUSES}")
        if kickoff is not None and generated_at is not None and status in GAME_STATUSES:
            check((status == "scheduled") == (kickoff > generated_at), f"{where}: status inconsistent with kickoff_utc")
        venue = game["venue"]
        venue_ok = isinstance(venue, dict) and all(
            k in venue and (venue[k] is None or isinstance(venue[k], bool)) for k in _VENUE_KEYS
        )
        check(venue_ok, f"{where}: venue must hold bool-or-null {_VENUE_KEYS}")
        check(game["net_rest_days"] is None or _is_int(game["net_rest_days"]), f"{where}: net_rest_days must be an int")

        pairs_ok = check_pair(game["model"], f"{where}.model") & check_pair(game["final"], f"{where}.final")
        market = game["market"]
        if market is not None:
            if not isinstance(market, dict):
                problems.append(f"{where}: market must be an object or null")
                continue
            pairs_ok &= check_pair(market, f"{where}.market")
            check(market.get("source") in MARKET_SOURCES, f"{where}: market.source must be one of {MARKET_SOURCES}")
            check(
                market.get("method") in (*MARKET_METHODS, None),
                f"{where}: market.method must be one of {MARKET_METHODS}",
            )
            check(
                market.get("books") is None or (_is_int(market["books"]) and market["books"] >= 0),
                f"{where}: market.books invalid",
            )
            check(
                market.get("home_spread") is None or _is_num(market["home_spread"]),
                f"{where}: market.home_spread invalid",
            )
            as_of = market.get("as_of")
            if as_of is not None:
                as_of_dt = _parse_iso(as_of)
                check(
                    as_of_dt is not None and (generated_at is None or as_of_dt <= generated_at),
                    f"{where}: market.as_of invalid or after generated_at",
                )
        if not pairs_ok:
            continue

        # Cross-field: blend, then every decision field re-derived from the published probabilities.
        if market is not None and _is_prob(weight):
            blended = weight * market["home_wp"] + (1 - weight) * game["model"]["home_wp"]
            check(
                abs(game["final"]["home_wp"] - blended) <= 2e-4,
                f"{where}: final != market_weight blend of market and model",
            )
        elif market is None:
            check(game["final"] == game["model"], f"{where}: final must equal model when there is no market line")
        expected = _decision_fields(game["model"], market, game["final"])
        pick = game["pick"]
        if not isinstance(pick, dict):
            problems.append(f"{where}: pick must be an object")
            continue
        check(pick.get("side") == expected["side"], f"{where}: pick.side must be the side final favours")
        check(pick.get("team") == game.get(f"{expected['side']}_team"), f"{where}: pick.team does not match pick.side")
        check(
            _close(pick.get("win_prob"), expected["pick_wp"], 1e-9),
            f"{where}: pick.win_prob must equal the final probability of the pick",
        )
        edge = game["model_edge_pct"]
        if expected["model_edge_pct"] is None:
            check(edge is None, f"{where}: model_edge_pct must be null without a market line")
        else:
            check(
                _close(edge, expected["model_edge_pct"], 0.011),
                f"{where}: model_edge_pct must equal 100*(model - market) on the pick side",
            )
        value_team = None if expected["value_side"] is None else game[f"{expected['value_side']}_team"]
        check(
            game["value_team"] == value_team, f"{where}: value_team must be the side the model rates above the market"
        )
        if expected["value_edge_pct"] is None:
            check(game["value_edge_pct"] is None, f"{where}: value_edge_pct must be null without a value side")
        else:
            check(
                _close(game["value_edge_pct"], expected["value_edge_pct"], 0.011),
                f"{where}: value_edge_pct inconsistent",
            )
        check(game["confidence_tier"] in TIERS, f"{where}: confidence_tier must be one of {TIERS}")
        if _is_prob(pick.get("win_prob")) and (edge is None or _is_num(edge)):
            tier = confidence_tier(pick["win_prob"], edge, game["model"][f"{expected['side']}_wp"])
            check(
                game["confidence_tier"] == tier,
                f"{where}: confidence_tier {game['confidence_tier']!r} contradicts tier_rules (expected {tier!r})",
            )

        drivers = game["top_drivers"]
        if isinstance(drivers, list) and len(drivers) <= TOP_DRIVERS:
            impacts = []
            for d in drivers:
                ok = (
                    isinstance(d, dict)
                    and d.get("feature") in F.FEATURE_COLUMNS
                    and isinstance(d.get("label"), str)
                    and _is_num(d.get("value"))
                    and _is_num(d.get("impact_logit"))
                    and d["impact_logit"] != 0
                    and d.get("favors") == (home if d["impact_logit"] > 0 else away)
                )
                check(ok, f"{where}: malformed driver {d}")
                if ok:
                    impacts.append(abs(d["impact_logit"]))
            check(impacts == sorted(impacts, reverse=True), f"{where}: top_drivers must be ordered by |impact_logit|")
        else:
            problems.append(f"{where}: top_drivers must be a list of at most {TOP_DRIVERS}")
        imputed = game["imputed_inputs"]
        check(
            isinstance(imputed, list) and set(imputed) <= set(MODEL_INPUT_DEFAULTS), f"{where}: imputed_inputs invalid"
        )
        result = game["result"]
        if status == "final":
            check(
                isinstance(result, dict) and _is_int(result.get("home_score")) and _is_int(result.get("away_score")),
                f"{where}: final games need integer home_score/away_score",
            )
        else:
            check(result is None, f"{where}: result must be null unless status is final")


# --------------------------------------------------------------------------- #
# Production: predict the current week
# --------------------------------------------------------------------------- #
def predict_week(
    features: pl.DataFrame,
    schedules: pl.DataFrame,
    week_market: pl.DataFrame,
    *,
    season: int,
    week: int,
    now: datetime,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train the production fold for ``season`` and return the validated Contract B payload for ``week``."""
    now = _as_utc(now)
    history = features.join(
        nflverse_market_probs(schedules).select("game_id", "market_home_wp"), on="game_id", how="left"
    )
    data = labeled_rows(history.filter(pl.col("season") < season))
    split = production_split(season, data["season"].unique().to_list())
    fold = fit_fold(data, split, random_state=random_state)

    week_rows = features.filter((pl.col("season") == season) & (pl.col("week") == week)).join(
        _normalize_market(week_market), on="game_id", how="left"
    )
    if week_rows.is_empty():
        raise ModelError(f"no feature rows for {season} week {week}; rebuild features")
    _warn_on_serving_missingness(_season_rows(data, split.train_seasons), week_rows)
    payload = build_payload(week_rows, fold, season=season, week=week, now=now)
    validate_contract_b(payload)
    return payload


def _warn_on_serving_missingness(train: pl.DataFrame, serving: pl.DataFrame, *, rare: float = 0.01) -> None:
    """Flag serving rows with a feature missing that training almost never saw missing.

    XGBoost would route such rows down a missing-value branch fitted on a handful of
    unrepresentative games. Venue flags in :data:`MODEL_INPUT_DEFAULTS` are reported
    at INFO level, since they are defaulted on purpose.
    """
    for column in F.FEATURE_COLUMNS:
        serving_nulls = serving[column].null_count()
        if not serving_nulls:
            continue
        if column in MODEL_INPUT_DEFAULTS:
            logger.info(
                "%d game(s) have unknown %s before kickoff; using %g",
                serving_nulls,
                column,
                MODEL_INPUT_DEFAULTS[column],
            )
            continue
        train_rate = train[column].null_count() / max(train.height, 1)
        if train_rate < rare:
            logger.warning(
                "%d game(s) are missing %s, which was missing in only %.2f%% of training rows; their probabilities "
                "rely on a sparsely trained branch",
                serving_nulls,
                column,
                100 * train_rate,
            )


def write_payload(payload: dict[str, Any], path: Path) -> Path:
    ingest.write_text_atomic(json.dumps(payload, indent=2, allow_nan=False) + "\n", path)
    logger.info("Wrote %s (%d games)", path, len(payload["games"]))
    return path


def run_current_week(
    cfg: Settings = settings,
    *,
    now: datetime | None = None,
    use_odds_api: bool = True,
    rebuild_features: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Load cached data, predict the current NFL week and write ``data/predictions/current_week.json``."""
    now = _as_utc(now or datetime.now(UTC))
    cache = ingest.get_cache() if cfg is settings else ingest.NflverseCache(cfg)
    features = F.build_and_save(cfg) if rebuild_features else F.load_features(cfg)
    schedules = cache.read("schedules", refresh=False)
    lines_fetched_at = schedules_fetched_at(cfg)  # provenance of the moneylines in `schedules`
    week_games = ingest.current_week_games(schedules, now)
    if week_games.is_empty():
        raise ModelError("no upcoming games in the current season schedule")
    season, week = int(week_games["season"][0]), int(week_games["week"][0])

    market = current_market_probs(
        week_games, schedules, cfg, use_odds_api=use_odds_api, now=now, schedules_fetched_at_override=lines_fetched_at
    )
    payload = predict_week(features, schedules, market, season=season, week=week, now=now)
    if write:
        write_payload(payload, cfg.current_week_json)
    return payload


def run_backtest(
    cfg: Settings = settings, *, first_test_season: int | None = None, rebuild_features: bool = True
) -> BacktestResult:
    """Walk-forward backtest over the cached history; writes predictions and a JSON report."""
    cache = ingest.get_cache() if cfg is settings else ingest.NflverseCache(cfg)
    features = F.build_and_save(cfg) if rebuild_features else F.load_features(cfg)
    schedules = cache.read("schedules", refresh=False)
    frame = features.join(
        nflverse_market_probs(schedules).select("game_id", "market_home_wp"), on="game_id", how="left"
    )
    result = walk_forward_backtest(frame, first_test_season=first_test_season)
    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
    ingest.write_parquet_atomic(result.predictions, cfg.predictions_dir / "backtest_predictions.parquet")
    ingest.write_text_atomic(
        json.dumps(result.report, indent=2, allow_nan=False) + "\n", cfg.predictions_dir / "backtest_report.json"
    )
    pooled = result.report["pooled"]
    logger.info(
        "Pooled out-of-sample %s: Brier model %.4f | market %.4f | final %.4f",
        f"{result.report['test_seasons'][0]}-{result.report['test_seasons'][-1]}",
        pooled["model"]["brier"],
        pooled["market"]["brier"],
        pooled["final"]["brier"],
    )
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_week(payload: dict[str, Any]) -> None:
    print(f"\n{payload['season']} week {payload['week']}  (market weight {payload['model']['market_weight']:.2f})")
    for g in payload["games"]:
        market = f"{g['market']['home_wp']:.3f}" if g["market"] else "  -  "
        edge = f"{g['model_edge_pct']:+.1f}" if g["model_edge_pct"] is not None else "  - "
        print(
            f"  {g['away_team']:>3} @ {g['home_team']:<3}  model {g['model']['home_wp']:.3f}  market {market}  "
            f"final {g['final']['home_wp']:.3f}  pick {g['pick']['team']:<3} {g['pick']['win_prob']:.1%}  "
            f"edge {edge:>5}  {g['confidence_tier']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model", description="Train, evaluate and run the win-probability model."
    )
    parser.add_argument("command", nargs="?", default="all", choices=("backtest", "predict", "all"))
    parser.add_argument("--first-test-season", type=int, default=None, help="earliest season to score in the backtest")
    parser.add_argument("--no-odds", action="store_true", help="do not call The Odds API")
    parser.add_argument("--no-rebuild-features", action="store_true", help="use data/processed/features.parquet as-is")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ...")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    settings.ensure_dirs()
    rebuild = not args.no_rebuild_features
    try:
        if args.command in ("backtest", "all"):
            run_backtest(first_test_season=args.first_test_season, rebuild_features=rebuild)
            rebuild = False  # features were just rebuilt
        if args.command in ("predict", "all"):
            _print_week(run_current_week(use_odds_api=not args.no_odds, rebuild_features=rebuild))
    except (ModelError, ContractError, F.FeatureBuildError, ingest.IngestError, FileNotFoundError) as exc:
        logger.error("%s failed: %s", args.command, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
