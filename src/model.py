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
from sklearn.metrics import roc_auc_score
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
LOCK_MIN_WIN_PROB: Final[float] = 0.75  # final win probability of the pick
VALUE_EDGE_MIN_PCT: Final[float] = 5.0  # |model - market| in percentage points
LEAN_MIN_WIN_PROB: Final[float] = 0.55
TOP_DRIVERS: Final[int] = 3

CONTRACT_NAME: Final[str] = "B"
CONTRACT_VERSION: Final[str] = "1.0.0"
GAME_STATUSES: Final[tuple[str, ...]] = ("scheduled", "in_progress", "final")
MARKET_SOURCES: Final[tuple[str, ...]] = ("odds_api", "odds_api_snapshot", "nflverse_close")

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


def to_matrix(frame: pl.DataFrame, columns: Sequence[str] = F.FEATURE_COLUMNS) -> np.ndarray:
    """Float64 design matrix; nulls become NaN, which XGBoost treats as missing."""
    return frame.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()


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
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n_games": len(y),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
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
    has_market = np.isfinite(market)
    validation = {
        "model": probability_metrics(y[has_market], crossfit[has_market]),
        "market": probability_metrics(y[has_market], market[has_market]),
        "final": probability_metrics(y, blend_probabilities(crossfit, market, blend.weight)),
    }
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

    ``frame`` is the feature matrix joined with ``market_home_wp``. Metrics for
    model, market and final are compared on the games that have a market line;
    ``model_all_games`` also covers games without one.
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
        has_market = np.isfinite(market_p)
        metrics = {
            "model": probability_metrics(y[has_market], model_p[has_market]),
            "market": probability_metrics(y[has_market], market_p[has_market]),
            "final": probability_metrics(y, final_p),
            "model_all_games": probability_metrics(y, model_p),
        }
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
            metrics["final"]["brier"],
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
    has_market = np.isfinite(market_p)
    report = {
        "generated_at": _iso(datetime.now(UTC)),
        "algorithm": _algorithm_description(),
        "hyperparameters": dict(XGB_PARAMS),
        "test_seasons": [s.test_season for s in splits],
        "folds": folds,
        "pooled": {
            "model": probability_metrics(y[has_market], model_p[has_market]),
            "market": probability_metrics(y[has_market], market_p[has_market]),
            "final": probability_metrics(y, final_p),
        },
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
    "market_books": pl.UInt32,
    "market_as_of": pl.Datetime("us", "UTC"),
}


def _normalize_market(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.col(name).cast(dtype) for name, dtype in _MARKET_COLUMNS.items())


def _odds_consensus_to_market(consensus: pl.DataFrame, source: str) -> pl.DataFrame:
    return _normalize_market(
        consensus.filter(pl.col("game_id").is_not_null()).select(
            "game_id",
            "market_home_wp",
            pl.col("consensus_home_spread").alias("market_home_spread"),
            pl.lit(source).alias("market_source"),
            pl.col("n_books_h2h").alias("market_books"),
            pl.col("fetched_at").alias("market_as_of"),
        )
    )


def latest_odds_snapshot(cfg: Settings, season: int, week: int) -> pl.DataFrame | None:
    """Most recent consensus snapshot saved by ``src.ingest`` for this season/week."""
    files = sorted((cfg.odds_dir / str(season)).glob(f"odds_{season}_wk{week:02d}_*_consensus.parquet"))
    if not files:
        return None
    logger.info("Using saved odds snapshot %s", files[-1].name)
    return pl.read_parquet(files[-1])


def current_market_probs(
    week_games: pl.DataFrame,
    schedules: pl.DataFrame,
    cfg: Settings = settings,
    *,
    use_odds_api: bool = True,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Best available de-vigged market probability per current-week game.

    Priority: live Odds API consensus, then the latest saved Odds API snapshot for
    the week, then nflverse moneylines. Games that already kicked off are absent
    from the Odds API, so they fall back to nflverse closing lines.
    """
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
    snapshot = latest_odds_snapshot(cfg, season, week)
    if snapshot is not None:
        candidates.append(_odds_consensus_to_market(snapshot, "odds_api_snapshot"))
    candidates.append(_normalize_market(nflverse_market_probs(schedules)))

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


def confidence_tier(pick_win_prob: float, model_edge_pct: float | None) -> Tier:
    """Map a pick to a tier (first match wins).

    LOCK        the final win probability of the pick is >= 75%
    VALUE EDGE  the model and the market disagree by >= 5 percentage points
    LEAN        the final win probability of the pick is >= 55%
    COIN FLIP   everything else
    """
    if pick_win_prob >= LOCK_MIN_WIN_PROB:
        return "LOCK"
    if model_edge_pct is not None and abs(model_edge_pct) >= VALUE_EDGE_MIN_PCT:
        return "VALUE EDGE"
    if pick_win_prob >= LEAN_MIN_WIN_PROB:
        return "LEAN"
    return "COIN FLIP"


def top_drivers(
    contributions: np.ndarray,
    values: np.ndarray,
    home_team: str,
    away_team: str,
    *,
    feature_columns: Sequence[str] = F.FEATURE_COLUMNS,
    k: int = TOP_DRIVERS,
) -> list[dict[str, Any]]:
    """The ``k`` features with the largest absolute contribution to this game's raw log-odds."""
    order = np.argsort(-np.abs(contributions), kind="stable")
    drivers = []
    for i in order[:k]:
        impact = float(contributions[i])
        if impact == 0.0:
            break
        column = feature_columns[i]
        drivers.append(
            {
                "feature": column,
                "label": FEATURE_LABELS.get(column, column),
                "value": _round(float(values[i])),
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


def build_payload(
    week_rows: pl.DataFrame,
    fold: FittedFold,
    *,
    season: int,
    week: int,
    now: datetime,
) -> dict[str, Any]:
    """Assemble the Contract B payload.

    ``week_rows`` is the feature matrix for the week's games, left-joined with the
    current market columns (:data:`_MARKET_COLUMNS`). Per game:

    * ``market`` / ``model`` / ``final``: home and away win probabilities (``market`` is null without a line);
    * ``pick``: the side ``final`` favours, with its win probability;
    * ``model_edge_pct``: ``100 * (model - market)`` for the pick side, in percentage points
      (negative means the market rates the pick higher than the model does); ``value_team`` is the
      side the model rates above the market;
    * ``confidence_tier``: :func:`confidence_tier`;
    * ``top_drivers``: :func:`top_drivers`, explained against the uncalibrated log-odds.
    """
    rows = week_rows.sort("kickoff_utc", "game_id")
    model_p = fold.model.predict_proba(rows)
    market_p = _column(rows, "market_home_wp")
    final_p = blend_probabilities(model_p, market_p, fold.blend.weight)
    contributions = fold.model.contributions(rows)
    values = to_matrix(rows)

    games = []
    for i, row in enumerate(rows.iter_rows(named=True)):
        home, away = row["home_team"], row["away_team"]
        pick_home = final_p[i] >= 0.5
        pick_wp = final_p[i] if pick_home else 1.0 - final_p[i]
        has_market = bool(np.isfinite(market_p[i]))
        edge_home = 100.0 * (model_p[i] - market_p[i]) if has_market else None
        edge_pick = None if edge_home is None else (edge_home if pick_home else -edge_home)
        value_team = None if not edge_home else (home if edge_home > 0 else away)
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
                        **_probability_pair(market_p[i]),
                        "home_spread": _round(row["market_home_spread"], 2),
                        "source": row["market_source"],
                        "books": row["market_books"],
                        "as_of": _iso(row["market_as_of"]),
                    }
                    if has_market
                    else None
                ),
                "model": _probability_pair(model_p[i]),
                "final": _probability_pair(final_p[i]),
                "pick": {
                    "team": home if pick_home else away,
                    "side": "home" if pick_home else "away",
                    "win_prob": round(float(pick_wp), 4),
                },
                "model_edge_pct": _round(edge_pick, 2),
                "value_team": value_team,
                "confidence_tier": confidence_tier(float(pick_wp), edge_pick),
                "top_drivers": top_drivers(contributions[i], values[i], home, away),
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
            "validation": {
                "season": split.calibration_season,
                "method": "cross-fitted calibration (odd/even weeks)",
                "brier_model": _round(blend.brier_model),
                "brier_market": _round(blend.brier_market),
                "brier_final": _round(blend.brier_final),
                "n_games": blend.n_games,
            },
        },
        "tier_rules": {
            "LOCK": f"pick win probability >= {LOCK_MIN_WIN_PROB:.0%}",
            "VALUE EDGE": f"|model - market| >= {VALUE_EDGE_MIN_PCT:g} percentage points",
            "LEAN": f"pick win probability >= {LEAN_MIN_WIN_PROB:.0%}",
            "COIN FLIP": "otherwise",
        },
        "games": games,
    }


def validate_contract_b(payload: dict[str, Any]) -> None:
    """Raise :class:`ContractError` listing every violation of Contract B."""
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def is_prob(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0

    def check_pair(pair: Any, where: str) -> None:
        if not isinstance(pair, dict) or not is_prob(pair.get("home_wp")) or not is_prob(pair.get("away_wp")):
            problems.append(f"{where}: home_wp/away_wp must be probabilities")
        elif abs(pair["home_wp"] + pair["away_wp"] - 1.0) > 1e-3:
            problems.append(f"{where}: home_wp + away_wp must equal 1")

    def is_iso(x: Any) -> bool:
        try:
            return isinstance(x, str) and datetime.fromisoformat(x.replace("Z", "+00:00")) is not None
        except ValueError:
            return False

    check(payload.get("contract") == CONTRACT_NAME, f"contract must be {CONTRACT_NAME!r}")
    check(isinstance(payload.get("schema_version"), str), "schema_version must be a string")
    check(is_iso(payload.get("generated_at")), "generated_at must be ISO-8601")
    check(isinstance(payload.get("season"), int), "season must be an int")
    check(isinstance(payload.get("week"), int), "week must be an int")
    model = payload.get("model")
    if isinstance(model, dict):
        check(is_prob(model.get("market_weight")), "model.market_weight must be in [0, 1]")
        check(model.get("market_weight_source") in ("optimized", "default"), "model.market_weight_source invalid")
        check(
            isinstance(model.get("train_seasons"), list) and bool(model.get("train_seasons")),
            "model.train_seasons empty",
        )
    else:
        problems.append("model must be an object")
    games = payload.get("games")
    if not isinstance(games, list):
        problems.append("games must be a list")
        games = []
    ids = [g.get("game_id") for g in games if isinstance(g, dict)]
    check(len(ids) == len(set(ids)), "game_id values must be unique")

    required = (
        "game_id", "season", "week", "game_type", "kickoff_utc", "status", "home_team", "away_team", "venue",
        "net_rest_days", "market", "model", "final", "pick", "model_edge_pct", "value_team", "confidence_tier",
        "top_drivers", "result",
    )  # fmt: skip
    for n, game in enumerate(games):
        where = f"games[{n}]"
        if not isinstance(game, dict):
            problems.append(f"{where} must be an object")
            continue
        missing = [key for key in required if key not in game]
        if missing:
            problems.append(f"{where} missing {missing}")
            continue
        where = f"games[{n}] ({game['game_id']})"
        check(is_iso(game["kickoff_utc"]), f"{where}: kickoff_utc must be ISO-8601")
        check(game["status"] in GAME_STATUSES, f"{where}: status must be one of {GAME_STATUSES}")
        check_pair(game["model"], f"{where}.model")
        check_pair(game["final"], f"{where}.final")
        if game["market"] is not None:
            check_pair(game["market"], f"{where}.market")
            check(
                game["market"].get("source") in MARKET_SOURCES,
                f"{where}: market.source must be one of {MARKET_SOURCES}",
            )
        else:
            check(game["model_edge_pct"] is None, f"{where}: model_edge_pct must be null without a market line")
        pick = game["pick"]
        if isinstance(pick, dict) and pick.get("side") in ("home", "away") and isinstance(game["final"], dict):
            check(pick.get("team") == game[f"{pick['side']}_team"], f"{where}: pick.team does not match pick.side")
            check(
                abs(pick.get("win_prob", -1) - game["final"].get(f"{pick['side']}_wp", -2)) < 1e-3,
                f"{where}: pick.win_prob must equal the final probability of the pick side",
            )
            check(pick.get("win_prob", 0) >= 0.5 - 1e-9, f"{where}: pick must be the side final favours")
        else:
            problems.append(f"{where}: pick must have side 'home' or 'away'")
        check(game["confidence_tier"] in TIERS, f"{where}: confidence_tier must be one of {TIERS}")
        edge = game["model_edge_pct"]
        check(
            edge is None or (isinstance(edge, (int, float)) and math.isfinite(edge)), f"{where}: model_edge_pct invalid"
        )
        check(game["value_team"] in (None, game["home_team"], game["away_team"]), f"{where}: value_team invalid")
        drivers = game["top_drivers"]
        if isinstance(drivers, list) and len(drivers) <= TOP_DRIVERS:
            for d in drivers:
                ok = isinstance(d, dict) and {"feature", "label", "value", "impact_logit", "favors"} <= d.keys()
                check(ok and d["favors"] in (game["home_team"], game["away_team"]), f"{where}: malformed driver {d}")
        else:
            problems.append(f"{where}: top_drivers must be a list of at most {TOP_DRIVERS}")
        check(
            (game["result"] is not None) == (game["status"] == "final"), f"{where}: result must be set only when final"
        )

    if problems:
        raise ContractError("Contract B violations:\n  - " + "\n  - ".join(problems))


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
    payload = build_payload(week_rows, fold, season=season, week=week, now=now)
    validate_contract_b(payload)
    return payload


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
    now = now or datetime.now(UTC)
    cache = ingest.get_cache() if cfg is settings else ingest.NflverseCache(cfg)
    features = F.build_and_save(cfg) if rebuild_features else F.load_features(cfg)
    schedules = cache.read("schedules", refresh=False)
    week_games = ingest.current_week_games(schedules, now)
    if week_games.is_empty():
        raise ModelError("no upcoming games in the current season schedule")
    season, week = int(week_games["season"][0]), int(week_games["week"][0])

    market = current_market_probs(week_games, schedules, cfg, use_odds_api=use_odds_api, now=now)
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
