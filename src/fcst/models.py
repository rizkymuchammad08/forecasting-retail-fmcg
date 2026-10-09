"""Models, scale statistics, conformal calibration, selection and bundles (P4).

Models
------
``SeasonalNaive`` (``seasonal_naive``, the baseline) forecasts the window
``(o, o + H]`` with the demand of the same window shifted back by
``ceil(H / S) * S`` periods (``S`` = season length). The shift is at least
``H``, so the shifted window ends at or before the origin and is fully observed.
It reads the panel (only periods ``<= o``), not the feature matrix. If the
shifted window starts before the first row of a series, the available part is
scaled up to ``H`` periods (nothing available: 0).

``GbmLag`` (``gbm_lag``, the candidate) is a ``HistGradientBoostingRegressor``
on the P3 features. Series differ in size by three orders of magnitude, so the
target and the demand-valued features (``lag_*``, ``roll_mean_*``, ``roll_std_*``)
are divided by a per-series **level** (mean target of the series on the training
rows, floored at ``calibration.scale_floor``). That level is the preprocessing
state; it is learned from training rows only (``fit_scale``) and a series not
seen in training uses its own as-of ``roll_mean_H * H`` instead (never a later
value). Forecasts are multiplied back and clipped at 0.

Both models have ``fit(X, y)`` and ``predict(X, panel)``; ``X`` is indexed by
``(series_id, origin_date)``.

Selection
---------
``select_model`` scores both models on every rolling-origin fold of the
training rows with the primary metric and picks the lowest mean; ties go to the
baseline (``use_baseline`` is a legitimate verdict). Calibration rows, test rows
and their targets are never read. ``train_bundle`` then fits on all training
rows, calibrates the selected model on the calibration rows and returns a
``Bundle``.

Calibration
-----------
``conformal_scaled``: scaled residuals ``(y - yhat) / scale`` on the calibration
rows; for each tau the offset is the empirical quantile at level
``min(1, ceil((n + 1) * tau) / n)`` (finite-sample conformal rule). Quantile
forecast = ``max(0, yhat + scale * offset)``. Offsets are monotone in tau, so
quantiles are monotone in tau, and the clip keeps them >= 0. ``taus`` is the
union of ``tau_star`` (from ``config.cost_rates``) and ``tradeoff.taus``.
Marginal coverage only: it holds on average over exchangeable rows, not per
series, and a time shift between calibration and test can break it.

Persistence
-----------
``save_bundle`` writes ``bundle.joblib`` and ``bundle.json`` (hashes and
metadata) into a directory (default ``artifacts/``). ``load_bundle`` refuses a
file whose SHA-256 differs from the one in the JSON, because a joblib file is a
pickle and must not be loaded blindly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fcst.config import DEFAULT_CONFIG_PATH, FcstError, cost_rates
from fcst.evaluate import primary_score, quantile_column
from fcst.features import (
    TARGET_COLUMN,
    _dense,
    _origin_keys,
    _resolve,
    feature_names,
    make_features,
    make_split,
    rolling_origin_folds,
)

BASELINE = "seasonal_naive"
CANDIDATE = "gbm_lag"
CALIBRATION_METHOD = "conformal_scaled"
DEFAULT_ARTIFACTS_DIR = DEFAULT_CONFIG_PATH.parent / "artifacts"
BUNDLE_FILE = "bundle.joblib"
META_FILE = "bundle.json"
_BUNDLE_FORMAT = 1

# Not in config.yaml (P0 fixes the config keys); module constants on purpose.
GBM_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}
_PREPROCESS = "level_scaled_v1"
_LEVEL_SCALED_PREFIXES = ("lag_", "roll_mean_", "roll_std_")
_TAU_MERGE = 1e-9

PREDICT_COLUMNS = ("series_id", "origin_date", "yhat")  # then one q_<tau> per tau


def _bad(msg: str) -> FcstError:
    return FcstError("E_PREDICT_BAD_INPUT", msg)


# --- taus -------------------------------------------------------------------------


def tau_star(cfg: dict) -> float:
    """Critical ratio ``c_under / (c_under + c_over)``; computed, never stored."""
    c_under, c_over = cost_rates(cfg["business"])
    return c_under / (c_under + c_over)


def calibration_taus(cfg: dict) -> list[float]:
    """``{tau_star} ∪ tradeoff.taus``, ascending, values closer than 1e-9 merged."""
    raw = sorted({float(t) for t in cfg["params"]["tradeoff"]["taus"]} | {tau_star(cfg)})
    out: list[float] = []
    for t in raw:
        if not out or t - out[-1] > _TAU_MERGE:
            out.append(t)
    return out


# --- scale statistics (training rows only) ------------------------------------------


@dataclass(frozen=True)
class ScaleStats:
    """Per-series scale learned from training rows.

    ``level``: mean target per series (floored). ``naive_mse``: mean squared
    difference between consecutive-origin targets of the series (a naive
    in-sample forecast; floored at ``floor ** 2``), the RMSSE denominator.
    ``fallback_*``: medians over series, for a series with no training rows.
    """

    floor: float
    level: dict
    naive_mse: dict
    fallback_level: float
    fallback_naive_mse: float

    def level_for(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """Level per row of ``X``. Unseen series: as-of ``roll_mean_H * H``, else the fallback."""
        sid = pd.Series(np.asarray(X.index.get_level_values("series_id")).astype(str))
        known = sid.map(self.level).to_numpy(dtype="float64")
        col = f"roll_mean_{horizon}"
        if col in X.columns:
            own = X[col].to_numpy(dtype="float64") * horizon
        else:
            own = np.full(len(X), self.fallback_level)
        level = np.where(np.isnan(known), own, known)
        return np.maximum(np.nan_to_num(level, nan=self.fallback_level), self.floor)

    def naive_mse_for(self, series_ids) -> pd.Series:
        """RMSSE scale for the unique series in ``series_ids`` (index = series id)."""
        ids = pd.unique(np.asarray(series_ids).astype(str))
        return pd.Series(
            [self.naive_mse.get(s, self.fallback_naive_mse) for s in ids], index=ids, dtype="float64"
        )


def fit_scale(y: pd.Series, cfg: dict) -> ScaleStats:
    """Scale statistics from the training targets ``y`` (index ``(series_id, origin_date)``)."""
    if not isinstance(y, pd.Series) or not {"series_id", "origin_date"} <= set(y.index.names):
        raise _bad("y must be a Series indexed by (series_id, origin_date)")
    if len(y) == 0:
        raise _bad("no training rows to compute the scale from")
    values = y.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise _bad("training targets must be finite")
    floor = float(cfg["params"]["calibration"]["scale_floor"])
    frame = pd.DataFrame(
        {
            "s": np.asarray(y.index.get_level_values("series_id")).astype(str),
            "t": np.asarray(y.index.get_level_values("origin_date")),
            "y": values,
        }
    ).sort_values(["s", "t"], kind="stable", ignore_index=True)
    level = np.maximum(frame.groupby("s")["y"].mean(), floor)
    sq = frame.groupby("s")["y"].diff() ** 2
    mse = np.maximum(sq.groupby(frame["s"]).mean().dropna(), floor**2)
    return ScaleStats(
        floor=floor,
        level={str(k): float(v) for k, v in level.items()},
        naive_mse={str(k): float(v) for k, v in mse.items()},
        fallback_level=float(max(np.median(level.to_numpy()), floor)),
        fallback_naive_mse=float(max(np.median(mse.to_numpy()) if len(mse) else 0.0, floor**2)),
    )


# --- models -------------------------------------------------------------------------


def _check_X(X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame) or not {"series_id", "origin_date"} <= set(X.index.names):
        raise _bad("X must be a DataFrame indexed by (series_id, origin_date)")
    missing = [c for c in names if c not in X.columns]
    if missing:
        raise _bad(f"X is missing feature columns: {missing[:5]}")
    X = X[names]
    if not np.isfinite(X.to_numpy(dtype="float64")).all():
        raise _bad("X has non-finite values")
    return X


class SeasonalNaive:
    """Baseline: the shifted, fully observed demand window."""

    name = BASELINE

    def __init__(self, cfg: dict) -> None:
        p = cfg["params"]
        self.horizon = int(p["horizon_periods"])
        self.season = int(p["season_length_periods"])
        self.shift = math.ceil(self.horizon / self.season) * self.season
        self.seed = int(p["seed"])
        self.feature_names = feature_names(cfg)

    @property
    def params(self) -> dict:
        return {"horizon": self.horizon, "season_length": self.season, "shift": self.shift}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SeasonalNaive:
        """Nothing is learned; kept so every model is fitted the same way."""
        return self

    def predict(self, X: pd.DataFrame, panel: pd.DataFrame | None = None) -> np.ndarray:
        if panel is None:
            raise _bad("seasonal_naive needs the panel")
        dn = _dense(panel)
        codes, slots = _resolve(dn, X.index)
        pos = slots[:, None] - self.shift + np.arange(1, self.horizon + 1)[None, :]
        inside = pos >= 0
        window = np.where(inside, dn.values[codes[:, None], np.where(inside, pos, 0)], np.nan)
        seen = np.isfinite(window).sum(axis=1)
        total = np.nansum(window, axis=1)
        yhat = np.where(seen > 0, total * self.horizon / np.maximum(seen, 1), 0.0)
        return np.clip(yhat, 0.0, None)


class GbmLag:
    """Candidate: gradient boosting on level-scaled lag/rolling features."""

    name = CANDIDATE

    def __init__(self, cfg: dict, params: dict | None = None) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.horizon = int(cfg["params"]["horizon_periods"])
        self.seed = int(cfg["params"]["seed"])
        self.feature_names = feature_names(cfg)
        self.hgb_params = {**GBM_PARAMS, **(params or {})}
        self.scale_: ScaleStats | None = None
        self.model_: HistGradientBoostingRegressor | None = None

    @property
    def params(self) -> dict:
        return {"preprocess": _PREPROCESS, **self.hgb_params}

    def scale_state(self) -> ScaleStats:
        if self.scale_ is None:
            raise _bad("model is not fitted")
        return self.scale_

    def _design(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        level = self.scale_state().level_for(X, self.horizon)
        per_period = level / self.horizon
        Z = X.copy()
        for col in Z.columns:
            if col.startswith(_LEVEL_SCALED_PREFIXES):
                Z[col] = X[col].to_numpy(dtype="float64") / per_period
        return Z.to_numpy(dtype="float64"), level

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GbmLag:
        X = _check_X(X, self.feature_names)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise _bad("y must be a Series with the same index as X")
        if len(X) == 0:
            raise _bad("no training rows")
        self.scale_ = fit_scale(y, self.cfg)  # training rows only
        Z, level = self._design(X)
        self.model_ = HistGradientBoostingRegressor(random_state=self.seed, **self.hgb_params)
        self.model_.fit(Z, y.to_numpy(dtype="float64") / level)
        return self

    def predict(self, X: pd.DataFrame, panel: pd.DataFrame | None = None) -> np.ndarray:
        if self.model_ is None:
            raise _bad("model is not fitted")
        Z, level = self._design(_check_X(X, self.feature_names))
        return np.clip(self.model_.predict(Z) * level, 0.0, None)


def model_hash(model) -> str:
    """SHA-256 of the model's identity: name, parameters, seed and feature names.

    Same inputs give the same hash; changing any model parameter, the seed or the
    feature list changes it. It does not depend on training data or fitted state.
    """
    payload = {
        "name": model.name,
        "params": model.params,
        "seed": model.seed,
        "features": list(model.feature_names),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# --- calibration --------------------------------------------------------------------


@dataclass(frozen=True)
class Calibrator:
    """Fitted ``conformal_scaled`` state: one offset (in scale units) per tau."""

    method: str
    taus: tuple
    offsets: tuple
    n_calib: int
    scale_floor: float

    def quantiles(self, yhat, scale, index=None) -> pd.DataFrame:
        """Quantile forecasts, one ``q_<tau>`` column per tau; monotone in tau and >= 0."""
        yh = np.clip(np.asarray(yhat, dtype="float64"), 0.0, None)
        sc = np.maximum(np.asarray(scale, dtype="float64"), self.scale_floor)
        if yh.ndim != 1 or yh.shape != sc.shape or not np.isfinite(yh).all():
            raise _bad("yhat and scale must be finite 1-D arrays of the same length")
        q = yh[:, None] + sc[:, None] * np.asarray(self.offsets)[None, :]
        q = np.maximum.accumulate(np.clip(q, 0.0, None), axis=1)
        return pd.DataFrame(q, columns=[quantile_column(t) for t in self.taus], index=index)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_calib": self.n_calib,
            "scale_floor": self.scale_floor,
            "offsets": {quantile_column(t): o for t, o in zip(self.taus, self.offsets, strict=True)},
        }


def fit_calibrator(yhat_calib, y_calib, scale_calib, taus, cfg: dict) -> Calibrator:
    """Fit ``conformal_scaled`` on the calibration rows. Depends on nothing else."""
    yh = np.asarray(yhat_calib, dtype="float64")
    y = np.asarray(y_calib, dtype="float64")
    sc = np.asarray(scale_calib, dtype="float64")
    if yh.ndim != 1 or not (yh.shape == y.shape == sc.shape):
        raise _bad("yhat_calib, y_calib and scale_calib must be 1-D and of the same length")
    if len(yh) == 0:
        raise _bad("no calibration rows")
    if not (np.isfinite(yh).all() and np.isfinite(y).all() and np.isfinite(sc).all()):
        raise _bad("calibration inputs must be finite")
    grid = sorted({float(t) for t in taus})
    if not grid or not all(0 < t < 1 for t in grid):
        raise FcstError("E_CONFIG", "taus: need at least one value and every value in (0, 1)")
    floor = float(cfg["params"]["calibration"]["scale_floor"])
    scores = (y - yh) / np.maximum(sc, floor)  # the floor also covers a zero scale
    n = len(scores)
    offsets = tuple(
        float(np.quantile(scores, min(1.0, math.ceil((n + 1) * t) / n), method="higher"))
        for t in grid
    )
    return Calibrator(CALIBRATION_METHOD, tuple(grid), offsets, n, floor)


# --- selection and bundle -------------------------------------------------------------


@dataclass
class Bundle:
    """Everything a forecast needs: fitted models, calibrator, scale state, hashes."""

    selected: str
    verdict: str  # "use_model" or "use_baseline"
    models: dict
    calibrator: Calibrator
    scale: ScaleStats
    feature_names: list
    model_hash: str
    hashes: dict
    selection: dict
    seed: int

    @property
    def model(self):
        return self.models[self.selected]


def _check_model_names(cfg: dict) -> None:
    m = cfg["params"]["models"]
    if m["baseline"] != BASELINE:
        raise FcstError("E_CONFIG", f"params.models.baseline: only {BASELINE!r} is implemented")
    if m["candidate"] != CANDIDATE:
        raise FcstError("E_CONFIG", f"params.models.candidate: only {CANDIDATE!r} is implemented")


def _check_table(table: pd.DataFrame, names: list[str]) -> None:
    if not isinstance(table, pd.DataFrame):
        raise _bad("table must be a DataFrame")
    missing = [c for c in [*names, TARGET_COLUMN] if c not in table.columns]
    if missing:
        raise _bad(f"table is missing columns: {missing[:5]}")
    if not {"series_id", "origin_date"} <= set(table.index.names):
        raise _bad("table must be indexed by (series_id, origin_date)")


def _series_of(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame.index.get_level_values("series_id")).astype(str)


def select_model(train: pd.DataFrame, panel: pd.DataFrame, cfg: dict, gbm_params=None) -> dict:
    """Pick the model with the lowest mean primary metric over rolling-origin folds.

    Only the training rows are passed in, and every fit and every scale statistic
    inside a fold sees that fold's training part only. Returns a report with the
    per-fold scores of both models; ``verdict`` is ``use_baseline`` unless the
    candidate is strictly better on average.
    """
    _check_model_names(cfg)
    names = feature_names(cfg)
    _check_table(train, names)
    metric = cfg["params"]["models"]["primary_metric"]
    plan = rolling_origin_folds(train, cfg)

    folds, per_model = [], {BASELINE: [], CANDIDATE: []}
    for j, fold in enumerate(plan):
        tr, va = fold.train, fold.valid
        scale = fit_scale(tr[TARGET_COLUMN], cfg)
        rmsse_scale = scale.naive_mse_for(_series_of(va))
        models = {
            BASELINE: SeasonalNaive(cfg).fit(tr[names], tr[TARGET_COLUMN]),
            CANDIDATE: GbmLag(cfg, gbm_params).fit(tr[names], tr[TARGET_COLUMN]),
        }
        scores = {}
        for name, model in models.items():
            yhat = model.predict(va[names], panel)
            scores[name] = primary_score(
                metric, va[TARGET_COLUMN], yhat, _series_of(va), rmsse_scale
            )
            per_model[name].append(scores[name])
        folds.append({"fold": j, "n_train": len(tr), "n_valid": len(va), "scores": scores})

    mean = {name: float(np.mean(v)) for name, v in per_model.items()}
    candidate_wins = mean[CANDIDATE] < mean[BASELINE]
    return {
        "primary_metric": metric,
        "folds": folds,
        "mean": mean,
        "selected": CANDIDATE if candidate_wins else BASELINE,
        "verdict": "use_model" if candidate_wins else "use_baseline",
        **plan.summary(),
    }


def train_bundle(
    table: pd.DataFrame, panel: pd.DataFrame, cfg: dict, gbm_params: dict | None = None
) -> Bundle:
    """Split, select on train folds, fit on train, calibrate on calib. Test is never read.

    ``table`` is the P3 table (features joined with ``y``, indexed by
    ``(series_id, origin_date)``); ``panel`` is the cleaned demand panel.
    """
    _check_model_names(cfg)
    names = feature_names(cfg)
    _check_table(table, names)
    split = make_split(table, cfg)
    train, calib = split.train, split.calib

    selection = select_model(train, panel, cfg, gbm_params)
    scale = fit_scale(train[TARGET_COLUMN], cfg)
    models = {
        BASELINE: SeasonalNaive(cfg).fit(train[names], train[TARGET_COLUMN]),
        CANDIDATE: GbmLag(cfg, gbm_params).fit(train[names], train[TARGET_COLUMN]),
    }
    chosen = models[selection["selected"]]
    yhat_calib = chosen.predict(calib[names], panel)
    calibrator = fit_calibrator(
        yhat_calib,
        calib[TARGET_COLUMN].to_numpy(dtype="float64"),
        scale.level_for(calib[names], cfg["params"]["horizon_periods"]),
        calibration_taus(cfg),
        cfg,
    )
    hashes = {name: model_hash(m) for name, m in models.items()}
    return Bundle(
        selected=selection["selected"],
        verdict=selection["verdict"],
        models=models,
        calibrator=calibrator,
        scale=scale,
        feature_names=names,
        model_hash=hashes[selection["selected"]],
        hashes=hashes,
        selection=selection,
        seed=int(cfg["params"]["seed"]),
    )


# --- predict ---------------------------------------------------------------------------


def _check_bundle(bundle, cfg: dict) -> None:
    if not isinstance(bundle, Bundle):
        raise _bad("bundle must be a Bundle")
    if list(bundle.feature_names) != feature_names(cfg):
        raise _bad("bundle was trained with different features than the config implies")
    if bundle.model_hash != model_hash(bundle.model):
        raise _bad("bundle model_hash does not match its model")


def _features(panel: pd.DataFrame, origins, cfg: dict) -> pd.DataFrame:
    """Features as-of the origins; data problems become ``E_PREDICT_BAD_INPUT``."""
    if not isinstance(panel, pd.DataFrame):
        raise _bad("panel must be a DataFrame")
    try:
        keys = _origin_keys(origins)
        if len(keys) == 0:
            raise _bad("no origins to forecast")
        return make_features(panel, keys, cfg)
    except FcstError as exc:
        if exc.code.startswith("E_DATA_"):
            raise _bad(exc.message) from exc
        raise


def predict_point(bundle: Bundle, panel, origins, cfg: dict, which: str | None = None) -> pd.DataFrame:
    """Point forecasts only (``series_id, origin_date, yhat``) of one model of the bundle.

    ``which`` is a model name (default: the selected one). Use it to get the
    baseline forecast for comparison; quantiles are only calibrated for the
    selected model, which is why ``predict`` does not take ``which``.
    """
    _check_bundle(bundle, cfg)
    name = bundle.selected if which is None else which
    if name not in bundle.models:
        raise _bad(f"unknown model {name!r}")
    X = _features(panel, origins, cfg)
    try:
        yhat = bundle.models[name].predict(X, panel)
    except FcstError as exc:
        raise (_bad(exc.message) if exc.code.startswith("E_DATA_") else exc) from exc
    return _frame(X, yhat)


def _frame(X: pd.DataFrame, yhat: np.ndarray) -> pd.DataFrame:
    if not np.isfinite(yhat).all() or (yhat < 0).any():
        raise _bad("model produced a non-finite or negative forecast")
    idx = X.index
    return pd.DataFrame(
        {
            "series_id": np.asarray(idx.get_level_values("series_id")),
            "origin_date": np.asarray(idx.get_level_values("origin_date")),
            "yhat": yhat,
        }
    )


def predict(bundle: Bundle, panel: pd.DataFrame, origins, cfg: dict) -> pd.DataFrame:
    """Calibrated forecast for every origin: ``series_id, origin_date, yhat, q_<tau>...``.

    Rows keep the order and number of ``origins``. No target is needed: the
    latest origin of a series (no complete target window yet) is allowed. Bad
    input (not a panel, no origins, unknown series, dates off the grid or
    outside the series, duplicates, config that does not match the bundle)
    raises ``E_PREDICT_BAD_INPUT``. All outputs are >= 0 and ``q_*`` is
    non-decreasing in tau.
    """
    _check_bundle(bundle, cfg)
    X = _features(panel, origins, cfg)
    try:
        yhat = bundle.model.predict(X, panel)
    except FcstError as exc:
        raise (_bad(exc.message) if exc.code.startswith("E_DATA_") else exc) from exc
    out = _frame(X, yhat)
    scale = bundle.scale.level_for(X, cfg["params"]["horizon_periods"])
    quantiles = bundle.calibrator.quantiles(out["yhat"].to_numpy(), scale)
    return pd.concat([out, quantiles], axis=1)


# --- persistence -------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(bundle: Bundle, directory: str | Path | None = None) -> Path:
    """Write ``bundle.joblib`` and ``bundle.json`` to ``directory`` (default ``artifacts/``)."""
    if not isinstance(bundle, Bundle):
        raise _bad("bundle must be a Bundle")
    out = Path(directory) if directory is not None else DEFAULT_ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / BUNDLE_FILE)
    meta = {
        "format": _BUNDLE_FORMAT,
        "file": BUNDLE_FILE,
        "sha256": _sha256_file(out / BUNDLE_FILE),
        "model_hash": bundle.model_hash,
        "hashes": bundle.hashes,
        "selected": bundle.selected,
        "verdict": bundle.verdict,
        "seed": bundle.seed,
        "feature_names": list(bundle.feature_names),
        "calibration": bundle.calibrator.to_dict(),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_bundle(directory: str | Path | None = None) -> Bundle:
    """Load a bundle after checking the file hash and the model hash."""
    src = Path(directory) if directory is not None else DEFAULT_ARTIFACTS_DIR
    try:
        meta = json.loads((src / META_FILE).read_text(encoding="utf-8"))
        file = src / meta["file"]
        if not file.is_file():
            raise _bad(f"bundle file not found: {file}")
    except (OSError, ValueError, KeyError) as exc:
        raise _bad(f"cannot read bundle metadata in '{src}': {exc}") from exc
    if _sha256_file(file) != meta.get("sha256"):
        raise _bad("bundle file does not match its recorded SHA-256")
    bundle = joblib.load(file)
    if not isinstance(bundle, Bundle) or bundle.model_hash != meta.get("model_hash"):
        raise _bad("bundle content does not match its recorded model_hash")
    if bundle.model_hash != model_hash(bundle.model):
        raise _bad("bundle model_hash does not match its model")
    return bundle
