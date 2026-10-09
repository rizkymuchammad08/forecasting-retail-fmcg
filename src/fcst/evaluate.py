"""Accuracy metrics, quantile metrics and the paired bootstrap (P4).

Conventions
-----------
* ``y`` is the realised target (total demand in the window), ``yhat`` the point
  forecast, ``q`` a quantile forecast for level ``tau``. All inputs are 1-D,
  finite and of the same length; anything else raises ``E_PREDICT_BAD_INPUT``
  (the P0 code list has no dedicated code for malformed metric inputs).
* ``wape``, ``bias``: a zero total in the denominator raises
  ``E_EVAL_ZERO_DENOMINATOR``; they never return NaN or inf.
* ``bias`` is relative and signed: ``(sum(yhat) - sum(y)) / sum(y)``, so a
  positive value means over-forecasting.
* ``rmsse``: squared error per series, averaged over the rows of the series,
  divided by the series' scale (the MSE of a naive forecast on the training
  rows, see ``fcst.models.fit_scale``), averaged over series, square root.
  Series are weighted equally. A scale that is missing for a series is
  ``E_PREDICT_BAD_INPUT``; a scale that is not > 0 is
  ``E_EVAL_ZERO_DENOMINATOR``.
* ``coverage`` is ``mean(y <= q)``; ``coverage_error`` is ``coverage - tau``
  (signed: negative means the quantile is too low, which means stockouts).
* ``bootstrap_delta`` is a paired bootstrap that resamples whole series, so
  correlated rows of one series (overlapping target windows) stay together
  and both models are always scored on the same resample.

Nothing here imports ``fcst.models``; the dependency goes the other way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fcst.config import FcstError


def quantile_column(tau: float) -> str:
    """Column name of a quantile forecast (same naming as the truth table of ``fcst.data``)."""
    return f"q_{tau:g}"


# --- input checks --------------------------------------------------------------


def _bad(msg: str) -> FcstError:
    return FcstError("E_PREDICT_BAD_INPUT", msg)


def _vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1:
        raise _bad(f"{name} must be 1-D")
    if len(arr) == 0:
        raise _bad(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise _bad(f"{name} has non-finite values")
    return arr


def _pair(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    a, b = _vector(y, "y"), _vector(yhat, "yhat")
    if len(a) != len(b):
        raise _bad(f"y and yhat have different lengths ({len(a)} vs {len(b)})")
    return a, b


def _tau(tau: float) -> float:
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not 0 < tau < 1:
        raise _bad(f"tau must be in (0, 1), got {tau!r}")
    return float(tau)


def _series(series, n: int) -> np.ndarray:
    arr = np.asarray(series).astype(str)
    if arr.ndim != 1 or len(arr) != n:
        raise _bad("series must be 1-D and as long as y")
    return arr


def _zero(what: str) -> FcstError:
    return FcstError("E_EVAL_ZERO_DENOMINATOR", f"{what} is zero")


# --- point-forecast metrics -----------------------------------------------------


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: ``sum|y - yhat| / sum|y|``."""
    a, b = _pair(y, yhat)
    den = float(np.abs(a).sum())
    if den <= 0.0:
        raise _zero("sum of |y| (WAPE denominator)")
    return float(np.abs(a - b).sum() / den)


def bias(y, yhat) -> float:
    """Relative signed bias: ``(sum(yhat) - sum(y)) / sum(y)``."""
    a, b = _pair(y, yhat)
    den = float(a.sum())
    if den == 0.0:
        raise _zero("sum of y (bias denominator)")
    return float((b.sum() - den) / den)


def _scale_by_series(scale, ids: pd.Index) -> np.ndarray:
    """Look up a per-series scale for the unique series ``ids``."""
    if not isinstance(scale, (pd.Series, Mapping)):
        raise _bad("scale must be a mapping or a Series keyed by series id")
    lookup = {str(k): v for k, v in scale.items()}
    missing = [s for s in ids if s not in lookup]
    if missing:
        raise _bad(f"scale is missing for series: {missing[:5]}")
    out = np.asarray([lookup[s] for s in ids], dtype="float64")
    if not np.isfinite(out).all() or (out <= 0).any():
        raise _zero("series scale (RMSSE denominator)")
    return out


def _rmsse_terms(y, yhat, series, scale) -> np.ndarray:
    a, b = _pair(y, yhat)
    sid = _series(series, len(a))
    mse = pd.Series((a - b) ** 2).groupby(sid, sort=True).mean()
    return mse.to_numpy() / _scale_by_series(scale, mse.index)


def rmsse(y, yhat, series, scale) -> float:
    """Root mean squared scaled error over series (series weighted equally)."""
    return float(math.sqrt(_rmsse_terms(y, yhat, series, scale).mean()))


def primary_score(metric: str, y, yhat, series, scale=None) -> float:
    """Score for ``params.models.primary_metric``; ``scale`` is only used by RMSSE."""
    if metric == "wape":
        return wape(y, yhat)
    if metric == "rmsse":
        return rmsse(y, yhat, series, scale)
    raise FcstError("E_CONFIG", f"params.models.primary_metric: unknown metric {metric!r}")


# --- quantile metrics -------------------------------------------------------------


def pinball_loss(y, q, tau: float) -> float:
    """Mean pinball (quantile) loss of the quantile forecast ``q`` at level ``tau``."""
    a, b = _pair(y, q)
    t = _tau(tau)
    d = a - b
    return float(np.mean(np.maximum(t * d, (t - 1.0) * d)))


def coverage(y, q) -> float:
    """Share of observations at or below the quantile forecast."""
    a, b = _pair(y, q)
    return float(np.mean(a <= b))


def coverage_error(y, q, tau: float) -> float:
    """Signed coverage error ``coverage(y, q) - tau``."""
    return coverage(y, q) - _tau(tau)


def calibration_report(y, quantiles: pd.DataFrame, taus, cfg: dict) -> dict:
    """Coverage and pinball loss per level, reported as they are.

    ``quantiles`` has one column per tau (``quantile_column``). The tolerance is
    ``params.evaluation.calibration_tolerance``. ``within_tolerance`` is a fact
    about this sample and never raises: with a small test set it can be False
    through sampling noise alone (the binomial standard error of a coverage near
    0.9 on 100 rows is 0.03), and the pipeline must still go on and say so.
    ``n`` is in the report so the reader can judge that.
    """
    a = _vector(y, "y")
    tol = float(cfg["params"]["evaluation"]["calibration_tolerance"])
    rows = []
    for tau in taus:
        col = quantile_column(tau)
        if col not in quantiles.columns:
            raise _bad(f"quantiles has no column {col!r}")
        q = quantiles[col].to_numpy(dtype="float64")
        cov = coverage(a, q)
        err = cov - _tau(tau)
        rows.append(
            {
                "tau": float(tau),
                "coverage": cov,
                "coverage_error": err,
                "within_tolerance": bool(abs(err) <= tol + 1e-12),
                "pinball_loss": pinball_loss(a, q, tau),
            }
        )
    return {
        "n": len(a),
        "tolerance": tol,
        "rows": rows,
        "max_abs_coverage_error": max((abs(r["coverage_error"]) for r in rows), default=0.0),
        "all_within_tolerance": all(r["within_tolerance"] for r in rows),
    }


# --- paired bootstrap ----------------------------------------------------------------


def _series_stats(metric: str, y, yhat, series, scale):
    """Per-series (numerator, denominator, finish): metric = finish(sum num / sum den)."""
    a, b = _pair(y, yhat)
    sid = _series(series, len(a))
    if metric == "wape":
        frame = pd.DataFrame({"s": sid, "num": np.abs(a - b), "den": np.abs(a)})
        grouped = frame.groupby("s", sort=True).sum()
        return grouped["num"].to_numpy(), grouped["den"].to_numpy(), (lambda r: r)
    if metric == "rmsse":
        terms = _rmsse_terms(a, b, sid, scale)
        return terms, np.ones(len(terms)), np.sqrt
    raise FcstError("E_CONFIG", f"params.models.primary_metric: unknown metric {metric!r}")


def bootstrap_delta(
    y,
    yhat_candidate,
    yhat_baseline,
    series,
    cfg: dict,
    metric: str | None = None,
    scale=None,
    seed: int | None = None,
) -> dict:
    """Paired series-level bootstrap of ``metric(candidate) - metric(baseline)``.

    ``value`` is the difference on the full sample (negative: the candidate is
    better). Each of ``evaluation.n_boot`` replicates draws series with
    replacement and scores *both* models on the same draw, so the interval is
    for the difference and not for two independent scores. The interval is the
    percentile interval at ``evaluation.ci_level``, from
    ``default_rng(seed)`` with ``seed`` defaulting to ``params.seed``: the same
    inputs and seed give the same numbers. Identical forecasts give exactly 0.
    The interval is widened to contain ``value`` if skew would leave it
    outside. Replicates whose drawn series have a zero denominator (WAPE with
    no demand) are dropped; if the whole sample has a zero denominator the
    metric itself raises ``E_EVAL_ZERO_DENOMINATOR``.
    """
    p = cfg["params"]
    metric = metric if metric is not None else p["models"]["primary_metric"]
    n_boot = int(p["evaluation"]["n_boot"])
    level = float(p["evaluation"]["ci_level"])
    seed = int(p["seed"] if seed is None else seed)

    num_c, den, finish = _series_stats(metric, y, yhat_candidate, series, scale)
    num_b, _, _ = _series_stats(metric, y, yhat_baseline, series, scale)
    total_den = float(den.sum())
    if total_den <= 0.0:
        raise _zero("sum of |y| (WAPE denominator)")
    value = float(finish(num_c.sum() / total_den) - finish(num_b.sum() / total_den))

    n_series = len(den)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_series, size=(n_boot, n_series))
    d = den[draw].sum(axis=1)
    ok = d > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = finish(num_c[draw].sum(axis=1) / d) - finish(num_b[draw].sum(axis=1) / d)
    delta = delta[ok]
    if len(delta) == 0:
        low = high = value
    else:
        low, high = np.quantile(delta, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return {
        "metric": metric,
        "value": value,
        "ci_low": float(min(low, value)),
        "ci_high": float(max(high, value)),
        "ci_level": level,
        "n_boot": n_boot,
        "n_series": n_series,
        "seed": seed,
    }


def compare_to_baseline(y, yhat_candidate, yhat_baseline, series, cfg: dict, scale=None) -> dict:
    """Primary metric of both models and their bootstrapped difference.

    ``delta_primary`` is ``candidate - baseline`` on ``params.models.primary_metric``
    (negative: the candidate wins). A model compared with itself gives 0.
    """
    metric = cfg["params"]["models"]["primary_metric"]
    boot = bootstrap_delta(
        y, yhat_candidate, yhat_baseline, series, cfg, metric=metric, scale=scale
    )
    return {
        "primary_metric": metric,
        "candidate": primary_score(metric, y, yhat_candidate, series, scale),
        "baseline": primary_score(metric, y, yhat_baseline, series, scale),
        "delta_primary": boot["value"],
        "delta_ci_low": boot["ci_low"],
        "delta_ci_high": boot["ci_high"],
        "ci_level": boot["ci_level"],
        "n_boot": boot["n_boot"],
        "n_series": boot["n_series"],
    }
