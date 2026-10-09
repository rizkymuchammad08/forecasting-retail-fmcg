"""Newsvendor value layer (P5, parts 1a and 1b).

Simulation is always ``newsvendor_per_origin``: each origin row is one independent
cycle that starts with no stock; ``S`` is the order-up-to level covering the horizon
``H`` and ``D`` the realised demand in ``(o, o + H]``.

    sold = min(S, D)      leftover = max(S - D, 0)      stockout = 1 if D > S (strict)
    net = c_under * sold - c_over * leftover            avg_inv = (S + leftover) / 2
    avg_inventory_value = unit_cost * mean(avg_inv)
    dio_periods = H * sum(avg_inv) / sum(sold)          (None when nothing was sold)

``c_under`` / ``c_over`` come from ``config.cost_rates`` (one implementation); ``c_over``
already holds one review period of holding cost plus the expected write-off.

Policies on the test origins: ``status_quo`` S = (1 + status_quo_buffer) * baseline point
forecast; ``baseline_newsvendor`` / ``model_newsvendor`` S = quantile at tau*. Curves use
every tau of ``models.calibration_taus(cfg)``. The caller supplies both quantile frames
(the bundle calibrates only the selected model; the baseline frame comes from
``fit_calibrator`` on the baseline's own calibration forecasts); rows are matched by position.

Equal service: the model's stockout rate at tau* is the target; the baseline's inventory at
that rate is a linear interpolation of its curve, never extrapolated (equal stockout rates
keep the smallest inventory). Outside the baseline range, or with zero baseline inventory:
``available = False`` and null values. Always a "conditional value".

Bootstrap: whole series are drawn with replacement from per-series aggregates (rows of a
series overlap in time). ``net_value_delta`` is a total over the test origins (resampled
sums are rescaled to the original origin count). Resamples without an equal-service match
are dropped and counted in ``n_boot_available``. Percentile intervals are widened to
contain the point value.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fcst.config import FcstError, cost_rates
from fcst.evaluate import quantile_column
from fcst.features import TARGET_COLUMN
from fcst.models import calibration_taus

SIMULATION = "newsvendor_per_origin"
POLICIES = ("status_quo", "baseline_newsvendor", "model_newsvendor")
VALUE_LABEL = "conditional value"
_N, _STOCK, _SOLD, _INV, _NET = range(5)  # columns of the per-series aggregates
_TAU_MATCH = 1e-9
_EPS = 1e-12


def _bad(msg: str) -> FcstError:
    return FcstError("E_PREDICT_BAD_INPUT", msg)


# --- costs and critical ratio ---


def costs(business: dict) -> tuple[float, float]:
    """``(c_under, c_over)`` per unit; delegates to ``config.cost_rates``."""
    try:
        c_under, c_over = cost_rates(business)
    except (KeyError, TypeError) as exc:
        raise FcstError("E_CONFIG", f"business: missing or invalid value ({exc})") from exc
    return float(c_under), float(c_over)


def tau_star(business: dict) -> float:
    """Critical ratio ``c_under / (c_under + c_over)``; computed, never stored."""
    c_under, c_over = costs(business)
    if c_under < 0 or c_over < 0 or c_under + c_over <= 0:
        raise FcstError("E_CONFIG", "business: costs must be >= 0 with a positive sum")
    return c_under / (c_under + c_over)


# --- order-up-to from quantile forecasts ---


def order_up_to(quantiles: pd.DataFrame, tau: float) -> np.ndarray:
    """Order-up-to level at ``tau``: the exact ``q_<tau>`` column if present, else a linear
    interpolation between the neighbouring tau columns (non-decreasing in tau). Outside
    the available range: ``E_PREDICT_BAD_INPUT`` (no extrapolation).
    """
    if not isinstance(quantiles, pd.DataFrame):
        raise _bad("quantiles must be a DataFrame with q_<tau> columns")
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not 0 < tau < 1:
        raise _bad(f"tau must be in (0, 1), got {tau!r}")
    if quantile_column(tau) in quantiles.columns:
        return quantiles[quantile_column(tau)].to_numpy(dtype="float64")
    grid = []
    for col in map(str, quantiles.columns):
        if col.startswith("q_") and col[2:].replace(".", "", 1).replace("e-", "").isdigit():
            grid.append((float(col[2:]), col))
    grid.sort()
    if not grid or tau < grid[0][0] - _TAU_MATCH or tau > grid[-1][0] + _TAU_MATCH:
        raise _bad(f"tau {tau!r} is outside the quantile columns; no extrapolation")
    taus = np.array([g[0] for g in grid])
    hi = min(max(int(np.searchsorted(taus, tau)), 1), len(taus) - 1)
    w = (tau - taus[hi - 1]) / (taus[hi] - taus[hi - 1])
    lo_q = quantiles[grid[hi - 1][1]].to_numpy(dtype="float64")
    return lo_q + w * (quantiles[grid[hi][1]].to_numpy(dtype="float64") - lo_q)


def _exact_column(quantiles: pd.DataFrame, tau: float, name: str) -> np.ndarray:
    col = quantile_column(tau)
    if col not in quantiles.columns:
        raise _bad(f"{name} has no column {col!r}")
    return quantiles[col].to_numpy(dtype="float64")


# --- simulation ---


def _vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1 or len(arr) == 0:
        raise _bad(f"{name} must be a non-empty 1-D array")
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise _bad(f"{name} must be finite and >= 0")
    return arr


def _rows(S: np.ndarray, D: np.ndarray, c_under: float, c_over: float) -> np.ndarray:
    """Per-origin ``(n, stockout, sold, avg_inv, net)`` as an (n, 5) array."""
    sold, left = np.minimum(S, D), np.maximum(S - D, 0.0)
    return np.column_stack(
        [np.ones(len(S)), (D > S).astype(float), sold, left + sold / 2.0, c_under * sold - c_over * left]
    )


def _summary(tot: np.ndarray, H: int, unit_cost: float) -> dict:
    """Metrics from the five column sums (also used for curve points)."""
    n, stock, sold, inv, net = (float(v) for v in tot)
    return {
        "n_origins": int(n),
        "stockout_rate": stock / n,
        "avg_inventory_units": inv / n,
        "avg_inventory_value": unit_cost * inv / n,
        "dio_periods": (H * inv / sold) if sold > 0 else None,
        "net_value": net,
    }


def _check_h(H) -> int:
    if not isinstance(H, (int, np.integer)) or isinstance(H, bool) or H < 1:
        raise _bad(f"H must be an integer >= 1, got {H!r}")
    return int(H)


def simulate_policy(S, D, costs, H, *, unit_cost: float) -> dict:
    """Run one policy over the origins. ``S`` and ``D`` are 1-D, finite and >= 0."""
    s, d = _vector(S, "S"), _vector(D, "D")
    if len(s) != len(d):
        raise _bad(f"S and D have different lengths ({len(s)} vs {len(d)})")
    try:
        c_under, c_over = (float(v) for v in costs)
    except (TypeError, ValueError) as exc:
        raise _bad("costs must be a (c_under, c_over) pair") from exc
    if not all(math.isfinite(v) and v >= 0 for v in (c_under, c_over, unit_cost)):
        raise _bad("costs and unit_cost must be finite and >= 0")
    rows = _rows(s, d, c_under, c_over)
    out = _summary(rows.sum(axis=0), _check_h(H), float(unit_cost))
    out.update(
        demand=float(d.sum()),
        sold=float(rows[:, _SOLD].sum()),
        leftover=float(np.maximum(s - d, 0.0).sum()),
        simulation=SIMULATION,
    )
    return out


# --- aggregates, curves, equal service ---


@dataclass(frozen=True)
class PolicyAggregates:
    """Per-series aggregates ``(n, stockouts, sold, avg_inv, net)`` of every policy."""

    series: np.ndarray
    taus: tuple
    tau_star: float
    unit_cost: float
    horizon: int
    status_quo: np.ndarray  # (n_series, 5)
    baseline: np.ndarray  # (n_taus, n_series, 5)
    model: np.ndarray  # (n_taus, n_series, 5)

    @property
    def tau_index(self) -> int:
        return int(np.argmin(np.abs(np.asarray(self.taus) - self.tau_star)))


def aggregate_policies(
    test: pd.DataFrame, q_baseline: pd.DataFrame, q_model: pd.DataFrame, cfg: dict
) -> PolicyAggregates:
    """Simulate the three policies and both curves on the test origins.

    ``test`` has ``y`` (realised demand), ``yhat_baseline`` (baseline point forecast) and
    ``series_id`` (column or index level). Every tau needs its exact ``q_<tau>`` column.
    """
    for col in (TARGET_COLUMN, "yhat_baseline"):
        if col not in test.columns:
            raise _bad(f"test frame has no column {col!r}")
    if "series_id" in test.columns:
        sid = test["series_id"].to_numpy()
    elif "series_id" in test.index.names:
        sid = np.asarray(test.index.get_level_values("series_id"))
    else:
        raise _bad("test frame has no series_id")
    business = cfg["business"]
    c_under, c_over = costs(business)
    D = _vector(test[TARGET_COLUMN], "y")
    yhat = _vector(test["yhat_baseline"], "yhat_baseline")
    for name, q in (("q_baseline", q_baseline), ("q_model", q_model)):
        if len(q) != len(D):
            raise _bad(f"{name} has {len(q)} rows, test has {len(D)}")
    taus = tuple(calibration_taus(cfg))

    ids, inv = np.unique(np.asarray(sid).astype(str), return_inverse=True)

    def agg(S) -> np.ndarray:
        mat = np.zeros((len(ids), 5))
        np.add.at(mat, inv, _rows(S, D, c_under, c_over))
        return mat

    status_quo = agg((1.0 + float(business["status_quo_buffer"])) * yhat)
    base = np.stack([agg(_exact_column(q_baseline, t, "q_baseline")) for t in taus])
    model = np.stack([agg(_exact_column(q_model, t, "q_model")) for t in taus])
    h = _check_h(cfg["params"]["horizon_periods"])
    unit_cost = float(business["avg_unit_cost"])
    return PolicyAggregates(ids, taus, tau_star(business), unit_cost, h, status_quo, base, model)


def tradeoff_curves(agg: PolicyAggregates) -> dict:
    """``{"baseline": [...], "model": [...]}``: one point per tau, ascending."""

    def curve(totals):
        pts = []
        for tau, tot in zip(agg.taus, totals, strict=True):
            p = {"tau": float(tau), "is_tau_star": abs(tau - agg.tau_star) <= _TAU_MATCH}
            pts.append(p | _summary(tot, agg.horizon, agg.unit_cost))
        return pts

    return {"baseline": curve(agg.baseline.sum(axis=1)), "model": curve(agg.model.sum(axis=1))}


def policy_summaries(agg: PolicyAggregates) -> dict:
    """Metrics of the three policies at tau* (the status quo has no tau)."""
    i = agg.tau_index
    tots = (agg.status_quo.sum(axis=0), agg.baseline[i].sum(axis=0), agg.model[i].sum(axis=0))
    return {k: _summary(t, agg.horizon, agg.unit_cost) for k, t in zip(POLICIES, tots, strict=True)}


def _interp_inventory(s_base, inv_base, target: float) -> tuple[float | None, str | None]:
    """Baseline inventory at stockout rate ``target``; ``(None, reason)`` if not available."""
    s = np.asarray(s_base, dtype="float64")
    inv = np.asarray(inv_base, dtype="float64")
    if len(s) == 0 or not (np.isfinite(s).all() and np.isfinite(inv).all()):
        return None, "baseline curve is empty or not finite"
    frontier: dict[float, float] = {}
    for x, y in zip(s, inv, strict=True):  # same service: keep the smallest inventory
        frontier[float(x)] = min(float(y), frontier.get(float(x), math.inf))
    xs = np.array(sorted(frontier))
    if target < xs[0] - _EPS or target > xs[-1] + _EPS:
        return None, "service target is outside the baseline stockout range"
    value = float(np.interp(target, xs, [frontier[x] for x in xs]))
    return (value, None) if value > 0.0 else (None, "baseline inventory at the target is zero")


def _equal_service(s_base, inv_base, target: float, inv_model: float) -> dict:
    base, reason = _interp_inventory(s_base, inv_base, target)
    ok = base is not None
    return {
        "method": "linear_interpolation",
        "available": ok,
        "target_stockout_rate": float(target),
        "inventory_model": float(inv_model),
        "inventory_baseline": base,
        "inventory_reduction_abs": (base - inv_model) if ok else None,
        "inventory_reduction_rel": ((base - inv_model) / base) if ok else None,
        "reason": reason,
    }


def equal_service(baseline_curve: list[dict], model_curve: list[dict], tau: float) -> dict:
    """Inventory difference at the model's stockout rate at ``tau`` ("conditional value").

    Curves are lists of points with ``tau``, ``stockout_rate`` and ``avg_inventory_value``
    (as ``tradeoff_curves`` returns). Positive ``inventory_reduction_rel``: the model needs
    less inventory for the same service.
    """
    need = {"tau", "stockout_rate", "avg_inventory_value"}
    if not all(c and all(need <= set(p) for p in c) for c in (baseline_curve, model_curve)):
        raise _bad(f"both curves need points with {', '.join(sorted(need))}")
    match = [p for p in model_curve if abs(p["tau"] - tau) <= _TAU_MATCH]
    if not match:
        raise _bad(f"model_curve has no point at tau {tau!r}")
    out = _equal_service(
        [p["stockout_rate"] for p in baseline_curve],
        [p["avg_inventory_value"] for p in baseline_curve],
        match[0]["stockout_rate"],
        match[0]["avg_inventory_value"],
    )
    return out | {"tau": float(tau)}


# --- bootstrap ---


def _interval(values: np.ndarray, point: float | None, level: float):
    if point is None or len(values) == 0:
        return None, None
    lo, hi = np.quantile(values, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return float(min(lo, point)), float(max(hi, point))


def bootstrap_business(
    agg: PolicyAggregates, cfg: dict, *, n_boot: int | None = None, seed: int | None = None
) -> dict:
    """Series bootstrap of ``net_value_delta.model_vs_status_quo`` and ``inventory_reduction_rel``."""
    ev = cfg["params"]["evaluation"]
    n_boot = int(ev["n_boot"] if n_boot is None else n_boot)
    level = float(ev["ci_level"])
    seed = int(cfg["params"]["seed"] if seed is None else seed)
    if n_boot < 1:
        raise FcstError("E_CONFIG", "params.evaluation.n_boot: must be >= 1")
    ti, uc, n_series = agg.tau_index, agg.unit_cost, len(agg.series)

    def inventory_cut(base: np.ndarray, model: np.ndarray) -> dict:
        s_b, i_b = base[:, _STOCK] / base[:, _N], uc * base[:, _INV] / base[:, _N]
        target = model[ti, _STOCK] / model[ti, _N]
        return _equal_service(s_b, i_b, target, uc * model[ti, _INV] / model[ti, _N])

    n_all = float(agg.status_quo[:, _N].sum())
    full_model = agg.model.sum(axis=1)
    delta_value = float(full_model[ti, _NET] - agg.status_quo[:, _NET].sum())
    full_cut = inventory_cut(agg.baseline.sum(axis=1), full_model)

    draw = np.random.default_rng(seed).integers(0, n_series, size=(n_boot, n_series))
    deltas, cuts = np.empty(n_boot), []
    for b, idx in enumerate(draw):
        sq, mo = agg.status_quo[idx].sum(axis=0), agg.model[:, idx, :].sum(axis=1)
        deltas[b] = (mo[ti, _NET] - sq[_NET]) * n_all / sq[_N]
        res = inventory_cut(agg.baseline[:, idx, :].sum(axis=1), mo)
        if res["available"]:
            cuts.append(res["inventory_reduction_rel"])

    common = {"ci_level": level, "n_boot": n_boot, "n_series": n_series, "seed": seed}
    lo, hi = _interval(deltas, delta_value, level)
    reduction = full_cut["inventory_reduction_rel"]
    r_lo, r_hi = _interval(np.asarray(cuts), reduction, level)
    delta = {"value": delta_value, "ci_low": lo, "ci_high": hi, "n_boot_available": n_boot}
    cut = {"available": full_cut["available"], "value": reduction, "ci_low": r_lo, "ci_high": r_hi}
    return {
        "net_value_delta": {
            "model_vs_status_quo": delta | {"scope": "total over test origins"} | common
        },
        "inventory_reduction_rel": cut
        | {"n_boot_available": len(cuts), "label": VALUE_LABEL}
        | common,
    }


# --- metrics.json block ---


def business_block(
    agg: PolicyAggregates, cfg: dict, *, n_boot: int | None = None, seed: int | None = None
) -> dict:
    """The ``business`` block of ``metrics.json`` (numbers only; the report reads them)."""
    curves = tradeoff_curves(agg)
    c_under, c_over = costs(cfg["business"])
    return {
        "simulation": SIMULATION,
        "value_label": VALUE_LABEL,
        "horizon_periods": agg.horizon,
        "n_origins": int(agg.status_quo[:, _N].sum()),
        "n_series": len(agg.series),
        "c_under": c_under,
        "c_over": c_over,
        "tau_star": agg.tau_star,
        "status_quo_buffer": float(cfg["business"]["status_quo_buffer"]),
        "policies": policy_summaries(agg),
        "tradeoff": {"taus": list(agg.taus)} | curves,
        "equal_service": equal_service(
            curves["baseline"], curves["model"], agg.taus[agg.tau_index]
        ),
    } | bootstrap_business(agg, cfg, n_boot=n_boot, seed=seed)


def _np_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_business_metrics(block: dict, path: str | Path) -> Path:
    """Set the ``business`` key of ``metrics.json``; other keys are kept as they are."""
    target, data = Path(path), {}
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FcstError("E_CONFIG", f"metrics file '{target}' is not readable JSON") from exc
        if not isinstance(data, dict):
            raise FcstError("E_CONFIG", f"metrics file '{target}' must hold a JSON object")
    data["business"] = block
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=_np_default)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target
