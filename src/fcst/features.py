"""Target, features, leakage guards, time split and rolling-origin CV (P3).

Conventions (fixed here, used by every later part)
--------------------------------------------------
* An **origin** is the last period whose demand is known. Everything a row uses
  is as-of the origin: demand of periods ``<= origin`` and the calendar.
* The **target** of an origin is the total demand in ``(origin, origin + H]``
  with ``H = params.horizon_periods`` (origin excluded, ``origin + H`` included).
* ``lag_k`` is the demand ``k - 1`` periods before the origin, so ``lag_1`` is
  the demand at the origin and ``lag_S`` (``S`` = season length) is the demand
  one season before the first target period.
* Rolling statistics use the last ``w`` periods up to and including the origin
  (population std, ``ddof=0``).
* Origins live on one global grid: first period of the panel, then every
  ``params.origin_step_periods``. Dates are mapped to integer period slots with
  the period length inferred by ``fcst.data`` (most frequent spacing).
* The feature matrix and the labels are indexed by ``(series_id, origin_date)``
  and share the row order of the origins they were built from.
* The time axis of a table is its ``origin_date`` (column or index level).

Leakage is handled twice: ``assert_no_forbidden`` and ``check_correlation`` are
fences (cheap, can miss things); ``perturb_future_check`` is the proof (changes
or deletes everything after each origin and requires identical features).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from fcst.config import FcstError
from fcst.data import _infer_step  # one rule for the period length, shared with P2

KEY_COLUMNS = ("series_id", "origin_date")
TARGET_COLUMN = "y"

# Column names (and name prefixes) that must never be a model input, on top of
# every schema column. Identifiers are excluded too: a model must not memorise
# series across the split.
_FORBIDDEN_NAMES = frozenset(
    {
        "y",
        "target",
        "label",
        "labels",
        "demand",
        "period_start",
        "origin_date",
        "target_end_date",
    }
)
FORBIDDEN_PREFIXES = ("target", "label", "future", "next_", "y_")


# --- feature specification ---------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    lags: tuple[int, ...]
    windows: tuple[int, ...]
    span: int


def _spec(cfg: dict) -> _Spec:
    """Lags and windows derived from the season length, horizon and min history.

    Nothing here is a free parameter: every value is a multiple of the season
    length, the horizon, or a short fixed lag, and is capped by
    ``domain.min_history_periods`` so each origin has all the history it needs.
    """
    p = cfg["params"]
    season = p["season_length_periods"]
    horizon = p["horizon_periods"]
    min_hist = p["domain"]["min_history_periods"]
    lags = sorted({1, 2, 3} | {season * m for m in (1, 2, 3, 4)})
    windows = sorted({season, horizon} | {season * m for m in (2, 4, 8)})
    lags = tuple(k for k in lags if k <= min_hist)
    windows = tuple(w for w in windows if w <= min_hist)
    if season > min_hist or not lags or not windows:
        raise FcstError(
            "E_CONFIG",
            "params.domain.min_history_periods: must be >= params.season_length_periods",
        )
    return _Spec(lags=lags, windows=windows, span=max(max(lags), max(windows)))


def feature_names(cfg: dict) -> list[str]:
    """Columns of the feature matrix, in matrix order."""
    spec = _spec(cfg)
    names = [f"lag_{k}" for k in spec.lags]
    for stat in ("mean", "std", "zero_share"):
        names += [f"roll_{stat}_{w}" for w in spec.windows]
    names.append("age_periods")
    names += ["cal_month", "cal_annual_sin", "cal_annual_cos", "cal_weekend_periods"]
    return names


# --- dense panel -------------------------------------------------------------


@dataclass(frozen=True)
class _Dense:
    ids: pd.Index  # sorted unique series ids
    t0: pd.Timestamp  # date of slot 0 (first period of the whole panel)
    step_ns: int  # period length in ns
    values: np.ndarray  # (n_series, n_slots), NaN outside the active range
    first: np.ndarray  # first active slot per series
    last: np.ndarray  # last active slot per series

    def dates(self, slots: np.ndarray) -> pd.DatetimeIndex:
        offset = (np.asarray(slots, dtype="int64") * self.step_ns).astype("timedelta64[ns]")
        return pd.DatetimeIndex(self.t0.to_datetime64() + offset)


def _dense(panel: pd.DataFrame) -> _Dense:
    """Series x period matrix of demand. Reads only series_id, period_start, demand."""
    if not isinstance(panel, pd.DataFrame) or not {
        "series_id",
        "period_start",
        "demand",
    } <= set(panel.columns):
        raise FcstError(
            "E_DATA_COLUMNS_MISSING", "panel needs series_id, period_start and demand"
        )
    if len(panel) == 0:
        raise FcstError("E_DATA_COLUMNS_MISSING", "panel is empty")
    dates = pd.to_datetime(panel["period_start"], errors="coerce")
    if dates.isna().any():
        raise FcstError("E_DATA_DATE_INVALID", "panel has unparseable period_start")
    dates = dates.astype("datetime64[ns]")
    demand = pd.to_numeric(panel["demand"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(demand).all():
        raise FcstError("E_DATA_DEMAND_INVALID", "panel has non-finite demand")
    if (demand < 0).any():
        raise FcstError("E_DATA_DEMAND_NEGATIVE", "panel has negative demand")

    sid = panel["series_id"].astype(str)
    step = _infer_step(sid, dates)
    d = dates.to_numpy(dtype="datetime64[ns]")
    t0 = d.min()
    delta = (d - t0).astype("int64")
    if (delta % step.value).any():
        raise FcstError("E_DATA_DATE_INVALID", "panel dates are not on one period grid")
    pos = delta // step.value
    n_slots = int(pos.max()) + 1

    codes, ids = pd.factorize(sid, sort=True)
    if pd.Series(codes.astype("int64") * n_slots + pos).duplicated().any():
        raise FcstError("E_DATA_DUPLICATE_KEY", "panel has duplicate (series, period) rows")

    values = np.full((len(ids), n_slots), np.nan)
    values[codes, pos] = demand
    first = np.full(len(ids), n_slots, dtype="int64")
    last = np.full(len(ids), -1, dtype="int64")
    np.minimum.at(first, codes, pos)
    np.maximum.at(last, codes, pos)
    slot = np.arange(n_slots)[None, :]
    inside = (slot >= first[:, None]) & (slot <= last[:, None])
    values[inside & np.isnan(values)] = 0.0  # a gap inside the active range is 0 (P0)
    return _Dense(
        ids=pd.Index(ids),
        t0=pd.Timestamp(t0),
        step_ns=int(step.value),
        values=values,
        first=first,
        last=last,
    )


def _origin_keys(origins) -> pd.DataFrame:
    if isinstance(origins, pd.MultiIndex):
        origins = origins.to_frame(index=False)
    if isinstance(origins, pd.DataFrame):
        if set(KEY_COLUMNS) <= set(origins.columns):
            return origins[list(KEY_COLUMNS)]
        if set(KEY_COLUMNS) <= set(origins.index.names):
            return origins.index.to_frame(index=False)[list(KEY_COLUMNS)]
    raise FcstError("E_DATA_COLUMNS_MISSING", "origins need series_id and origin_date")


def _resolve(dn: _Dense, origins) -> tuple[np.ndarray, np.ndarray]:
    """Map (series_id, origin_date) rows to (series code, slot). Validates them."""
    keys = _origin_keys(origins)
    codes = dn.ids.get_indexer(keys["series_id"].astype(str).to_numpy())
    if (codes < 0).any():
        raise FcstError("E_DATA_COLUMNS_MISSING", "origins contain series not in the panel")
    when = pd.to_datetime(keys["origin_date"], errors="coerce")
    if when.isna().any():
        raise FcstError("E_DATA_DATE_INVALID", "origins contain unparseable origin_date")
    delta = (
        when.astype("datetime64[ns]").to_numpy() - dn.t0.to_datetime64()
    ).astype("int64")
    if (delta < 0).any() or (delta % dn.step_ns).any():
        raise FcstError("E_DATA_DATE_INVALID", "origin_date is not on the panel period grid")
    slots = delta // dn.step_ns
    if ((slots < dn.first[codes]) | (slots > dn.last[codes])).any():
        raise FcstError("E_DATA_DATE_INVALID", "origin_date is outside the series active range")
    if pd.DataFrame({"c": codes, "s": slots}).duplicated().any():
        raise FcstError("E_DATA_DUPLICATE_KEY", "origins contain duplicate keys")
    return codes.astype("int64"), slots.astype("int64")


def _index(dn: _Dense, codes: np.ndarray, slots: np.ndarray) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [dn.ids.take(codes), dn.dates(slots)], names=list(KEY_COLUMNS)
    )


# --- origins -----------------------------------------------------------------


def build_origins(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """All admissible ``(series_id, origin_date)`` rows, sorted by series then date.

    An origin is kept when it is on the origin grid and, as-of the origin:
    the series has at least ``domain.min_history_periods`` periods of history,
    it is not inactive (some demand in the last ``domain.inactive_periods``
    periods, window clipped to the history), and the target window
    ``(origin, origin + H]`` ends at or before the last row of the series.
    """
    p = cfg["params"]
    horizon = p["horizon_periods"]
    min_hist = p["domain"]["min_history_periods"]
    inactive = p["domain"]["inactive_periods"]
    dn = _dense(panel)
    n_series, n_slots = dn.values.shape

    slots = np.arange(0, n_slots, p["origin_step_periods"], dtype="int64")[None, :]
    first, last = dn.first[:, None], dn.last[:, None]
    ok = (slots >= first) & (slots - first + 1 >= min_hist) & (slots + horizon <= last)

    nonzero = np.concatenate(
        [np.zeros((n_series, 1), dtype="int64"), np.cumsum(dn.values > 0, axis=1)], axis=1
    )
    rows = np.arange(n_series)[:, None]
    lo = np.maximum(slots - inactive + 1, first)
    recent = nonzero[rows, slots + 1] - nonzero[rows, lo]
    ok &= recent > 0

    s_idx, c_idx = np.nonzero(ok)
    return pd.DataFrame(
        {
            "series_id": dn.ids.take(s_idx),
            "origin_date": dn.dates(slots[0, c_idx]),
        }
    )


# --- features and labels -------------------------------------------------------


def _calendar(dn: _Dense, slots: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Calendar of the target window, from dates only (one pass per distinct origin)."""
    uniq, inv = np.unique(slots, return_inverse=True)
    inv = inv.reshape(-1)
    origin = dn.t0.to_datetime64() + (uniq * dn.step_ns).astype("timedelta64[ns]")
    ahead = (np.arange(1, horizon + 1) * dn.step_ns).astype("timedelta64[ns]")
    target = origin[:, None] + ahead[None, :]
    dow = pd.DatetimeIndex(target.ravel()).dayofweek.to_numpy().reshape(len(uniq), horizon)
    weekend = (dow >= 5).sum(axis=1)
    mid = pd.DatetimeIndex(origin + np.timedelta64((horizon * dn.step_ns) // 2, "ns"))
    angle = 2.0 * math.pi * mid.dayofyear.to_numpy() / 365.25
    return {
        "cal_month": mid.month.to_numpy(dtype="float64")[inv],
        "cal_annual_sin": np.sin(angle)[inv],
        "cal_annual_cos": np.cos(angle)[inv],
        "cal_weekend_periods": weekend.astype("float64")[inv],
    }


def make_features(panel: pd.DataFrame, origins, cfg: dict) -> pd.DataFrame:
    """Feature matrix as-of each origin; columns equal ``feature_names(cfg)``.

    ``origins`` is a frame (or MultiIndex) with ``series_id`` and ``origin_date``.
    The result is indexed by ``(series_id, origin_date)`` in the order of the
    origins. Only demand at periods ``<= origin`` and dates are read, so an
    origin with no complete target window (the latest one, for a live forecast)
    is allowed. Never touches the target.
    """
    spec = _spec(cfg)
    names = feature_names(cfg)
    assert_no_forbidden(names, cfg)
    dn = _dense(panel)
    codes, slots = _resolve(dn, origins)
    history = slots - dn.first[codes] + 1
    if (history < spec.span).any():
        raise FcstError(
            "E_DATA_DATE_INVALID",
            f"an origin has fewer than {spec.span} periods of history",
        )

    back = np.arange(spec.span)  # position j holds lag_{j+1}
    recent = dn.values[codes[:, None], slots[:, None] - back[None, :]]
    cols: dict[str, np.ndarray] = {}
    for k in spec.lags:
        cols[f"lag_{k}"] = recent[:, k - 1]
    for stat in ("mean", "std", "zero_share"):
        for w in spec.windows:
            window = recent[:, :w]
            if stat == "mean":
                cols[f"roll_mean_{w}"] = window.mean(axis=1)
            elif stat == "std":
                cols[f"roll_std_{w}"] = window.std(axis=1, ddof=0)
            else:
                cols[f"roll_zero_share_{w}"] = (window == 0).mean(axis=1)
    cols["age_periods"] = history.astype("float64")
    cols.update(_calendar(dn, slots, cfg["params"]["horizon_periods"]))

    matrix = pd.DataFrame(cols, index=_index(dn, codes, slots))[names]
    assert_no_forbidden(matrix.columns, cfg)
    return matrix


def build_labels(panel: pd.DataFrame, origins, cfg: dict) -> pd.Series:
    """Target ``y`` = total demand in ``(origin, origin + H]``, indexed like the features.

    Raises ``E_DATA_DATE_INVALID`` if a window runs past the last row of its series.
    """
    horizon = cfg["params"]["horizon_periods"]
    dn = _dense(panel)
    codes, slots = _resolve(dn, origins)
    if (slots + horizon > dn.last[codes]).any():
        raise FcstError(
            "E_DATA_DATE_INVALID", "a target window runs past the last period of its series"
        )
    ahead = np.arange(1, horizon + 1)
    window = dn.values[codes[:, None], slots[:, None] + ahead[None, :]]
    return pd.Series(
        window.sum(axis=1), index=_index(dn, codes, slots), name=TARGET_COLUMN, dtype="float64"
    )


# --- leakage -------------------------------------------------------------------


def forbidden_columns(cfg: dict) -> frozenset[str]:
    """Exact column names banned from the feature matrix (schema + target names)."""
    schema = cfg["params"]["schema"]["required_columns"]
    return frozenset(_FORBIDDEN_NAMES | set(schema))


def assert_no_forbidden(columns, cfg: dict) -> None:
    """Raise ``E_LEAKAGE`` if any column is forbidden by name or by prefix.

    ``columns`` is an iterable of names or a DataFrame. Case-insensitive.
    """
    names = list(columns.columns) if isinstance(columns, pd.DataFrame) else list(columns)
    banned = {n.lower() for n in forbidden_columns(cfg)}
    bad = [
        str(n)
        for n in names
        if str(n).lower() in banned or str(n).lower().startswith(FORBIDDEN_PREFIXES)
    ]
    if bad:
        raise FcstError("E_LEAKAGE", f"forbidden columns in feature matrix: {sorted(bad)}")


def check_correlation(X: pd.DataFrame, y: pd.Series, cfg: dict) -> pd.Series:
    """Fence: raise ``E_LEAKAGE`` if a feature has ``|corr| > leakage.max_abs_corr`` with y.

    The statistic is the Pearson correlation after removing each series' mean
    from the feature and from y (within-series), when the index has a
    ``series_id`` level; otherwise the plain pooled Pearson. Pooled correlation
    is dominated by the spread of series sizes: on the default sample an honest
    ``roll_mean_56`` already reaches 0.977 pooled but only about 0.4 within a
    series, while a copy of the target (times a constant, plus noise, plus any
    per-series offset) stays near 1 either way. Pairwise on non-missing values;
    constant columns count as 0. Returns the absolute correlations (largest
    first) when it passes. This catches a copy of the target, not a subtle
    leak: see ``perturb_future_check``.
    """
    if not X.index.equals(y.index):
        raise FcstError("E_PREDICT_BAD_INPUT", "X and y must share the same index")
    limit = cfg["params"]["leakage"]["max_abs_corr"]
    target = y.astype("float64").to_numpy()
    group = X.index.get_level_values("series_id") if "series_id" in X.index.names else None

    def centered(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
        out = np.where(ok, values, 0.0)
        if group is None:
            return np.where(ok, out - out[ok].mean(), 0.0)
        frame = pd.DataFrame({"g": group, "v": np.where(ok, values, np.nan)})
        return np.nan_to_num(values - frame.groupby("g")["v"].transform("mean").to_numpy())

    corr = {}
    for name in X.columns:
        col = X[name].astype("float64").to_numpy()
        ok = np.isfinite(col) & np.isfinite(target)
        if ok.sum() < 2:
            corr[name] = 0.0
            continue
        a, b = centered(col, ok), centered(target, ok)
        a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
        denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        corr[name] = abs(float((a * b).sum()) / denom) if denom > 0 else 0.0
    out = pd.Series(corr, dtype="float64").sort_values(ascending=False)
    bad = out[out > limit]
    if len(bad):
        detail = ", ".join(f"{k} ({v:.4f})" for k, v in bad.items())
        raise FcstError("E_LEAKAGE", f"|corr| with target above {limit}: {detail}")
    return out


def _differing_columns(base: pd.DataFrame, other: pd.DataFrame) -> list[str]:
    if not base.index.equals(other.index) or list(base.columns) != list(other.columns):
        return list(base.columns)
    a, b = base.to_numpy(dtype="float64"), other.to_numpy(dtype="float64")
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return [c for c, ok in zip(base.columns, same.all(axis=0), strict=True) if not ok]


def perturb_future_check(
    panel: pd.DataFrame,
    origins,
    cfg: dict,
    feature_fn: Callable[[pd.DataFrame, pd.DataFrame, dict], pd.DataFrame] | None = None,
) -> dict:
    """Proof of no look-ahead: features must not change when the future changes.

    For every distinct origin date O, two perturbed panels are built: one where
    every row after O has its demand replaced by unrelated random values, and
    one where every row after O is deleted. The features of the origins at O are
    recomputed on each and must be identical (exact equality) to the originals.
    Raises ``E_LEAKAGE`` naming the columns that moved; otherwise returns a small
    evidence dict. ``feature_fn(panel, origins, cfg)`` defaults to ``make_features``.
    """
    fn = feature_fn if feature_fn is not None else make_features
    keys = _origin_keys(origins).reset_index(drop=True)
    when = pd.to_datetime(keys["origin_date"]).to_numpy()
    base = fn(panel, keys, cfg)
    if len(base) != len(keys):
        raise FcstError("E_LEAKAGE", "feature function changed the number of rows")
    rng = np.random.default_rng(cfg["params"]["seed"])
    period = pd.to_datetime(panel["period_start"])

    moved: dict[str, set[str]] = {}
    n_rows = 0
    dates = np.unique(when)
    for date in dates:
        at = when == date
        rows = keys[at]
        expected = base.iloc[np.nonzero(at)[0]]
        after = (period > pd.Timestamp(date)).to_numpy()
        modified = panel.copy()
        modified.loc[after, "demand"] = rng.uniform(0.0, 1e4, int(after.sum()))
        deleted = panel[~after]
        for variant, perturbed in (("modify", modified), ("delete", deleted)):
            for col in _differing_columns(expected, fn(perturbed, rows, cfg)):
                moved.setdefault(col, set()).add(variant)
        n_rows += len(rows)
    if moved:
        detail = ", ".join(f"{c} ({'+'.join(sorted(v))})" for c, v in sorted(moved.items()))
        raise FcstError("E_LEAKAGE", f"features changed when the future changed: {detail}")
    return {
        "passed": True,
        "n_origin_dates": len(dates),
        "n_rows": n_rows,
        "variants": ["modify", "delete"],
    }


# --- time axis, split, rolling-origin CV -------------------------------------------


def _origin_axis(table) -> np.ndarray:
    """``origin_date`` of every row as datetime64[ns]; ``E_SPLIT_NO_TIME_AXIS`` if absent."""
    if not isinstance(table, pd.DataFrame):
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "table must be a DataFrame with origin_date")
    if "origin_date" in table.columns:
        raw = table["origin_date"]
    elif "origin_date" in table.index.names:
        raw = table.index.get_level_values("origin_date").to_series(index=table.index)
    else:
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "table has no origin_date column")
    if len(raw) == 0:
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "table is empty")
    if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "origin_date is not a date")
    dates = pd.to_datetime(raw, errors="coerce")
    if dates.isna().any():
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "origin_date has missing or invalid dates")
    return dates.astype("datetime64[ns]").to_numpy()


def _cut(fraction: float, n: int) -> int:
    return math.floor(fraction * n + 0.5 + 1e-9)


class Split(NamedTuple):
    """Row subsets of the input table, in input order."""

    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    purged: pd.DataFrame


def make_split(table: pd.DataFrame, cfg: dict) -> Split:
    """Temporal split on distinct origin dates: ``train < calib < test``.

    The sorted distinct origin dates are cut by ``split.train`` / ``calib`` /
    ``test`` (rounded cumulative cuts). Then the last origins of ``train`` and of
    ``calib`` are purged, so the next side starts at least ``purge_gap_periods``
    later (``ceil(purge_gap / origin_step)`` origin steps; the gap is therefore
    >= H). Purging takes from the earlier side, so ``test`` keeps its exact
    share. All rows of one origin date go to the same side. Assumes the origin
    dates are on the origin grid of ``build_origins`` (spacing >= one step).
    Returns ``Split(train, calib, test, purged)``.
    """
    dates = _origin_axis(table)
    uniq, inv = np.unique(dates, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(uniq)
    if n < 3:
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "need at least 3 distinct origin dates to split")

    s = cfg["params"]["split"]
    gap = math.ceil(s["purge_gap_periods"] / cfg["params"]["origin_step_periods"])
    b1 = _cut(s["train"], n)
    b2 = _cut(s["train"] + s["calib"], n)
    role = np.empty(n, dtype="int8")  # 0 train, 1 calib, 2 test, 3 purged
    role[:b1], role[b1:b2], role[b2:] = 0, 1, 2
    role[max(b1 - (gap - 1), 0) : b1] = 3
    role[max(b2 - (gap - 1), b1) : b2] = 3
    if not all((role == r).any() for r in (0, 1, 2)):
        raise FcstError(
            "E_SPLIT_NO_TIME_AXIS", f"{n} origin dates are too few for train/calib/test with purge"
        )
    row_role = role[inv]
    return Split(*(table[row_role == r] for r in (0, 1, 2, 3)))


class Fold(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame


@dataclass(frozen=True)
class CVPlan:
    """Expanding-window folds plus how many were asked for and how many fit."""

    folds: tuple[Fold, ...]
    n_folds_requested: int
    n_folds_effective: int

    def __iter__(self) -> Iterator[Fold]:
        return iter(self.folds)

    def __len__(self) -> int:
        return len(self.folds)

    def summary(self) -> dict:
        return {
            "n_folds_requested": self.n_folds_requested,
            "n_folds_effective": self.n_folds_effective,
        }


def rolling_origin_folds(table: pd.DataFrame, cfg: dict) -> CVPlan:
    """Expanding-window rolling-origin folds over the distinct origin dates.

    Fold ``j`` validates on a block of ``v`` consecutive origins and trains on
    every origin up to ``g`` origin steps before the block, where
    ``g = H // origin_step + 1`` makes the validation origin strictly later than
    the last training origin plus H. Blocks are the latest origins, back to
    back, all of size ``v``. ``v`` is the largest size that leaves the first
    fold at least ``cv.min_train_origins`` training origins. If ``cv.n_folds``
    folds do not fit, fewer are used (``n_folds_effective``); fewer than 2 raises
    ``E_CV_TOO_SHORT``.
    """
    dates = _origin_axis(table)
    uniq, inv = np.unique(dates, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(uniq)
    p = cfg["params"]
    requested = p["cv"]["n_folds"]
    min_train = p["cv"]["min_train_origins"]
    gap = p["horizon_periods"] // p["origin_step_periods"] + 1

    room = n - (min_train - 1 + gap)  # origins available for validation blocks
    effective = min(requested, max(room, 0))
    if effective < 2:
        raise FcstError(
            "E_CV_TOO_SHORT",
            f"{n} origin dates allow {max(effective, 0)} fold(s); need at least 2 "
            f"(min_train_origins={min_train}, gap={gap} origin steps)",
        )
    size = room // effective
    folds = []
    for j in range(effective):
        start = n - (effective - j) * size
        train_mask = inv <= start - gap
        valid_mask = (inv >= start) & (inv < start + size)
        folds.append(Fold(table[train_mask], table[valid_mask]))
    return CVPlan(tuple(folds), requested, effective)
