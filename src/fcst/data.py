"""Data adapter: load, validate, clean, hash, and synthetic generation.

The panel has one row per (series_id, period_start) with a ``demand`` column
(units sold per period; a proxy for true demand, since there is no stockout
column). Only columns listed in ``schema.required_columns`` are ever read;
everything else is dropped (guard 4). A missing real-data file is a hard
error and never falls back to synthetic data (guard 1).

``validate`` returns error codes and never raises for broken data; ``load_panel``
and ``clean`` raise ``FcstError`` with the first code.

The period length is not a config key. It is inferred as the most frequent
positive spacing between consecutive rows of a series (ties: the smaller one);
a panel with no such spacing is treated as daily.

CLI: ``python -m fcst.data --write-sample | --check-sample``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fcst.config import DEFAULT_CONFIG_PATH, FcstError, cost_rates, load_config

SYNTHETIC = "synthetic"
REAL = "real"

# Repository constraint for data/sample.csv (not a model parameter).
MAX_SAMPLE_BYTES = 1_500_000
SAMPLE_PATH = DEFAULT_CONFIG_PATH.parent / "data" / "sample.csv"

_BANNER = "*** SYNTHETIC DATA: illustrative only, not real sales ***"
_KEY_COLUMNS = ("sku_id", "location_id", "series_id")


# --- helpers ---------------------------------------------------------------


def _schema_columns(cfg: dict) -> list[str]:
    return list(cfg["params"]["schema"]["required_columns"])


def make_series_id(sku_id, location_id):
    """Deterministic series id from SKU and location (works on scalars or Series)."""
    if isinstance(sku_id, pd.Series):
        return sku_id.astype(str) + "@" + location_id.astype(str)
    return f"{sku_id}@{location_id}"


def _series_ids(df: pd.DataFrame) -> pd.Series:
    if "series_id" in df.columns:
        return df["series_id"].astype(str)
    return make_series_id(df["sku_id"], df["location_id"])


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse to naive datetime64[ns]; unparseable values become NaT (ISO 8601 only)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values)
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(None)
    else:
        parsed = pd.to_datetime(
            values.astype("string"), errors="coerce", format="ISO8601", utc=True
        ).dt.tz_convert(None)
    return parsed.astype("datetime64[ns]")


def _sorted_frame(sid: pd.Series, dates: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"s": sid.to_numpy(), "d": dates.to_numpy()})
    return frame.sort_values(["s", "d"], kind="stable").reset_index(drop=True)


def _infer_step(sid: pd.Series, dates: pd.Series) -> pd.Timedelta:
    frame = _sorted_frame(sid, dates)
    diffs = frame.groupby("s", sort=False)["d"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(days=1)
    counts = diffs.value_counts()
    return counts[counts == counts.max()].index.min()


def _grid_and_gaps(sid: pd.Series, dates: pd.Series) -> tuple[bool, int]:
    """Return (all dates on one grid, largest gap in missing periods)."""
    step = _infer_step(sid, dates)
    off_grid = bool(((dates - dates.min()) % step != pd.Timedelta(0)).any())
    if off_grid:
        return False, 0
    frame = _sorted_frame(sid, dates)
    diffs = frame.groupby("s", sort=False)["d"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return True, 0
    return True, int((diffs // step).max()) - 1


# --- validate / clean ------------------------------------------------------


def validate(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return the data error codes found in ``df`` (empty list = valid). Never raises."""
    if not isinstance(df, pd.DataFrame):
        return ["E_DATA_COLUMNS_MISSING"]
    present = set(df.columns)
    needed = [c for c in _schema_columns(cfg) if c not in present]
    if "series_id" in needed and {"sku_id", "location_id"} <= present:
        needed.remove("series_id")
    if needed or len(df) == 0:
        return ["E_DATA_COLUMNS_MISSING"]
    for col in _KEY_COLUMNS:
        if col in present:
            text = df[col].astype("string")
            if text.isna().any() or (text.str.strip() == "").any():
                return ["E_DATA_COLUMNS_MISSING"]

    errors: list[str] = []
    sid = _series_ids(df)
    dates = _parse_dates(df["period_start"])
    demand = pd.to_numeric(df["demand"], errors="coerce")
    dates_ok = bool(dates.notna().all())

    valid = dates.notna()
    keys = pd.DataFrame({"s": sid[valid].to_numpy(), "d": dates[valid].to_numpy()})
    if keys.duplicated().any():
        errors.append("E_DATA_DUPLICATE_KEY")

    off_grid, max_gap = (False, 0)
    if dates_ok:
        grid_ok, max_gap = _grid_and_gaps(sid, dates)
        off_grid = not grid_ok
    if not dates_ok or off_grid:
        errors.append("E_DATA_DATE_INVALID")

    finite = np.isfinite(demand.to_numpy(dtype="float64", na_value=np.nan))
    if not finite.all():
        errors.append("E_DATA_DEMAND_INVALID")
    if (demand[finite] < 0).any():
        errors.append("E_DATA_DEMAND_NEGATIVE")
    if dates_ok and not off_grid and max_gap > cfg["params"]["data"]["max_gap_periods"]:
        errors.append("E_DATA_GAP_TOO_LONG")
    return errors


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return the canonical panel; raise ``FcstError`` if ``df`` is invalid.

    Parses dates, converts types, keeps only the required columns, builds
    ``series_id`` when absent, fills allowed gaps with 0 and sorts by
    (series_id, period_start). Idempotent. Does not modify ``df``.
    """
    errors = validate(df, cfg)
    if errors:
        raise FcstError(errors[0], f"panel failed validation: {', '.join(errors)}")

    work = pd.DataFrame(
        {
            "sku_id": df["sku_id"].astype(str).to_numpy(),
            "location_id": df["location_id"].astype(str).to_numpy(),
            "series_id": _series_ids(df).to_numpy(),
            "period_start": _parse_dates(df["period_start"]).to_numpy(),
            "demand": pd.to_numeric(df["demand"]).to_numpy(dtype="float64"),
        }
    )
    work = work.sort_values(["series_id", "period_start"], kind="stable").reset_index(drop=True)

    step = _infer_step(work["series_id"], work["period_start"])
    origin = work["period_start"].min()
    pos = ((work["period_start"] - origin) // step).to_numpy()
    series = work["series_id"].to_numpy()
    same = series[1:] == series[:-1]
    missing = np.where(same, np.maximum(np.diff(pos) - 1, 0), 0)
    if missing.sum() > 0:
        prev_rows = np.nonzero(missing)[0]
        counts = missing[prev_rows]
        rep = np.repeat(prev_rows, counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        new_pos = pos[rep] + (np.arange(counts.sum()) - starts + 1)
        filler = pd.DataFrame(
            {
                "sku_id": work["sku_id"].to_numpy()[rep],
                "location_id": work["location_id"].to_numpy()[rep],
                "series_id": series[rep],
                "period_start": origin + pd.to_timedelta(new_pos * step.value, unit="ns"),
                "demand": 0.0,
            }
        )
        work = pd.concat([work, filler], ignore_index=True)
        work = work.sort_values(["series_id", "period_start"], kind="stable")

    out = work[_schema_columns(cfg)].reset_index(drop=True)
    out["period_start"] = out["period_start"].astype("datetime64[ns]")
    out["demand"] = out["demand"].astype("float64")
    return out


def load_panel(path, cfg: dict) -> pd.DataFrame:
    """Read a real panel from a CSV file and return the cleaned panel.

    A missing file raises ``E_DATA_FILE_MISSING``; there is never a fallback to
    synthetic data. Only schema columns are read.
    """
    p = Path(path)
    if not p.is_file():
        raise FcstError("E_DATA_FILE_MISSING", f"data file not found: {p}")
    wanted = set(_schema_columns(cfg))
    try:
        raw = pd.read_csv(
            p,
            dtype=str,
            usecols=lambda c: c in wanted,
            keep_default_na=False,
            na_values=[""],
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise FcstError("E_DATA_COLUMNS_MISSING", f"cannot parse {p}: {exc}") from exc
    return clean(raw, cfg)


# --- hash and banner -------------------------------------------------------


def data_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the panel content; independent of row order and extra columns."""
    dates = _parse_dates(df["period_start"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    demand = pd.to_numeric(df["demand"]).astype("float64").map(lambda x: repr(float(x)))
    rows = (
        df["sku_id"].astype(str)
        + "|"
        + df["location_id"].astype(str)
        + "|"
        + _series_ids(df)
        + "|"
        + dates
        + "|"
        + demand
    )
    payload = "\n".join(sorted(rows.tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def banner(data_source: str) -> str:
    """Banner text for synthetic data; empty string for real data."""
    if data_source == SYNTHETIC:
        return _BANNER
    if data_source == REAL:
        return ""
    raise ValueError(f"unknown data_source: {data_source!r}")


# --- sampling --------------------------------------------------------------


def make_sample(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Seeded subset of whole series (full history) from a cleaned panel.

    Size is ``params.data.synthetic.n_series`` (or all series if fewer).
    """
    ids = np.sort(raw["series_id"].astype(str).unique())
    k = min(cfg["params"]["data"]["synthetic"]["n_series"], len(ids))
    rng = np.random.default_rng(cfg["params"]["seed"])
    chosen = rng.choice(ids, size=k, replace=False)
    out = raw[raw["series_id"].astype(str).isin(chosen)]
    return out.sort_values(["series_id", "period_start"], kind="stable").reset_index(drop=True)


# --- synthetic generator ---------------------------------------------------
# The constants below describe the synthetic world (shape of the data), not a
# decision parameter. Only the size comes from params.data.synthetic.

_START = pd.Timestamp("2023-01-01")
_N_LOCATIONS = 4
_GROUPS = ("high_volume", "low_volume", "intermittent")
_GROUP_PARAMS = {
    # base level range, gamma-Poisson shape (smaller = noisier), weekly amplitude range
    "high_volume": {"base": (60.0, 200.0), "shape": 20.0, "weekly": (0.15, 0.40)},
    "low_volume": {"base": (6.0, 25.0), "shape": 8.0, "weekly": (0.10, 0.35)},
    "intermittent": {"base": (0.2, 0.9), "shape": 3.0, "weekly": (0.0, 0.2)},
}
_DROP_PROB = 0.3  # chance that an isolated zero-sales row is not recorded


def generate(seed: int, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic panel and its ground truth.

    Demand is a gamma-Poisson (negative binomial) count around
    trend x weekly x annual x promo. Three behaviour groups, some outlier
    spikes, new series (late start) and delisted series (early end). Isolated
    zero-sales rows are left out of the output, so ``clean`` has gaps to fill.

    Returns ``(df, truth)``. ``truth`` is for tests only: ``table`` (full grid
    with the true ``mean`` and true quantiles), ``quantile_cols`` (tau -> column),
    ``group``, ``new_series``, ``delisted``, ``outliers``, ``tau_star``.
    """
    n_series = cfg["params"]["data"]["synthetic"]["n_series"]
    n_periods = cfg["params"]["data"]["synthetic"]["n_periods"]
    season = cfg["params"]["season_length_periods"]
    per_year = float(cfg["business"]["periods_per_year"])
    rng = np.random.default_rng(seed)

    group_of = np.array([_GROUPS[i % len(_GROUPS)] for i in range(n_series)])
    rng.shuffle(group_of)
    perm = rng.permutation(n_series)
    n_special = max(2, round(0.1 * n_series))
    delisted_idx = set(perm[:n_special].tolist())
    new_idx = set(perm[n_special : 2 * n_special].tolist())
    annual_phase = rng.uniform(0, 2 * np.pi)
    weekly_phase = int(rng.integers(0, max(season, 1)))

    parts: list[dict] = []
    for i in range(n_series):
        params = _GROUP_PARAMS[group_of[i]]
        start, end = 0, n_periods
        if i in new_idx:
            start = int(rng.integers(int(0.25 * n_periods), n_periods - 30))
        if i in delisted_idx:
            end = int(rng.integers(int(0.4 * n_periods), int(0.85 * n_periods) + 1))
        t = np.arange(start, end)

        base = rng.uniform(*params["base"])
        slope = rng.uniform(-0.25, 0.5)
        weekly_amp = rng.uniform(*params["weekly"])
        series_phase = weekly_phase + int(rng.integers(-1, 2))
        annual_amp = rng.uniform(0.05, 0.25)
        annual_jitter = rng.normal(0, 0.3)
        trend = np.maximum(1.0 + slope * t / n_periods, 0.2)
        weekly = 1.0 + weekly_amp * np.sin(2 * np.pi * (t + series_phase) / max(season, 1))
        annual = 1.0 + annual_amp * np.sin(2 * np.pi * t / per_year + annual_phase + annual_jitter)
        promo = np.ones(len(t))
        for _ in range(rng.poisson(len(t) / 60.0)):
            p_start = int(rng.integers(0, len(t)))
            p_len = int(rng.integers(3, 8))
            promo[p_start : p_start + p_len] *= rng.uniform(1.3, 2.0)
        mu = np.maximum(base * trend * weekly * annual * promo, 1e-6)
        shape = params["shape"]
        y = rng.poisson(mu * rng.gamma(shape, 1.0 / shape, size=len(t))).astype(float)
        parts.append({"i": i, "t": t, "mu": mu, "y": y, "shape": np.full(len(t), shape)})

    idx = np.concatenate([np.full(len(p["t"]), p["i"]) for p in parts])
    t_all = np.concatenate([p["t"] for p in parts])
    mu = np.concatenate([p["mu"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    shape_all = np.concatenate([p["shape"] for p in parts])
    first = np.concatenate([np.arange(len(p["t"])) == 0 for p in parts])
    last = np.concatenate([np.arange(len(p["t"])) == len(p["t"]) - 1 for p in parts])
    n_rows = len(y)

    # outlier spikes on non-intermittent series
    eligible = np.nonzero(group_of[idx] != "intermittent")[0]
    n_out = min(max(3, int(0.003 * n_rows)), len(eligible))
    out_rows = np.sort(rng.choice(eligible, size=n_out, replace=False))
    y[out_rows] = np.ceil(mu[out_rows] * rng.uniform(4, 8, size=n_out) + 5)

    # leave out isolated zero-sales rows (never the first or last row of a series)
    u = rng.random(n_rows)
    candidate = (y == 0) & ~first & ~last
    dropped = np.zeros(n_rows, dtype=bool)
    for j in np.nonzero(candidate)[0]:
        if u[j] < _DROP_PROB and not dropped[j - 1]:
            dropped[j] = True

    sku = np.array([f"SKU{k // _N_LOCATIONS + 1:03d}" for k in range(n_series)])[idx]
    loc = np.array([f"LOC{k % _N_LOCATIONS + 1:02d}" for k in range(n_series)])[idx]
    sid = make_series_id(pd.Series(sku), pd.Series(loc)).to_numpy()
    dates = _START + pd.to_timedelta(t_all, unit="D")
    full = pd.DataFrame(
        {
            "sku_id": sku,
            "location_id": loc,
            "series_id": sid,
            "period_start": dates,
            "demand": y,
        }
    )
    df = full[~dropped].reset_index(drop=True)
    df["period_start"] = df["period_start"].astype("datetime64[ns]")

    _, c_over = cost_rates(cfg["business"])
    c_under = cfg["business"]["avg_unit_price"] - cfg["business"]["avg_unit_cost"]
    tau_star = c_under / (c_under + c_over)
    taus = sorted(set(cfg["params"]["tradeoff"]["taus"]) | {tau_star})
    p_nb = shape_all / (shape_all + mu)
    table = pd.DataFrame(
        {"series_id": sid, "period_start": full["period_start"].astype("datetime64[ns]"), "mean": mu}
    )
    quantile_cols = {}
    for tau in taus:
        name = f"q_{tau:g}"
        table[name] = stats.nbinom.ppf(tau, shape_all, p_nb)
        quantile_cols[tau] = name

    series_ids = [f"SKU{k // _N_LOCATIONS + 1:03d}@LOC{k % _N_LOCATIONS + 1:02d}" for k in range(n_series)]
    truth = {
        "table": table,
        "quantile_cols": quantile_cols,
        "tau_star": tau_star,
        "group": {series_ids[k]: str(group_of[k]) for k in range(n_series)},
        "new_series": sorted(series_ids[k] for k in new_idx),
        "delisted": sorted(series_ids[k] for k in delisted_idx),
        "outliers": full.iloc[out_rows][["series_id", "period_start"]].reset_index(drop=True),
    }
    return df, truth


# --- CSV and CLI -----------------------------------------------------------


def to_csv_bytes(df: pd.DataFrame, cfg: dict) -> bytes:
    """Serialize schema columns to deterministic CSV bytes (ISO dates, integer counts)."""
    out = df[_schema_columns(cfg)].copy()
    dates = _parse_dates(out["period_start"])
    midnight = bool((dates == dates.dt.normalize()).all())
    out["period_start"] = dates.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%dT%H:%M:%S")
    demand = pd.to_numeric(out["demand"])
    if bool((demand == demand.round()).all()):
        demand = demand.astype("int64")
    out["demand"] = demand
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_sample(cfg: dict, out_path: Path) -> int:
    print(banner(SYNTHETIC))
    df, _ = generate(cfg["params"]["seed"], cfg)
    payload = to_csv_bytes(df, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    print(f"wrote {out_path} ({len(df)} rows, {len(payload)} bytes)")
    print(f"data_hash {data_hash(clean(df, cfg))}")
    return 0


def _check_sample(cfg: dict, out_path: Path) -> int:
    print(banner(SYNTHETIC))
    seed = cfg["params"]["seed"]
    df1, _ = generate(seed, cfg)
    df2, _ = generate(seed, cfg)
    h1 = data_hash(clean(df1, cfg))
    h2 = data_hash(clean(df2, cfg))
    problems = []
    if h1 != h2:
        problems.append("generating twice gave different data_hash values")
    size = len(to_csv_bytes(df1, cfg))
    if size > MAX_SAMPLE_BYTES:
        problems.append(f"generated sample is {size} bytes, above the {MAX_SAMPLE_BYTES} limit")
    if out_path.is_file():
        if out_path.stat().st_size > MAX_SAMPLE_BYTES:
            problems.append(f"{out_path} is above the {MAX_SAMPLE_BYTES} byte limit")
        if data_hash(load_panel(out_path, cfg)) != h1:
            problems.append(f"{out_path} does not match the generator output; run `make sample`")
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"check-sample ok: data_hash {h1}, {size} bytes")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fcst.data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-sample", action="store_true", help="generate data/sample.csv")
    mode.add_argument("--check-sample", action="store_true", help="check sample determinism and size")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--out", default=str(SAMPLE_PATH), help="sample CSV path")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        out_path = Path(args.out)
        if args.write_sample:
            return _write_sample(cfg, out_path)
        return _check_sample(cfg, out_path)
    except FcstError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
