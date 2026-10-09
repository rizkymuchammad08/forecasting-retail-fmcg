"""Feature tests (P3): origins, features, labels, leakage, split and CV."""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

import fcst.features as ffeat
from fcst import data as fdata
from fcst.config import FcstError, load_config, validate_config
from fcst.features import (
    assert_no_forbidden,
    build_labels,
    build_origins,
    check_correlation,
    feature_names,
    forbidden_columns,
    make_features,
    make_split,
    perturb_future_check,
    rolling_origin_folds,
)

DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2024-01-01")  # a Monday


def _cfg(*, step=2, season=2, lead=1, review=2, min_history=8, inactive=4, **split_cv):
    """Small valid config: H = lead + review, everything else from config.yaml."""
    cfg = copy.deepcopy(load_config())
    cfg["business"]["lead_time_periods"] = lead
    cfg["business"]["review_period_periods"] = review
    p = cfg["params"]
    p["horizon_periods"] = lead + review
    p["season_length_periods"] = season
    p["origin_step_periods"] = step
    p["domain"] = {"min_history_periods": min_history, "inactive_periods": inactive}
    p["split"]["purge_gap_periods"] = lead + review
    for key, value in split_cv.items():
        group, leaf = key.split("__")
        p[group][leaf] = value
    validate_config(cfg)
    return cfg


def _rows(series_id, start, values):
    return [
        {
            "sku_id": series_id,
            "location_id": "L1",
            "series_id": series_id,
            "period_start": START + (start + t) * DAY,
            "demand": float(v),
        }
        for t, v in enumerate(values)
    ]


def _a(t):  # series A: (5 t) mod 9, written out in the hand-computed test
    return (5 * t) % 9


def _small_panel():
    """Six small series, deterministic. Slots are days since 2024-01-01."""
    rows = []
    rows += _rows("A", 0, [_a(t) for t in range(30)])  # slots 0..29, plain
    rows += _rows("B", 0, [1 if t % 3 == 0 else 0 for t in range(30)])  # intermittent
    rows += _rows("C", 0, [2, 3, 1, 4, 2, 5, 1])  # 7 periods: shorter than min history
    rows += _rows("D", 0, [3 + t % 4 for t in range(18)] + [0] * 12)  # dies at slot 18
    rows += _rows("E", 6, [2 + t % 5 for t in range(24)])  # starts at slot 6, ends 29
    rows += _rows("F", 0, [1 + (3 * t) % 5 for t in range(20)])  # delisted: ends at slot 19
    df = pd.DataFrame(rows)
    df["period_start"] = df["period_start"].astype("datetime64[ns]")
    return df.sort_values(["series_id", "period_start"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def cfg():
    return _cfg()


@pytest.fixture(scope="module")
def panel():
    return _small_panel()


@pytest.fixture(scope="module")
def origins(panel, cfg):
    return build_origins(panel, cfg)


def _slots(frame, series_id):
    d = frame.loc[frame["series_id"] == series_id, "origin_date"]
    return [int((x - START) / DAY) for x in d]


# --- origins -----------------------------------------------------------------


def test_features_origins_match_hand_count(origins):
    # step 2, H = 3, min history 8, inactive 4, grid starts at slot 0
    assert _slots(origins, "A") == [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    assert _slots(origins, "B") == [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    # E has history >= 8 from slot 13, so the first even slot is 14
    assert _slots(origins, "E") == [14, 16, 18, 20, 22, 24, 26]
    # F ends at slot 19: slot + 3 <= 19
    assert _slots(origins, "F") == [8, 10, 12, 14, 16]


def test_features_short_series_is_dropped(origins):
    assert "C" not in set(origins["series_id"])


def test_features_inactive_origins_are_dropped(origins):
    # D has demand up to slot 17, zeros from slot 18. Origin 20: last 4 periods
    # (17..20) still contain slot 17. Origin 22: slots 19..22 are all zero.
    slots = _slots(origins, "D")
    assert slots == [8, 10, 12, 14, 16, 18, 20]
    assert 22 not in slots


def test_features_window_past_the_end_is_dropped(panel):
    cfg1 = _cfg(step=1)
    slots = _slots(build_origins(panel, cfg1), "A")
    assert slots[-1] == 26  # 26 + 3 == 29, the last row: kept
    assert 27 not in slots  # 27 + 3 = 30 runs past the end: dropped
    assert slots == list(range(7, 27))  # history 8 first at slot 7


def test_features_origins_are_on_the_grid_and_sorted(origins):
    assert origins.columns.tolist() == ["series_id", "origin_date"]
    assert not origins.duplicated().any()
    assert origins.equals(
        origins.sort_values(["series_id", "origin_date"]).reset_index(drop=True)
    )
    assert all(s % 2 == 0 for s in _slots(origins, "A") + _slots(origins, "E"))


def test_features_origins_need_a_valid_panel(cfg):
    with pytest.raises(FcstError) as exc:
        build_origins(pd.DataFrame({"series_id": ["A"]}), cfg)
    assert exc.value.code == "E_DATA_COLUMNS_MISSING"


# --- features: hand-computed and reference ----------------------------------------


def test_features_names_follow_config(cfg):
    assert feature_names(cfg) == [
        *[f"lag_{k}" for k in (1, 2, 3, 4, 6, 8)],
        *[f"roll_mean_{w}" for w in (2, 3, 4, 8)],
        *[f"roll_std_{w}" for w in (2, 3, 4, 8)],
        *[f"roll_zero_share_{w}" for w in (2, 3, 4, 8)],
        "age_periods",
        "cal_month",
        "cal_annual_sin",
        "cal_annual_cos",
        "cal_weekend_periods",
    ]
    default = feature_names(load_config())
    assert "lag_7" in default and "roll_mean_28" in default and "roll_std_56" in default
    assert len(default) == len(set(default))


def test_features_names_reject_history_shorter_than_a_season():
    bad = _cfg(season=9, min_history=8)
    with pytest.raises(FcstError) as exc:
        feature_names(bad)
    assert exc.value.code == "E_CONFIG"
    assert "min_history_periods" in exc.value.message


def test_features_row_matches_hand_computation(panel, cfg):
    # Series A, origin slot 10 = 2024-01-11 (Thursday). Demand a_t = 5t mod 9:
    # a3..a13 = 6 2 7 3 8 4 0 5 1 6 2   (a10 = 5)
    key = pd.DataFrame({"series_id": ["A"], "origin_date": [START + 10 * DAY]})
    row = make_features(panel, key, cfg).iloc[0]
    expected = {
        # lag_k = a[10 - (k - 1)]
        "lag_1": 5, "lag_2": 0, "lag_3": 4, "lag_4": 8, "lag_6": 7, "lag_8": 6,
        # windows end at the origin: w=2 -> 5,0 ; w=3 -> 5,0,4 ; w=4 -> 5,0,4,8
        # w=8 -> 5,0,4,8,3,7,2,6
        "roll_mean_2": 2.5, "roll_mean_3": 3.0, "roll_mean_4": 4.25, "roll_mean_8": 4.375,
        "roll_std_2": 2.5,  # sqrt(((2.5)^2 + (2.5)^2) / 2)
        "roll_std_3": math.sqrt(14 / 3),  # deviations 2, -3, 1
        "roll_std_4": math.sqrt(8.1875),  # sum of squares 32.75 / 4
        "roll_std_8": math.sqrt(203 / 8 - 4.375**2),
        "roll_zero_share_2": 1 / 2, "roll_zero_share_3": 1 / 3,
        "roll_zero_share_4": 1 / 4, "roll_zero_share_8": 1 / 8,
        "age_periods": 11,  # slots 0..10
        # target window = slots 11..13 = Fri 12, Sat 13, Sun 14 January
        "cal_weekend_periods": 2,
        "cal_month": 1,
        # window midpoint = origin + 1.5 days = 12 January 12:00 -> day of year 12
        "cal_annual_sin": math.sin(2 * math.pi * 12 / 365.25),
        "cal_annual_cos": math.cos(2 * math.pi * 12 / 365.25),
    }
    assert set(expected) == set(feature_names(cfg))
    for name, value in expected.items():
        assert row[name] == pytest.approx(value, abs=1e-12), name


def _reference_features(panel, key_rows, cfg):
    """Slow, loop-based re-implementation of the feature definitions."""
    p = cfg["params"]
    horizon = p["horizon_periods"]
    out = []
    for sid, origin in key_rows:
        s = panel[panel["series_id"] == sid].set_index("period_start")["demand"]
        first = s.index.min()
        hist = [s[origin - i * DAY] for i in range(8)]  # hist[j] = lag_{j+1}
        row = {f"lag_{k}": hist[k - 1] for k in (1, 2, 3, 4, 6, 8)}
        for w in (2, 3, 4, 8):
            window = hist[:w]
            mean = sum(window) / w
            row[f"roll_mean_{w}"] = mean
            row[f"roll_std_{w}"] = math.sqrt(sum((v - mean) ** 2 for v in window) / w)
            row[f"roll_zero_share_{w}"] = sum(v == 0 for v in window) / w
        row["age_periods"] = (origin - first) / DAY + 1
        targets = [origin + (j + 1) * DAY for j in range(horizon)]
        row["cal_weekend_periods"] = sum(d.dayofweek >= 5 for d in targets)
        mid = origin + 0.5 * horizon * DAY
        row["cal_month"] = mid.month
        row["cal_annual_sin"] = math.sin(2 * math.pi * mid.dayofyear / 365.25)
        row["cal_annual_cos"] = math.cos(2 * math.pi * mid.dayofyear / 365.25)
        out.append(row)
    return pd.DataFrame(out)


def test_features_match_reference_loop_on_every_row(panel, origins, cfg):
    X = make_features(panel, origins, cfg)
    keys = list(zip(origins["series_id"], origins["origin_date"], strict=True))
    ref = _reference_features(panel, keys, cfg)
    assert len(X) == len(ref) == len(origins)
    np.testing.assert_allclose(
        X[feature_names(cfg)].to_numpy(), ref[feature_names(cfg)].to_numpy(), atol=1e-12
    )


def test_features_columns_equal_feature_names_and_keys_are_unique(panel, origins, cfg):
    X = make_features(panel, origins, cfg)
    assert list(X.columns) == feature_names(cfg)
    assert X.index.names == ["series_id", "origin_date"]
    assert X.index.is_unique
    assert X.index.equals(
        pd.MultiIndex.from_frame(origins[["series_id", "origin_date"]])
    )
    assert not X.isna().any().any()
    assert (X.dtypes == "float64").all()


def test_features_labels_match_hand_computation(panel, origins, cfg):
    key = pd.DataFrame({"series_id": ["A"], "origin_date": [START + 10 * DAY]})
    y = build_labels(panel, key, cfg)
    # (origin, origin + H] = slots 11, 12, 13 -> a = 1, 6, 2
    assert y.iloc[0] == 9.0
    # shifted windows would give a different number: [10..12] = 12, [12..14] = 15
    assert y.iloc[0] not in (_a(10) + _a(11) + _a(12), _a(12) + _a(13) + _a(14))
    assert y.name == "y"


def test_features_labels_cover_every_origin_and_are_clean(panel, origins, cfg):
    y = build_labels(panel, origins, cfg)
    assert len(y) == len(origins)
    assert y.index.is_unique
    assert not y.isna().any()
    assert (y >= 0).all()
    assert y.index.equals(make_features(panel, origins, cfg).index)
    # every label equals the plain sum over the next H periods
    for sid, origin, value in zip(
        origins["series_id"], origins["origin_date"], y.to_numpy(), strict=True
    ):
        s = panel[panel["series_id"] == sid].set_index("period_start")["demand"]
        assert value == sum(s[origin + (j + 1) * DAY] for j in range(3))


def test_features_labels_reject_a_window_past_the_end(panel, cfg):
    # series A ends at slot 29: origin 28 would need slots 29, 30, 31
    key = pd.DataFrame({"series_id": ["A"], "origin_date": [START + 28 * DAY]})
    with pytest.raises(FcstError) as exc:
        build_labels(panel, key, cfg)
    assert exc.value.code == "E_DATA_DATE_INVALID"
    # the boundary itself is fine: origin 26 + H = 29
    ok = pd.DataFrame({"series_id": ["A"], "origin_date": [START + 26 * DAY]})
    assert build_labels(panel, ok, cfg).iloc[0] == _a(27) + _a(28) + _a(29)


def test_features_live_origin_without_a_window_is_allowed(panel, cfg):
    last = pd.DataFrame({"series_id": ["A"], "origin_date": [START + 29 * DAY]})
    X = make_features(panel, last, cfg)  # forecast time: no future exists
    assert X.iloc[0]["lag_1"] == _a(29)


def test_features_bad_origins_are_rejected(panel, cfg):
    def code(series, day):
        key = pd.DataFrame({"series_id": [series], "origin_date": [START + day * DAY]})
        with pytest.raises(FcstError) as exc:
            make_features(panel, key, cfg)
        return exc.value.code

    assert code("ZZZ", 10) == "E_DATA_COLUMNS_MISSING"
    assert code("A", 40) == "E_DATA_DATE_INVALID"  # outside the active range
    assert code("A", 3) == "E_DATA_DATE_INVALID"  # not enough history for lag_8
    dup = pd.DataFrame({"series_id": ["A", "A"], "origin_date": [START + 10 * DAY] * 2})
    with pytest.raises(FcstError) as exc:
        make_features(panel, dup, cfg)
    assert exc.value.code == "E_DATA_DUPLICATE_KEY"


def test_features_are_deterministic_and_ignore_row_order(panel, origins, cfg):
    X1 = make_features(panel, origins, cfg)
    X2 = make_features(panel, origins, cfg)
    pd.testing.assert_frame_equal(X1, X2)
    shuffled = panel.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(make_features(shuffled, origins, cfg), X1)
    pd.testing.assert_series_equal(
        build_labels(shuffled, origins, cfg), build_labels(panel, origins, cfg)
    )
    pd.testing.assert_frame_equal(
        build_origins(shuffled, cfg), origins
    )


# --- features never depend on the target ---------------------------------------------


def test_features_do_not_call_build_labels(panel, origins, cfg, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("make_features must never build the target")

    monkeypatch.setattr(ffeat, "build_labels", boom)
    X = make_features(panel, origins, cfg)
    assert len(X) == len(origins)


def test_features_unchanged_when_the_whole_target_is_replaced(panel, origins, cfg):
    X = make_features(panel, origins, cfg)
    horizon = cfg["params"]["horizon_periods"]
    rng = np.random.default_rng(0)
    for date in sorted(origins["origin_date"].unique()):
        rows = origins[origins["origin_date"] == date]
        window = (panel["period_start"] > date) & (
            panel["period_start"] <= date + horizon * DAY
        )
        replaced = panel.copy()
        replaced.loc[window, "demand"] = rng.uniform(100, 900, int(window.sum()))
        y_new = build_labels(replaced, rows, cfg)
        y_old = build_labels(panel, rows, cfg)
        assert not np.allclose(y_new, y_old)  # the target really changed
        got = make_features(replaced, rows, cfg)
        pd.testing.assert_frame_equal(got, X.loc[got.index])


def test_features_ignore_extra_columns(panel, origins, cfg):
    noisy = panel.copy()
    noisy["y"] = 123.0
    noisy["target"] = -5.0
    noisy["units_sold_next_week"] = 99.0
    pd.testing.assert_frame_equal(
        make_features(noisy, origins, cfg), make_features(panel, origins, cfg)
    )


# --- leakage: forbidden columns -------------------------------------------------------


def test_leakage_forbidden_columns_are_not_in_the_matrix(panel, origins, cfg):
    X = make_features(panel, origins, cfg)
    assert set(X.columns).isdisjoint(forbidden_columns(cfg))
    assert_no_forbidden(X, cfg)
    assert_no_forbidden(feature_names(load_config()), load_config())
    for name in ("y", "demand", "period_start", "origin_date", "series_id", "sku_id"):
        assert name in forbidden_columns(cfg)


@pytest.mark.parametrize(
    "bad",
    ["y", "Demand", "series_id", "target_next", "label_x", "Future_sales", "next_week", "y_hat"],
)
def test_leakage_assert_no_forbidden_rejects(cfg, bad):
    with pytest.raises(FcstError) as exc:
        assert_no_forbidden(["lag_1", bad], cfg)
    assert exc.value.code == "E_LEAKAGE"
    assert bad in exc.value.message


def test_leakage_assert_no_forbidden_rejects_a_dataframe_with_the_target(panel, origins, cfg):
    X = make_features(panel, origins, cfg)
    X["y"] = build_labels(panel, origins, cfg)
    with pytest.raises(FcstError) as exc:
        assert_no_forbidden(X, cfg)
    assert exc.value.code == "E_LEAKAGE"


# --- leakage on the generated sample ----------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    big = load_config()
    raw, _ = fdata.generate(big["params"]["seed"], big)
    pnl = fdata.clean(raw, big)
    org = build_origins(pnl, big)
    X = make_features(pnl, org, big)
    y = build_labels(pnl, org, big)
    return {"cfg": big, "panel": pnl, "origins": org, "X": X, "y": y, "table": X.join(y)}


def test_features_sample_is_clean(sample):
    X, y, org = sample["X"], sample["y"], sample["origins"]
    assert len(org) > 500
    assert list(X.columns) == feature_names(sample["cfg"])
    assert X.index.is_unique and y.index.equals(X.index)
    assert not X.isna().any().any() and not y.isna().any()
    assert (y >= 0).all()


def test_leakage_perturb_check_passes_on_official_features(sample):
    report = perturb_future_check(sample["panel"], sample["origins"], sample["cfg"])
    assert report["passed"] is True
    assert report["n_rows"] == len(sample["origins"])
    assert report["n_origin_dates"] == sample["origins"]["origin_date"].nunique()


def test_leakage_perturb_check_passes_on_the_small_fixture(panel, origins, cfg):
    assert perturb_future_check(panel, origins, cfg)["passed"] is True


def test_leakage_correlation_fence_passes_on_official_features(sample):
    corr = check_correlation(sample["X"], sample["y"], sample["cfg"])
    limit = sample["cfg"]["params"]["leakage"]["max_abs_corr"]
    assert set(corr.index) == set(sample["X"].columns)
    assert corr.max() < limit - 0.2  # a clear margin, not a near miss


# --- negative controls built here -----------------------------------------------------------


def test_leakage_control_target_copy_is_caught_by_the_correlation_fence(sample):
    X, y, cfg = sample["X"].copy(), sample["y"], sample["cfg"]
    rng = np.random.default_rng(1)
    X["leaky_feature"] = 3.0 * y.to_numpy() + rng.normal(0, 0.05 * y.std(), len(y))
    with pytest.raises(FcstError) as exc:
        check_correlation(X, y, cfg)
    assert exc.value.code == "E_LEAKAGE"
    assert "leaky_feature" in exc.value.message
    assert "roll_mean_7" not in exc.value.message  # honest features are not blamed


def test_leakage_control_target_copy_with_series_offsets_is_still_caught(sample):
    X, y, cfg = sample["X"].copy(), sample["y"], sample["cfg"]
    sid = X.index.get_level_values("series_id")
    offsets = pd.Series(np.random.default_rng(2).uniform(0, 5000, sid.nunique()), index=sid.unique())
    X["leaky_feature"] = 2.0 * y.to_numpy() + offsets.loc[sid].to_numpy()
    with pytest.raises(FcstError) as exc:
        check_correlation(X, y, cfg)
    assert exc.value.code == "E_LEAKAGE"


def test_leakage_correlation_fence_works_without_series_index(sample):
    X, y, cfg = sample["X"].copy(), sample["y"], sample["cfg"]
    X["leaky_feature"] = -y.to_numpy() * 0.5
    flat_X, flat_y = X.reset_index(drop=True), y.reset_index(drop=True)
    with pytest.raises(FcstError) as exc:
        check_correlation(flat_X, flat_y, cfg)
    assert exc.value.code == "E_LEAKAGE"


def test_leakage_correlation_fence_needs_aligned_index(sample):
    with pytest.raises(FcstError) as exc:
        check_correlation(sample["X"], sample["y"].iloc[:-1], sample["cfg"])
    assert exc.value.code == "E_PREDICT_BAD_INPUT"


def _leaky_rolling_no_shift(window):
    """Rolling mean over ``window`` periods, read at origin + H instead of at the origin.

    This is the classic mistake: compute a rolling mean on the whole series and
    join it on the row, forgetting that the row's information stops at the origin.
    """

    def fn(panel, origins, cfg):
        horizon = cfg["params"]["horizon_periods"]
        honest = make_features(panel, origins, cfg)
        roll = (
            panel.sort_values(["series_id", "period_start"])
            .set_index(["series_id", "period_start"])
            .groupby(level="series_id")["demand"]
            .rolling(window)
            .mean()
            .droplevel(0)
        )
        keys = _origin_keys(origins)
        lookup = pd.MultiIndex.from_arrays(
            [keys["series_id"], keys["origin_date"] + horizon * DAY],
            names=["series_id", "period_start"],
        )
        out = honest.copy()
        out["roll_mean_unshifted"] = roll.reindex(lookup).to_numpy()
        return out

    return fn


def _origin_keys(origins):
    return origins[["series_id", "origin_date"]]


def test_leakage_control_unshifted_rolling_is_caught_by_the_perturbation_check(panel, origins, cfg):
    with pytest.raises(FcstError) as exc:
        perturb_future_check(panel, origins, cfg, feature_fn=_leaky_rolling_no_shift(3))
    assert exc.value.code == "E_LEAKAGE"
    assert "roll_mean_unshifted" in exc.value.message
    # only the leaky column is blamed, not the honest ones
    assert "lag_1" not in exc.value.message


def test_leakage_control_unshifted_rolling_on_the_sample(sample):
    with pytest.raises(FcstError) as exc:
        perturb_future_check(
            sample["panel"],
            sample["origins"].iloc[::20],
            sample["cfg"],
            feature_fn=_leaky_rolling_no_shift(7),
        )
    assert exc.value.code == "E_LEAKAGE"


def test_leakage_control_the_correlation_fence_alone_would_miss_it(sample):
    # the unshifted rolling mean looks like a normal feature to the fence: this is
    # why the perturbation check is the proof and the fence is only a guard
    fn = _leaky_rolling_no_shift(7)
    X = fn(sample["panel"], sample["origins"], sample["cfg"])
    assert check_correlation(X, sample["y"], sample["cfg"])["roll_mean_unshifted"] < 0.98


@pytest.mark.parametrize("ahead", [1, 3])
def test_leakage_control_lookahead_lag_is_caught(panel, origins, cfg, ahead):
    def fn(p, o, c):
        out = make_features(p, o, c)
        keys = _origin_keys(o)
        lookup = pd.MultiIndex.from_arrays(
            [keys["series_id"], keys["origin_date"] + ahead * DAY],
            names=["series_id", "period_start"],
        )
        demand = p.set_index(["series_id", "period_start"])["demand"]
        out["future_peek"] = demand.reindex(lookup).to_numpy()
        return out

    with pytest.raises(FcstError) as exc:
        perturb_future_check(panel, origins, cfg, feature_fn=fn)
    assert exc.value.code == "E_LEAKAGE"
    assert "future_peek" in exc.value.message


def test_leakage_perturb_check_flags_each_variant(panel, origins, cfg):
    # a feature that reads only the first future row: both the modify and the
    # delete variant must move it
    def fn(p, o, c):
        out = make_features(p, o, c)
        keys = _origin_keys(o)
        lookup = pd.MultiIndex.from_arrays(
            [keys["series_id"], keys["origin_date"] + DAY], names=["series_id", "period_start"]
        )
        out["peek"] = p.set_index(["series_id", "period_start"])["demand"].reindex(lookup).to_numpy()
        return out

    with pytest.raises(FcstError) as exc:
        perturb_future_check(panel, origins, cfg, feature_fn=fn)
    assert "delete+modify" in exc.value.message


# --- split -----------------------------------------------------------------------------------


def _dates(frame):
    if "origin_date" in frame.columns:
        return pd.DatetimeIndex(frame["origin_date"])
    return frame.index.get_level_values("origin_date")


def _grid_table(n_dates, n_series=3, step=7):
    rows = [
        {"series_id": f"S{s}", "origin_date": START + 56 * DAY + i * step * DAY, "y": float(i + s)}
        for i in range(n_dates)
        for s in range(n_series)
    ]
    return pd.DataFrame(rows)


def test_split_order_disjoint_and_gap_on_the_sample(sample):
    cfg = sample["cfg"]
    train, calib, test, purged = make_split(sample["table"], cfg)
    d_tr, d_ca, d_te = _dates(train), _dates(calib), _dates(test)
    assert d_tr.max() < d_ca.min() and d_ca.max() < d_te.min()
    horizon = cfg["params"]["horizon_periods"]
    assert d_ca.min() - d_tr.max() >= horizon * DAY
    assert d_te.min() - d_ca.max() >= horizon * DAY
    assert d_te.min() - d_tr.max() >= horizon * DAY
    gap = cfg["params"]["split"]["purge_gap_periods"]
    assert d_ca.min() - d_tr.max() >= gap * DAY
    # disjoint rows; train + calib + test + purged cover the table exactly once
    sides = [train, calib, test, purged]
    keys = [set(s.index) for s in sides]
    assert sum(len(k) for k in keys) == len(sample["table"])
    assert set().union(*keys) == set(sample["table"].index)
    for i in range(4):
        for j in range(i + 1, 4):
            assert keys[i].isdisjoint(keys[j])
    # purged origins sit between the sides, never inside test
    assert len(purged) > 0
    assert _dates(purged).max() < d_te.min()


def _half_up(x):
    return math.floor(x + 0.5 + 1e-9)


def test_split_sizes_follow_the_config(sample):
    cfg = copy.deepcopy(sample["cfg"])
    table = _grid_table(50)  # 50 distinct origin dates, 3 series each
    p = cfg["params"]
    removed = math.ceil(p["split"]["purge_gap_periods"] / p["origin_step_periods"]) - 1
    for fractions in ((0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.7, 0.15, 0.15)):
        p["split"].update(zip(("train", "calib", "test"), fractions, strict=True))
        train, calib, test, purged = make_split(table, cfg)
        # cumulative cuts of the 50 distinct dates, rounded half up
        cut1 = _half_up(fractions[0] * 50)
        cut2 = _half_up((fractions[0] + fractions[1]) * 50)
        n_train, n_calib, n_test = cut1, cut2 - cut1, 50 - cut2
        n = {k: len(set(_dates(v))) for k, v in zip(("tr", "ca", "te", "pu"), (train, calib, test, purged), strict=True)}
        assert n["te"] == n_test  # purging never touches test
        assert n["tr"] == n_train - removed
        assert n["ca"] == n_calib - removed
        assert n["pu"] == 2 * removed
        assert n["tr"] + n["ca"] + n["te"] + n["pu"] == 50
        # sizes track the configured shares within one origin date
        for got, share in ((n_train, fractions[0]), (n_calib, fractions[1]), (n_test, fractions[2])):
            assert abs(got - share * 50) <= 1
        # each side holds whole origin dates: all 3 series per date
        assert len(train) == 3 * n["tr"] and len(test) == 3 * n["te"]


def test_split_is_deterministic_and_ignores_row_order(sample):
    cfg, table = sample["cfg"], sample["table"]
    first = make_split(table, cfg)
    again = make_split(table, cfg)
    for a, b in zip(first, again, strict=True):
        pd.testing.assert_frame_equal(a, b)
    shuffled = table.sample(frac=1.0, random_state=11)
    for a, b in zip(first, make_split(shuffled, cfg), strict=True):
        assert set(a.index) == set(b.index)


def test_split_accepts_origin_date_as_a_column_or_an_index_level(sample):
    cfg = sample["cfg"]
    as_column = sample["table"].reset_index()
    by_index = make_split(sample["table"], cfg)
    by_column = make_split(as_column, cfg)
    for a, b in zip(by_index, by_column, strict=True):
        assert len(a) == len(b)


@pytest.mark.parametrize(
    "broken",
    [
        lambda t: t.drop(columns=["origin_date"]),
        lambda t: t.assign(origin_date=np.arange(len(t))),  # numbers are not dates
        lambda t: t.assign(origin_date=pd.NaT),
        lambda t: t.assign(origin_date=["not a date"] * len(t)),
        lambda t: t.iloc[0:0],
        lambda t: t.to_numpy(),
    ],
    ids=["no-column", "numeric", "all-NaT", "garbage", "empty", "not-a-frame"],
)
def test_split_without_a_time_axis_raises(cfg, broken):
    with pytest.raises(FcstError) as exc:
        make_split(broken(_grid_table(20)), cfg)
    assert exc.value.code == "E_SPLIT_NO_TIME_AXIS"


def test_split_rejects_a_table_with_only_a_period_column(cfg):
    table = _grid_table(20).rename(columns={"origin_date": "period_start"})
    with pytest.raises(FcstError) as exc:
        make_split(table, cfg)
    assert exc.value.code == "E_SPLIT_NO_TIME_AXIS"


def test_split_rejects_too_few_origin_dates(cfg):
    for n in (1, 2):
        with pytest.raises(FcstError) as exc:
            make_split(_grid_table(n), cfg)
        assert exc.value.code == "E_SPLIT_NO_TIME_AXIS"


def test_split_gap_holds_for_other_steps():
    cfg = _cfg(step=5, season=5, lead=3, review=4, min_history=10)  # H = 7, step 5
    table = _grid_table(30, step=5)
    train, calib, test, _ = make_split(table, cfg)
    for left, right in ((train, calib), (calib, test)):
        assert _dates(right).min() - _dates(left).max() >= 7 * DAY


# --- rolling-origin CV --------------------------------------------------------------------------


def _expected_folds(n):
    """Hand count for cv.n_folds=3, min_train_origins=8, H=14, step 7 (gap = 3 origins)."""
    return max(0, min(3, n - (8 - 1 + 3)))


@pytest.mark.parametrize("n", [12, 13, 14, 17, 20, 33])
def test_cv_effective_folds_follow_the_history(cfg_default, n):
    plan = rolling_origin_folds(_grid_table(n), cfg_default)
    assert plan.n_folds_requested == 3
    assert plan.n_folds_effective == _expected_folds(n) == len(plan)
    assert plan.summary() == {"n_folds_requested": 3, "n_folds_effective": len(plan)}


@pytest.fixture(scope="module")
def cfg_default():
    return load_config()


@pytest.mark.parametrize("n", [0, 5, 10, 11])
def test_cv_too_short_history_raises(cfg_default, n):
    table = _grid_table(max(n, 1))
    if n == 0:
        table = table.iloc[0:0]
    with pytest.raises(FcstError) as exc:
        rolling_origin_folds(table, cfg_default)
    expected = "E_SPLIT_NO_TIME_AXIS" if n == 0 else "E_CV_TOO_SHORT"
    assert exc.value.code == expected


@pytest.mark.parametrize("n", [12, 13, 20, 33])
def test_cv_fold_properties(cfg_default, n):
    cfg = cfg_default
    horizon = cfg["params"]["horizon_periods"]
    min_train = cfg["params"]["cv"]["min_train_origins"]
    table = _grid_table(n)
    plan = rolling_origin_folds(table, cfg)
    previous_train: set = set()
    previous_valid_end = None
    for fold in plan:
        train_dates, valid_dates = set(_dates(fold.train)), set(_dates(fold.valid))
        assert len(train_dates) >= min_train
        assert min(valid_dates) > max(train_dates) + horizon * DAY  # strictly after + H
        assert train_dates.isdisjoint(valid_dates)  # no origin on both sides
        assert set(fold.train.index).isdisjoint(set(fold.valid.index))
        assert previous_train <= train_dates  # expanding window
        if previous_valid_end is not None:
            assert min(valid_dates) > previous_valid_end  # validation blocks do not overlap
        previous_train, previous_valid_end = train_dates, max(valid_dates)
    assert previous_valid_end == table["origin_date"].max()  # latest origins are used
    sizes = {len(set(_dates(f.valid))) for f in plan}
    assert len(sizes) == 1  # blocks have equal size


def test_cv_hand_count_for_twenty_origins(cfg_default):
    # 20 origins, gap 3, min_train 8, 3 folds: room 10, block size 3,
    # valid blocks = ranks [11,14) [14,17) [17,20); train = ranks 0..8, 0..11, 0..14
    plan = rolling_origin_folds(_grid_table(20, n_series=1), cfg_default)
    assert [(len(f.train), len(f.valid)) for f in plan] == [(9, 3), (12, 3), (15, 3)]


def test_cv_fewer_folds_are_reported_not_hidden(cfg_default):
    plan = rolling_origin_folds(_grid_table(12), cfg_default)
    assert plan.n_folds_requested == 3 and plan.n_folds_effective == 2


def test_cv_more_requested_folds_are_capped_by_history(cfg_default):
    cfg = copy.deepcopy(cfg_default)
    cfg["params"]["cv"]["n_folds"] = 6
    assert rolling_origin_folds(_grid_table(14), cfg).n_folds_effective == 4
    assert rolling_origin_folds(_grid_table(40), cfg).n_folds_effective == 6


def test_cv_without_a_time_axis_raises(cfg_default):
    with pytest.raises(FcstError) as exc:
        rolling_origin_folds(_grid_table(20).drop(columns=["origin_date"]), cfg_default)
    assert exc.value.code == "E_SPLIT_NO_TIME_AXIS"


def test_cv_is_deterministic_and_ignores_row_order(cfg_default):
    table = _grid_table(25)
    a = rolling_origin_folds(table, cfg_default)
    b = rolling_origin_folds(table.sample(frac=1.0, random_state=5), cfg_default)
    for fa, fb in zip(a, b, strict=True):
        assert set(fa.train.index) == set(fb.train.index)
        assert set(fa.valid.index) == set(fb.valid.index)


def test_cv_runs_on_the_train_side_of_the_sample_split(sample):
    train, _, _, _ = make_split(sample["table"], sample["cfg"])
    plan = rolling_origin_folds(train, sample["cfg"])
    assert plan.n_folds_effective == 3
    last_train_date = _dates(train).max()
    assert all(_dates(f.valid).max() <= last_train_date for f in plan)


def test_cv_gap_holds_when_the_step_does_not_divide_the_horizon():
    cfg = _cfg(step=5, season=5, lead=3, review=4, min_history=10, cv__min_train_origins=4)
    plan = rolling_origin_folds(_grid_table(20, step=5), cfg)  # H = 7, step 5 -> gap 2 origins
    for fold in plan:
        assert _dates(fold.valid).min() > _dates(fold.train).max() + 7 * DAY
