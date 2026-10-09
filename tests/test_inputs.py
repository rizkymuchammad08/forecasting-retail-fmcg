"""Input tests: configuration (P1) and data (P2)."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fcst import data as fdata
from fcst.config import (
    ERROR_CODES,
    REQUIRED_FIELDS,
    FcstError,
    cost_rates,
    load_config,
    validate_config,
)

EXPECTED_CODES = (
    "E_CONFIG",
    "E_DATA_FILE_MISSING",
    "E_DATA_COLUMNS_MISSING",
    "E_DATA_DUPLICATE_KEY",
    "E_DATA_DATE_INVALID",
    "E_DATA_DEMAND_INVALID",
    "E_DATA_DEMAND_NEGATIVE",
    "E_DATA_GAP_TOO_LONG",
    "E_LEAKAGE",
    "E_SPLIT_NO_TIME_AXIS",
    "E_CV_TOO_SHORT",
    "E_EVAL_ZERO_DENOMINATOR",
    "E_PREDICT_BAD_INPUT",
    "E_RENDER_MISSING_KEY",
)


@pytest.fixture
def cfg():
    return load_config()


def _set(cfg, path, value):
    node = cfg
    *parents, leaf = path.split(".")
    for part in parents:
        node = node[part]
    node[leaf] = value


def _delete(cfg, path):
    node = cfg
    *parents, leaf = path.split(".")
    for part in parents:
        node = node[part]
    del node[leaf]


def _assert_config_error(cfg, field):
    with pytest.raises(FcstError) as exc:
        validate_config(cfg)
    assert exc.value.code == "E_CONFIG"
    assert field in exc.value.message


# --- FcstError and codes ---------------------------------------------------


def test_config_error_codes_match_contract():
    assert ERROR_CODES == EXPECTED_CODES


def test_config_fcsterror_carries_code_and_message():
    err = FcstError("E_LEAKAGE", "something")
    assert err.code == "E_LEAKAGE"
    assert err.message == "something"
    assert isinstance(err, Exception)


def test_config_fcsterror_rejects_unknown_code():
    with pytest.raises(ValueError):
        FcstError("E_NOT_A_CODE", "x")


# --- loading ---------------------------------------------------------------


def test_config_default_file_loads_and_validates(cfg):
    assert validate_config(cfg) is None
    assert cfg["params"]["horizon_periods"] == (
        cfg["business"]["lead_time_periods"] + cfg["business"]["review_period_periods"]
    )


def test_config_load_works_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config()["params"]["seed"] >= 0


def test_config_load_returns_fresh_dict_each_time():
    first = load_config()
    first["params"]["seed"] = 999
    assert load_config()["params"]["seed"] != 999


def test_config_load_custom_path(tmp_path, cfg):
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert load_config(path) == cfg


def test_config_load_missing_file_raises(tmp_path):
    with pytest.raises(FcstError) as exc:
        load_config(tmp_path / "nope.yaml")
    assert exc.value.code == "E_CONFIG"


def test_config_load_invalid_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(FcstError) as exc:
        load_config(path)
    assert exc.value.code == "E_CONFIG"


def test_config_load_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(FcstError) as exc:
        load_config(path)
    assert exc.value.code == "E_CONFIG"


# --- required keys and types -----------------------------------------------


@pytest.mark.parametrize("path", sorted(REQUIRED_FIELDS))
def test_config_missing_key_is_rejected(cfg, path):
    _delete(cfg, path)
    _assert_config_error(cfg, path)


@pytest.mark.parametrize("section", ["business", "params"])
def test_config_missing_section_is_rejected(cfg, section):
    del cfg[section]
    with pytest.raises(FcstError) as exc:
        validate_config(cfg)
    assert exc.value.code == "E_CONFIG"


def test_config_non_mapping_is_rejected():
    with pytest.raises(FcstError) as exc:
        validate_config(["not", "a", "mapping"])
    assert exc.value.code == "E_CONFIG"


@pytest.mark.parametrize("path", sorted(REQUIRED_FIELDS))
def test_config_null_value_is_rejected(cfg, path):
    _set(cfg, path, None)
    _assert_config_error(cfg, path)


@pytest.mark.parametrize(
    "path", sorted(p for p, kind in REQUIRED_FIELDS.items() if kind in ("int", "num"))
)
def test_config_bool_is_not_a_number(cfg, path):
    _set(cfg, path, True)
    _assert_config_error(cfg, path)


@pytest.mark.parametrize(
    "path", sorted(p for p, kind in REQUIRED_FIELDS.items() if kind == "int")
)
def test_config_integer_field_rejects_float(cfg, path):
    _set(cfg, path, 7.5)
    _assert_config_error(cfg, path)


# --- value checks (each names the field) -------------------------------------

BAD_VALUES = [
    ("business.avg_unit_cost", 0, "business.avg_unit_cost"),
    ("business.avg_unit_cost", -1.0, "business.avg_unit_cost"),
    ("business.avg_unit_price", 5.0, "business.avg_unit_price"),
    ("business.avg_unit_price", 6.0, "business.avg_unit_price"),
    ("business.holding_cost_rate_annual", 0, "business.holding_cost_rate_annual"),
    ("business.holding_cost_rate_annual", -0.1, "business.holding_cost_rate_annual"),
    ("business.salvage_ratio", -0.1, "business.salvage_ratio"),
    ("business.salvage_ratio", 1.1, "business.salvage_ratio"),
    ("business.overstock_writeoff_share", -0.1, "business.overstock_writeoff_share"),
    ("business.overstock_writeoff_share", 1.1, "business.overstock_writeoff_share"),
    ("business.status_quo_buffer", -0.01, "business.status_quo_buffer"),
    ("business.lead_time_periods", -1, "business.lead_time_periods"),
    ("business.review_period_periods", 0, "business.review_period_periods"),
    ("business.periods_per_year", 0, "business.periods_per_year"),
    ("params.horizon_periods", 13, "params.horizon_periods"),
    ("business.lead_time_periods", 8, "params.horizon_periods"),
    ("params.seed", -1, "params.seed"),
    ("params.season_length_periods", 0, "params.season_length_periods"),
    ("params.origin_step_periods", 0, "params.origin_step_periods"),
    ("params.data.path", "", "params.data.path"),
    ("params.data.max_gap_periods", -1, "params.data.max_gap_periods"),
    ("params.data.synthetic.n_series", 9, "params.data.synthetic.n_series"),
    ("params.data.synthetic.n_periods", 98, "params.data.synthetic.n_periods"),
    ("params.domain.min_history_periods", 0, "params.domain.min_history_periods"),
    ("params.domain.inactive_periods", 0, "params.domain.inactive_periods"),
    ("params.leakage.max_abs_corr", 0, "params.leakage.max_abs_corr"),
    ("params.leakage.max_abs_corr", 1.5, "params.leakage.max_abs_corr"),
    ("params.split.train", 0.7, "params.split"),
    ("params.split.calib", 0, "params.split.calib"),
    ("params.split.purge_gap_periods", 13, "params.split.purge_gap_periods"),
    ("params.cv.n_folds", 1, "params.cv.n_folds"),
    ("params.cv.min_train_origins", 0, "params.cv.min_train_origins"),
    ("params.models.candidate", "seasonal_naive", "params.models.candidate"),
    ("params.models.primary_metric", "mae", "params.models.primary_metric"),
    ("params.calibration.scale_floor", 0, "params.calibration.scale_floor"),
    ("params.tradeoff.taus", [], "params.tradeoff.taus"),
    ("params.tradeoff.taus", [0.5, 1.0], "params.tradeoff.taus"),
    ("params.tradeoff.taus", [0.0, 0.5], "params.tradeoff.taus"),
    ("params.tradeoff.taus", [0.5, -0.1], "params.tradeoff.taus"),
    ("params.evaluation.ci_level", 1.0, "params.evaluation.ci_level"),
    ("params.evaluation.ci_level", 0, "params.evaluation.ci_level"),
    ("params.evaluation.n_boot", 0, "params.evaluation.n_boot"),
    (
        "params.evaluation.calibration_tolerance",
        -0.1,
        "params.evaluation.calibration_tolerance",
    ),
    (
        "params.evaluation.calibration_tolerance",
        1.0,
        "params.evaluation.calibration_tolerance",
    ),
    ("params.report.min_test_series", 0, "params.report.min_test_series"),
    ("params.report.min_test_obs", 0, "params.report.min_test_obs"),
    (
        "params.schema.required_columns",
        ["sku_id", "location_id", "series_id", "period_start"],
        "params.schema.required_columns",
    ),
    (
        "params.schema.required_columns",
        ["sku_id", "location_id", "series_id", "period_start", "demand", "demand"],
        "params.schema.required_columns",
    ),
]


@pytest.mark.parametrize(("path", "value", "field"), BAD_VALUES)
def test_config_bad_value_is_rejected_and_names_field(cfg, path, value, field):
    _set(cfg, path, copy.deepcopy(value))
    _assert_config_error(cfg, field)


# --- boundary values that must stay valid ------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("business.lead_time_periods", 0),
        ("business.salvage_ratio", 0),
        ("business.salvage_ratio", 1),
        ("business.overstock_writeoff_share", 0),
        ("business.overstock_writeoff_share", 1),
        ("business.status_quo_buffer", 0),
        ("params.data.max_gap_periods", 0),
        ("params.leakage.max_abs_corr", 1),
        ("params.evaluation.calibration_tolerance", 0),
        ("params.cv.n_folds", 2),
        ("params.cv.min_train_origins", 1),
        ("params.data.synthetic.n_series", 10),
        ("params.data.synthetic.n_periods", 99),
        ("params.models.primary_metric", "rmsse"),
    ],
)
def test_config_boundary_value_is_accepted(cfg, path, value):
    if path == "business.lead_time_periods":
        cfg["params"]["horizon_periods"] = value + cfg["business"]["review_period_periods"]
        cfg["params"]["split"]["purge_gap_periods"] = cfg["params"]["horizon_periods"]
    _set(cfg, path, value)
    validate_config(cfg)


def test_config_purge_gap_equal_to_horizon_is_accepted(cfg):
    cfg["params"]["split"]["purge_gap_periods"] = cfg["params"]["horizon_periods"]
    validate_config(cfg)


def test_config_split_sum_uses_float_tolerance(cfg):
    cfg["params"]["split"].update({"train": 0.7, "calib": 0.2, "test": 0.1})
    validate_config(cfg)


# --- structure: no business values or tau_star under params ----------------


@pytest.mark.parametrize(
    "path",
    [
        "params.avg_unit_price",
        "params.lead_time_periods",
        "params.business",
        "params.tau_star",
        "params.evaluation.tau_star",
        "params.data.synthetic.avg_unit_cost",
    ],
)
def test_config_business_keys_and_tau_star_under_params_are_rejected(cfg, path):
    _set(cfg, path, 1.0)
    _assert_config_error(cfg, path)


def test_config_default_file_has_no_stored_tau_star(cfg):
    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from keys(value)

    assert not {k for k in keys(cfg) if "tau_star" in k or "tau*" in k}


# --- derived decision quantities ---------------------------------------------


def test_config_costs_and_tau_star_follow_the_formulas(cfg):
    b = cfg["business"]
    c_under, c_over = cost_rates(b)
    assert c_under == pytest.approx(b["avg_unit_price"] - b["avg_unit_cost"])
    holding = (
        b["avg_unit_cost"]
        * b["holding_cost_rate_annual"]
        * b["review_period_periods"]
        / b["periods_per_year"]
    )
    writeoff = b["avg_unit_cost"] * (1 - b["salvage_ratio"]) * b["overstock_writeoff_share"]
    assert c_over == pytest.approx(holding + writeoff)
    assert 0 < c_under / (c_under + c_over) < 1


def test_config_tau_star_is_in_open_unit_interval_across_valid_extremes(cfg):
    cfg["business"].update(
        {"avg_unit_price": 6.0001, "overstock_writeoff_share": 1.0, "salvage_ratio": 0.0}
    )
    validate_config(cfg)
    c_under, c_over = cost_rates(cfg["business"])
    assert 0 < c_under / (c_under + c_over) < 1


# =============================================================================
# Data (P2)
# =============================================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
MAX_SAMPLE_BYTES = 1_500_000  # 1.5 MB repository constraint for data/sample.csv


@pytest.fixture(scope="module")
def default_cfg():
    return load_config()


@pytest.fixture(scope="module")
def generated(default_cfg):
    return fdata.generate(default_cfg["params"]["seed"], default_cfg)


def _tiny_panel(n=20, series=("S1", "S2")):
    rows = []
    for k, sku in enumerate(series):
        for d in range(n):
            rows.append(
                {
                    "sku_id": sku,
                    "location_id": "L1",
                    "period_start": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).strftime(
                        "%Y-%m-%d"
                    ),
                    "demand": float(d + k),
                }
            )
    return pd.DataFrame(rows)


def _with_series_id(df):
    out = df.copy()
    out["series_id"] = fdata.make_series_id(out["sku_id"], out["location_id"])
    return out


def _broken(kind):
    df = _with_series_id(_tiny_panel())
    if kind == "E_DATA_COLUMNS_MISSING":
        return df.drop(columns=["demand"])
    if kind == "E_DATA_DUPLICATE_KEY":
        return pd.concat([df, df.iloc[[3]]], ignore_index=True)
    if kind == "E_DATA_DATE_INVALID":
        df.loc[5, "period_start"] = "2024-13-45"
        return df
    if kind == "E_DATA_DEMAND_INVALID":
        df["demand"] = df["demand"].astype(object)
        df.loc[4, "demand"] = "abc"
        return df
    if kind == "E_DATA_DEMAND_NEGATIVE":
        df.loc[4, "demand"] = -3.0
        return df
    if kind == "E_DATA_GAP_TOO_LONG":
        return df.drop(index=[5, 6, 7, 8]).reset_index(drop=True)  # 4 missing > max gap 3
    raise AssertionError(kind)


DATA_CODES = [
    "E_DATA_COLUMNS_MISSING",
    "E_DATA_DUPLICATE_KEY",
    "E_DATA_DATE_INVALID",
    "E_DATA_DEMAND_INVALID",
    "E_DATA_DEMAND_NEGATIVE",
    "E_DATA_GAP_TOO_LONG",
]


# --- validate ----------------------------------------------------------------


def test_data_generated_sample_passes_validate(generated, default_cfg):
    df, _ = generated
    assert fdata.validate(df, default_cfg) == []
    assert fdata.validate(fdata.clean(df, default_cfg), default_cfg) == []


def test_data_valid_tiny_panel_passes_validate(default_cfg):
    assert fdata.validate(_tiny_panel(), default_cfg) == []


@pytest.mark.parametrize("code", DATA_CODES)
def test_data_broken_fixture_gives_exactly_that_code(default_cfg, code):
    assert fdata.validate(_broken(code), default_cfg) == [code]


@pytest.mark.parametrize("code", DATA_CODES)
def test_data_clean_raises_the_same_code(default_cfg, code):
    with pytest.raises(FcstError) as exc:
        fdata.clean(_broken(code), default_cfg)
    assert exc.value.code == code


@pytest.mark.parametrize("code", DATA_CODES)
def test_data_load_panel_raises_the_same_code(default_cfg, tmp_path, code):
    path = tmp_path / "panel.csv"
    _broken(code).to_csv(path, index=False)
    with pytest.raises(FcstError) as exc:
        fdata.load_panel(path, default_cfg)
    assert exc.value.code == code


def test_data_validate_never_raises_on_garbage(default_cfg):
    assert fdata.validate(pd.DataFrame(), default_cfg) == ["E_DATA_COLUMNS_MISSING"]
    assert fdata.validate(None, default_cfg) == ["E_DATA_COLUMNS_MISSING"]
    assert fdata.validate(_tiny_panel().iloc[0:0], default_cfg) == ["E_DATA_COLUMNS_MISSING"]


def test_data_blank_identifier_is_rejected(default_cfg):
    df = _tiny_panel()
    df.loc[2, "sku_id"] = ""
    assert fdata.validate(df, default_cfg) == ["E_DATA_COLUMNS_MISSING"]


def test_data_validate_reports_several_codes_in_contract_order(default_cfg):
    df = _with_series_id(_tiny_panel())
    df.loc[4, "demand"] = -1.0
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    assert fdata.validate(df, default_cfg) == ["E_DATA_DUPLICATE_KEY", "E_DATA_DEMAND_NEGATIVE"]


def test_data_gap_equal_to_max_gap_is_allowed_and_filled_with_zero(default_cfg):
    df = _tiny_panel().drop(index=[5, 6, 7]).reset_index(drop=True)  # 3 missing == max gap
    assert fdata.validate(df, default_cfg) == []
    out = fdata.clean(df, default_cfg)
    s1 = out[out["series_id"] == "S1@L1"]
    assert len(s1) == 20
    assert s1["demand"].iloc[5:8].tolist() == [0.0, 0.0, 0.0]


def test_data_off_grid_date_is_invalid(default_cfg):
    df = _tiny_panel()
    df.loc[3, "period_start"] = "2024-01-04T12:00:00"
    assert fdata.validate(df, default_cfg) == ["E_DATA_DATE_INVALID"]


def test_data_infinite_demand_is_invalid(default_cfg):
    df = _tiny_panel()
    df["demand"] = df["demand"].astype(object)
    df.loc[2, "demand"] = "inf"
    assert fdata.validate(df, default_cfg) == ["E_DATA_DEMAND_INVALID"]


# --- clean ---------------------------------------------------------------------


def test_data_clean_is_idempotent(generated, default_cfg):
    df, _ = generated
    once = fdata.clean(df, default_cfg)
    twice = fdata.clean(once, default_cfg)
    pd.testing.assert_frame_equal(once, twice)


def test_data_clean_drops_unlisted_columns(default_cfg):
    df = _with_series_id(_tiny_panel())
    df["stockout_flag"] = 1
    df["revenue"] = 99.0
    out = fdata.clean(df, default_cfg)
    assert list(out.columns) == default_cfg["params"]["schema"]["required_columns"]


def test_data_clean_builds_series_id_when_absent(default_cfg):
    out = fdata.clean(_tiny_panel(), default_cfg)
    assert set(out["series_id"]) == {"S1@L1", "S2@L1"}
    assert fdata.make_series_id("a", "b") != fdata.make_series_id("b", "a")
    assert fdata.make_series_id("a", "b") == fdata.make_series_id("a", "b")


def test_data_clean_keeps_existing_series_id(default_cfg):
    df = _tiny_panel()
    df["series_id"] = "custom-" + df["sku_id"]
    assert set(fdata.clean(df, default_cfg)["series_id"]) == {"custom-S1", "custom-S2"}


def test_data_clean_types_sorting_and_no_input_mutation(default_cfg):
    df = _tiny_panel().sample(frac=1.0, random_state=1).reset_index(drop=True)
    df = df.astype({"demand": str})
    before = df.copy()
    out = fdata.clean(df, default_cfg)
    pd.testing.assert_frame_equal(df, before)
    assert str(out["period_start"].dtype) == "datetime64[ns]"
    assert out["demand"].dtype == np.float64
    keys = list(zip(out["series_id"], out["period_start"]))
    assert keys == sorted(keys)


def test_data_clean_fills_generated_gaps_back_to_the_full_grid(generated, default_cfg):
    df, truth = generated
    out = fdata.clean(df, default_cfg)
    table = truth["table"]
    assert len(out) > len(df)
    assert len(out) == len(table)
    assert (out["series_id"].to_numpy() == table["series_id"].to_numpy()).all()
    assert (out["period_start"].to_numpy() == table["period_start"].to_numpy()).all()
    added = out.merge(df[["series_id", "period_start"]], how="left", indicator=True)
    assert (added.loc[added["_merge"] == "left_only", "demand"] == 0).all()


# --- hash and banner -----------------------------------------------------------


def test_data_hash_is_deterministic_and_order_independent(generated, default_cfg):
    df = fdata.clean(generated[0], default_cfg)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert fdata.data_hash(df) == fdata.data_hash(df.copy())
    assert fdata.data_hash(df) == fdata.data_hash(shuffled)
    assert fdata.data_hash(df) == fdata.data_hash(df[df.columns[::-1]])


def test_data_hash_ignores_unlisted_columns_and_detects_changes(default_cfg):
    df = fdata.clean(_tiny_panel(), default_cfg)
    extra = df.assign(revenue=1.0)
    assert fdata.data_hash(extra) == fdata.data_hash(df)
    changed = df.copy()
    changed.loc[0, "demand"] += 1
    assert fdata.data_hash(changed) != fdata.data_hash(df)
    shorter = df.iloc[:-1]
    assert fdata.data_hash(shorter) != fdata.data_hash(df)


def test_data_hash_same_seed_same_hash_and_different_seed_different_hash(default_cfg):
    a, _ = fdata.generate(5, default_cfg)
    b, _ = fdata.generate(5, default_cfg)
    c, _ = fdata.generate(6, default_cfg)
    assert fdata.data_hash(a) == fdata.data_hash(b)
    assert fdata.data_hash(a) != fdata.data_hash(c)


def test_data_banner_shows_for_synthetic_only():
    assert "SYNTHETIC" in fdata.banner("synthetic")
    assert fdata.banner("real") == ""
    with pytest.raises(ValueError):
        fdata.banner("other")


# --- generator -----------------------------------------------------------------


def _span(df):
    g = df.groupby("series_id")["period_start"]
    return g.min(), g.max()


def test_data_generator_matches_configured_size(generated, default_cfg):
    df, _ = generated
    size = default_cfg["params"]["data"]["synthetic"]
    first, last = _span(df)
    assert df["series_id"].nunique() == size["n_series"]
    assert ((last - first).dt.days + 1).max() == size["n_periods"]
    assert (last - first).dt.days.max() + 1 <= size["n_periods"]
    assert (df["demand"] >= 0).all()
    assert (df["demand"] == df["demand"].round()).all()


def test_data_generator_size_follows_config_only():
    cfg = load_config()
    cfg["params"]["data"]["synthetic"] = {"n_series": 12, "n_periods": 150}
    df, truth = fdata.generate(1, cfg)
    assert df["series_id"].nunique() == 12
    first, last = _span(df)
    assert ((last - first).dt.days + 1).max() == 150
    assert len(truth["group"]) == 12


def test_data_generator_has_three_behaviour_groups(generated):
    df, truth = generated
    groups = pd.Series(truth["group"])
    assert set(groups) == {"high_volume", "low_volume", "intermittent"}
    mean_by_group = (
        df.assign(group=df["series_id"].map(truth["group"])).groupby("group")["demand"].mean()
    )
    assert mean_by_group["high_volume"] > mean_by_group["low_volume"] > mean_by_group["intermittent"]
    zero_share = (
        df.assign(group=df["series_id"].map(truth["group"]))
        .assign(zero=lambda d: d["demand"] == 0)
        .groupby("group")["zero"]
        .mean()
    )
    assert zero_share["intermittent"] > 0.3
    assert zero_share["high_volume"] < 0.05


def test_data_generator_has_outliers_above_the_true_upper_quantile(generated):
    df, truth = generated
    outliers = truth["outliers"]
    assert len(outliers) >= 3
    top = truth["quantile_cols"][0.99]
    merged = outliers.merge(df, on=["series_id", "period_start"]).merge(
        truth["table"], on=["series_id", "period_start"]
    )
    assert len(merged) == len(outliers)
    assert (merged["demand"] > merged[top]).all()


def test_data_generator_has_new_and_delisted_series(generated):
    df, truth = generated
    first, last = _span(df)
    assert truth["new_series"] and truth["delisted"]
    assert not set(truth["new_series"]) & set(truth["delisted"])
    assert (first[truth["new_series"]] > first.min()).all()
    assert (last[truth["delisted"]] < last.max()).all()
    others = first.index.difference(truth["new_series"] + truth["delisted"])
    assert (first[others] == first.min()).all()
    assert (last[others] == last.max()).all()


def test_data_generator_truth_quantiles_are_sensible(generated, default_cfg):
    df, truth = generated
    table = truth["table"]
    taus = sorted(truth["quantile_cols"])
    cols = [truth["quantile_cols"][t] for t in taus]
    assert (table["mean"] > 0).all()
    assert (np.diff(table[cols].to_numpy(), axis=1) >= 0).all()
    assert truth["tau_star"] in truth["quantile_cols"]
    full = fdata.clean(df, default_cfg).merge(table, on=["series_id", "period_start"])
    outlier_keys = set(zip(truth["outliers"]["series_id"], truth["outliers"]["period_start"]))
    keep = [k not in outlier_keys for k in zip(full["series_id"], full["period_start"])]
    full = full[keep]
    coverage = (full["demand"] <= full[truth["quantile_cols"][0.9]]).mean()
    assert 0.88 <= coverage <= 0.96
    assert abs(full["demand"].mean() / full["mean"].mean() - 1) < 0.03


def test_data_generator_is_deterministic_per_seed(default_cfg):
    a, ta = fdata.generate(3, default_cfg)
    b, tb = fdata.generate(3, default_cfg)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta["table"], tb["table"])


def test_data_generate_twice_gives_same_hash_and_csv_is_small_enough(default_cfg):
    a, _ = fdata.generate(default_cfg["params"]["seed"], default_cfg)
    b, _ = fdata.generate(default_cfg["params"]["seed"], default_cfg)
    assert fdata.data_hash(fdata.clean(a, default_cfg)) == fdata.data_hash(fdata.clean(b, default_cfg))
    payload = fdata.to_csv_bytes(a, default_cfg)
    assert payload == fdata.to_csv_bytes(b, default_cfg)
    assert len(payload) <= MAX_SAMPLE_BYTES


# --- make_sample ---------------------------------------------------------------


def test_data_make_sample_keeps_full_history_and_is_deterministic(generated, default_cfg):
    full = fdata.clean(generated[0], default_cfg)
    cfg = copy.deepcopy(default_cfg)
    cfg["params"]["data"]["synthetic"]["n_series"] = 12
    a = fdata.make_sample(full, cfg)
    b = fdata.make_sample(full, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert a["series_id"].nunique() == 12
    for sid, part in a.groupby("series_id"):
        source = full[full["series_id"] == sid].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), source)


def test_data_make_sample_depends_on_seed(generated, default_cfg):
    full = fdata.clean(generated[0], default_cfg)
    cfg = copy.deepcopy(default_cfg)
    cfg["params"]["data"]["synthetic"]["n_series"] = 12
    other = copy.deepcopy(cfg)
    other["params"]["seed"] = cfg["params"]["seed"] + 1
    assert set(fdata.make_sample(full, cfg)["series_id"]) != set(
        fdata.make_sample(full, other)["series_id"]
    )


def test_data_make_sample_never_exceeds_available_series(generated, default_cfg):
    full = fdata.clean(generated[0], default_cfg)
    cfg = copy.deepcopy(default_cfg)
    cfg["params"]["data"]["synthetic"]["n_series"] = 10_000
    assert fdata.make_sample(full, cfg)["series_id"].nunique() == full["series_id"].nunique()


# --- load_panel ----------------------------------------------------------------


def test_data_missing_real_file_raises_and_never_falls_back(default_cfg, tmp_path):
    with pytest.raises(FcstError) as exc:
        fdata.load_panel(tmp_path / "absent.csv", default_cfg)
    assert exc.value.code == "E_DATA_FILE_MISSING"
    with pytest.raises(FcstError) as exc:
        fdata.load_panel(tmp_path, default_cfg)  # a directory is not a data file
    assert exc.value.code == "E_DATA_FILE_MISSING"
    with pytest.raises(FcstError) as exc:
        fdata.load_panel(default_cfg["params"]["data"]["path"], default_cfg)
    assert exc.value.code == "E_DATA_FILE_MISSING"


def test_data_load_panel_round_trips_the_generator_output(generated, default_cfg, tmp_path):
    df, _ = generated
    path = tmp_path / "panel.csv"
    path.write_bytes(fdata.to_csv_bytes(df, default_cfg))
    loaded = fdata.load_panel(path, default_cfg)
    pd.testing.assert_frame_equal(loaded, fdata.clean(df, default_cfg))
    assert fdata.data_hash(loaded) == fdata.data_hash(fdata.clean(df, default_cfg))


def test_data_load_panel_never_reads_unlisted_columns(default_cfg, tmp_path):
    df = _with_series_id(_tiny_panel())
    df["stockout_flag"] = 1
    df["revenue"] = 123.0
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    loaded = fdata.load_panel(path, default_cfg)
    assert list(loaded.columns) == default_cfg["params"]["schema"]["required_columns"]


def test_data_load_panel_empty_file_raises_columns_missing(default_cfg, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(FcstError) as exc:
        fdata.load_panel(path, default_cfg)
    assert exc.value.code == "E_DATA_COLUMNS_MISSING"


# --- CLI (runs in a temp dir; never touches the repository data/ folder) -----------


def _run_cli(args, cwd):
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    return subprocess.run(
        [sys.executable, "-m", "fcst.data", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )


def test_data_cli_write_then_check_sample(tmp_path):
    out = tmp_path / "sample.csv"
    wrote = _run_cli(["--write-sample", "--out", str(out)], tmp_path)
    assert wrote.returncode == 0, wrote.stderr
    assert "SYNTHETIC" in wrote.stdout
    assert 0 < out.stat().st_size <= MAX_SAMPLE_BYTES
    checked = _run_cli(["--check-sample", "--out", str(out)], tmp_path)
    assert checked.returncode == 0, checked.stderr
    assert "SYNTHETIC" in checked.stdout


def test_data_cli_check_sample_fails_on_stale_file(tmp_path):
    out = tmp_path / "sample.csv"
    assert _run_cli(["--write-sample", "--out", str(out)], tmp_path).returncode == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    lines[5] = lines[5].rsplit(",", 1)[0] + ",9999"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _run_cli(["--check-sample", "--out", str(out)], tmp_path)
    assert result.returncode == 1
    assert "make sample" in result.stderr


def test_data_cli_requires_a_mode(tmp_path):
    assert _run_cli([], tmp_path).returncode != 0
