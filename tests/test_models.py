"""Model, calibration and accuracy tests (P4)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import fcst.models as fm
from fcst import data as fdata
from fcst import evaluate as ev
from fcst.config import FcstError, load_config, validate_config
from fcst.features import (
    build_labels,
    build_origins,
    feature_names,
    make_features,
    make_split,
)

DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2024-01-01")


def _code(exc) -> str:
    return exc.value.code


# --- fixtures ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def world(cfg):
    """Default synthetic panel, its truth, the P3 table and a trained bundle."""
    raw, truth = fdata.generate(cfg["params"]["seed"], cfg)
    panel = fdata.clean(raw, cfg)
    origins = build_origins(panel, cfg)
    X = make_features(panel, origins, cfg)
    table = X.join(build_labels(panel, origins, cfg))
    bundle = fm.train_bundle(table, panel, cfg)
    return {"panel": panel, "truth": truth, "table": table, "bundle": bundle}


@pytest.fixture(scope="module")
def split(world, cfg):
    return make_split(world["table"], cfg)


def _names(cfg):
    return feature_names(cfg)


# --- baseline ---------------------------------------------------------------------------


def _small_cfg(season=2, lead=1, review=2):
    c = copy.deepcopy(load_config())
    c["business"]["lead_time_periods"] = lead
    c["business"]["review_period_periods"] = review
    p = c["params"]
    p["horizon_periods"] = lead + review
    p["season_length_periods"] = season
    p["origin_step_periods"] = 2
    p["domain"] = {"min_history_periods": 8, "inactive_periods": 4}
    p["split"]["purge_gap_periods"] = lead + review
    validate_config(c)
    return c


def _panel(values, series_id="S1"):
    return pd.DataFrame(
        {
            "sku_id": series_id,
            "location_id": "L1",
            "series_id": series_id,
            "period_start": [START + t * DAY for t in range(len(values))],
            "demand": [float(v) for v in values],
        }
    )


def _x_at(panel, cfg, slots, series_id="S1"):
    origins = pd.DataFrame(
        {"series_id": series_id, "origin_date": [START + s * DAY for s in slots]}
    )
    return make_features(panel, origins, cfg)


def test_seasonal_naive_matches_hand_count_and_shift_is_a_whole_number_of_seasons():
    cfg = _small_cfg(season=2, lead=1, review=2)  # H = 3, S = 2 -> shift = 4
    model = fm.SeasonalNaive(cfg)
    assert model.shift == 4 and model.shift % 2 == 0 and model.shift >= 3
    demand = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7]
    panel = _panel(demand)
    X = _x_at(panel, cfg, [9, 11])
    # origin slot 9: window (9, 12] shifted back 4 -> slots 6, 7, 8 -> 2 + 6 + 5
    # origin slot 11: window (11, 14] shifted back 4 -> slots 8, 9, 10 -> 5 + 3 + 5
    assert model.predict(X, panel).tolist() == [13.0, 13.0]


def test_seasonal_naive_shift_when_season_divides_horizon():
    cfg = _small_cfg(season=3, lead=1, review=2)  # H = 3 = S -> shift = 3
    assert fm.SeasonalNaive(cfg).shift == 3
    cfg = _small_cfg(season=7, lead=7, review=7)  # H = 14 = 2 S
    assert fm.SeasonalNaive(cfg).shift == 14


def test_seasonal_naive_window_is_fully_observed_at_the_origin_and_ignores_the_future():
    cfg = _small_cfg(season=2, lead=1, review=2)
    model = fm.SeasonalNaive(cfg)
    rng = np.random.default_rng(0)
    demand = rng.integers(0, 20, 30)
    panel = _panel(demand)
    X = _x_at(panel, cfg, [9, 13, 17])
    base = model.predict(X, panel)
    later = panel.copy()
    later.loc[later["period_start"] > START + 17 * DAY, "demand"] = 1e6
    assert model.predict(X, later).tolist() == base.tolist()
    cut = panel[panel["period_start"] <= START + 17 * DAY]
    assert model.predict(X, cut).tolist() == base.tolist()


def test_seasonal_naive_needs_the_panel_and_is_non_negative(world, split, cfg):
    names = _names(cfg)
    model = fm.SeasonalNaive(cfg)
    with pytest.raises(FcstError) as exc:
        model.predict(split.test[names])
    assert _code(exc) == "E_PREDICT_BAD_INPUT"
    assert (model.predict(split.test[names], world["panel"]) >= 0).all()


# --- scale statistics ----------------------------------------------------------------------


def _y(series, values, start=0):
    idx = pd.MultiIndex.from_arrays(
        [series, [START + (start + i) * 7 * DAY for i in range(len(series))]],
        names=["series_id", "origin_date"],
    )
    return pd.Series(np.asarray(values, dtype="float64"), index=idx, name="y")


def test_fit_scale_values_and_zero_series_do_not_divide_by_zero(cfg):
    y = _y(["a", "a", "a", "z", "z"], [10, 20, 40, 0, 0])
    s = fm.fit_scale(y, cfg)
    floor = cfg["params"]["calibration"]["scale_floor"]
    assert s.level["a"] == pytest.approx(70 / 3)
    assert s.level["z"] == floor  # all zeros in train: floored, not 0
    assert s.naive_mse["a"] == pytest.approx((100 + 400) / 2)
    assert s.naive_mse["z"] == floor**2
    assert s.fallback_level >= floor and s.fallback_naive_mse >= floor**2
    X = pd.DataFrame({"roll_mean_14": [0.0, 0.0, 5.0]}, index=y.index[[3, 0, 0]])
    level = s.level_for(X, 14)
    assert np.isfinite(level).all() and (level >= floor).all()
    unseen = pd.DataFrame(
        {"roll_mean_14": [3.0]},
        index=pd.MultiIndex.from_tuples([("new", START)], names=["series_id", "origin_date"]),
    )
    assert s.level_for(unseen, 14)[0] == pytest.approx(42.0)  # own as-of level, not train
    assert s.naive_mse_for(["new", "a"]).tolist() == [s.fallback_naive_mse, s.naive_mse["a"]]


def test_fit_scale_rejects_bad_input(cfg):
    with pytest.raises(FcstError) as exc:
        fm.fit_scale(pd.Series([1.0, 2.0]), cfg)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"
    with pytest.raises(FcstError) as exc:
        fm.fit_scale(_y(["a"], [np.nan]), cfg)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"


# --- leakage: fit and scale only see train rows ----------------------------------------------


def _spy(monkeypatch, seen):
    real_scale, real_gbm, real_naive = fm.fit_scale, fm.GbmLag.fit, fm.SeasonalNaive.fit

    def scale(y, c):
        seen["scale"].append(set(y.index))
        return real_scale(y, c)

    def gbm(self, X, y):
        seen["gbm"].append(set(X.index) | set(y.index))
        return real_gbm(self, X, y)

    def naive(self, X, y=None):
        seen["naive"].append(set(X.index))
        return real_naive(self, X, y)

    monkeypatch.setattr(fm, "fit_scale", scale)
    monkeypatch.setattr(fm.GbmLag, "fit", gbm)
    monkeypatch.setattr(fm.SeasonalNaive, "fit", naive)


def test_every_fit_and_scale_statistic_sees_only_train_rows(world, split, cfg, monkeypatch):
    seen = {"scale": [], "gbm": [], "naive": []}
    _spy(monkeypatch, seen)
    fm.train_bundle(world["table"], world["panel"], cfg)
    train_rows = set(split.train.index)
    for kind, calls in seen.items():
        assert calls, f"{kind} was never called"
        for rows in calls:
            assert rows and rows <= train_rows, kind
    # 3 CV folds + the final fit; one scale per fold, one in the bundle, one per GBM fit
    assert len(seen["gbm"]) == 4 and len(seen["naive"]) == 4
    assert len(seen["scale"]) == 3 + 1 + 4
    blocked = set(split.calib.index) | set(split.test.index) | set(split.purged.index)
    assert not any(rows & blocked for calls in seen.values() for rows in calls)


def _perturbed(table, split, which, seed):
    """Copy of the table with the chosen sides changed (features and target)."""
    rng = np.random.default_rng(seed)
    out = table.copy()
    rows = out.index.isin(pd.concat([getattr(split, w) for w in which]).index)
    noise = rng.uniform(0.2, 5.0, size=(int(rows.sum()), out.shape[1]))
    out.loc[rows] = out.loc[rows].to_numpy() * noise
    return out


def test_changing_calib_or_test_rows_does_not_change_preprocessing_state(world, split, cfg):
    base = world["bundle"]
    other = fm.train_bundle(
        _perturbed(world["table"], split, ("calib", "test"), 1), world["panel"], cfg
    )
    assert other.scale == base.scale
    assert other.models["gbm_lag"].scale_state() == base.models["gbm_lag"].scale_state()
    assert other.selection == base.selection
    assert other.model_hash == base.model_hash and other.hashes == base.hashes
    X = split.train[_names(cfg)]
    for name in base.models:
        assert np.array_equal(
            other.models[name].predict(X, world["panel"]),
            base.models[name].predict(X, world["panel"]),
        )
    assert other.calibrator != base.calibrator  # calib rows do reach the calibrator


def test_changing_test_targets_does_not_change_the_selected_model(world, split, cfg):
    base = world["bundle"]
    table = world["table"].copy()
    test_rows = table.index.isin(split.test.index)
    table.loc[test_rows, "y"] = table.loc[test_rows, "y"] * 7.0 + 123.0
    other = fm.train_bundle(table, world["panel"], cfg)
    assert other.selected == base.selected and other.verdict == base.verdict
    assert other.selection == base.selection
    assert other.model_hash == base.model_hash
    assert other.calibrator == base.calibrator
    X = split.test[_names(cfg)]
    assert np.array_equal(
        fm.predict(other, world["panel"], X.index, cfg).drop(columns="origin_date").to_numpy(),
        fm.predict(base, world["panel"], X.index, cfg).drop(columns="origin_date").to_numpy(),
    )


# --- selection ---------------------------------------------------------------------------------


def test_selection_reports_per_fold_scores_for_both_models_and_picks_the_lowest_mean(world, cfg):
    sel = world["bundle"].selection
    assert sel["primary_metric"] == cfg["params"]["models"]["primary_metric"]
    assert sel["n_folds_effective"] == len(sel["folds"]) >= 2
    for fold in sel["folds"]:
        assert set(fold["scores"]) == {"seasonal_naive", "gbm_lag"}
        assert fold["n_train"] > 0 and fold["n_valid"] > 0
    for name in ("seasonal_naive", "gbm_lag"):
        scores = [f["scores"][name] for f in sel["folds"]]
        assert sel["mean"][name] == pytest.approx(np.mean(scores))
    best = min(sel["mean"], key=sel["mean"].get)
    assert sel["selected"] == best
    assert sel["verdict"] == ("use_model" if best == "gbm_lag" else "use_baseline")


def test_a_candidate_that_loses_gives_use_baseline(world, cfg, monkeypatch):
    monkeypatch.setattr(fm.GbmLag, "predict", lambda self, X, panel=None: np.zeros(len(X)))
    lost = fm.train_bundle(world["table"], world["panel"], cfg)
    assert (lost.selected, lost.verdict) == ("seasonal_naive", "use_baseline")
    assert lost.model_hash == lost.hashes["seasonal_naive"]


def test_candidate_wins_when_baseline_is_bad(world, cfg, monkeypatch):
    monkeypatch.setattr(
        fm.SeasonalNaive, "predict", lambda self, X, panel=None: np.zeros(len(X))
    )
    won = fm.train_bundle(world["table"], world["panel"], cfg)
    assert (won.selected, won.verdict) == ("gbm_lag", "use_model")


def test_tie_goes_to_the_baseline(world, cfg, monkeypatch):
    monkeypatch.setattr(
        fm.GbmLag,
        "predict",
        lambda self, X, panel=None: fm.SeasonalNaive(cfg).predict(X, world["panel"]),
    )
    tied = fm.select_model(
        make_split(world["table"], cfg).train, world["panel"], cfg
    )
    assert tied["mean"]["gbm_lag"] == tied["mean"]["seasonal_naive"]
    assert tied["selected"] == "seasonal_naive"


def test_unsupported_model_names_are_a_config_error(world, cfg):
    bad = copy.deepcopy(cfg)
    bad["params"]["models"]["candidate"] = "prophet"
    with pytest.raises(FcstError) as exc:
        fm.train_bundle(world["table"], world["panel"], bad)
    assert _code(exc) == "E_CONFIG"


# --- hash, save, load --------------------------------------------------------------------------


def test_model_hash_is_stable_for_the_same_seed_and_changes_with_a_param_or_the_seed(cfg):
    a, b = fm.GbmLag(cfg), fm.GbmLag(cfg)
    assert fm.model_hash(a) == fm.model_hash(b)
    assert fm.model_hash(fm.GbmLag(cfg, {"max_iter": 150})) != fm.model_hash(a)
    assert fm.model_hash(fm.GbmLag(cfg, {"learning_rate": 0.06})) != fm.model_hash(a)
    other_seed = copy.deepcopy(cfg)
    other_seed["params"]["seed"] = 43
    assert fm.model_hash(fm.GbmLag(other_seed)) != fm.model_hash(a)
    assert fm.model_hash(fm.SeasonalNaive(cfg)) == fm.model_hash(fm.SeasonalNaive(cfg))
    assert fm.model_hash(fm.SeasonalNaive(cfg)) != fm.model_hash(a)


def test_bundle_hash_follows_the_model_params(world, cfg):
    changed = fm.train_bundle(world["table"], world["panel"], cfg, {"max_iter": 120})
    assert changed.hashes["gbm_lag"] != world["bundle"].hashes["gbm_lag"]
    assert changed.hashes["seasonal_naive"] == world["bundle"].hashes["seasonal_naive"]


def test_save_then_load_gives_identical_predictions(world, split, cfg, tmp_path):
    bundle, panel = world["bundle"], world["panel"]
    out = fm.save_bundle(bundle, tmp_path / "art")
    assert (out / fm.BUNDLE_FILE).is_file() and (out / fm.META_FILE).is_file()
    loaded = fm.load_bundle(out)
    assert loaded.model_hash == bundle.model_hash and loaded.hashes == bundle.hashes
    assert loaded.calibrator == bundle.calibrator and loaded.scale == bundle.scale
    origins = split.test.index
    a = fm.predict(bundle, panel, origins, cfg)
    b = fm.predict(loaded, panel, origins, cfg)
    pd.testing.assert_frame_equal(a, b)
    for name in bundle.models:
        pd.testing.assert_frame_equal(
            fm.predict_point(bundle, panel, origins, cfg, name),
            fm.predict_point(loaded, panel, origins, cfg, name),
        )


def test_the_candidate_is_saved_and_loaded_too_when_it_is_not_selected(world, tmp_path):
    loaded = fm.load_bundle(fm.save_bundle(world["bundle"], tmp_path))
    assert set(loaded.models) == {"seasonal_naive", "gbm_lag"}


def test_load_refuses_a_tampered_or_missing_bundle(world, tmp_path):
    out = fm.save_bundle(world["bundle"], tmp_path)
    with pytest.raises(FcstError) as exc:
        fm.load_bundle(tmp_path / "nowhere")
    assert _code(exc) == "E_PREDICT_BAD_INPUT"
    with (out / fm.BUNDLE_FILE).open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(FcstError) as exc:
        fm.load_bundle(out)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"


# --- calibrator ----------------------------------------------------------------------------------


def _calib_inputs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    scale = rng.uniform(1, 50, n)
    yhat = scale * rng.uniform(0.5, 2, n)
    y = np.maximum(yhat + scale * rng.normal(0, 0.4, n), 0)
    return yhat, y, scale


def test_taus_are_tau_star_plus_the_tradeoff_grid(cfg):
    taus = fm.calibration_taus(cfg)
    assert taus == sorted(taus) and len(set(taus)) == len(taus)
    assert set(cfg["params"]["tradeoff"]["taus"]) <= set(taus)
    assert any(math.isclose(t, fm.tau_star(cfg)) for t in taus)
    assert len(taus) == len(cfg["params"]["tradeoff"]["taus"]) + 1
    near = copy.deepcopy(cfg)
    near["params"]["tradeoff"]["taus"] = [0.5, fm.tau_star(cfg) + 1e-12]
    assert len(fm.calibration_taus(near)) == 2  # no duplicate column names


def test_calibrator_depends_only_on_the_calibration_inputs(cfg):
    yhat, y, scale = _calib_inputs()
    taus = fm.calibration_taus(cfg)
    a = fm.fit_calibrator(yhat, y, scale, taus, cfg)
    assert a == fm.fit_calibrator(yhat.copy(), y.copy(), scale.copy(), taus, cfg)
    assert a == fm.fit_calibrator(yhat[::-1], y[::-1], scale[::-1], taus, cfg)  # order is irrelevant
    y2 = y.copy()
    y2[0] += 1000.0
    assert fm.fit_calibrator(yhat, y2, scale, taus, cfg) != a
    assert set(vars(a)) == {"method", "taus", "offsets", "n_calib", "scale_floor"}
    assert a.n_calib == len(y) and a.method == "conformal_scaled"


def test_bundle_calibrator_is_refit_from_calib_rows_alone(world, split, cfg):
    bundle, names = world["bundle"], _names(cfg)
    yhat = bundle.model.predict(split.calib[names], world["panel"])
    again = fm.fit_calibrator(
        yhat,
        split.calib["y"].to_numpy(),
        bundle.scale.level_for(split.calib[names], cfg["params"]["horizon_periods"]),
        fm.calibration_taus(cfg),
        cfg,
    )
    assert again == bundle.calibrator


def test_quantiles_are_monotone_in_tau_and_non_negative(cfg):
    yhat, y, scale = _calib_inputs()
    cal = fm.fit_calibrator(yhat, y, scale, fm.calibration_taus(cfg), cfg)
    rng = np.random.default_rng(5)
    for yh, sc in (
        (rng.uniform(0, 100, 400), rng.uniform(0, 40, 400)),
        (np.zeros(50), np.zeros(50)),
        (rng.uniform(0, 1, 50), np.full(50, 1e-9)),
    ):
        q = cal.quantiles(yh, sc)
        assert list(q.columns) == [ev.quantile_column(t) for t in cal.taus]
        arr = q.to_numpy()
        assert (arr >= 0).all() and np.isfinite(arr).all()
        assert (np.diff(arr, axis=1) >= 0).all()
    # the offsets themselves rise with tau
    assert list(cal.offsets) == sorted(cal.offsets)


def test_quantiles_cover_the_calibration_rows_at_least_at_tau(cfg):
    yhat, y, scale = _calib_inputs()
    taus = fm.calibration_taus(cfg)
    cal = fm.fit_calibrator(yhat, y, scale, taus, cfg)
    q = cal.quantiles(yhat, scale)
    for tau in taus:
        assert ev.coverage(y, q[ev.quantile_column(tau)]) >= tau - 1e-12


def test_zero_scale_in_train_and_in_calib_never_divides_by_zero(cfg):
    yhat, y, _ = _calib_inputs()
    with np.errstate(all="raise"):
        cal = fm.fit_calibrator(yhat, y, np.zeros(len(y)), [0.5, 0.9], cfg)
        q = cal.quantiles(yhat, np.zeros(len(y)))
    assert np.isfinite(cal.offsets).all() and np.isfinite(q.to_numpy()).all()
    zero_train = _y(["a"] * 5, [0, 0, 0, 0, 0])
    with np.errstate(all="raise"):
        s = fm.fit_scale(zero_train, cfg)
    assert s.level["a"] == cfg["params"]["calibration"]["scale_floor"]


def test_fit_calibrator_rejects_bad_input(cfg):
    yhat, y, scale = _calib_inputs(10)
    for args in (
        (yhat[:5], y, scale, [0.5]),
        (yhat[:0], y[:0], scale[:0], [0.5]),
        (np.where(np.arange(10) == 0, np.nan, yhat), y, scale, [0.5]),
    ):
        with pytest.raises(FcstError) as exc:
            fm.fit_calibrator(*args, cfg)
        assert _code(exc) == "E_PREDICT_BAD_INPUT"
    for taus in ([], [0.0], [1.0], [0.5, 1.2]):
        with pytest.raises(FcstError) as exc:
            fm.fit_calibrator(yhat, y, scale, taus, cfg)
        assert _code(exc) == "E_CONFIG"


def test_calibration_improves_coverage_on_synthetic_data_with_known_truth(cfg):
    """Truth: y ~ gamma-Poisson around a known mean. Point forecast = true mean."""
    rng = np.random.default_rng(11)

    def draw(n):
        mu = rng.uniform(5, 200, n)
        y = rng.poisson(mu * rng.gamma(10.0, 1.0 / 10.0, n)).astype(float)
        return mu, y

    mu_c, y_c = draw(4000)
    mu_t, y_t = draw(20000)
    floor = cfg["params"]["calibration"]["scale_floor"]
    scale_c, scale_t = np.maximum(mu_c, floor), np.maximum(mu_t, floor)
    taus = fm.calibration_taus(cfg)
    cal = fm.fit_calibrator(mu_c, y_c, scale_c, taus, cfg)
    after = cal.quantiles(mu_t, scale_t)
    tol = cfg["params"]["evaluation"]["calibration_tolerance"]
    before_errors, after_errors = [], []
    for tau in taus:
        before = ev.coverage_error(y_t, mu_t, tau)  # no calibration: the point forecast
        err = ev.coverage_error(y_t, after[ev.quantile_column(tau)], tau)
        before_errors.append(abs(before))
        after_errors.append(abs(err))
        assert abs(err) <= tol, tau
        if tau >= 0.6:
            assert abs(err) <= abs(before), tau
    assert np.mean(after_errors) < np.mean(before_errors)
    report = ev.calibration_report(y_t, after, taus, cfg)
    assert report["all_within_tolerance"] and report["n"] == len(y_t)


def test_small_test_set_reports_within_tolerance_as_is_and_does_not_fail(cfg):
    yhat, y, scale = _calib_inputs(200, seed=3)
    taus = fm.calibration_taus(cfg)
    cal = fm.fit_calibrator(yhat, y, scale, taus, cfg)
    # a tiny test set whose realisations sit far above the forecast
    y_small = yhat[:12] * 3 + 50
    report = ev.calibration_report(y_small, cal.quantiles(yhat[:12], scale[:12]), taus, cfg)
    assert report["n"] == 12
    flags = [r["within_tolerance"] for r in report["rows"]]
    assert all(isinstance(f, bool) for f in flags)
    assert not report["all_within_tolerance"] and not all(flags)
    assert report["max_abs_coverage_error"] > report["tolerance"]
    for r in report["rows"]:
        assert r["within_tolerance"] == (abs(r["coverage_error"]) <= report["tolerance"] + 1e-12)


def test_the_real_pipeline_reports_calibration_on_the_test_split_without_failing(world, split, cfg):
    bundle = world["bundle"]
    pred = fm.predict(bundle, world["panel"], split.test.index, cfg)
    q = pred[[ev.quantile_column(t) for t in bundle.calibrator.taus]]
    report = ev.calibration_report(split.test["y"], q, bundle.calibrator.taus, cfg)
    assert report["n"] == len(split.test)
    assert len(report["rows"]) == len(bundle.calibrator.taus)


# --- predict -------------------------------------------------------------------------------------


def test_predict_columns_order_rows_and_signs(world, split, cfg):
    bundle, panel = world["bundle"], world["panel"]
    rows = split.test.index.to_frame(index=False).sample(frac=1.0, random_state=4)
    pred = fm.predict(bundle, panel, rows, cfg)
    expected = [*fm.PREDICT_COLUMNS, *[ev.quantile_column(t) for t in bundle.calibrator.taus]]
    assert list(pred.columns) == expected
    assert len(pred) == len(rows)
    assert pred["series_id"].tolist() == rows["series_id"].tolist()
    assert pd.to_datetime(pred["origin_date"]).tolist() == pd.to_datetime(rows["origin_date"]).tolist()
    numeric = pred.drop(columns=["series_id", "origin_date"]).to_numpy(dtype="float64")
    assert np.isfinite(numeric).all() and (numeric >= 0).all()
    q = pred[expected[3:]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert pred["yhat"].tolist() == fm.predict_point(bundle, panel, rows, cfg)["yhat"].tolist()


def test_predict_accepts_a_multiindex_and_other_columns_are_ignored(world, split, cfg):
    bundle, panel = world["bundle"], world["panel"]
    from_index = fm.predict(bundle, panel, split.test.index, cfg)
    frame = split.test.index.to_frame(index=False).assign(y=-1.0, extra="x")
    pd.testing.assert_frame_equal(from_index, fm.predict(bundle, panel, frame, cfg))


def test_predict_needs_no_target_and_serves_the_live_origin(world, cfg):
    bundle, panel = world["bundle"], world["panel"]
    last = panel.groupby("series_id")["period_start"].max()
    live = last[last == panel["period_start"].max()].rename("origin_date").reset_index()
    assert len(live) > 5
    pred = fm.predict(bundle, panel, live, cfg)  # no window after these origins
    assert len(pred) == len(live) and (pred["yhat"] >= 0).all()
    with pytest.raises(FcstError) as exc:
        build_labels(panel, live, cfg)
    assert _code(exc) == "E_DATA_DATE_INVALID"  # there is no target to read, and none is needed


def test_predict_does_not_read_the_future_of_the_panel(world, split, cfg):
    bundle, panel = world["bundle"], world["panel"]
    when = split.test.index.get_level_values("origin_date").min()
    rows = split.test[split.test.index.get_level_values("origin_date") == when].index
    base = fm.predict(bundle, panel, rows, cfg)
    scrambled = panel.copy()
    scrambled.loc[scrambled["period_start"] > when, "demand"] = 999.0
    pd.testing.assert_frame_equal(base, fm.predict(bundle, scrambled, rows, cfg))


def _origin(series_id, date):
    return pd.DataFrame({"series_id": [series_id], "origin_date": [pd.Timestamp(date)]})


def test_predict_bad_input_is_e_predict_bad_input(world, split, cfg):
    bundle, panel = world["bundle"], world["panel"]
    ok = split.test.index.to_frame(index=False).iloc[:3]
    known = ok.iloc[0]["series_id"]
    first_day = panel["period_start"].min()
    cases = {
        "origins missing a column": ok[["series_id"]],
        "origins not a frame": [1, 2, 3],
        "no origins": ok.iloc[:0],
        "unknown series": _origin("nope@nowhere", ok.iloc[0]["origin_date"]),
        "date off the period grid": _origin(known, pd.Timestamp(ok.iloc[0]["origin_date"]) + pd.Timedelta(hours=5)),
        "date before the data": _origin(known, first_day - 10 * DAY),
        "date after the series": _origin(known, panel["period_start"].max() + 30 * DAY),
        "too little history": _origin(known, first_day + 3 * DAY),
        "duplicate origins": pd.concat([ok.iloc[:1], ok.iloc[:1]]),
        "unparseable date": _origin(known, "2024-01-01").assign(origin_date=["not a date"]),
    }
    for label, origins in cases.items():
        with pytest.raises(FcstError) as exc:
            fm.predict(bundle, panel, origins, cfg)
        assert _code(exc) == "E_PREDICT_BAD_INPUT", label
    for label, bad_panel in {
        "panel not a frame": "panel",
        "panel without demand": panel.drop(columns=["demand"]),
        "empty panel": panel.iloc[:0],
    }.items():
        with pytest.raises(FcstError) as exc:
            fm.predict(bundle, bad_panel, ok, cfg)
        assert _code(exc) == "E_PREDICT_BAD_INPUT", label
    with pytest.raises(FcstError) as exc:
        fm.predict("not a bundle", panel, ok, cfg)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"
    with pytest.raises(FcstError) as exc:
        fm.predict_point(bundle, panel, ok, cfg, which="prophet")
    assert _code(exc) == "E_PREDICT_BAD_INPUT"


def test_predict_rejects_a_config_that_does_not_match_the_bundle(world, split, cfg):
    other = copy.deepcopy(cfg)
    other["params"]["domain"]["min_history_periods"] = 28  # fewer lags and windows
    with pytest.raises(FcstError) as exc:
        fm.predict(world["bundle"], world["panel"], split.test.index, other)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"


# --- metrics --------------------------------------------------------------------------------------


def test_a_perfect_forecast_has_zero_wape_and_rmsse_and_zero_bias():
    y = np.array([3.0, 8.0, 0.0, 21.0, 5.0, 5.0])
    s = np.array(["a", "a", "a", "b", "b", "b"])
    assert ev.wape(y, y) == 0.0
    assert ev.rmsse(y, y, s, {"a": 2.0, "b": 9.0}) == 0.0
    assert ev.bias(y, y) == 0.0
    assert ev.pinball_loss(y, y, 0.9) == 0.0 and ev.coverage(y, y) == 1.0


def test_metrics_match_a_hand_count():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    yhat = np.array([12.0, 18.0, 33.0, 36.0])
    s = np.array(["A", "A", "B", "B"])
    assert ev.wape(y, yhat) == pytest.approx(11 / 100)
    assert ev.bias(y, yhat) == pytest.approx(-1 / 100)
    # A: mse (4 + 4) / 2 = 4, scale 2 -> 2; B: mse (9 + 16) / 2 = 12.5, scale 5 -> 2.5
    assert ev.rmsse(y, yhat, s, {"A": 2.0, "B": 5.0}) == pytest.approx(math.sqrt((2 + 2.5) / 2))
    assert ev.rmsse(y, yhat, s, pd.Series({"A": 2.0, "B": 5.0})) == pytest.approx(1.5)
    # pinball at 0.8: (0.2*2 + 0.8*2 + 0.2*3 + 0.8*4) / 4
    assert ev.pinball_loss(y, yhat, 0.8) == pytest.approx((0.4 + 1.6 + 0.6 + 3.2) / 4)
    assert ev.coverage(y, yhat) == 0.5
    assert ev.coverage_error(y, yhat, 0.8) == pytest.approx(-0.3)
    assert ev.primary_score("wape", y, yhat, s) == ev.wape(y, yhat)
    assert ev.primary_score("rmsse", y, yhat, s, {"A": 2.0, "B": 5.0}) == ev.rmsse(
        y, yhat, s, {"A": 2.0, "B": 5.0}
    )


def test_rmsse_weights_series_equally_not_rows():
    y = np.array([0.0] * 9 + [0.0])
    yhat = np.array([1.0] * 9 + [3.0])
    s = np.array(["big"] * 9 + ["small"])
    # big: mse 1, small: mse 9, both scale 1 -> sqrt((1 + 9) / 2); row-weighted would differ
    assert ev.rmsse(y, yhat, s, {"big": 1.0, "small": 1.0}) == pytest.approx(math.sqrt(5.0))


def test_true_quantiles_have_the_right_coverage_and_the_lowest_pinball_loss():
    rng = np.random.default_rng(2)
    n = 60000
    shape, scale = rng.uniform(2, 6, n), rng.uniform(5, 40, n)
    y = rng.gamma(shape, scale)
    for tau in (0.5, 0.8, 0.95):
        q = stats.gamma.ppf(tau, shape, scale=scale)
        assert ev.coverage(y, q) == pytest.approx(tau, abs=0.01)
        assert abs(ev.coverage_error(y, q, tau)) < 0.01
        best = ev.pinball_loss(y, q, tau)
        for factor in (0.8, 0.9, 1.1, 1.25):
            assert best < ev.pinball_loss(y, q * factor, tau)
        assert best < ev.pinball_loss(y, q + 3.0, tau) and best < ev.pinball_loss(y, q - 3.0, tau)


def test_generator_truth_quantiles_cover_at_tau_for_high_volume_series(world, cfg):
    truth = world["truth"]
    merged = world["panel"].merge(truth["table"], on=["series_id", "period_start"], how="inner")
    high = merged[merged["series_id"].map(truth["group"]) == "high_volume"]
    assert len(high) > 1000
    y = high["demand"].to_numpy()
    for tau, col in truth["quantile_cols"].items():
        assert ev.coverage(y, high[col]) == pytest.approx(tau, abs=0.03), tau
    losses = {t: ev.pinball_loss(y, high[c], t) for t, c in truth["quantile_cols"].items()}
    for tau, col in truth["quantile_cols"].items():
        assert losses[tau] < ev.pinball_loss(y, high[col] * 1.3, tau)
        assert losses[tau] < ev.pinball_loss(y, high[col] * 0.7, tau)


def test_zero_denominators_raise_their_code():
    zeros = np.zeros(4)
    with pytest.raises(FcstError) as exc:
        ev.wape(zeros, np.array([1.0, 0, 0, 0]))
    assert _code(exc) == "E_EVAL_ZERO_DENOMINATOR"
    with pytest.raises(FcstError) as exc:
        ev.bias(zeros, zeros)
    assert _code(exc) == "E_EVAL_ZERO_DENOMINATOR"
    for scale in ({"a": 0.0}, {"a": -1.0}, {"a": float("nan")}):
        with pytest.raises(FcstError) as exc:
            ev.rmsse(np.ones(3), np.ones(3), ["a"] * 3, scale)
        assert _code(exc) == "E_EVAL_ZERO_DENOMINATOR"
    with pytest.raises(FcstError) as exc:
        ev.bootstrap_delta(zeros, zeros, zeros, ["a", "a", "b", "b"], load_config())
    assert _code(exc) == "E_EVAL_ZERO_DENOMINATOR"


def test_metric_inputs_are_checked():
    bad = [
        (ev.wape, ([1.0, 2.0], [1.0])),
        (ev.wape, ([1.0, np.nan], [1.0, 1.0])),
        (ev.wape, ([], [])),
        (ev.pinball_loss, ([1.0], [1.0], 1.0)),
        (ev.pinball_loss, ([1.0], [1.0], 0.0)),
        (ev.rmsse, ([1.0], [1.0], ["a"], {"b": 1.0})),
        (ev.rmsse, ([1.0], [1.0], ["a"], 1.0)),
    ]
    for fn, args in bad:
        with pytest.raises(FcstError) as exc:
            fn(*args)
        assert _code(exc) == "E_PREDICT_BAD_INPUT", fn.__name__


# --- bootstrap_delta --------------------------------------------------------------------------------


def _boot_data(seed=0, n_series=12, rows=6):
    rng = np.random.default_rng(seed)
    series = np.repeat([f"s{i:02d}" for i in range(n_series)], rows)
    level = np.repeat(rng.uniform(5, 100, n_series), rows)
    y = rng.poisson(level).astype(float)
    base = level * 1.0 + rng.normal(0, 8, len(y))
    cand = y + rng.normal(0, 3, len(y))  # closer to the truth than the baseline
    return y, np.maximum(cand, 0), np.maximum(base, 0), series


def test_bootstrap_delta_is_deterministic_per_seed_and_the_seed_matters(cfg):
    y, cand, base, s = _boot_data()
    a = ev.bootstrap_delta(y, cand, base, s, cfg)
    assert a == ev.bootstrap_delta(y, cand, base, s, cfg)
    other = copy.deepcopy(cfg)
    other["params"]["seed"] = 7
    b = ev.bootstrap_delta(y, cand, base, s, other)
    assert b["value"] == a["value"]
    assert (b["ci_low"], b["ci_high"]) != (a["ci_low"], a["ci_high"])
    assert ev.bootstrap_delta(y, cand, base, s, cfg, seed=7) == b | {"seed": 7}


def test_bootstrap_delta_value_is_the_metric_difference_and_the_ci_contains_it(cfg):
    y, cand, base, s = _boot_data(seed=1)
    res = ev.bootstrap_delta(y, cand, base, s, cfg, metric="wape")
    assert res["value"] == pytest.approx(ev.wape(y, cand) - ev.wape(y, base), abs=1e-12)
    assert res["ci_low"] <= res["value"] <= res["ci_high"]
    assert res["value"] < 0 and res["ci_high"] < 0  # the candidate is better in every series
    assert res["n_series"] == 12 and res["n_boot"] == cfg["params"]["evaluation"]["n_boot"]
    assert res["ci_level"] == cfg["params"]["evaluation"]["ci_level"]
    scale = {f"s{i:02d}": 25.0 for i in range(12)}
    r2 = ev.bootstrap_delta(y, cand, base, s, cfg, metric="rmsse", scale=scale)
    assert r2["value"] == pytest.approx(
        ev.rmsse(y, cand, s, scale) - ev.rmsse(y, base, s, scale), abs=1e-12
    )
    assert r2["ci_low"] <= r2["value"] <= r2["ci_high"]


def test_bootstrap_delta_is_zero_for_identical_scores(cfg):
    y, _, base, s = _boot_data(seed=2)
    for metric, scale in (("wape", None), ("rmsse", {x: 9.0 for x in np.unique(s)})):
        res = ev.bootstrap_delta(y, base, base.copy(), s, cfg, metric=metric, scale=scale)
        assert (res["value"], res["ci_low"], res["ci_high"]) == (0.0, 0.0, 0.0)


def test_bootstrap_delta_resamples_series_not_rows(cfg):
    y, cand, base, s = _boot_data(seed=3)
    ref = ev.bootstrap_delta(y, cand, base, s, cfg)
    order = np.random.default_rng(9).permutation(len(y))
    shuffled = ev.bootstrap_delta(y[order], cand[order], base[order], s[order], cfg)
    assert shuffled == ref  # row order is irrelevant; series are the units
    # one series only: nothing to resample, so the interval collapses to the value
    one = ev.bootstrap_delta(y[:6], cand[:6], base[:6], s[:6], cfg)
    assert one["n_series"] == 1 and one["ci_low"] == one["value"] == one["ci_high"]


def test_bootstrap_delta_is_paired(cfg):
    """Two noisy models that differ by a constant: the paired interval stays tight."""
    y, _, base, s = _boot_data(seed=4, n_series=20)
    cand = base + 0.5
    paired = ev.bootstrap_delta(y, cand, base, s, cfg)
    width = paired["ci_high"] - paired["ci_low"]
    unpaired = []
    rng = np.random.default_rng(0)
    ids = np.unique(s)
    for _ in range(200):
        pick = rng.choice(ids, len(ids))
        a = np.concatenate([np.flatnonzero(s == p) for p in pick])
        unpaired.append(ev.wape(y[a], cand[a]))
    assert width < 0.5 * (np.quantile(unpaired, 0.975) - np.quantile(unpaired, 0.025))


def test_compare_to_baseline_against_itself_is_zero(world, split, cfg):
    bundle, names = world["bundle"], _names(cfg)
    test = split.test
    base = bundle.models["seasonal_naive"].predict(test[names], world["panel"])
    series = test.index.get_level_values("series_id")
    scale = bundle.scale.naive_mse_for(series)
    out = ev.compare_to_baseline(test["y"], base, base, series, cfg, scale)
    assert out["delta_primary"] == 0.0 and out["candidate"] == out["baseline"]
    assert out["delta_ci_low"] <= out["delta_primary"] <= out["delta_ci_high"]
    assert out["primary_metric"] == cfg["params"]["models"]["primary_metric"]


def test_compare_to_baseline_sign_and_rmsse_metric(world, split, cfg):
    bundle, names = world["bundle"], _names(cfg)
    test = split.test
    cand = bundle.models["gbm_lag"].predict(test[names], world["panel"])
    base = bundle.models["seasonal_naive"].predict(test[names], world["panel"])
    series = test.index.get_level_values("series_id")
    scale = bundle.scale.naive_mse_for(series)
    rm = copy.deepcopy(cfg)
    rm["params"]["models"]["primary_metric"] = "rmsse"
    out = ev.compare_to_baseline(test["y"], cand, base, series, rm, scale)
    assert out["primary_metric"] == "rmsse"
    assert out["delta_primary"] == pytest.approx(out["candidate"] - out["baseline"])
    assert out["delta_ci_low"] <= out["delta_primary"] <= out["delta_ci_high"]
