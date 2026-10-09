"""Business value tests: newsvendor and report parts (P5) and the end-to-end pipeline (P6)."""

from __future__ import annotations

import copy
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fcst.data as fd
import fcst.models as fm
import fcst.newsvendor as nv
import fcst.pipeline as pl
from fcst.config import FcstError, cost_rates, load_config
from fcst.evaluate import coverage, quantile_column, wape


def _code(exc) -> str:
    return exc.value.code


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _business(cfg, **changes) -> dict:
    b = copy.deepcopy(cfg["business"])
    b.update(changes)
    return b


# --- shared synthetic world (no model training; quantiles come from a calibrator) ------------


def _world(cfg, *, n_series=8, per_series=6, seed=0, model_noise=0.15, base_noise=0.35):
    """Test frame plus calibrated quantile frames for a 'baseline' and a 'model'.

    The model forecast is closer to the truth than the baseline's, so its
    quantiles are tighter. Calibration rows are drawn the same way as test rows.
    """
    rng = np.random.default_rng(seed)
    taus = fm.calibration_taus(cfg)

    def draw(n, sids):
        level = np.exp(rng.normal(3.0, 0.8, size=n_series))[sids]
        y = rng.poisson(level * 14 / 7).astype(float)
        truth = level * 14 / 7
        yb = truth * np.exp(rng.normal(0, base_noise, n))
        ym = truth * np.exp(rng.normal(0, model_noise, n))
        return y, yb, ym, np.maximum(level, 1.0)

    n = n_series * per_series
    sids = np.tile(np.arange(n_series), per_series)
    y_c, yb_c, ym_c, sc_c = draw(4 * n, np.tile(np.arange(n_series), 4 * per_series))
    y, yb, ym, sc = draw(n, sids)
    qb = fm.fit_calibrator(yb_c, y_c, sc_c, taus, cfg).quantiles(yb, sc)
    qm = fm.fit_calibrator(ym_c, y_c, sc_c, taus, cfg).quantiles(ym, sc)
    test = pd.DataFrame({"series_id": [f"s{i}" for i in sids], "y": y, "yhat_baseline": yb})
    return test, qb, qm


# === 1a: costs, tau*, simulation, policies, curves ===============================================


def test_costs_delegate_to_config_and_tau_star_matches_models(cfg):
    assert nv.costs(cfg["business"]) == cost_rates(cfg["business"])
    assert nv.tau_star(cfg["business"]) == pytest.approx(fm.tau_star(cfg), abs=1e-15)


def test_tau_star_matches_hand_calculation(cfg):
    # defaults: price 10, cost 6, holding 0.2/yr, 7 review periods, 365 periods/yr,
    # salvage 0.1, write-off 0.05
    c_under = 10.0 - 6.0
    c_over = 6.0 * 0.20 * 7 / 365 + 6.0 * (1 - 0.10) * 0.05
    assert nv.costs(cfg["business"]) == pytest.approx((c_under, c_over), abs=1e-12)
    assert nv.tau_star(cfg["business"]) == pytest.approx(c_under / (c_under + c_over), abs=1e-12)
    assert nv.tau_star(cfg["business"]) == pytest.approx(0.93175, abs=1e-5)


def test_tau_star_edge_cases_and_missing_keys(cfg):
    # c_over -> 0 gives tau* -> 1, c_under -> 0 gives tau* -> 0
    near_zero_over = _business(cfg, holding_cost_rate_annual=1e-12, overstock_writeoff_share=0.0)
    assert nv.tau_star(near_zero_over) > 1 - 1e-9
    assert nv.costs(_business(cfg, overstock_writeoff_share=0.0))[1] > 0
    near_zero_under = _business(cfg, avg_unit_price=6.0 + 1e-12)
    assert nv.tau_star(near_zero_under) < 1e-9
    b = _business(cfg)
    del b["avg_unit_price"]
    with pytest.raises(FcstError) as exc:
        nv.costs(b)
    assert _code(exc) == "E_CONFIG"


def _grid_quantiles():
    taus = [0.05, 0.2, 0.4, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
    base = np.array([10.0, 20.0, 0.0])
    return pd.DataFrame(
        {quantile_column(t): base + 10 * i + 5 * np.arange(3) for i, t in enumerate(taus)}
    )


def test_higher_price_does_not_lower_tau_star_or_order_up_to(cfg):
    q = _grid_quantiles()
    prev_tau, prev_S = -1.0, None
    for price in (7.0, 8.0, 10.0, 15.0, 40.0):
        ts = nv.tau_star(_business(cfg, avg_unit_price=price))
        S = nv.order_up_to(q, ts)
        assert ts >= prev_tau
        if prev_S is not None:
            assert (S >= prev_S - 1e-12).all()
        prev_tau, prev_S = ts, S


def test_higher_holding_cost_does_not_raise_tau_star_or_order_up_to(cfg):
    q = _grid_quantiles()
    prev_tau, prev_S = 2.0, None
    for rate in (0.05, 0.2, 1.0, 5.0, 50.0):
        ts = nv.tau_star(_business(cfg, holding_cost_rate_annual=rate))
        S = nv.order_up_to(q, ts)
        assert ts <= prev_tau
        if prev_S is not None:
            assert (S <= prev_S + 1e-12).all()
        prev_tau, prev_S = ts, S


def test_order_up_to_exact_column_interpolation_and_range(cfg):
    q = _grid_quantiles()
    assert nv.order_up_to(q, 0.9).tolist() == q["q_0.9"].tolist()
    mid = nv.order_up_to(q, 0.8)  # halfway between 0.7 and 0.9
    assert mid == pytest.approx(0.5 * (q["q_0.7"] + q["q_0.9"]).to_numpy())
    for bad in (0.01, 0.9995, 0.0, 1.0, True, "x"):
        with pytest.raises(FcstError) as exc:
            nv.order_up_to(q, bad)
        assert _code(exc) == "E_PREDICT_BAD_INPUT"
    with pytest.raises(FcstError):
        nv.order_up_to(np.zeros(3), 0.9)


def test_zero_stock_sells_nothing_and_always_stocks_out(cfg):
    D = np.array([3.0, 8.0, 1.0, 20.0])
    r = nv.simulate_policy(np.zeros(4), D, nv.costs(cfg["business"]), 14, unit_cost=6.0)
    assert r["net_value"] == 0.0
    assert r["stockout_rate"] == 1.0
    assert r["avg_inventory_value"] == 0.0
    assert r["sold"] == 0.0
    assert r["dio_periods"] is None
    assert r["simulation"] == "newsvendor_per_origin"


def test_stock_equal_to_demand_earns_full_margin_and_never_stocks_out(cfg):
    D = np.array([3.0, 8.0, 1.0, 20.0])
    c_under, c_over = nv.costs(cfg["business"])
    r = nv.simulate_policy(D, D, (c_under, c_over), 14, unit_cost=6.0)
    assert r["net_value"] == pytest.approx(c_under * D.sum(), abs=1e-12)
    assert r["stockout_rate"] == 0.0
    assert r["leftover"] == 0.0


def test_overstock_is_charged_c_over_and_understock_loses_sales(cfg):
    costs = (4.0, 0.5)
    r = nv.simulate_policy([10.0, 2.0], [6.0, 5.0], costs, 14, unit_cost=6.0)
    # origin 1: sold 6, left 4; origin 2: sold 2, lost 3
    assert r["net_value"] == pytest.approx(4.0 * 8 - 0.5 * 4)
    assert r["stockout_rate"] == 0.5
    assert r["leftover"] == 4.0


def test_identical_policies_have_zero_incremental_value(cfg):
    D = np.array([3.0, 8.0, 1.0, 20.0])
    S = np.array([4.0, 5.0, 2.0, 25.0])
    a = nv.simulate_policy(S, D, nv.costs(cfg["business"]), 14, unit_cost=6.0)
    b = nv.simulate_policy(S.copy(), D, nv.costs(cfg["business"]), 14, unit_cost=6.0)
    assert a["net_value"] - b["net_value"] == 0.0
    test, qb, _ = _world(cfg)
    agg = nv.aggregate_policies(test, qb, qb.copy(), cfg)
    pol = nv.policy_summaries(agg)
    assert pol["model_newsvendor"]["net_value"] - pol["baseline_newsvendor"]["net_value"] == 0.0


def test_dio_and_inventory_value_match_hand_calculation(cfg):
    # origin 1: S=10, D=6 -> sold 6, left 4, avg inventory (10 + 4) / 2 = 7
    # origin 2: S=4,  D=9 -> sold 4, left 0, avg inventory (4 + 0) / 2 = 2
    r = nv.simulate_policy([10.0, 4.0], [6.0, 9.0], (4.0, 0.3), 14, unit_cost=6.0)
    assert r["avg_inventory_units"] == pytest.approx(4.5)
    assert r["avg_inventory_value"] == pytest.approx(6.0 * 4.5)
    assert r["dio_periods"] == pytest.approx(14 * 9.0 / 10.0)


@pytest.mark.parametrize(
    "S, D, costs, H, unit_cost",
    [
        ([1.0, 2.0], [1.0], (4.0, 0.3), 14, 6.0),  # length mismatch
        ([-1.0], [1.0], (4.0, 0.3), 14, 6.0),  # negative S
        ([1.0], [float("nan")], (4.0, 0.3), 14, 6.0),  # non-finite D
        ([], [], (4.0, 0.3), 14, 6.0),  # empty
        ([1.0], [1.0], (4.0, -0.3), 14, 6.0),  # negative cost
        ([1.0], [1.0], (4.0, 0.3), 0, 6.0),  # bad horizon
        ([1.0], [1.0], (4.0, 0.3), 14, -1.0),  # bad unit cost
    ],
)
def test_simulate_policy_rejects_bad_input(S, D, costs, H, unit_cost):
    with pytest.raises(FcstError) as exc:
        nv.simulate_policy(S, D, costs, H, unit_cost=unit_cost)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"


def test_three_policies_and_curve_grid(cfg):
    test, qb, qm = _world(cfg)
    agg = nv.aggregate_policies(test, qb, qm, cfg)
    pol = nv.policy_summaries(agg)
    assert tuple(pol) == nv.POLICIES
    assert all(p["n_origins"] == len(test) for p in pol.values())
    curves = nv.tradeoff_curves(agg)
    expected = set(cfg["params"]["tradeoff"]["taus"]) | {nv.tau_star(cfg["business"])}
    for name in ("baseline", "model"):
        taus = [p["tau"] for p in curves[name]]
        assert taus == sorted(taus)
        assert len(taus) == len(expected)
        assert sum(p["is_tau_star"] for p in curves[name]) == 1
    # the status quo is the baseline point forecast plus the buffer
    D = test["y"].to_numpy()
    sq = nv.simulate_policy(
        (1 + cfg["business"]["status_quo_buffer"]) * test["yhat_baseline"].to_numpy(),
        D,
        nv.costs(cfg["business"]),
        14,
        unit_cost=cfg["business"]["avg_unit_cost"],
    )
    assert pol["status_quo"]["net_value"] == pytest.approx(sq["net_value"])
    assert pol["status_quo"]["stockout_rate"] == pytest.approx(sq["stockout_rate"])


def test_curves_are_monotone_in_tau(cfg):
    test, qb, qm = _world(cfg, seed=3)
    curves = nv.tradeoff_curves(nv.aggregate_policies(test, qb, qm, cfg))
    for name in ("baseline", "model"):
        pts = curves[name]
        stock = [p["stockout_rate"] for p in pts]
        inv = [p["avg_inventory_value"] for p in pts]
        assert all(a >= b - 1e-12 for a, b in itertools.pairwise(stock))
        assert all(a <= b + 1e-12 for a, b in itertools.pairwise(inv))
    # curve point at tau* equals the policy summary at tau*
    pol = nv.policy_summaries(nv.aggregate_policies(test, qb, qm, cfg))
    star = next(p for p in curves["model"] if p["is_tau_star"])
    assert star["net_value"] == pytest.approx(pol["model_newsvendor"]["net_value"])


def test_aggregate_policies_rejects_bad_frames(cfg):
    test, qb, qm = _world(cfg)
    with pytest.raises(FcstError) as exc:
        nv.aggregate_policies(test.drop(columns="yhat_baseline"), qb, qm, cfg)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"
    with pytest.raises(FcstError) as exc:
        nv.aggregate_policies(test, qb.iloc[:-1], qm, cfg)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"
    with pytest.raises(FcstError) as exc:
        nv.aggregate_policies(test, qb.drop(columns=quantile_column(0.9)), qm, cfg)
    assert _code(exc) == "E_PREDICT_BAD_INPUT"


# === 1b: equal service, bootstrap, metrics block ====================================================


def _curve(taus, stockout, inventory):
    return [
        {"tau": t, "stockout_rate": s, "avg_inventory_value": i}
        for t, s, i in zip(taus, stockout, inventory, strict=True)
    ]


_TAUS = [0.5, 0.7, 0.9]
_BASE = _curve(_TAUS, [0.5, 0.3, 0.1], [10.0, 20.0, 30.0])


def test_equal_service_identical_curves_give_zero_reduction():
    for tau in _TAUS:
        r = nv.equal_service(_BASE, _BASE, tau)
        assert r["available"] is True
        assert r["inventory_reduction_rel"] == pytest.approx(0.0, abs=1e-12)
        assert r["inventory_reduction_abs"] == pytest.approx(0.0, abs=1e-12)
        assert r["method"] == "linear_interpolation"


def test_equal_service_lower_model_inventory_gives_positive_reduction():
    model = _curve(_TAUS, [0.5, 0.3, 0.1], [8.0, 16.0, 24.0])  # 20% lower everywhere
    for tau in _TAUS:
        r = nv.equal_service(_BASE, model, tau)
        assert r["inventory_reduction_rel"] == pytest.approx(0.2)
    # a service level between two baseline points is interpolated linearly:
    # stockout 0.2 lies halfway between (0.3, 20) and (0.1, 30) -> 25
    mid = _curve([0.8], [0.2], [20.0])
    r = nv.equal_service(_BASE, mid, 0.8)
    assert r["inventory_baseline"] == pytest.approx(25.0)
    assert r["inventory_reduction_rel"] == pytest.approx(0.2)
    assert r["inventory_reduction_abs"] == pytest.approx(5.0)


def test_equal_service_never_extrapolates():
    for target in (0.02, 0.6):  # below and above the baseline stockout range [0.1, 0.5]
        model = _curve([0.9], [target], [1.0])
        r = nv.equal_service(_BASE, model, 0.9)
        assert r["available"] is False
        assert r["inventory_baseline"] is None
        assert r["inventory_reduction_rel"] is None
        assert r["inventory_reduction_abs"] is None
        assert r["reason"]
    edge = nv.equal_service(_BASE, _curve([0.9], [0.1], [24.0]), 0.9)  # exactly on the edge
    assert edge["available"] is True


def test_equal_service_same_stockout_keeps_the_smallest_baseline_inventory():
    base = _curve([0.5, 0.7, 0.9], [0.4, 0.2, 0.2], [10.0, 30.0, 20.0])
    r = nv.equal_service(base, _curve([0.9], [0.2], [15.0]), 0.9)
    assert r["inventory_baseline"] == pytest.approx(20.0)


def test_equal_service_rejects_bad_input():
    for base, model, tau in (
        ([], _BASE, 0.9),
        (_BASE, [{"tau": 0.9}], 0.9),
        (_BASE, _BASE, 0.6),  # tau not on the model curve
    ):
        with pytest.raises(FcstError) as exc:
            nv.equal_service(base, model, tau)
        assert _code(exc) == "E_PREDICT_BAD_INPUT"


def _hand_aggregates(*, base_stock, model_stock):
    """Four series with one origin each, two taus (0.5, 0.9), tau* = 0.9."""
    z = np.ones(4)

    def block(stock, inv):
        return np.column_stack([z, np.asarray(stock, float), z, np.asarray(inv, float), z])

    return nv.PolicyAggregates(
        series=np.array(["a", "b", "c", "d"]),
        taus=(0.5, 0.9),
        tau_star=0.9,
        unit_cost=1.0,
        horizon=14,
        status_quo=block([0, 0, 0, 0], [2, 2, 2, 2]),
        baseline=np.stack([block([1, 1, 1, 1], [2] * 4), block(base_stock, [5] * 4)]),
        model=np.stack([block([1, 1, 1, 1], [1] * 4), block(model_stock, [3] * 4)]),
    )


def _small(cfg, n=200, seed=None):
    c = copy.deepcopy(cfg)
    c["params"]["evaluation"]["n_boot"] = n
    if seed is not None:
        c["params"]["seed"] = seed
    return c


def test_bootstrap_identical_curves_give_zero_reduction(cfg):
    test, qb, _ = _world(cfg)
    agg = nv.aggregate_policies(test, qb, qb.copy(), cfg)
    boot = nv.bootstrap_business(agg, cfg, n_boot=60)
    r = boot["inventory_reduction_rel"]
    assert r["available"] is True
    assert r["value"] == pytest.approx(0.0, abs=1e-12)
    assert r["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert r["ci_high"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_boot_available"] == 60


def test_bootstrap_lower_model_inventory_gives_positive_reduction(cfg):
    test, qb, qm = _world(cfg, n_series=12, per_series=8, model_noise=0.05, base_noise=0.5)
    agg = nv.aggregate_policies(test, qb, qm, cfg)
    r = nv.bootstrap_business(agg, cfg, n_boot=100)["inventory_reduction_rel"]
    assert r["available"] is True
    assert r["value"] > 0
    assert r["ci_low"] <= r["value"] <= r["ci_high"]
    # same number as the point-estimate function
    curves = nv.tradeoff_curves(agg)
    point = nv.equal_service(curves["baseline"], curves["model"], agg.taus[agg.tau_index])
    assert r["value"] == pytest.approx(point["inventory_reduction_rel"])


def test_bootstrap_outside_baseline_range_is_unavailable_with_null_values(cfg):
    test, qb, qm = _world(cfg)
    test = test.assign(y=10.0)
    low = qb.mul(0.0).add(5.0)  # baseline always stocks out: range [1, 1]
    high = qm.mul(0.0).add(20.0)  # model never stocks out: target 0
    agg = nv.aggregate_policies(test, low, high, cfg)
    r = nv.bootstrap_business(agg, cfg, n_boot=30)["inventory_reduction_rel"]
    assert r["available"] is False
    assert r["value"] is None and r["ci_low"] is None and r["ci_high"] is None
    assert r["n_boot_available"] == 0
    block = nv.business_block(agg, cfg, n_boot=30)
    assert block["equal_service"]["available"] is False
    assert block["equal_service"]["inventory_reduction_rel"] is None


def test_bootstrap_drops_resamples_without_a_match_and_counts_them(cfg):
    # full sample: model stockout 0.25 equals the baseline minimum (available); resamples
    # that draw series d more often than series a put the target below that minimum.
    agg = _hand_aggregates(base_stock=[0, 0, 0, 1], model_stock=[1, 0, 0, 0])
    r = nv.bootstrap_business(agg, cfg, n_boot=300, seed=1)["inventory_reduction_rel"]
    assert r["available"] is True
    assert 0 < r["n_boot_available"] < r["n_boot"] == 300
    assert r["ci_low"] <= r["value"] <= r["ci_high"]
    # the net-value interval keeps every resample
    d = nv.bootstrap_business(agg, cfg, n_boot=300, seed=1)["net_value_delta"]["model_vs_status_quo"]
    assert d["n_boot_available"] == 300


def test_bootstrap_is_deterministic_per_seed_and_orders_its_interval(cfg):
    test, qb, qm = _world(cfg, seed=5)
    agg = nv.aggregate_policies(test, qb, qm, cfg)
    a = nv.bootstrap_business(agg, cfg, n_boot=80, seed=7)
    b = nv.bootstrap_business(agg, cfg, n_boot=80, seed=7)
    c = nv.bootstrap_business(agg, cfg, n_boot=80, seed=8)
    assert a == b
    assert a != c
    d = a["net_value_delta"]["model_vs_status_quo"]
    assert d["ci_low"] <= d["value"] <= d["ci_high"]
    assert d["seed"] == 7 and d["n_boot"] == 80 and d["n_series"] == len(agg.series)
    r = a["inventory_reduction_rel"]
    if r["available"]:
        assert r["ci_low"] <= r["value"] <= r["ci_high"]
    # defaults come from params.seed and params.evaluation.n_boot
    x = nv.bootstrap_business(agg, _small(cfg, 40, seed=11))
    assert x["net_value_delta"]["model_vs_status_quo"]["seed"] == 11
    assert x["net_value_delta"]["model_vs_status_quo"]["n_boot"] == 40
    with pytest.raises(FcstError) as exc:
        nv.bootstrap_business(agg, cfg, n_boot=0)
    assert _code(exc) == "E_CONFIG"


def test_net_value_delta_is_model_minus_status_quo_on_the_full_sample(cfg):
    test, qb, qm = _world(cfg, seed=2)
    agg = nv.aggregate_policies(test, qb, qm, cfg)
    pol = nv.policy_summaries(agg)
    d = nv.bootstrap_business(agg, cfg, n_boot=20)["net_value_delta"]["model_vs_status_quo"]
    assert d["value"] == pytest.approx(
        pol["model_newsvendor"]["net_value"] - pol["status_quo"]["net_value"]
    )
    assert d["scope"] == "total over test origins"


def test_business_block_is_complete_and_always_newsvendor_per_origin(cfg):
    test, qb, qm = _world(cfg)
    agg = nv.aggregate_policies(test, qb, qm, cfg)
    block = nv.business_block(agg, cfg, n_boot=20)
    assert block["simulation"] == "newsvendor_per_origin"
    assert block["value_label"] == "conditional value"
    assert block["tau_star"] == pytest.approx(nv.tau_star(cfg["business"]))
    assert (block["c_under"], block["c_over"]) == nv.costs(cfg["business"])
    assert block["n_origins"] == len(test) and block["horizon_periods"] == 14
    assert tuple(block["policies"]) == nv.POLICIES
    assert block["tradeoff"]["taus"] == fm.calibration_taus(cfg)
    assert block["equal_service"]["tau"] == pytest.approx(block["tau_star"])
    assert block["inventory_reduction_rel"]["label"] == "conditional value"
    # the simulation label does not depend on the data or the cost inputs
    other = copy.deepcopy(cfg)
    other["business"]["avg_unit_price"] = 25.0
    t2, qb2, qm2 = _world(other, seed=9)
    assert nv.business_block(nv.aggregate_policies(t2, qb2, qm2, other), other, n_boot=5)[
        "simulation"
    ] == "newsvendor_per_origin"


def test_write_business_metrics_merges_and_serialises_nulls(cfg, tmp_path):
    import json

    test, qb, qm = _world(cfg)
    agg = nv.aggregate_policies(test.assign(y=10.0), qb.mul(0.0).add(5.0), qm.mul(0.0).add(20.0), cfg)
    block = nv.business_block(agg, cfg, n_boot=10)
    path = tmp_path / "reports" / "metrics.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"accuracy": {"wape": 0.2}, "business": {"old": 1}}))
    out = nv.write_business_metrics(block, path)
    data = json.loads(out.read_text())
    assert data["accuracy"] == {"wape": 0.2}  # other blocks stay
    assert "old" not in data["business"]  # the business block is replaced
    assert data["business"]["simulation"] == "newsvendor_per_origin"
    assert data["business"]["inventory_reduction_rel"]["value"] is None
    assert not list(path.parent.glob("*.tmp"))
    # missing file or missing folder is created; a corrupt file is refused, not overwritten
    fresh = nv.write_business_metrics(block, tmp_path / "new" / "metrics.json")
    assert json.loads(fresh.read_text())["business"]["tau_star"] == pytest.approx(block["tau_star"])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(FcstError) as exc:
        nv.write_business_metrics(block, bad)
    assert _code(exc) == "E_CONFIG"
    assert bad.read_text() == "{not json"


def test_module_stays_within_size_budget():
    from pathlib import Path

    lines = Path(nv.__file__).read_text(encoding="utf-8").count("\n")
    assert lines <= 420, "newsvendor.py passed the P5 budget (~400 lines); log in NOTES.md and trim"


# === 1c: decide, metrics shape, strict render, chart ===============================================

import json
import re

import fcst.report as rp


@pytest.fixture(scope="module")
def _metrics_base(cfg):
    test, qb, qm = _world(cfg)
    agg = nv.aggregate_policies(test, qb, qm, cfg)
    return {
        "data_source": "real",
        "data_hash": "d" * 64,
        "model_hash": "m" * 64,
        "test": {"n_obs": 800, "n_series": 40},
        "leakage": {"passed": True, "n_origin_dates": 33, "n_rows": 1156},
        "accuracy": {
            "primary_metric": "wape",
            "baseline_name": "seasonal_naive",
            "candidate_name": "gbm_lag",
            "selection_verdict": "use_model",
            "baseline": 0.30,
            "candidate": 0.25,
            "delta_primary": -0.05,
            "delta_ci_low": -0.08,
            "delta_ci_high": -0.02,
            "ci_level": 0.95,
            "cv_mean_baseline": 0.20,
            "cv_mean_candidate": 0.18,
            "cv_folds": [
                {"fold": i, "n_train": 100 + i, "n_valid": 50, "baseline": 0.2, "candidate": 0.18}
                for i in range(3)
            ],
        },
        "calibration": {
            "n": 800,
            "tolerance": 0.03,
            "max_abs_coverage_error": 0.01,
            "all_within_tolerance": True,
            "rows": [
                {"tau": t, "coverage": t, "coverage_error": 0.0, "within_tolerance": True,
                 "pinball_loss": 1.5}
                for t in (0.5, 0.9)
            ],
        },
        "business": json.loads(json.dumps(nv.business_block(agg, cfg, n_boot=50), default=float)),
    }


@pytest.fixture
def m(_metrics_base):
    """A valid metrics dict that currently gives ``pilot_then_scale`` and a headline."""
    out = copy.deepcopy(_metrics_base)
    b = out["business"]
    b["equal_service"] |= {
        "available": True, "inventory_baseline": 120.0, "inventory_model": 100.0,
        "inventory_reduction_abs": 20.0, "inventory_reduction_rel": 0.2, "reason": None,
    }
    b["inventory_reduction_rel"] |= {"available": True, "value": 0.2, "ci_low": 0.1, "ci_high": 0.3}
    b["net_value_delta"]["model_vs_status_quo"] |= {"value": 500.0, "ci_low": 100.0, "ci_high": 900.0}
    b["policies"]["model_newsvendor"]["avg_inventory_value"] = 50.0
    b["policies"]["status_quo"]["avg_inventory_value"] = 60.0
    return out


def _set(d, path, value):
    *head, last = path.split(".")
    for k in head:
        d = d[k]
    d[last] = value
    return d


def _decide(cfg, m, **sets):
    for path, value in sets.items():
        _set(m, path.replace("__", "."), value)
    return rp.decide(m, cfg)


BASELINE_SETS = {"accuracy__selection_verdict": "use_baseline"}
QUO_SETS = {"business__net_value_delta__model_vs_status_quo__value": -1.0}


def test_valid_fixture_passes_shape_check_and_decides_pilot(cfg, m):
    rp.check_metrics_shape(m)
    out = rp.decide(m, cfg)
    assert out["verdict"] == "pilot_then_scale"
    assert out["headline_supported"] is True
    assert out["flags"] == []


def test_verdict_use_baseline_when_cv_says_so(cfg, m):
    out = _decide(cfg, m, **BASELINE_SETS)
    assert (out["verdict"], out["reasons"]) == ("use_baseline", ["R_CV_CANDIDATE_NOT_BETTER"])


def test_verdict_use_baseline_when_test_delta_not_negative(cfg, m):
    out = _decide(cfg, m, accuracy__delta_primary=0.0)
    assert (out["verdict"], out["reasons"]) == ("use_baseline", ["R_TEST_DELTA_NOT_NEGATIVE"])


@pytest.mark.parametrize(
    ("sets", "code"),
    [
        ({"test__n_series": 1}, "R_TEST_SERIES_BELOW_MIN"),
        ({"test__n_obs": 1}, "R_TEST_OBS_BELOW_MIN"),
        ({"business__equal_service__available": False}, "R_EQUAL_SERVICE_UNAVAILABLE"),
        (QUO_SETS, "R_NET_VALUE_NOT_POSITIVE"),
    ],
)
def test_verdict_keep_status_quo_for_each_reason(cfg, m, sets, code):
    out = _decide(cfg, m, **sets)
    assert (out["verdict"], out["reasons"]) == ("keep_status_quo", [code])
    assert out["headline_supported"] is False


def test_verdict_pilot_then_scale(cfg, m):
    out = rp.decide(m, cfg)
    assert (out["verdict"], out["reasons"]) == (
        "pilot_then_scale", ["R_CANDIDATE_BEATS_BASELINE", "R_VALUE_CHECKS_PASSED"]
    )


def test_priority_use_baseline_over_keep_status_quo_over_pilot(cfg, m):
    both = _decide(cfg, copy.deepcopy(m), **BASELINE_SETS, **QUO_SETS)
    assert both["verdict"] == "use_baseline"
    assert both["reasons"] == ["R_CV_CANDIDATE_NOT_BETTER"]  # status-quo reasons are not mixed in
    assert _decide(cfg, copy.deepcopy(m), **QUO_SETS)["verdict"] == "keep_status_quo"
    assert rp.decide(m, cfg)["verdict"] == "pilot_then_scale"


FLAG_SETS = {
    "F_SYNTHETIC_DATA": {"data_source": "synthetic"},
    "F_CALIBRATION_OUT_OF_TOLERANCE": {"calibration__all_within_tolerance": False},
    "F_ACCURACY_CI_INCLUDES_ZERO": {"accuracy__delta_ci_high": 0.01},
    "F_NET_VALUE_CI_INCLUDES_ZERO": {"business__net_value_delta__model_vs_status_quo__ci_low": -5.0},
    "F_MORE_INVENTORY_THAN_STATUS_QUO": {"business__policies__model_newsvendor__avg_inventory_value": 70.0},
    "F_LEAKAGE_CHECK_NOT_PASSED": {"leakage__passed": False},
}


def test_every_flag_has_a_case_and_text():
    assert set(FLAG_SETS) == set(rp.FLAGS) == set(rp.FLAG_TEXT)
    assert set(rp.REASON_TEXT) >= {"R_CANDIDATE_BEATS_BASELINE", "R_CV_CANDIDATE_NOT_BETTER"}


@pytest.mark.parametrize("flag", rp.FLAGS)
def test_each_flag_is_raised_alone_and_does_not_change_the_verdict(cfg, m, flag):
    clean = rp.decide(copy.deepcopy(m), cfg)
    out = _decide(cfg, m, **FLAG_SETS[flag])
    assert out["flags"] == [flag]
    assert out["verdict"] == clean["verdict"]
    assert out["reasons"] == clean["reasons"]
    assert out["headline_supported"] == clean["headline_supported"]


@pytest.mark.parametrize("sets", [BASELINE_SETS, QUO_SETS, {}])
def test_all_flags_together_leave_verdict_and_reasons_unchanged(cfg, m, sets):
    clean = _decide(cfg, copy.deepcopy(m), **sets)
    allsets = {k: v for s in FLAG_SETS.values() for k, v in s.items()}
    out = _decide(cfg, m, **sets, **allsets)
    assert out["flags"] == list(rp.FLAGS)
    assert (out["verdict"], out["reasons"]) == (clean["verdict"], clean["reasons"])


def test_reasons_and_flags_are_codes_only(cfg, m):
    for sets in (BASELINE_SETS, QUO_SETS, {}, {"test__n_series": 1, "test__n_obs": 1}):
        out = _decide(cfg, copy.deepcopy(m), **sets, **FLAG_SETS["F_SYNTHETIC_DATA"])
        assert out["verdict"] in rp.VERDICTS
        assert all(re.fullmatch(r"R_[A-Z_]+", c) and c in rp.REASON_TEXT for c in out["reasons"])
        assert all(re.fullmatch(r"F_[A-Z_]+", c) and c in rp.FLAG_TEXT for c in out["flags"])


def test_decide_is_deterministic_and_does_not_mutate_inputs(cfg, m):
    m_before, cfg_before = copy.deepcopy(m), copy.deepcopy(cfg)
    first, second = rp.decide(m, cfg), rp.decide(m, cfg)
    assert first == second
    assert m == m_before and cfg == cfg_before


@pytest.mark.parametrize(
    ("sets", "expected"),
    [
        ({}, True),
        ({"business__inventory_reduction_rel__ci_low": 0.0}, False),
        ({"business__inventory_reduction_rel__ci_low": -0.1}, False),
        ({"business__inventory_reduction_rel__ci_low": None}, False),
        ({"business__equal_service__available": False}, False),  # also keep_status_quo
        (BASELINE_SETS, False),  # use_baseline, even with ci_low > 0
        (QUO_SETS, False),
    ],
)
def test_headline_supported_only_for_pilot_with_available_and_positive_ci_low(cfg, m, sets, expected):
    assert _decide(cfg, m, **sets)["headline_supported"] is expected


@pytest.mark.parametrize(
    "path",
    [
        "accuracy.selection_verdict", "accuracy.delta_primary", "test.n_series", "test.n_obs",
        "business.equal_service.available", "business.net_value_delta", "business.policies.status_quo",
        "business.inventory_reduction_rel.ci_low", "calibration.all_within_tolerance", "leakage",
        "data_source",
    ],
)
def test_decide_missing_metrics_key_raises(cfg, m, path):
    *head, last = path.split(".")
    node = m
    for k in head:
        node = node[k]
    del node[last]
    with pytest.raises(FcstError) as exc:
        rp.decide(m, cfg)
    assert _code(exc) == "E_RENDER_MISSING_KEY"


def test_decide_null_in_non_nullable_raises(cfg, m):
    m["accuracy"]["delta_primary"] = None
    with pytest.raises(FcstError) as exc:
        rp.decide(m, cfg)
    assert _code(exc) == "E_RENDER_MISSING_KEY"


def test_decide_missing_config_key_raises(cfg, m):
    bad = copy.deepcopy(cfg)
    del bad["params"]["report"]["min_test_obs"]
    with pytest.raises(FcstError) as exc:
        rp.decide(m, bad)
    assert _code(exc) == "E_RENDER_MISSING_KEY"


# --- shape checker -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("data_hash"),
        lambda d: d["business"].pop("tradeoff"),
        lambda d: d["business"]["equal_service"].pop("available"),
        lambda d: d["accuracy"].update(delta_primary="0.1"),
        lambda d: d["accuracy"].update(delta_primary=True),
        lambda d: d["accuracy"].update(delta_primary=float("nan")),
        lambda d: d["accuracy"].update(cv_folds=[]),
        lambda d: d["accuracy"]["cv_folds"][0].update(fold=1.5),
        lambda d: d["test"].update(n_obs=None),
        lambda d: d["leakage"].update(passed="yes"),
        lambda d: d["business"]["tradeoff"].update(taus=[0.5, "x"]),
        lambda d: d["business"]["tradeoff"]["model"][0].pop("stockout_rate"),
    ],
)
def test_shape_checker_rejects_malformed_metrics(m, mutate):
    mutate(m)
    with pytest.raises(FcstError) as exc:
        rp.check_metrics_shape(m)
    assert _code(exc) == "E_RENDER_MISSING_KEY"
    assert "metrics" in exc.value.message


def test_shape_checker_rejects_non_objects_and_names_the_path(m):
    for bad in (None, [], "x", 3):
        with pytest.raises(FcstError) as exc:
            rp.check_metrics_shape(bad)
        assert _code(exc) == "E_RENDER_MISSING_KEY"
    del m["business"]["policies"]["status_quo"]["net_value"]
    with pytest.raises(FcstError) as exc:
        rp.check_metrics_shape(m)
    assert "business.policies.status_quo.net_value" in exc.value.message


def test_shape_checker_allows_nulls_where_documented_and_extra_keys(m):
    m["business"]["equal_service"] |= {
        "available": False, "inventory_baseline": None, "inventory_reduction_abs": None,
        "inventory_reduction_rel": None, "reason": "service target is outside the baseline stockout range",
    }
    m["business"]["policies"]["status_quo"]["dio_periods"] = None
    m["extra"] = {"anything": 1}
    rp.check_metrics_shape(m)


def test_real_business_block_fits_the_skeleton(m, cfg, tmp_path):
    # the block written by newsvendor.write_business_metrics round-trips through JSON and fits
    path = nv.write_business_metrics(m["business"], tmp_path / "metrics.json")
    loaded = json.loads(path.read_text())
    rp.check_metrics_shape(m | {"business": loaded["business"]})


# --- render --------------------------------------------------------------------------------


def _render(cfg, m, tmp_path, **kw):
    rp.chart(m, tmp_path / rp.CHART_NAME)
    path = rp.render_report(m, cfg, out_dir=tmp_path, **kw)
    return path.read_text(encoding="utf-8")


def test_render_writes_report_with_all_sections_in_order(cfg, m, tmp_path):
    text = _render(cfg, m, tmp_path)
    assert (tmp_path / "report.md").exists()
    heads = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert heads == [
        "Verdict", "Accuracy", "Calibration", "Stockout vs working capital", "Flags",
        "What NOT to conclude", "Limitations",
    ]
    assert "equal service" in text.lower() and "not at equal service" in text
    assert "tradeoff.png" in text


def _unavailable(m):
    m["business"]["equal_service"] |= {"available": False, "inventory_baseline": None,
                                       "inventory_reduction_abs": None,
                                       "inventory_reduction_rel": None, "reason": "no match"}
    m["business"]["inventory_reduction_rel"] |= {"available": False, "value": None, "ci_low": None,
                                                "ci_high": None}


def _apply_scenario(m, sets):
    """Apply a render scenario: dotted-path overrides, or ``{"unavailable": True}`` for a run
    without an equal-service match. Shared by the render tests below."""
    if sets.get("unavailable"):
        _unavailable(m)
        return
    for path, value in sets.items():
        _set(m, path.replace("__", "."), value)


RENDER_SCENARIOS = [{}, BASELINE_SETS, QUO_SETS, {"unavailable": True}]


@pytest.mark.parametrize("sets", RENDER_SCENARIOS)
def test_render_has_no_placeholders_and_always_labels_conditional_value(cfg, m, tmp_path, sets):
    _apply_scenario(m, sets)
    text = _render(cfg, m, tmp_path)
    assert "conditional value" in text
    for bad in ("{{", "}}", "{%", "None", "nan", "inf", "Undefined", "[FILL]"):
        assert bad not in text


def test_render_headline_only_when_supported(cfg, m, tmp_path):
    assert "**Headline (conditional value):**" in _render(cfg, copy.deepcopy(m), tmp_path)
    m["accuracy"]["selection_verdict"] = "use_baseline"
    text = _render(cfg, m, tmp_path)
    assert "Headline" not in text
    assert "No headline claim is supported" in text
    assert "keep using the baseline" in text


@pytest.mark.parametrize(
    ("sets", "verdict", "code"),
    [
        (BASELINE_SETS, "use_baseline", "R_CV_CANDIDATE_NOT_BETTER"),
        (QUO_SETS, "keep_status_quo", "R_NET_VALUE_NOT_POSITIVE"),
    ],
)
def test_render_shows_verdict_and_reason_text_for_both_non_pilot_outcomes(
    cfg, m, tmp_path, sets, verdict, code
):
    # a model that does not beat the baseline, and one that is more accurate but not more
    # profitable, are both reported openly: verdict code, reason code and reason wording
    for path, value in sets.items():
        _set(m, path.replace("__", "."), value)
    text = _render(cfg, m, tmp_path)
    assert f"Verdict code: `{verdict}`" in text
    assert f"- `{code}`: {rp.REASON_TEXT[code]}" in text
    assert "No headline claim is supported" in text


def test_render_flags_equal_service_as_not_empirical_when_baseline_selected(cfg, m, tmp_path):
    marker = "**Not an empirical result.**"
    assert marker not in _render(cfg, copy.deepcopy(m), tmp_path)  # candidate selected
    m["accuracy"]["selection_verdict"] = "use_baseline"
    text = _render(cfg, m, tmp_path)
    assert text.count(marker) == 1
    assert "zero by construction" in text
    # the note sits inside the equal-service section, before its figures
    section = text.split("**Equal service (", 1)[1]
    assert section.index(marker) < section.index("**Net value against the status quo.**")
    # a rendering note only: verdict, reasons and flags are those of ``decide``
    assert rp.decide(m, cfg)["verdict"] == "use_baseline"


# Claims about fewer stockouts or lower inventory exist only as the equal-service result of
# metrics.json, labelled "conditional value": the headline line and the equal-service block.
# A claim line has a claim word and a subject word. English first, then Indonesian, so a
# translated template is still covered. Keyword based: a phrase that uses none of these words
# still passes unseen, so wording changes need a read by hand too.
_CLAIM_WORD = re.compile(
    r"\b(less|fewer|lower|smaller|reduc\w*|decreas\w*|sav(?:e|es|ing|ings)|cuts?|down|drops?|dropped"
    r"|better|improv\w*|gains?|efficien\w*|higher"
    r"|turun|menurun|penurunan|kurang|berkurang|mengurangi|lebih\s+(?:sedikit|rendah|kecil|baik|hemat)"
    r"|hemat|penghematan|efisien|efisiensi|naik|meningkat)\b",
    re.IGNORECASE,
)
_CLAIM_SUBJECT = re.compile(
    r"inventor|stock|working capital|service|fill rate|capital"
    r"|persediaan|stok|modal kerja|layanan|pelayanan",
    re.IGNORECASE,
)
_EQUAL_SERVICE_START = "**Equal service ("
_EQUAL_SERVICE_END = "**Net value against the status quo.**"
_HEADLINE_START = "**Headline (conditional value):**"


def _claim_lines_outside_allowed_blocks(text: str) -> list[str]:
    """Lines that say less stock / fewer stockouts and are not the headline, the equal-service
    block or the one flag line that negates a stock reduction."""
    allowed_negation = rp.FLAG_TEXT["F_MORE_INVENTORY_THAN_STATUS_QUO"]
    stray, inside = [], False
    for line in text.splitlines():
        if line.startswith(_EQUAL_SERVICE_START):
            inside = True
        elif line.startswith(_EQUAL_SERVICE_END):
            inside = False
        if inside or line.startswith(_HEADLINE_START) or allowed_negation in line:
            continue
        if _CLAIM_WORD.search(line) and _CLAIM_SUBJECT.search(line):
            stray.append(line)
    return stray


@pytest.mark.parametrize("sets", RENDER_SCENARIOS)
def test_render_less_stock_claims_only_in_labelled_equal_service_blocks(cfg, m, tmp_path, sets):
    _apply_scenario(m, sets)
    text = _render(cfg, m, tmp_path)
    assert _claim_lines_outside_allowed_blocks(text) == []
    # the equal-service block carries the label inside itself
    block = text.split(_EQUAL_SERVICE_START, 1)[1].split(_EQUAL_SERVICE_END, 1)[0]
    assert m["business"]["value_label"] == "conditional value"
    assert "conditional value" in block
    # a headline line, when present, carries the label and the figures of metrics.json
    headline = [ln for ln in text.splitlines() if ln.startswith(_HEADLINE_START)]
    if rp.decide(m, cfg)["headline_supported"]:
        assert len(headline) == 1
        red = m["business"]["inventory_reduction_rel"]
        for key in ("value", "ci_low", "ci_high"):
            assert f"{100 * red[key]:.1f}%" in headline[0]
    else:
        assert headline == []
    if sets.get("unavailable"):
        assert "No inventory claim is made." in block


@pytest.mark.parametrize("sets", RENDER_SCENARIOS)
def test_render_policy_table_states_that_rows_are_not_a_claim(cfg, m, tmp_path, sets):
    _apply_scenario(m, sets)
    text = _render(cfg, m, tmp_path)
    section = text.split("## Stockout vs working capital", 1)[1].split(_EQUAL_SERVICE_START, 1)[0]
    lines = section.splitlines()
    last_row = max(i for i, ln in enumerate(lines) if ln.startswith("| Model newsvendor"))
    note = next(ln for ln in lines[last_row + 1:] if ln.strip())  # first text under the table
    assert "carry no claim" in note and "equal-service block" in note
    assert "conditional value" in note


def test_render_unsupported_headline_has_no_less_stock_claim_outside_equal_service(cfg, m, tmp_path):
    # equal service is available but its interval includes zero: no headline, and the only
    # "needs less" wording left is the generic sign rule inside the labelled block
    m["business"]["inventory_reduction_rel"] |= {"ci_low": -0.1}
    text = _render(cfg, m, tmp_path)
    assert "Headline" not in text and "No headline claim is supported" in text
    assert _claim_lines_outside_allowed_blocks(text) == []


CAUGHT_PHRASES = [
    "The model gives fewer stockouts.",
    "Stockouts went down with the model.",
    "The model delivers better service.",
    "Service improves and working capital drops.",
    "The model is more efficient on inventory.",
    "Stockout turun dengan model.",
    "Layanan lebih baik dengan model ini.",
    "Persediaan berkurang dan stok lebih rendah.",
    "Model ini lebih hemat modal kerja.",
]
NOT_CAUGHT_PHRASES = [
    "Primary metric: `wape` (lower is better).",
    "A negative difference means the candidate has the lower error.",
    "Do not change large safety stocks without a pilot.",
]


def test_claim_detector_catches_english_and_indonesian_claim_phrases():
    # the detector itself, so a change to the word lists cannot quietly weaken the check
    for phrase in CAUGHT_PHRASES:
        assert _CLAIM_WORD.search(phrase) and _CLAIM_SUBJECT.search(phrase), phrase
        assert _claim_lines_outside_allowed_blocks(phrase) == [phrase], phrase


def test_claim_detector_leaves_neutral_report_lines_alone():
    for phrase in NOT_CAUGHT_PHRASES:
        assert _claim_lines_outside_allowed_blocks(phrase) == [], phrase


def test_template_source_has_less_stock_wording_only_in_allowed_blocks():
    src = (rp.ROOT / "templates" / rp.TEMPLATE_NAME).read_text(encoding="utf-8")
    stray = _claim_lines_outside_allowed_blocks(src)
    assert stray == []
    # the label is a template variable on the headline line and in the equal-service heading
    assert _HEADLINE_START in src
    heading = next(ln for ln in src.splitlines() if ln.startswith(_EQUAL_SERVICE_START))
    assert "{{ m.business.value_label }}" in heading


@pytest.mark.parametrize(
    ("sets", "phrase"),
    [({}, "limited pilot"), (BASELINE_SETS, "keep using the baseline"), (QUO_SETS, "keep the current ordering")],
)
def test_render_action_text_follows_the_verdict(cfg, m, tmp_path, sets, phrase):
    for path, value in sets.items():
        _set(m, path.replace("__", "."), value)
    assert phrase in _render(cfg, m, tmp_path)


def test_render_banner_only_for_synthetic(cfg, m, tmp_path):
    synthetic = copy.deepcopy(m) | {"data_source": "synthetic"}
    first = _render(cfg, synthetic, tmp_path).splitlines()[0]
    assert "SYNTHETIC DATA" in first and first.startswith(">")
    assert "SYNTHETIC" not in _render(cfg, m, tmp_path)


def test_render_footer_has_source_hash_and_data_and_model_hashes(cfg, m, tmp_path):
    text = _render(cfg, m, tmp_path)
    digest = rp.source_hash(m)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert f"Source hash: `{digest}`" in text.splitlines()[-3]
    assert m["data_hash"] in text.splitlines()[-2] and m["model_hash"] in text.splitlines()[-1]


def test_source_hash_covers_config_and_metrics(cfg, m, tmp_path):
    cfg_a = tmp_path / "a.yaml"
    cfg_b = tmp_path / "b.yaml"
    base = rp.DEFAULT_CONFIG_PATH.read_bytes()
    cfg_a.write_bytes(base)
    cfg_b.write_bytes(base + b"# comment\n")
    h = rp.source_hash(m, cfg_a)
    assert h == rp.source_hash(copy.deepcopy(m), cfg_a)
    assert h == rp.source_hash(m)  # default path is config.yaml
    assert h != rp.source_hash(m, cfg_b)
    changed = copy.deepcopy(m)
    changed["accuracy"]["baseline"] = 0.31
    assert h != rp.source_hash(changed, cfg_a)
    assert rp.render_report(m, cfg, config_path=cfg_b, out_dir=tmp_path).read_text().count(
        rp.source_hash(m, cfg_b)
    ) == 1
    with pytest.raises(FcstError) as exc:
        rp.source_hash(m, tmp_path / "missing.yaml")
    assert _code(exc) == "E_CONFIG"


def test_render_is_deterministic(cfg, m, tmp_path):
    assert _render(cfg, m, tmp_path) == _render(cfg, m, tmp_path)


@pytest.mark.parametrize(
    "path", ["accuracy.delta_ci_low", "business.policies.status_quo", "calibration.rows", "data_hash", "leakage"]
)
def test_render_missing_key_raises_and_writes_nothing(cfg, m, tmp_path, path):
    *head, last = path.split(".")
    node = m
    for k in head:
        node = node[k]
    del node[last]
    with pytest.raises(FcstError) as exc:
        rp.render_report(m, cfg, out_dir=tmp_path)
    assert _code(exc) == "E_RENDER_MISSING_KEY"
    assert not (tmp_path / "report.md").exists()


def test_render_is_strict_about_undefined_names_in_the_template(cfg, m, tmp_path):
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / rp.TEMPLATE_NAME).write_text("value: {{ m.business.not_a_key }}\n", encoding="utf-8")
    with pytest.raises(FcstError) as exc:
        rp.render_report(m, cfg, out_dir=tmp_path / "out", template_dir=tpl)
    assert _code(exc) == "E_RENDER_MISSING_KEY"
    (tpl / rp.TEMPLATE_NAME).write_text("{{ unknown_name }}\n", encoding="utf-8")
    with pytest.raises(FcstError) as exc:
        rp.render_report(m, cfg, out_dir=tmp_path / "out", template_dir=tpl)
    assert _code(exc) == "E_RENDER_MISSING_KEY"
    assert not (tmp_path / "out" / "report.md").exists()


def test_template_has_no_numeric_literals_outside_the_whitelist():
    text = (rp.ROOT / "templates" / rp.TEMPLATE_NAME).read_text(encoding="utf-8")
    assert rp.template_number_literals(text) == []
    assert rp.template_number_literals("a 12 b 0.5 c x2 d 0") == ["12", "0.5", "x2"]


def test_template_has_the_english_section_titles_and_fixed_phrases():
    text = (rp.ROOT / "templates" / rp.TEMPLATE_NAME).read_text(encoding="utf-8")
    for needle in ("## Verdict", "## Accuracy", "## Calibration", "## Stockout vs working capital",
                   "## Flags", "## What NOT to conclude", "## Limitations", "not at equal service"):
        assert needle in text
    for key in ("no_carry_over", "supplier_constraints", "substitution", "demand_proxied_by_sales"):
        assert key in rp.LIMITATIONS


# --- chart ---------------------------------------------------------------------------------


def test_chart_writes_a_png_and_needs_valid_metrics(m, tmp_path):
    out = rp.chart(m, tmp_path / "sub" / "tradeoff.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5_000
    synthetic = rp.chart(copy.deepcopy(m) | {"data_source": "synthetic"}, tmp_path / "s.png")
    assert synthetic.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    del m["business"]["tradeoff"]
    with pytest.raises(FcstError) as exc:
        rp.chart(m, tmp_path / "x.png")
    assert _code(exc) == "E_RENDER_MISSING_KEY"
    assert not (tmp_path / "x.png").exists()


def test_report_module_has_no_default_for_missing_values():
    src = (rp.ROOT / "src" / "fcst" / "report.py").read_text(encoding="utf-8")
    assert ".get(" not in src


# === P6: pipeline, end to end ====================================================================

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
NOTES = (ROOT / "NOTES.md").read_text(encoding="utf-8")


def _synthetic_panel(cfg) -> pd.DataFrame:
    raw, _ = fd.generate(cfg["params"]["seed"], cfg)
    return fd.clean(raw, cfg)


def _section(text: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    assert match, f"missing section {title!r}"
    return match.group(1)


def _assert_close(got, want, path: str = "business") -> None:
    """Same structure, equal strings/bools/ints, floats within a relative 1e-9."""
    if isinstance(want, dict):
        assert got.keys() == want.keys(), path
        for key in want:
            _assert_close(got[key], want[key], f"{path}.{key}")
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_close(g, w, f"{path}[{i}]")
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), path
    else:
        assert got == want, path


class _Recorder:
    """Records the row index of every ``predict`` call of both models."""

    def __init__(self):
        self.calls: list[tuple[str, frozenset]] = []

    def patch(self, mp: pytest.MonkeyPatch) -> None:
        for cls in (fm.SeasonalNaive, fm.GbmLag):
            original = cls.predict

            def wrapped(self_, X, panel=None, _orig=original, _name=cls.__name__):
                self.calls.append((_name, frozenset(X.index)))
                return _orig(self_, X, panel)

            mp.setattr(cls, "predict", wrapped)


@pytest.fixture(scope="module")
def e2e(cfg, tmp_path_factory):
    """One full run on the synthetic sample, with every model ``predict`` call recorded."""
    out = tmp_path_factory.mktemp("e2e")
    recorder, steps = _Recorder(), []
    mp = pytest.MonkeyPatch()
    recorder.patch(mp)
    try:
        metrics = pl.run(cfg, None, True, out, log=steps.append)
    finally:
        mp.undo()
    pred = pd.read_csv(out / "artifacts" / pl.PREDICTIONS_NAME, parse_dates=["origin_date"])
    return {"out": out, "metrics": metrics, "calls": recorder.calls, "steps": steps, "pred": pred}


def _files(out: Path):
    return out / "reports", out / "artifacts"


def test_run_writes_every_output_and_metrics_json_equals_the_return_value(e2e):
    reports, artifacts = _files(e2e["out"])
    for name in ("metrics.json", "report.md", "tradeoff.png"):
        assert (reports / name).stat().st_size > 0
    for name in ("bundle.joblib", "bundle.json", pl.PREDICTIONS_NAME):
        assert (artifacts / name).stat().st_size > 0
    assert json.loads((reports / "metrics.json").read_text(encoding="utf-8")) == e2e["metrics"]


def test_steps_run_in_the_documented_order(e2e):
    seen = [s.strip("[]") for s in e2e["steps"] if s.startswith("[")]
    assert seen == list(pl.STEPS)
    assert pl.STEPS == (
        "load", "validate", "clean", "origins", "features", "split",
        "train", "predict_test", "metrics", "decide", "report", "write",
    )  # fmt: skip


def test_metrics_fit_the_skeleton_and_the_decision_is_the_decide_output(e2e, cfg):
    m = copy.deepcopy(e2e["metrics"])
    rp.check_metrics_shape(m)
    decision = m.pop("decision")
    assert decision == rp.decide(m, cfg)
    assert m["data_source"] == "synthetic"
    assert "F_SYNTHETIC_DATA" in decision["flags"]
    assert decision["verdict"] in rp.VERDICTS
    assert m["split"]["test"] == m["test"]["n_obs"] == len(e2e["pred"])
    assert m["test"]["n_series"] == e2e["pred"]["series_id"].nunique()


def test_hashes_identify_the_data_and_the_saved_model(e2e, cfg):
    _, artifacts = _files(e2e["out"])
    assert e2e["metrics"]["data_hash"] == fd.data_hash(_synthetic_panel(cfg))
    assert e2e["metrics"]["model_hash"] == fm.load_bundle(artifacts).model_hash


def test_report_has_the_banner_the_verdict_and_a_fresh_source_hash(e2e):
    reports, _ = _files(e2e["out"])
    text = (reports / "report.md").read_text(encoding="utf-8")
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert text.startswith("> " + fd.banner("synthetic"))
    assert f"`{metrics['decision']['verdict']}`" in text
    assert f"`{rp.source_hash(metrics)}`" in text  # report is not stale against metrics.json
    assert not rp.template_number_literals((ROOT / "templates" / "report.md.j2").read_text("utf-8"))


def test_each_model_predicts_the_test_origins_exactly_once(e2e):
    pred = e2e["pred"]
    test_keys = frozenset(zip(pred["series_id"], pred["origin_date"], strict=True))
    for name in ("SeasonalNaive", "GbmLag"):
        hits = [idx for who, idx in e2e["calls"] if who == name and idx & test_keys]
        assert len(hits) == 1, f"{name} saw test rows in {len(hits)} calls"
        assert hits[0] == test_keys  # exactly the test rows, nothing else mixed in


def test_every_number_after_prediction_comes_from_the_prediction_table(e2e, cfg):
    pred, m = e2e["pred"], e2e["metrics"]
    y = pred["y"].to_numpy()
    taus = fm.calibration_taus(cfg)
    assert m["accuracy"]["baseline"] == pytest.approx(wape(y, pred["yhat_baseline"]), rel=1e-9)
    assert m["accuracy"]["candidate"] == pytest.approx(wape(y, pred["yhat_candidate"]), rel=1e-9)
    for row in m["calibration"]["rows"]:
        col = quantile_column(row["tau"])
        assert row["coverage"] == pytest.approx(coverage(y, pred[col]), abs=1e-12)
    cols = [quantile_column(t) for t in taus]
    q_model = pred[cols].reset_index(drop=True)
    q_base = pred[["baseline_" + c for c in cols]].rename(columns=lambda c: c[len("baseline_"):])
    agg = nv.aggregate_policies(pred[["series_id", "y", "yhat_baseline"]], q_base, q_model, cfg)
    _assert_close(json.loads(json.dumps(nv.business_block(agg, cfg), default=float)), m["business"])


def test_selected_model_quantiles_are_the_ones_the_saved_bundle_gives(e2e, cfg):
    _, artifacts = _files(e2e["out"])
    bundle = fm.load_bundle(artifacts)
    pred = e2e["pred"]
    again = fm.predict(bundle, _synthetic_panel(cfg), pred[["series_id", "origin_date"]], cfg)
    cols = [quantile_column(t) for t in fm.calibration_taus(cfg)]
    selected_yhat = pred["yhat_baseline" if bundle.selected == fm.BASELINE else "yhat_candidate"]
    assert again["yhat"].to_numpy() == pytest.approx(selected_yhat.to_numpy(), rel=1e-9)
    assert again[cols].to_numpy() == pytest.approx(pred[cols].to_numpy(), rel=1e-9)


def test_quantiles_in_the_table_are_monotone_and_non_negative(e2e, cfg):
    pred = e2e["pred"]
    for prefix in ("", "baseline_"):
        cols = [prefix + quantile_column(t) for t in fm.calibration_taus(cfg)]
        q = pred[cols].to_numpy()
        assert (q >= 0).all()
        assert (np.diff(q, axis=1) >= -1e-12).all()


def test_two_runs_give_identical_files(e2e, cfg, tmp_path):
    pl.run(cfg, None, True, tmp_path)
    for rel in ("reports/metrics.json", "reports/report.md", f"artifacts/{pl.PREDICTIONS_NAME}"):
        assert (tmp_path / rel).read_bytes() == (e2e["out"] / rel).read_bytes(), rel


def test_real_data_run_reads_the_csv_and_has_no_synthetic_banner(e2e, cfg, tmp_path):
    csv = tmp_path / "panel.csv"
    csv.write_bytes(fd.to_csv_bytes(fd.generate(cfg["params"]["seed"], cfg)[0], cfg))
    small = copy.deepcopy(cfg)
    small["params"]["evaluation"]["n_boot"] = 50
    metrics = pl.run(small, csv, False, tmp_path / "out")
    assert metrics["data_source"] == "real"
    assert "F_SYNTHETIC_DATA" not in metrics["decision"]["flags"]
    assert metrics["data_hash"] == e2e["metrics"]["data_hash"]
    assert metrics["accuracy"]["baseline"] == e2e["metrics"]["accuracy"]["baseline"]
    text = (tmp_path / "out" / "reports" / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" not in text


def test_real_reader_gives_the_same_panel_as_data_load_panel(cfg, tmp_path):
    csv = tmp_path / "panel.csv"
    csv.write_bytes(fd.to_csv_bytes(fd.generate(cfg["params"]["seed"], cfg)[0], cfg))
    via_pipeline = fd.clean(pl._read_raw(csv, cfg), cfg)
    assert fd.data_hash(via_pipeline) == fd.data_hash(fd.load_panel(csv, cfg))


# --- failures: first failing step, its code, nothing written -------------------------------


def _nothing_written(out: Path) -> bool:
    return not (out / "reports").exists() and not (out / "artifacts").exists()


def test_missing_real_file_is_e_data_file_missing_at_load(cfg, tmp_path):
    with pytest.raises(FcstError) as exc:
        pl.run(cfg, tmp_path / "nope.csv", False, tmp_path / "out")
    assert _code(exc) == "E_DATA_FILE_MISSING"
    assert exc.value.step == "load"
    assert _nothing_written(tmp_path / "out")


def test_missing_file_at_the_configured_path_never_falls_back_to_synthetic(cfg, tmp_path):
    changed = copy.deepcopy(cfg)
    changed["params"]["data"]["path"] = str(tmp_path / "absent" / "panel.csv")
    with pytest.raises(FcstError) as exc:
        pl.run(changed, None, False, tmp_path / "out")
    assert _code(exc) == "E_DATA_FILE_MISSING"
    assert _nothing_written(tmp_path / "out")


def test_relative_configured_path_is_read_from_the_project_root(cfg):
    assert pl._resolve_data_path(cfg, None, None) == ROOT / cfg["params"]["data"]["path"]
    assert pl._resolve_data_path(cfg, "x.csv", None) == Path("x.csv")


def _bad_csv(cfg, tmp_path, mutate) -> Path:
    df = fd.generate(cfg["params"]["seed"], cfg)[0]
    path = tmp_path / "bad.csv"
    path.write_bytes(fd.to_csv_bytes(mutate(df.copy()), cfg))
    return path


@pytest.mark.parametrize(
    ("code", "mutate"),
    [
        ("E_DATA_DUPLICATE_KEY", lambda df: pd.concat([df, df.iloc[:1]], ignore_index=True)),
        ("E_DATA_DEMAND_NEGATIVE", lambda df: df.assign(demand=df["demand"].where(df.index != 5, -1))),
        ("E_DATA_GAP_TOO_LONG", lambda df: df.drop(index=range(20, 30))),
    ],
)
def test_invalid_data_stops_at_validate_and_no_later_step_runs(
    cfg, tmp_path, monkeypatch, code, mutate
):
    path = _bad_csv(cfg, tmp_path, mutate)
    reached = []
    for name in ("build_origins", "make_features", "train_bundle", "render_report"):
        monkeypatch.setattr(pl, name, lambda *a, _n=name, **k: reached.append(_n))
    with pytest.raises(FcstError) as exc:
        pl.run(cfg, path, False, tmp_path / "out")
    assert _code(exc) == code
    assert exc.value.step == "validate"
    assert reached == []
    assert _nothing_written(tmp_path / "out")


def test_missing_column_is_reported_at_validate(cfg, tmp_path):
    path = tmp_path / "cols.csv"
    path.write_text("sku_id,location_id,period_start\nA,L,2024-01-01\n", encoding="utf-8")
    with pytest.raises(FcstError) as exc:
        pl.run(cfg, path, False, tmp_path / "out")
    assert (_code(exc), exc.value.step) == ("E_DATA_COLUMNS_MISSING", "validate")


def test_sample_and_an_explicit_data_path_cannot_be_combined(cfg, tmp_path):
    with pytest.raises(FcstError) as exc:
        pl.run(cfg, tmp_path / "x.csv", True, tmp_path / "out")
    assert (_code(exc), exc.value.step) == ("E_CONFIG", "load")


def test_a_failing_leakage_check_stops_before_any_model_is_trained(cfg, tmp_path, monkeypatch):
    def leaking(*_a, **_k):
        raise FcstError("E_LEAKAGE", "forced")

    trained = []
    monkeypatch.setattr(pl, "perturb_future_check", leaking)
    monkeypatch.setattr(pl, "train_bundle", lambda *a, **k: trained.append(1))
    with pytest.raises(FcstError) as exc:
        pl.run(cfg, None, True, tmp_path / "out")
    assert (_code(exc), exc.value.step) == ("E_LEAKAGE", "features")
    assert trained == []
    assert _nothing_written(tmp_path / "out")


def test_a_failure_in_training_keeps_its_code_and_writes_no_report(cfg, tmp_path, monkeypatch):
    def too_short(*_a, **_k):
        raise FcstError("E_CV_TOO_SHORT", "forced")

    monkeypatch.setattr(pl, "perturb_future_check", lambda *a, **k: {
        "passed": True, "n_origin_dates": 1, "n_rows": 1, "variants": []
    })
    monkeypatch.setattr(pl, "train_bundle", too_short)
    with pytest.raises(FcstError) as exc:
        pl.run(cfg, None, True, tmp_path / "out")
    assert (_code(exc), exc.value.step) == ("E_CV_TOO_SHORT", "train")
    assert not (tmp_path / "out" / "reports").exists()


# --- CLI -----------------------------------------------------------------------------------


def test_cli_missing_file_exits_1_with_the_code_and_the_step(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "fcst.pipeline", "--data", str(tmp_path / "nope.csv"),
         "--out", str(tmp_path / "out")],
        cwd=ROOT, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip
    assert done.returncode == 1
    assert "E_DATA_FILE_MISSING" in done.stderr and "step: load" in done.stderr
    assert "SYNTHETIC" not in done.stdout
    assert _nothing_written(tmp_path / "out")


def test_cli_sample_prints_the_banner_first_and_last(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pl, "run", lambda *a, **k: {"decision": {"verdict": "use_baseline"}})
    assert pl.main(["--sample", "--out", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == lines[-1] == fd.banner("synthetic")
    assert "verdict: use_baseline" in lines


def test_cli_returns_1_and_names_the_code_when_a_step_fails(tmp_path, monkeypatch, capsys):
    def failing(*_a, **_k):
        err = FcstError("E_LEAKAGE", "forced")
        err.step = "features"
        raise err

    monkeypatch.setattr(pl, "run", failing)
    assert pl.main(["--out", str(tmp_path)]) == 1
    assert "E_LEAKAGE" in capsys.readouterr().err


# --- Makefile, README, NOTES ---------------------------------------------------------------


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_all_uses_real_data_and_all_sample_uses_the_sample_path():
    def dry(target: str) -> str:
        done = subprocess.run(
            ["make", "-n", target], cwd=ROOT, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    real = dry("all")
    assert re.search(r"-m fcst\.pipeline\s*$", real, re.MULTILINE)
    assert "--sample" not in real
    fast = dry("all-sample")
    assert fast.index("--write-sample") < fast.index("-m fcst.pipeline --sample")
    assert "not implemented yet" not in (ROOT / "Makefile").read_text(encoding="utf-8")


def test_readme_has_exactly_the_five_final_sections_in_order():
    assert re.findall(r"^## (.+)$", README, re.MULTILINE) == [
        "Problem", "Results", "Structure", "Run", "Limitations",
    ]  # fmt: skip
    assert "Stub" not in README


def test_readme_results_and_limitations_hold_no_numbers_and_point_to_reports():
    results, limits = _section(README, "Results"), _section(README, "Limitations")
    assert "reports/" in results
    assert not re.search(r"\d", results)
    assert not re.search(r"\d", limits)
    assert "demand is proxied by sales" in limits.lower()


def test_readme_structure_lists_real_files_and_run_lists_the_make_targets():
    structure = _section(README, "Structure")
    generated = {"data", "artifacts", "reports"}
    for path in re.findall(r"`([\w./-]+\.(?:py|md|yaml|j2|txt)|Makefile)`", structure):
        assert (ROOT / path).exists(), path
    for src in (ROOT / "src" / "fcst").glob("*.py"):
        assert src.name in structure, src.name
    for name in generated:
        assert name in structure
    run_section = _section(README, "Run")
    for target in ("make install", "make test", "make all-sample", "make all"):
        assert target in run_section


def test_notes_end_with_the_run_record_section():
    headings = re.findall(r"^## (.+)$", NOTES, re.MULTILINE)
    assert headings[-1] == "Run record"
    assert "## P6" in NOTES
    assert _section(NOTES, "Run record").strip()


# --- end to end in a temporary copy of the project -------------------------------------------

import hashlib

from fcst.data import banner as _data_banner

_COPY_ITEMS = ("src", "templates", "config.yaml")
_PLACEHOLDERS = ("{{", "}}", "{%", "%}", "None", "nan", "inf", "Undefined", "[FILL]", "TODO", "TBD", "XXX")


def _project_copy(tmp_path_factory, name: str) -> Path:
    """A fresh copy of the code and config, without data, artifacts, reports or caches."""
    dest = tmp_path_factory.mktemp(name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for item in _COPY_ITEMS:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, dest / item, ignore=ignore)
        else:
            shutil.copy2(src, dest / item)
    return dest


def _cli_in(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    return subprocess.run(
        [sys.executable, "-m", "fcst.pipeline", *args],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )


@pytest.fixture(scope="module")
def sample_copy(tmp_path_factory):
    root = _project_copy(tmp_path_factory, "copy_a")
    done = _cli_in(root, "--sample")
    return {"root": root, "done": done}


def test_sample_in_a_temporary_copy_writes_the_three_report_files(sample_copy):
    assert sample_copy["done"].returncode == 0, sample_copy["done"].stderr
    for name in ("metrics.json", "report.md", "tradeoff.png"):
        assert (sample_copy["root"] / "reports" / name).stat().st_size > 0, name


def test_copy_metrics_json_passes_the_shape_checker(sample_copy):
    text = (sample_copy["root"] / "reports" / "metrics.json").read_text(encoding="utf-8")
    rp.check_metrics_shape(json.loads(text))


def test_copy_run_shows_the_banner_in_cli_report_and_metrics(sample_copy):
    banner = _data_banner(fd.SYNTHETIC)
    lines = sample_copy["done"].stdout.splitlines()
    assert lines[0] == banner and lines[-1] == banner
    report = (sample_copy["root"] / "reports" / "report.md").read_text(encoding="utf-8")
    assert banner in report
    metrics = json.loads((sample_copy["root"] / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["data_source"] == fd.SYNTHETIC


def test_copy_report_has_no_placeholders(sample_copy):
    report = (sample_copy["root"] / "reports" / "report.md").read_text(encoding="utf-8")
    for bad in _PLACEHOLDERS:
        assert bad not in report, bad
    assert "conditional value" in report


def test_missing_real_panel_in_a_fresh_copy_stops_with_e_data_file_missing(tmp_path_factory):
    root = _project_copy(tmp_path_factory, "copy_missing")
    done = _cli_in(root)
    assert done.returncode == 1
    assert "E_DATA_FILE_MISSING" in done.stderr and "step: load" in done.stderr
    assert not (root / "reports").exists() and not (root / "artifacts").exists()


def test_same_seed_gives_the_same_metrics_json_hash(sample_copy, tmp_path_factory):
    other = _project_copy(tmp_path_factory, "copy_b")
    assert _cli_in(other, "--sample").returncode == 0

    def digest(root: Path) -> str:
        return hashlib.sha256((root / "reports" / "metrics.json").read_bytes()).hexdigest()

    assert digest(sample_copy["root"]) == digest(other)


# --- no hand-typed numbers in prose, comments and template -----------------------------------


def _hand_typed_digits(text: str) -> list[str]:
    """Digits in prose. Allowed: inside code spans, part labels, list markers, SHA-256."""
    found = []
    for line in text.splitlines():
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"\bP\d\b", "", line)
        line = re.sub(r"^\s*\d+\.\s", "", line)
        line = line.replace("SHA-256", "")
        found += re.findall(r"\S*\d\S*", line)
    return found


def _yaml_comment_text(text: str) -> str:
    return "\n".join(line.split("#", 1)[1] for line in text.splitlines() if "#" in line)


def test_docs_comments_and_template_hold_no_hand_typed_numbers():
    template = (ROOT / "templates" / rp.TEMPLATE_NAME).read_text(encoding="utf-8")
    config = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert _hand_typed_digits(README) == []
    assert _hand_typed_digits(NOTES) == []
    assert _hand_typed_digits(_yaml_comment_text(config)) == []
    assert rp.template_number_literals(template) == []
    assert _hand_typed_digits("a 0.93 b `x2` c P3 d\n1. item e SHA-256") == ["0.93"]


def test_notes_have_one_section_per_part_p1_to_p6_in_order():
    heads = re.findall(r"^## (P\d)$", NOTES, re.MULTILINE)
    assert heads == ["P1", "P2", "P3", "P4", "P5", "P6"]
