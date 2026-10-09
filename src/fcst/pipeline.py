"""End-to-end pipeline and CLI (P6): one run from the panel to ``reports/metrics.json``.

Steps, in order; the run stops at the first one that fails and the error keeps its code
(``FcstError.step`` names the step)::

    load -> validate -> clean -> origins -> features -> split -> train -> predict_test
         -> metrics -> decide -> report -> write

``train`` is CV and selection on the training origins, the fit, and the calibration on the
calibration origins (all inside ``models.train_bundle``; it never reads the test rows).

The test set is predicted once: each model predicts the test origins one time in
``predict_test`` and the result is one table (``pred``). Every number after that step
(accuracy, interval, calibration, business block) is computed from ``pred`` and from nothing
else that has seen the test rows. ``pred`` is saved as ``artifacts/test_predictions.csv``.

Layout of ``out_dir``: ``reports/`` (``metrics.json``, ``report.md``, ``tradeoff.png``) and
``artifacts/`` (``bundle.joblib``, ``bundle.json``, ``test_predictions.csv``).

CLI: ``python -m fcst.pipeline [--sample] [--data PATH] [--out DIR] [--config PATH]``.
``--sample`` runs on the synthetic generator; without it the real panel is read from
``--data`` or ``params.data.path`` and a missing file is ``E_DATA_FILE_MISSING``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fcst.config import DEFAULT_CONFIG_PATH, FcstError, load_config
from fcst.data import REAL, SYNTHETIC, banner, clean, data_hash, generate, validate
from fcst.evaluate import calibration_report, compare_to_baseline, quantile_column
from fcst.features import (
    TARGET_COLUMN,
    build_labels,
    build_origins,
    check_correlation,
    make_features,
    make_split,
    perturb_future_check,
)
from fcst.models import (
    BASELINE,
    CANDIDATE,
    calibration_taus,
    fit_calibrator,
    predict,
    predict_point,
    save_bundle,
    train_bundle,
)
from fcst.newsvendor import aggregate_policies, business_block
from fcst.report import chart, check_metrics_shape, decide, render_report

STEPS = (
    "load",
    "validate",
    "clean",
    "origins",
    "features",
    "split",
    "train",
    "predict_test",
    "metrics",
    "decide",
    "report",
    "write",
)
METRICS_NAME = "metrics.json"
PREDICTIONS_NAME = "test_predictions.csv"
BASELINE_PREFIX = "baseline_"  # prefix of the baseline quantile columns in ``pred``


# --- helpers -------------------------------------------------------------------------------


def _resolve_data_path(cfg: dict, data_path, config_path) -> Path:
    """Explicit path as given; otherwise ``params.data.path`` relative to the config file."""
    if data_path is not None:
        return Path(data_path)
    root = Path(config_path).resolve().parent if config_path else DEFAULT_CONFIG_PATH.parent
    configured = Path(cfg["params"]["data"]["path"])
    return configured if configured.is_absolute() else root / configured


def _read_raw(path: Path, cfg: dict) -> pd.DataFrame:
    """Read the schema columns of a real panel as strings; a missing file is a hard error."""
    if not path.is_file():
        raise FcstError("E_DATA_FILE_MISSING", f"data file not found: {path}")
    wanted = set(cfg["params"]["schema"]["required_columns"])
    try:
        return pd.read_csv(
            path,
            dtype=str,
            usecols=lambda c: c in wanted,
            keep_default_na=False,
            na_values=[""],
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise FcstError("E_DATA_COLUMNS_MISSING", f"cannot parse {path}: {exc}") from exc


def _plain(obj):
    """Round-trip through JSON so numpy scalars become Python numbers (NaN is kept, then
    rejected by ``check_metrics_shape``)."""

    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    return json.loads(json.dumps(obj, default=default))


def _write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _predict_test_once(bundle, split, panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The one table of test predictions: realised ``y``, both point forecasts, the selected
    model's calibrated quantiles (``q_<tau>``) and the baseline's (``baseline_q_<tau>``).

    Each model predicts the test origins exactly once. The baseline needs its own calibrator,
    fit on the calibration rows only. ``naive_mse`` is the train-only RMSSE scale of the row's
    series.
    """
    test, calib = split.test, split.calib
    names = bundle.feature_names
    horizon = cfg["params"]["horizon_periods"]
    taus = calibration_taus(cfg)
    origins = test.index.to_frame(index=False)

    selected = predict(bundle, panel, origins, cfg)  # selected model: yhat + q_<tau>
    other = CANDIDATE if bundle.selected == BASELINE else BASELINE
    other_point = predict_point(bundle, panel, origins, cfg, which=other)["yhat"].to_numpy()
    yhat = {bundle.selected: selected["yhat"].to_numpy(), other: other_point}

    scale_test = bundle.scale.level_for(test[names], horizon)
    scale_calib = bundle.scale.level_for(calib[names], horizon)
    calibrator = fit_calibrator(
        bundle.models[BASELINE].predict(calib[names], panel),
        calib[TARGET_COLUMN].to_numpy(dtype="float64"),
        scale_calib,
        taus,
        cfg,
    )
    q_base = calibrator.quantiles(yhat[BASELINE], scale_test).add_prefix(BASELINE_PREFIX)
    series = np.asarray(test.index.get_level_values("series_id")).astype(str)
    naive = bundle.scale.naive_mse_for(series)
    pred = pd.DataFrame(
        {
            "series_id": series,
            "origin_date": np.asarray(test.index.get_level_values("origin_date")),
            TARGET_COLUMN: test[TARGET_COLUMN].to_numpy(dtype="float64"),
            "yhat_baseline": yhat[BASELINE],
            "yhat_candidate": yhat[CANDIDATE],
            "naive_mse": naive.reindex(series).to_numpy(),
        }
    )
    quantiles = selected.drop(columns=["series_id", "origin_date", "yhat"])
    return pd.concat([pred, quantiles, q_base], axis=1)


def _quantile_frames(pred: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(selected, baseline)`` quantile frames from ``pred``, both with plain ``q_<tau>`` names."""
    cols = [quantile_column(t) for t in calibration_taus(cfg)]
    selected = pred[cols].reset_index(drop=True)
    baseline = pred[[BASELINE_PREFIX + c for c in cols]].rename(
        columns=lambda c: c[len(BASELINE_PREFIX) :]
    )
    return selected, baseline.reset_index(drop=True)


def _metrics_from_pred(pred: pd.DataFrame, bundle, cfg: dict) -> dict:
    """Accuracy, calibration and test-size blocks, from the prediction table only."""
    y = pred[TARGET_COLUMN].to_numpy()
    series = pred["series_id"].to_numpy()
    scale = pred.drop_duplicates("series_id").set_index("series_id")["naive_mse"]
    cmp = compare_to_baseline(
        y, pred["yhat_candidate"].to_numpy(), pred["yhat_baseline"].to_numpy(), series, cfg, scale
    )
    sel = bundle.selection
    folds = [
        {
            "fold": f["fold"],
            "n_train": f["n_train"],
            "n_valid": f["n_valid"],
            "baseline": f["scores"][BASELINE],
            "candidate": f["scores"][CANDIDATE],
        }
        for f in sel["folds"]
    ]
    accuracy = {
        "primary_metric": cmp["primary_metric"],
        "baseline_name": BASELINE,
        "candidate_name": CANDIDATE,
        "selection_verdict": sel["verdict"],
        "baseline": cmp["baseline"],
        "candidate": cmp["candidate"],
        "delta_primary": cmp["delta_primary"],
        "delta_ci_low": cmp["delta_ci_low"],
        "delta_ci_high": cmp["delta_ci_high"],
        "ci_level": cmp["ci_level"],
        "n_boot": cmp["n_boot"],
        "cv_mean_baseline": sel["mean"][BASELINE],
        "cv_mean_candidate": sel["mean"][CANDIDATE],
        "cv_folds": folds,
        "n_folds_requested": sel["n_folds_requested"],
        "n_folds_effective": sel["n_folds_effective"],
        "selected": bundle.selected,
    }
    selected_q, _ = _quantile_frames(pred, cfg)
    calibration = calibration_report(y, selected_q, calibration_taus(cfg), cfg)
    test = {"n_obs": len(pred), "n_series": int(pred["series_id"].nunique())}
    return {"test": test, "accuracy": accuracy, "calibration": calibration}


# --- the pipeline --------------------------------------------------------------------------


def run(
    cfg: dict,
    data_path=None,
    sample: bool = False,
    out_dir=None,
    *,
    config_path=None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run every step and return the metrics dict that was written to ``reports/metrics.json``.

    ``sample=True`` uses the synthetic generator (``data_source = "synthetic"``, banner
    everywhere); otherwise the real panel is read from ``data_path`` or ``params.data.path``.
    ``out_dir`` is the output root (default: the project root). Raises ``FcstError`` at the first
    failing step, with ``exc.step`` set to the step name; nothing is written after that step.
    """
    say = log if log is not None else (lambda _msg: None)
    state = {"step": STEPS[0]}

    def step(name: str) -> None:
        state["step"] = name
        say(f"[{name}]")

    root = Path(out_dir) if out_dir is not None else DEFAULT_CONFIG_PATH.parent
    try:
        return _run(cfg, data_path, sample, root, config_path, step, say)
    except FcstError as exc:
        exc.step = state["step"]
        raise


def _run(cfg, data_path, sample, root, config_path, step, say) -> dict:
    reports, artifacts = root / "reports", root / "artifacts"
    seed = cfg["params"]["seed"]

    step("load")
    if sample:
        if data_path is not None:
            raise FcstError("E_CONFIG", "data_path: cannot be combined with sample")
        source = SYNTHETIC
        raw, _truth = generate(seed, cfg)
    else:
        source = REAL
        raw = _read_raw(_resolve_data_path(cfg, data_path, config_path), cfg)
    say(f"  data_source = {source}, {len(raw)} raw rows")

    step("validate")
    errors = validate(raw, cfg)
    if errors:
        raise FcstError(errors[0], f"panel failed validation: {', '.join(errors)}")

    step("clean")
    panel = clean(raw, cfg)
    dhash = data_hash(panel)

    step("origins")
    origins = build_origins(panel, cfg)
    if len(origins) == 0:
        raise FcstError("E_SPLIT_NO_TIME_AXIS", "no admissible origins in the panel")

    step("features")
    X = make_features(panel, origins, cfg)
    y = build_labels(panel, origins, cfg)
    corr = check_correlation(X, y, cfg)
    leakage = perturb_future_check(panel, origins, cfg)
    leakage = {**leakage, "max_abs_corr": float(corr.abs().max())}
    table = X.join(y)

    step("split")
    split = make_split(table, cfg)
    sizes = {k: len(getattr(split, k)) for k in ("train", "calib", "test", "purged")}
    say(f"  rows: {sizes}")

    step("train")
    bundle = train_bundle(table, panel, cfg)
    save_bundle(bundle, artifacts)
    say(f"  selected = {bundle.selected} (CV verdict {bundle.verdict})")

    step("predict_test")
    pred = _predict_test_once(bundle, split, panel, cfg)
    artifacts.mkdir(parents=True, exist_ok=True)
    pred.to_csv(artifacts / PREDICTIONS_NAME, index=False)

    step("metrics")
    metrics = {
        "data_source": source,
        "data_hash": dhash,
        "model_hash": bundle.model_hash,
        "leakage": leakage,
        "split": sizes,
        **_metrics_from_pred(pred, bundle, cfg),
    }
    q_model, q_base = _quantile_frames(pred, cfg)
    frame = pred[["series_id", TARGET_COLUMN, "yhat_baseline"]]
    agg = aggregate_policies(frame, q_base, q_model, cfg)
    metrics["business"] = business_block(agg, cfg)
    metrics = _plain(metrics)
    check_metrics_shape(metrics)

    step("decide")
    metrics["decision"] = decide(metrics, cfg)
    say(f"  verdict = {metrics['decision']['verdict']}")

    step("report")
    chart(metrics, reports / "tradeoff.png")
    render_report(metrics, cfg, config_path=config_path, out_dir=reports)

    step("write")
    _write_json(metrics, reports / METRICS_NAME)
    return metrics


# --- CLI -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fcst.pipeline")
    parser.add_argument("--sample", action="store_true", help="run on the synthetic generator")
    parser.add_argument("--data", default=None, help="real panel CSV (default: params.data.path)")
    parser.add_argument("--out", default=None, help="output root (default: project root)")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    if args.sample:
        print(banner(SYNTHETIC))
    try:
        cfg = load_config(args.config)
        metrics = run(
            cfg, args.data, args.sample, args.out, config_path=args.config, log=print
        )
    except FcstError as exc:
        print(f"error: {exc} (step: {getattr(exc, 'step', 'config')})", file=sys.stderr)
        return 1
    root = Path(args.out) if args.out is not None else DEFAULT_CONFIG_PATH.parent
    print(f"verdict: {metrics['decision']['verdict']}")
    print(f"wrote {root / 'reports' / METRICS_NAME}")
    print(f"wrote {root / 'reports' / 'report.md'}")
    if args.sample:
        print(banner(SYNTHETIC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
