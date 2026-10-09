"""Verdict, strict report rendering and trade-off chart (P5, stop point 1c).

``decide(metrics, cfg)`` is a pure function of ``metrics.json`` and ``config.yaml``; a missing
input key raises ``E_RENDER_MISSING_KEY`` and never falls back to a default. ``reasons`` and
``flags`` are codes only; their wording lives in ``REASON_TEXT`` / ``FLAG_TEXT`` below.

Verdict priority (first match wins): ``use_baseline`` > ``keep_status_quo`` > ``pilot_then_scale``.

* ``use_baseline``: the candidate did not beat the baseline in CV, or its test delta on the
  primary metric is not negative.
* ``keep_status_quo``: the model may be fine, but the value is not shown: too few test series or
  test origin rows (the gate counts origin rows, ``test.n_obs``), no equal-service match, or a
  net value against the status quo that is not positive.
* ``pilot_then_scale``: everything else. Never "scale now".

Flags never change the verdict. When the CV selection is the baseline, the report adds a note that the equal-service result is an
identity (not a finding); this is a rendering note, not a flag, and changes nothing else.
``headline_supported`` is true only for ``pilot_then_scale`` with
an available equal-service comparison whose ``ci_low`` is above zero.

``metrics.json`` is shaped by ``METRICS_SKELETON`` (checked by ``check_metrics_shape``, no schema
library). ``source_hash`` = SHA-256 of the bytes of ``config.yaml`` plus the canonical JSON of the
metrics. The template holds no numeric literal outside ``TEMPLATE_NUMBER_WHITELIST``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from fcst.config import DEFAULT_CONFIG_PATH, FcstError
from fcst.data import banner as data_banner

ROOT = DEFAULT_CONFIG_PATH.parent
TEMPLATE_NAME = "report.md.j2"
CHART_NAME = "tradeoff.png"
REPORT_NAME = "report.md"
TEMPLATE_NUMBER_WHITELIST = frozenset({"0"})  # sign tests only

VERDICTS = ("use_baseline", "keep_status_quo", "pilot_then_scale")
FLAGS = (
    "F_SYNTHETIC_DATA",
    "F_CALIBRATION_OUT_OF_TOLERANCE",
    "F_ACCURACY_CI_INCLUDES_ZERO",
    "F_NET_VALUE_CI_INCLUDES_ZERO",
    "F_MORE_INVENTORY_THAN_STATUS_QUO",
    "F_LEAKAGE_CHECK_NOT_PASSED",
)
REASON_TEXT = {
    "R_CV_CANDIDATE_NOT_BETTER": "The candidate did not beat the baseline on the CV folds.",
    "R_TEST_DELTA_NOT_NEGATIVE": "The candidate is not better than the baseline on the test split.",
    "R_TEST_SERIES_BELOW_MIN": "Fewer test series than the configured minimum.",
    "R_TEST_OBS_BELOW_MIN": "Fewer test origin rows than the configured minimum.",
    "R_EQUAL_SERVICE_UNAVAILABLE": "No equal-service comparison is available.",
    "R_NET_VALUE_NOT_POSITIVE": "Net value against the status quo is not positive.",
    "R_CANDIDATE_BEATS_BASELINE": "The candidate beats the baseline in CV and on the test split.",
    "R_VALUE_CHECKS_PASSED": "Sample gates, equal service and net value checks passed.",
}
FLAG_TEXT = {
    "F_SYNTHETIC_DATA": "The data is synthetic; nothing here describes real sales.",
    "F_CALIBRATION_OUT_OF_TOLERANCE": "At least one quantile level is outside the coverage tolerance.",
    "F_ACCURACY_CI_INCLUDES_ZERO": "The accuracy difference interval includes zero.",
    "F_NET_VALUE_CI_INCLUDES_ZERO": "The net value interval against the status quo includes zero.",
    "F_MORE_INVENTORY_THAN_STATUS_QUO": (
        "The newsvendor policy holds more inventory than the status quo "
        "(it buys service with stock, which is not a stock reduction)."
    ),
    "F_LEAKAGE_CHECK_NOT_PASSED": "The future-perturbation leakage check did not pass.",
}
SELECTED_IS_BASELINE_NOTE = (
    "The model selected in cross-validation is the baseline itself, so the model and baseline "
    "newsvendor policies are the same policy. The equal-service difference is zero by "
    "construction (zero-width interval). It is not an empirical finding and says nothing about "
    "whether the candidate could save inventory."
)
LIMITATIONS = {
    "no_carry_over": "No carry-over: every cycle starts with no stock and is independent.",
    "supplier_constraints": "No supplier constraints: no minimum order, pack size or capacity.",
    "substitution": "No substitution between products or locations.",
    "demand_proxied_by_sales": "Demand is proxied by sales; there is no stockout column.",
    "delisting_tail": "The fade-out of delisted series is under-represented in train and test.",
}

# --- locked shape of metrics.json -------------------------------------------------------
# Leaf tokens: num, int, str, bool, nums; a trailing "?" allows null. {..} nests; [spec] is a
# list whose items all match spec.

_POLICY = {
    "n_origins": "int",
    "stockout_rate": "num",
    "avg_inventory_units": "num",
    "avg_inventory_value": "num",
    "dio_periods": "num?",
    "net_value": "num",
}
_POINT = _POLICY | {"tau": "num", "is_tau_star": "bool"}
_BOOT = {"ci_level": "num", "n_boot": "int", "n_series": "int", "seed": "int"}
METRICS_SKELETON = {
    "data_source": "str",
    "data_hash": "str",
    "model_hash": "str",
    "test": {"n_obs": "int", "n_series": "int"},
    "leakage": {"passed": "bool", "n_origin_dates": "int", "n_rows": "int"},
    "accuracy": {
        "primary_metric": "str",
        "baseline_name": "str",
        "candidate_name": "str",
        "selection_verdict": "str",
        "baseline": "num",
        "candidate": "num",
        "delta_primary": "num",
        "delta_ci_low": "num",
        "delta_ci_high": "num",
        "ci_level": "num",
        "cv_mean_baseline": "num",
        "cv_mean_candidate": "num",
        "cv_folds": [
            {"fold": "int", "n_train": "int", "n_valid": "int", "baseline": "num", "candidate": "num"}
        ],
    },
    "calibration": {
        "n": "int",
        "tolerance": "num",
        "max_abs_coverage_error": "num",
        "all_within_tolerance": "bool",
        "rows": [
            {
                "tau": "num",
                "coverage": "num",
                "coverage_error": "num",
                "within_tolerance": "bool",
                "pinball_loss": "num",
            }
        ],
    },
    "business": {
        "value_label": "str",
        "horizon_periods": "int",
        "n_origins": "int",
        "n_series": "int",
        "c_under": "num",
        "c_over": "num",
        "tau_star": "num",
        "status_quo_buffer": "num",
        "policies": {k: _POLICY for k in ("status_quo", "baseline_newsvendor", "model_newsvendor")},
        "tradeoff": {"taus": "nums", "baseline": [_POINT], "model": [_POINT]},
        "equal_service": {
            "available": "bool",
            "target_stockout_rate": "num",
            "inventory_model": "num",
            "inventory_baseline": "num?",
            "inventory_reduction_abs": "num?",
            "inventory_reduction_rel": "num?",
            "reason": "str?",
        },
        "net_value_delta": {
            "model_vs_status_quo": {
                "value": "num",
                "ci_low": "num",
                "ci_high": "num",
                "scope": "str",
            }
            | _BOOT
        },
        "inventory_reduction_rel": {
            "available": "bool",
            "value": "num?",
            "ci_low": "num?",
            "ci_high": "num?",
            "n_boot_available": "int",
            "label": "str",
        }
        | _BOOT,
    },
}


def _miss(path: str, why: str = "missing") -> FcstError:
    return FcstError("E_RENDER_MISSING_KEY", f"{path}: {why}")


def _leaf_ok(token: str, value) -> bool:
    if value is None:
        return token.endswith("?")
    token = token.rstrip("?")
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
    if token == "num":
        return is_num and math.isfinite(value)
    if token == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if token == "str":
        return isinstance(value, str)
    if token == "bool":
        return isinstance(value, bool)
    if token == "nums":
        return isinstance(value, list) and all(_leaf_ok("num", v) for v in value)
    raise ValueError(f"bad skeleton token {token!r}")


def check_metrics_shape(metrics, skeleton=METRICS_SKELETON, path: str = "metrics") -> None:
    """Raise ``E_RENDER_MISSING_KEY`` (naming the path) if ``metrics`` does not fit the skeleton.

    Extra keys are allowed. The code list is closed, so a wrong type uses the same code.
    """
    if isinstance(skeleton, dict):
        if not isinstance(metrics, dict):
            raise _miss(path, "expected an object")
        for key, spec in skeleton.items():
            if key not in metrics:
                raise _miss(f"{path}.{key}")
            check_metrics_shape(metrics[key], spec, f"{path}.{key}")
    elif isinstance(skeleton, list):
        if not isinstance(metrics, list) or not metrics:
            raise _miss(path, "expected a non-empty list")
        for i, item in enumerate(metrics):
            check_metrics_shape(item, skeleton[0], f"{path}[{i}]")
    elif not _leaf_ok(skeleton, metrics):
        raise _miss(path, f"expected {skeleton}, got {metrics!r}")


# --- decide -----------------------------------------------------------------------------


def _need(node, *keys, root: str = "metrics"):
    """Walk ``node`` along ``keys``; a missing key raises instead of defaulting."""
    path = root
    for key in keys:
        path += f".{key}"
        if not isinstance(node, dict) or key not in node:
            raise _miss(path)
        node = node[key]
    return node


def _val(m: dict, *keys, nullable: bool = False):
    value = _need(m, *keys)
    if value is None and not nullable:
        raise _miss("metrics." + ".".join(keys), "null is not allowed")
    return value


def decide(metrics: dict, cfg: dict) -> dict:
    """Verdict, reason codes, flag codes and ``headline_supported`` (see the module docstring)."""
    gate_series = _need(cfg, "params", "report", "min_test_series", root="config")
    gate_obs = _need(cfg, "params", "report", "min_test_obs", root="config")
    cv_verdict = _val(metrics, "accuracy", "selection_verdict")
    delta = _val(metrics, "accuracy", "delta_primary")
    n_series, n_obs = _val(metrics, "test", "n_series"), _val(metrics, "test", "n_obs")
    available = _val(metrics, "business", "equal_service", "available")
    nv_keys = ("business", "net_value_delta", "model_vs_status_quo")
    net = _val(metrics, *nv_keys, "value")
    net_low, net_high = _val(metrics, *nv_keys, "ci_low"), _val(metrics, *nv_keys, "ci_high")
    red_low = _val(metrics, "business", "inventory_reduction_rel", "ci_low", nullable=True)
    acc_low, acc_high = _val(metrics, "accuracy", "delta_ci_low"), _val(metrics, "accuracy", "delta_ci_high")
    policies = _val(metrics, "business", "policies")
    inv_model = _need(policies, "model_newsvendor", "avg_inventory_value")
    inv_status_quo = _need(policies, "status_quo", "avg_inventory_value")
    raised = {
        "F_SYNTHETIC_DATA": _val(metrics, "data_source") == "synthetic",
        "F_CALIBRATION_OUT_OF_TOLERANCE": not _val(metrics, "calibration", "all_within_tolerance"),
        "F_ACCURACY_CI_INCLUDES_ZERO": acc_low <= 0 <= acc_high,
        "F_NET_VALUE_CI_INCLUDES_ZERO": net_low <= 0 <= net_high,
        "F_MORE_INVENTORY_THAN_STATUS_QUO": inv_model > inv_status_quo,
        "F_LEAKAGE_CHECK_NOT_PASSED": not _val(metrics, "leakage", "passed"),
    }

    baseline_reasons = []
    if cv_verdict == "use_baseline":
        baseline_reasons.append("R_CV_CANDIDATE_NOT_BETTER")
    if delta >= 0:
        baseline_reasons.append("R_TEST_DELTA_NOT_NEGATIVE")
    status_quo_reasons = []
    if n_series < gate_series:
        status_quo_reasons.append("R_TEST_SERIES_BELOW_MIN")
    if n_obs < gate_obs:
        status_quo_reasons.append("R_TEST_OBS_BELOW_MIN")
    if not available:
        status_quo_reasons.append("R_EQUAL_SERVICE_UNAVAILABLE")
    if net <= 0:
        status_quo_reasons.append("R_NET_VALUE_NOT_POSITIVE")

    if baseline_reasons:
        verdict, reasons = "use_baseline", baseline_reasons
    elif status_quo_reasons:
        verdict, reasons = "keep_status_quo", status_quo_reasons
    else:
        verdict, reasons = "pilot_then_scale", ["R_CANDIDATE_BEATS_BASELINE", "R_VALUE_CHECKS_PASSED"]
    return {
        "verdict": verdict,
        "reasons": reasons,
        "flags": [f for f in FLAGS if raised[f]],
        "headline_supported": bool(
            verdict == "pilot_then_scale" and available and red_low is not None and red_low > 0
        ),
    }


# --- hashes, filters, rendering -----------------------------------------------------------


def source_hash(metrics: dict, config_path: str | Path | None = None) -> str:
    """SHA-256 over the bytes of ``config.yaml`` and the canonical JSON of the metrics."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    try:
        config_bytes = path.read_bytes()
    except OSError as exc:
        raise FcstError("E_CONFIG", f"config: cannot read '{path}': {exc}") from exc
    canon = json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256()
    digest.update(config_bytes)
    digest.update(b"\n--metrics--\n")
    digest.update(canon.encode("utf-8"))
    return digest.hexdigest()


def _na(fn):
    return lambda v: "n/a" if v is None else fn(v)


FILTERS = {  # fixed formats, so the template needs no digits
    "ratio": _na(lambda v: f"{v:.3f}"),
    "pct": _na(lambda v: f"{100 * v:.1f}%"),
    "money": _na(lambda v: f"{v:,.0f}"),
    "units": _na(lambda v: f"{v:,.1f}"),
    "count": _na(lambda v: f"{int(v):,}"),
    "signed": _na(lambda v: f"{v:+,.0f}"),
}


def template_number_literals(text: str) -> list[str]:
    """Numeric literals (and identifiers holding digits) in a template that are not in
    ``TEMPLATE_NUMBER_WHITELIST``.
    """
    found = re.findall(r"\w*\d[\w.]*", text)
    return [t for t in found if t not in TEMPLATE_NUMBER_WHITELIST]


def render_report(
    metrics: dict,
    cfg: dict,
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    template_dir: str | Path | None = None,
) -> Path:
    """Render ``reports/report.md`` with ``StrictUndefined`` and return its path.

    The report links ``tradeoff.png``; call ``chart`` first so the link resolves.
    """
    import jinja2

    check_metrics_shape(metrics)
    decision = decide(metrics, cfg)
    loader = jinja2.FileSystemLoader(str(Path(template_dir) if template_dir else ROOT / "templates"))
    env = jinja2.Environment(
        loader=loader,
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(FILTERS)
    context = {
        "m": metrics,
        "decision": decision,
        "banner": data_banner(metrics["data_source"]),
        "reason_text": REASON_TEXT,
        "flag_text": FLAG_TEXT,
        # CV chose the baseline: model and baseline policies coincide (see newsvendor notes).
        "selected_is_baseline": _val(metrics, "accuracy", "selection_verdict") == "use_baseline",
        "selected_is_baseline_note": SELECTED_IS_BASELINE_NOTE,
        "limitations": LIMITATIONS,
        "chart_name": CHART_NAME,
        "min_test_series": _need(cfg, "params", "report", "min_test_series", root="config"),
        "min_test_obs": _need(cfg, "params", "report", "min_test_obs", root="config"),
        "source_hash": source_hash(metrics, config_path),
    }
    try:
        text = env.get_template(TEMPLATE_NAME).render(**context)
    except jinja2.UndefinedError as exc:
        raise _miss("template", str(exc)) from exc
    except jinja2.TemplateNotFound as exc:
        raise FcstError("E_CONFIG", f"template '{exc}' not found") from exc
    out = Path(out_dir) if out_dir else ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    target = out / REPORT_NAME
    target.write_text(text, encoding="utf-8")
    return target


# --- chart --------------------------------------------------------------------------------


def chart(metrics: dict, out_path: str | Path | None = None) -> Path:
    """Write ``tradeoff.png``: stockout rate (x) against average inventory value (y) for the
    baseline and model curves, with the status quo and tau* marked. Synthetic data adds a banner.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    check_metrics_shape(metrics)
    biz = metrics["business"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, style in (("baseline", "o--"), ("model", "s-")):
        pts = biz["tradeoff"][name]
        ax.plot(
            [p["stockout_rate"] for p in pts],
            [p["avg_inventory_value"] for p in pts],
            style,
            markersize=4,
            label=f"{name} (quantile policy, per tau)",
        )
        star = [p for p in pts if p["is_tau_star"]]
        if star:
            ax.scatter(
                [p["stockout_rate"] for p in star],
                [p["avg_inventory_value"] for p in star],
                marker="*",
                s=260,
                zorder=5,
                edgecolors="black",
                label=f"{name} at tau*",
            )
    sq = biz["policies"]["status_quo"]
    ax.scatter(
        [sq["stockout_rate"]],
        [sq["avg_inventory_value"]],
        marker="X",
        s=140,
        color="black",
        zorder=5,
        label="status quo (not at equal service)",
    )
    ax.set_xlabel("Stockout rate (share of cycles with unmet demand)")
    ax.set_ylabel("Average inventory value")
    ax.set_title("Stockout rate vs working capital (conditional value)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    note = data_banner(metrics["data_source"])
    if note:
        fig.text(0.5, 0.01, note, ha="center", color="red", fontsize=9, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    target = Path(out_path) if out_path else ROOT / "reports" / CHART_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target
