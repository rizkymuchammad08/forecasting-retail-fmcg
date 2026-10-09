"""Configuration loading and validation, plus the project exception class.

``config.yaml`` is one of the two places where numbers live (the other is
``reports/metrics.json``). This module reads it and rejects anything
inconsistent before any other step runs. Every validation failure raises
``FcstError("E_CONFIG", ...)`` with the offending field name in the message.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NoReturn

import yaml

ERROR_CODES = (
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

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class FcstError(Exception):
    """The single exception class of the project.

    Tests assert on ``code``; ``message`` is for humans only.
    """

    def __init__(self, code: str, message: str = "") -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --- required fields and their types -------------------------------------

_INT = "int"
_NUM = "num"
_STR = "str"
_STRS = "list[str]"
_NUMS = "list[num]"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
    )


_TYPE_CHECKS = {
    _INT: (_is_int, "an integer"),
    _NUM: (_is_num, "a finite number"),
    _STR: (lambda v: isinstance(v, str), "a string"),
    _STRS: (
        lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
        "a list of strings",
    ),
    _NUMS: (
        lambda v: isinstance(v, list) and all(_is_num(x) for x in v),
        "a list of finite numbers",
    ),
}

REQUIRED_FIELDS: dict[str, str] = {
    "business.avg_unit_price": _NUM,
    "business.avg_unit_cost": _NUM,
    "business.holding_cost_rate_annual": _NUM,
    "business.salvage_ratio": _NUM,
    "business.overstock_writeoff_share": _NUM,
    "business.lead_time_periods": _INT,
    "business.review_period_periods": _INT,
    "business.periods_per_year": _NUM,
    "business.status_quo_buffer": _NUM,
    "params.seed": _INT,
    "params.horizon_periods": _INT,
    "params.season_length_periods": _INT,
    "params.origin_step_periods": _INT,
    "params.data.path": _STR,
    "params.data.max_gap_periods": _INT,
    "params.data.synthetic.n_series": _INT,
    "params.data.synthetic.n_periods": _INT,
    "params.schema.required_columns": _STRS,
    "params.domain.min_history_periods": _INT,
    "params.domain.inactive_periods": _INT,
    "params.leakage.max_abs_corr": _NUM,
    "params.split.train": _NUM,
    "params.split.calib": _NUM,
    "params.split.test": _NUM,
    "params.split.purge_gap_periods": _INT,
    "params.cv.n_folds": _INT,
    "params.cv.min_train_origins": _INT,
    "params.models.baseline": _STR,
    "params.models.candidate": _STR,
    "params.models.primary_metric": _STR,
    "params.calibration.scale_floor": _NUM,
    "params.tradeoff.taus": _NUMS,
    "params.evaluation.ci_level": _NUM,
    "params.evaluation.n_boot": _INT,
    "params.evaluation.calibration_tolerance": _NUM,
    "params.report.min_test_series": _INT,
    "params.report.min_test_obs": _INT,
}

REQUIRED_COLUMNS = ("sku_id", "location_id", "series_id", "period_start", "demand")
PRIMARY_METRICS = ("wape", "rmsse")

_BUSINESS_KEYS = frozenset(
    p.split(".", 1)[1] for p in REQUIRED_FIELDS if p.startswith("business.")
)
_FORBIDDEN_UNDER_PARAMS = _BUSINESS_KEYS | {"business", "tau_star", "tau*"}


# --- helpers ---------------------------------------------------------------


def _fail(field: str, msg: str) -> NoReturn:
    raise FcstError("E_CONFIG", f"{field}: {msg}")


def _require(cond: bool, field: str, msg: str) -> None:
    if not cond:
        _fail(field, msg)


def _get(cfg: dict, path: str) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            _fail(path, "missing required field")
        node = node[part]
    return node


def _scan_forbidden(node: Any, prefix: str) -> None:
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        full = f"{prefix}.{key}"
        if str(key) in _FORBIDDEN_UNDER_PARAMS:
            _fail(full, "business values and tau_star must not appear under params")
        _scan_forbidden(value, full)


def cost_rates(business: dict) -> tuple[float, float]:
    """Return ``(c_under, c_over)`` per unit from the business block.

    c_under = price - cost
    c_over  = cost * holding_rate * review_periods / periods_per_year
              + cost * (1 - salvage_ratio) * writeoff_share
    """
    price = business["avg_unit_price"]
    cost = business["avg_unit_cost"]
    c_under = price - cost
    c_over = (
        cost
        * business["holding_cost_rate_annual"]
        * business["review_period_periods"]
        / business["periods_per_year"]
        + cost
        * (1.0 - business["salvage_ratio"])
        * business["overstock_writeoff_share"]
    )
    return c_under, c_over


# --- public API ------------------------------------------------------------


def validate_config(cfg: Any) -> None:
    """Raise ``FcstError("E_CONFIG")`` naming the field if ``cfg`` is invalid."""
    _require(isinstance(cfg, dict), "config", "must be a mapping")
    _require(isinstance(cfg.get("params"), dict), "params", "must be a mapping")
    _scan_forbidden(cfg["params"], "params")

    v: dict[str, Any] = {}
    for path, kind in REQUIRED_FIELDS.items():
        value = _get(cfg, path)
        check, desc = _TYPE_CHECKS[kind]
        _require(check(value), path, f"must be {desc}")
        v[path] = value

    # business
    _require(v["business.avg_unit_cost"] > 0, "business.avg_unit_cost", "must be > 0")
    _require(
        v["business.avg_unit_price"] > v["business.avg_unit_cost"],
        "business.avg_unit_price",
        "must be > business.avg_unit_cost",
    )
    _require(
        v["business.holding_cost_rate_annual"] > 0,
        "business.holding_cost_rate_annual",
        "must be > 0",
    )
    for key in ("salvage_ratio", "overstock_writeoff_share"):
        _require(0 <= v[f"business.{key}"] <= 1, f"business.{key}", "must be in [0, 1]")
    _require(
        v["business.status_quo_buffer"] >= 0, "business.status_quo_buffer", "must be >= 0"
    )
    _require(
        v["business.lead_time_periods"] >= 0, "business.lead_time_periods", "must be >= 0"
    )
    _require(
        v["business.review_period_periods"] >= 1,
        "business.review_period_periods",
        "must be >= 1",
    )
    _require(
        v["business.periods_per_year"] > 0, "business.periods_per_year", "must be > 0"
    )

    # horizon and derived tau*
    horizon = v["params.horizon_periods"]
    _require(
        horizon
        == v["business.lead_time_periods"] + v["business.review_period_periods"],
        "params.horizon_periods",
        "must equal business.lead_time_periods + business.review_period_periods",
    )
    c_under, c_over = cost_rates(cfg["business"])
    _require(c_under > 0 and c_over > 0, "tau_star", "derived costs must be > 0")
    tau_star = c_under / (c_under + c_over)
    _require(0 < tau_star < 1, "tau_star", "derived value must be in (0, 1)")

    # params: scalars
    _require(v["params.seed"] >= 0, "params.seed", "must be >= 0")
    for key in ("season_length_periods", "origin_step_periods"):
        _require(v[f"params.{key}"] >= 1, f"params.{key}", "must be >= 1")

    # params: data
    _require(v["params.data.path"].strip() != "", "params.data.path", "must not be empty")
    _require(
        v["params.data.max_gap_periods"] >= 0, "params.data.max_gap_periods", "must be >= 0"
    )
    _require(
        v["params.data.synthetic.n_series"] >= 10,
        "params.data.synthetic.n_series",
        "must be >= 10",
    )
    min_periods = v["params.domain.min_history_periods"] + 3 * horizon
    _require(
        v["params.data.synthetic.n_periods"] > min_periods,
        "params.data.synthetic.n_periods",
        "must be > domain.min_history_periods + 3 * horizon_periods",
    )
    columns = v["params.schema.required_columns"]
    _require(
        len(set(columns)) == len(columns),
        "params.schema.required_columns",
        "must not contain duplicates",
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    _require(
        not missing,
        "params.schema.required_columns",
        f"must include {', '.join(missing)}",
    )

    # params: domain, leakage
    for key in ("min_history_periods", "inactive_periods"):
        _require(v[f"params.domain.{key}"] >= 1, f"params.domain.{key}", "must be >= 1")
    _require(
        0 < v["params.leakage.max_abs_corr"] <= 1,
        "params.leakage.max_abs_corr",
        "must be in (0, 1]",
    )

    # params: split
    for key in ("train", "calib", "test"):
        _require(v[f"params.split.{key}"] > 0, f"params.split.{key}", "must be > 0")
    fractions = sum(v[f"params.split.{k}"] for k in ("train", "calib", "test"))
    _require(
        math.isclose(fractions, 1.0, abs_tol=1e-9),
        "params.split",
        "train + calib + test must equal 1",
    )
    _require(
        v["params.split.purge_gap_periods"] >= horizon,
        "params.split.purge_gap_periods",
        "must be >= params.horizon_periods",
    )

    # params: cv
    _require(v["params.cv.n_folds"] >= 2, "params.cv.n_folds", "must be >= 2")
    _require(
        v["params.cv.min_train_origins"] >= 1, "params.cv.min_train_origins", "must be >= 1"
    )

    # params: models
    _require(
        v["params.models.baseline"] != v["params.models.candidate"],
        "params.models.candidate",
        "must differ from params.models.baseline",
    )
    _require(
        v["params.models.primary_metric"] in PRIMARY_METRICS,
        "params.models.primary_metric",
        f"must be one of {', '.join(PRIMARY_METRICS)}",
    )

    # params: calibration, tradeoff, evaluation, report
    _require(
        v["params.calibration.scale_floor"] > 0, "params.calibration.scale_floor", "must be > 0"
    )
    taus = v["params.tradeoff.taus"]
    _require(len(taus) > 0, "params.tradeoff.taus", "must not be empty")
    _require(
        all(0 < t < 1 for t in taus), "params.tradeoff.taus", "every value must be in (0, 1)"
    )
    _require(
        0 < v["params.evaluation.ci_level"] < 1,
        "params.evaluation.ci_level",
        "must be in (0, 1)",
    )
    _require(v["params.evaluation.n_boot"] >= 1, "params.evaluation.n_boot", "must be >= 1")
    _require(
        0 <= v["params.evaluation.calibration_tolerance"] < 1,
        "params.evaluation.calibration_tolerance",
        "must be in [0, 1)",
    )
    for key in ("min_test_series", "min_test_obs"):
        _require(v[f"params.report.{key}"] >= 1, f"params.report.{key}", "must be >= 1")


def load_config(path: str | Path | None = None) -> dict:
    """Read and validate a config file (default: ``config.yaml`` at the repo root)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FcstError("E_CONFIG", f"config: cannot read '{cfg_path}': {exc}") from exc
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise FcstError("E_CONFIG", f"config: invalid YAML in '{cfg_path}': {exc}") from exc
    validate_config(cfg)
    return cfg
