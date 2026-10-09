# Reference

Facts for configuring, feeding, running and extending the project. For the workflow see the [User Guide](user-guide.md); for the reasoning see [Methodology](methodology.md).

Values are not repeated here. `config.yaml` is the single source for inputs and `reports/metrics.json` for outputs.

## Contents

- [Commands](#commands)
- [Import path](#import-path)
- [Panel format](#panel-format)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Verdicts, reasons and flags](#verdicts-reasons-and-flags)
- [Error codes](#error-codes)
- [Modules](#modules)
- [Testing and linting](#testing-and-linting)
- [Packaging and dependencies](#packaging-and-dependencies)
- [Extending the project](#extending-the-project)
- [Glossary](#glossary)

## Commands

### Make targets

| Target | Does |
|---|---|
| `make install` | Installs the packages in `requirements.txt` |
| `make install-editable` | Installs the project editable with the `dev` extra (puts `fcst` on the import path and adds console scripts) |
| `make lint` | Runs ruff over the project |
| `make test` | Runs pytest over `tests/` |
| `make sample` | Writes the synthetic sample file |
| `make check-sample` | Regenerates the sample twice, compares data hashes, checks the file size limit and compares an existing sample file with the generator output |
| `make all-sample` | Lint, tests, sample, sample check, then the pipeline on synthetic data |
| `make all` | Lint, tests, then the pipeline on the real panel |
| `make clean` | Removes generated output and caches |

`make all` stops with `E_DATA_FILE_MISSING` if the panel does not exist, after lint and tests. `make all-sample` is never a source of results.

### Command line

```
python -m fcst.pipeline [--sample] [--data PATH] [--out DIR] [--config PATH]
python -m fcst.data --write-sample | --check-sample
```

| Option | Meaning |
|---|---|
| `--sample` | Run on the in-memory synthetic generator (not on the sample file), so a stale or missing sample file cannot change a run |
| `--data PATH` | Real panel CSV; default is `params.data.path` |
| `--out DIR` | Output root; default is the project root |
| `--config PATH` | Alternative `config.yaml` |

Console scripts after an editable install: `fcst-pipeline` is `python -m fcst.pipeline`, and `fcst-data` is `python -m fcst.data`.

Exit status is zero on success. On failure the CLI prints `error: CODE: message (step: name)` to stderr, exits with status one and writes nothing after the failing step. On success it prints the verdict and the paths it wrote; with `--sample` the synthetic banner is printed at the start and at the end.

## Import path

The code uses a `src/` layout, so `fcst` is importable only with `src` on the import path.

| Way | When | How |
|---|---|---|
| `make` targets | Default; the Makefile exports `PYTHONPATH=src` | `make test`, `make all-sample` |
| By hand | Direct calls outside `make` | Linux and macOS: `export PYTHONPATH=src`. PowerShell: `$env:PYTHONPATH = "src"`. cmd: `set PYTHONPATH=src` |
| Editable install | No `PYTHONPATH` at all, or the console scripts | `make install-editable` |

`pytest` needs no variable: `pyproject.toml` sets its `pythonpath`. The symptom of a missing path is `ModuleNotFoundError: No module named 'fcst'`.

Install editable, never as a regular wheel: `config.yaml` and `templates/` are read from the project root next to `src/`, and a regular install would copy only the code into site-packages and lose them.

## Panel format

A CSV, one row per series and period.

| Column | Meaning |
|---|---|
| `sku_id` | Product identifier |
| `location_id` | Store or location identifier |
| `series_id` | Series identifier; built as the SKU and location joined by an at sign if you only supply the other two |
| `period_start` | ISO date of the period start; all series on one regular grid |
| `demand` | Units sold in the period; finite and not negative |

Rules enforced by `validate`, `load` and `clean`:

- The key (`series_id`, `period_start`) is unique.
- Identifiers are never blank, and the panel is not empty.
- Gaps inside a series are filled with zero up to `params.data.max_gap_periods`; longer gaps are an error.
- Columns not listed in `params.schema.required_columns` are dropped. This is deliberate: realisation columns are a leakage risk.
- A missing file is `E_DATA_FILE_MISSING`; there is no fallback to synthetic data.

The period length is inferred from the data (most frequent positive spacing; daily if none). A panel that stores only sale rows and is very sparse may be inferred wrongly.

## Configuration

All inputs are in `config.yaml`. Loading validates the whole file and raises `E_CONFIG` naming the offending key. Values are illustrative assumptions, not measurements.

### Business block

| Key | Meaning | Rule |
|---|---|---|
| `business.avg_unit_price` | Average selling price | Greater than cost |
| `business.avg_unit_cost` | Average unit cost | Greater than zero |
| `business.holding_cost_rate_annual` | Holding cost per year as a share of unit cost | Greater than zero |
| `business.salvage_ratio` | Salvage value of leftover units as a share of unit cost | Between zero and one |
| `business.overstock_writeoff_share` | Share of leftover units written off (set with the product's shelf life in mind) | Between zero and one |
| `business.lead_time_periods` | Periods from order to receipt | Not negative |
| `business.review_period_periods` | Periods between orders | At least one |
| `business.periods_per_year` | Periods in a year (turns the annual holding rate into a per-period one) | Greater than zero |
| `business.status_quo_buffer` | Relative buffer on the baseline point forecast that models today's practice | Not negative |

Derived and never stored: the target service level, from the cost rates. It must fall strictly between zero and one.

### Parameters

| Key | Meaning | Rule |
|---|---|---|
| `params.seed` | Seed for all randomness | Not negative |
| `params.horizon_periods` | Forecast horizon | Must equal lead time plus review period |
| `params.season_length_periods` | Seasonal cycle in periods | At least one |
| `params.origin_step_periods` | Spacing of forecast origins | At least one |
| `params.data.path` | Real panel location | Not empty |
| `params.data.max_gap_periods` | Longest gap that is filled with zero | Not negative |
| `params.data.synthetic.n_series`, `.n_periods` | Size of the synthetic panel (the only place its size comes from) | Enough series; enough periods for minimum history plus several horizons |
| `params.schema.required_columns` | Columns that are read | No duplicates; must include all panel columns above |
| `params.domain.min_history_periods` | Minimum history before an origin counts | At least one; not below the season length |
| `params.domain.inactive_periods` | Window with no demand that marks a series inactive | At least one |
| `params.leakage.max_abs_corr` | Correlation fence between a feature and the target | Greater than zero, at most one |
| `params.split.train`, `.calib`, `.test` | Time shares | Each positive; sum to one |
| `params.split.purge_gap_periods` | Gap between split parts | At least the horizon |
| `params.cv.n_folds` | Cross-validation folds | At least two |
| `params.cv.min_train_origins` | Origins required in each training part | At least one |
| `params.models.baseline`, `.candidate` | Model names: `seasonal_naive`, `gbm_lag` | Must differ; other names raise `E_CONFIG` |
| `params.models.primary_metric` | Selection and comparison metric | `wape` or `rmsse` |
| `params.calibration.scale_floor` | Floor for per-series scale and level | Greater than zero |
| `params.tradeoff.taus` | Service levels for the trade-off curves | Non-empty; each strictly between zero and one |
| `params.evaluation.ci_level` | Interval level | Strictly between zero and one |
| `params.evaluation.n_boot` | Bootstrap resamples | At least one |
| `params.evaluation.calibration_tolerance` | Allowed coverage error | Zero up to, not including, one |
| `params.report.min_test_series` | Minimum test series for a value claim | At least one |
| `params.report.min_test_obs` | Minimum test origin rows for a value claim | At least one |

Business values and the service target must never appear under `params`; the loader rejects them. Comments in `config.yaml` carry no hand-typed numbers.

## Outputs

Written under the output root (default: the project root). Generated, not committed.

| File | Content |
|---|---|
| `reports/report.md` | Rendered report with source hash in the footer |
| `reports/metrics.json` | Every number of the run |
| `reports/tradeoff.png` | Trade-off chart (stockout against inventory), banner on synthetic data |
| `artifacts/bundle.joblib` | Fitted models and calibrators (a pickle) |
| `artifacts/bundle.json` | SHA-256 of the bundle, model hash, per-model hashes, calibrated levels, offsets |
| `artifacts/test_predictions.csv` | The single test prediction table behind every later number |

### `metrics.json` layout

Shape checked by `check_metrics_shape`; extra keys are allowed.

| Key | Content |
|---|---|
| `data_source`, `data_hash`, `model_hash` | Identify the run (synthetic or real; hashes of data and model) |
| `test` | Number of test origin rows and test series |
| `split` | Row counts per part |
| `leakage` | Whether the future-perturbation proof passed, origin dates and rows checked, the largest correlation seen |
| `accuracy` | Primary metric, baseline and candidate names, selection verdict, scores, difference with interval, cross-validation means and per-fold scores |
| `calibration` | Per-level coverage, coverage error, tolerance check and pinball loss |
| `business` | Cost rates, target service level, the three policies, trade-off curves, equal-service block, net-value difference, inventory reduction, value label |
| `decision` | Verdict, reason codes, flag codes, whether the headline is supported |

A report is rendered from this file only, strictly: a missing key stops the run.

## Verdicts, reasons and flags

### Verdicts

`use_baseline`, `keep_status_quo`, `pilot_then_scale`. Priority in that order; the rules are in [Methodology](methodology.md#verdict-and-flags).

### Reason codes

| Code | Verdict | Meaning |
|---|---|---|
| `R_CV_CANDIDATE_NOT_BETTER` | `use_baseline` | The candidate did not beat the baseline on the cross-validation folds |
| `R_TEST_DELTA_NOT_NEGATIVE` | `use_baseline` | The candidate is not better than the baseline on the test split |
| `R_TEST_SERIES_BELOW_MIN` | `keep_status_quo` | Fewer test series than `report.min_test_series` |
| `R_TEST_OBS_BELOW_MIN` | `keep_status_quo` | Fewer test origin rows than `report.min_test_obs` |
| `R_EQUAL_SERVICE_UNAVAILABLE` | `keep_status_quo` | No equal-service comparison is available |
| `R_NET_VALUE_NOT_POSITIVE` | `keep_status_quo` | Net value against the status quo is not positive |
| `R_CANDIDATE_BEATS_BASELINE` | `pilot_then_scale` | The candidate wins in cross-validation and on the test split |
| `R_VALUE_CHECKS_PASSED` | `pilot_then_scale` | Sample gates, equal service and net value checks passed |

Reasons hold only the codes of the chosen verdict.

### Flags

Flags never change the verdict.

| Code | Raised when |
|---|---|
| `F_SYNTHETIC_DATA` | The data is synthetic; nothing describes real sales |
| `F_CALIBRATION_OUT_OF_TOLERANCE` | Coverage misses the target at one or more levels by more than the tolerance |
| `F_ACCURACY_CI_INCLUDES_ZERO` | The accuracy difference may be noise |
| `F_NET_VALUE_CI_INCLUDES_ZERO` | The net-value difference may be noise |
| `F_MORE_INVENTORY_THAN_STATUS_QUO` | The policy holds more inventory than the status quo, so it buys service rather than saving stock |
| `F_LEAKAGE_CHECK_NOT_PASSED` | The future-perturbation proof did not pass |

Wording for reasons and flags lives in `report.py` (`REASON_TEXT`, `FLAG_TEXT`), not in the codes or the template.

## Error codes

One exception class, `FcstError(code, message)`. The code list is closed. Tests assert on the **code**, never the message. `validate` returns data codes and does not raise for bad data; loaders and pipeline steps raise.

| Code | Raised for |
|---|---|
| `E_CONFIG` | Missing or inconsistent config field (message names it); unsupported model name; unreadable config or template; corrupt metrics file |
| `E_DATA_FILE_MISSING` | No file at the real-data path |
| `E_DATA_COLUMNS_MISSING` | Missing required column, empty panel, blank or null identifier |
| `E_DATA_DUPLICATE_KEY` | Repeated series and period |
| `E_DATA_DATE_INVALID` | Unparseable dates or dates off one regular grid |
| `E_DATA_DEMAND_INVALID` | Non-numeric or non-finite demand |
| `E_DATA_DEMAND_NEGATIVE` | Negative demand |
| `E_DATA_GAP_TOO_LONG` | A gap longer than `data.max_gap_periods` |
| `E_LEAKAGE` | Forbidden feature column, correlation above the fence, or a failed future-perturbation proof |
| `E_SPLIT_NO_TIME_AXIS` | Too few origin dates for a split side, or a missing or non-date origin date |
| `E_CV_TOO_SHORT` | Not enough history for the configured folds |
| `E_EVAL_ZERO_DENOMINATOR` | A metric would divide by zero (zero total or non-positive scale) |
| `E_PREDICT_BAD_INPUT` | Malformed arrays, shape mismatch, bad level, missing series scale, missing or tampered bundle, or data errors while predicting |
| `E_RENDER_MISSING_KEY` | Missing key, wrong type or undefined name when deciding or rendering; names the path |

The pipeline attaches the failing step to the exception; steps are `load`, `validate`, `clean`, `origins`, `features`, `split`, `train`, `predict_test`, `metrics`, `decide`, `report`, `write`.

## Modules

| Module | Responsibility |
|---|---|
| `src/fcst/config.py` | Config loading and validation, `FcstError`, error codes, cost rates |
| `src/fcst/data.py` | Load, validate, clean, hash; synthetic generator; sample CLI |
| `src/fcst/features.py` | Origins, as-of features, labels, leakage checks, time split, rolling-origin CV |
| `src/fcst/models.py` | Baseline and candidate, scale statistics, selection, calibration, bundle save and load, predict |
| `src/fcst/evaluate.py` | Accuracy and quantile metrics, calibration report, paired bootstrap |
| `src/fcst/newsvendor.py` | Cost rates, service target, policy simulation, equal service, business block |
| `src/fcst/report.py` | Verdict rules, strict rendering, chart, metrics shape check |
| `src/fcst/pipeline.py` | End-to-end run and CLI |
| `templates/report.md.j2` | Report template, no numeric literals |

Dependency direction: `evaluate` does not import `models`; the reverse holds. Features read the period length through the same rule as the data layer, so one rule defines it.

## Testing and linting

| Test file | Covers |
|---|---|
| `tests/test_inputs.py` | Config and data |
| `tests/test_features.py` | Origins, features, leakage, split, cross-validation |
| `tests/test_models.py` | Models, calibration, metrics, bundles |
| `tests/test_business.py` | Newsvendor, report, pipeline end to end, README and notes checks |

Some tests enforce documentation rules: the README keeps exactly its fixed sections, the README, notes, config comments and template hold no hand-typed numbers, a size limit applies to `newsvendor.py`, and `.get(` must not appear in `report.py`. End-to-end tests copy the project into a temporary directory and run the CLI there.

`pyproject.toml` has no ruff section, so ruff's defaults apply.

## Packaging and dependencies

- Runtime packages: `numpy`, `pandas`, `pyyaml`, `scikit-learn`, `scipy`, `matplotlib`, `jinja2`. `pyproject.toml` also lists joblib, which the bundle code imports directly. Keep `requirements.txt` and `pyproject.toml` in step.
- Dev extra: pytest and ruff.
- Entry points: `fcst-pipeline` and `fcst-data`.
- Generated and not committed: `data/`, `artifacts/`, `reports/`, and packaging output.

## Extending the project

| Task | Where |
|---|---|
| Add a new input or parameter | `config.yaml` plus a rule in `config.py`; business values never go under `params` |
| Add a model | Implement `fit(X, y)` and `predict(X, panel)`, register the name, and extend the allowed names in config validation |
| Add a feature | `features.py`; it must be as of the origin and pass the leakage proof |
| Add a verdict reason or flag | Code in `report.py`, wording in `REASON_TEXT` or `FLAG_TEXT`, shape in the metrics skeleton, and a test |
| Add an error code | Only deliberately: the list is closed and a test pins it |
| Change report wording | `report.py` for fixed texts; the template for layout (no numeric literals) |

Rules to keep: all randomness from `params.seed`; numbers only in `config.yaml` and `metrics.json`; no silent defaults for missing keys.

## Glossary

| Term | Meaning |
|---|---|
| Series | One SKU at one location |
| Origin | Last period with known demand |
| Horizon | Lead time plus review period |
| Target | Total demand after the origin, over the horizon |
| Order-up-to level | Stock position ordered up to at each review |
| Baseline | Seasonal naive reference model |
| Candidate | Gradient-boosted model that must beat the baseline |
| Status quo | Baseline point forecast plus a relative buffer |
| Tau | Service level (quantile level) |
| Target service level | Margin over margin plus overage cost, derived from the cost inputs |
| Equal service | Comparison at the same stockout rate |
| Conditional value | A business figure that holds only under the stated assumptions and at equal service |
| WAPE | Weighted absolute percentage error |
| RMSSE | Root mean squared scaled error |
| Conformal calibration | Quantile offsets from scaled residuals on held-out rows |
| Purge gap | Gap between split parts so target windows do not overlap |
| Leakage | Using information from after the origin |

---

Back to the [User Guide](user-guide.md), the [Methodology](methodology.md) or the [README](../README.md).
