# User Guide

How to run the project, read its report and decide what to do next. For the reasoning behind the numbers see [Methodology](methodology.md); for every config key, file format and error code see [Reference](reference.md).

This guide holds no result figures and no copied config values. Values live in `config.yaml` (inputs) and `reports/metrics.json` (outputs).

## Contents

- [What this project answers](#what-this-project-answers)
- [Before you start](#before-you-start)
- [Path A: try it on synthetic data](#path-a-try-it-on-synthetic-data)
- [Path B: run it on your own panel](#path-b-run-it-on-your-own-panel)
- [What happens in a run](#what-happens-in-a-run)
- [Reading the report](#reading-the-report)
- [What to do after each verdict](#what-to-do-after-each-verdict)
- [Adapting the inputs](#adapting-the-inputs)
- [When a run stops](#when-a-run-stops)
- [Questions people ask](#questions-people-ask)

## What this project answers

One question: **should we replace the current way of setting order-up-to quantities with a forecasting model, and is there evidence that it pays?**

It does not hand you a forecast to trust. It forecasts, turns the forecast into an order-up-to level, compares that with a simple baseline and with the current practice (the status quo), and ends with one of three verdicts. The honest answer is often "keep the baseline", and that is a valid result.

## Before you start

| You need | Why |
|---|---|
| Python with the packages in `requirements.txt` | The pipeline is plain Python; `make install` installs them. |
| A panel CSV for real data | One row per series and period, with the columns `sku_id`, `location_id`, `series_id`, `period_start`, `demand`. Not needed for the sample path. |
| Cost inputs you can defend | Price, cost, holding cost, salvage and write-off share drive the target service level and every business figure. The shipped values are illustrative. |

Know these terms first:

| Term | Meaning |
|---|---|
| **Series** | One SKU at one location. |
| **Origin** | The last period whose demand is known when a forecast is made. |
| **Horizon** | Lead time plus review period. The target is the total demand after the origin, over the horizon. |
| **Order-up-to level** | The stock position you order up to at each review. |
| **Baseline** | The simple reference model (seasonal naive). |
| **Candidate** | The model that has to beat it (gradient boosting on lag features). |
| **Status quo** | Today's practice, modelled as the baseline point forecast plus a relative buffer. |
| **Equal service** | Comparing inventory at the same stockout rate, the only fair way to compare. |

## Path A: try it on synthetic data

Use this to see the whole flow once. Its output says nothing about real sales.

```
make install
make all-sample
```

The target runs lint, tests, writes and checks the sample file, then runs the pipeline on synthetic data. A banner marks the CLI output, the report and the chart as synthetic. If you only want the pipeline:

```
make install
python -m fcst.pipeline --sample
```

Outside `make`, put `src` on the import path first (see [Reference](reference.md#import-path)).

## Path B: run it on your own panel

1. **Prepare the panel.** One CSV, one row per series and period. Required columns are listed in [Reference](reference.md#panel-format). Other columns are dropped on purpose.
2. **Set the path.** Put the file where `params.data.path` in `config.yaml` points, or pass `--data`.
3. **Replace the cost inputs.** Edit the `business` block with your own price, cost, holding cost, salvage ratio, write-off share, lead time and review period. Keep `params.horizon_periods` equal to lead time plus review period, or the config is rejected.
4. **Check the periods.** Season length, origin step and the domain rules in `params` must fit your data (see [Adapting the inputs](#adapting-the-inputs)).
5. **Run.**

   ```
   make all
   ```

   or, for the pipeline alone:

   ```
   python -m fcst.pipeline --data path/to/panel.csv
   ```

   If the file does not exist the run stops with `E_DATA_FILE_MISSING`. It never falls back to synthetic data.
6. **Read `reports/report.md`.** Start with the verdict.

Note that `make all` runs lint and tests before the pipeline, so a missing panel is reported last. Calling the pipeline directly fails at the first step.

## What happens in a run

The pipeline stops at the first step that fails and prints the error code and the step.

| Step | What it does |
|---|---|
| `load` | Reads the panel (or generates the synthetic one). |
| `validate` | Collects data error codes without raising. |
| `clean` | Keeps only schema columns, fills short gaps with zero, fixes dtypes. |
| `origins` | Builds the grid of forecast origins per series. |
| `features` | Computes as-of features and the target for each origin. |
| `split` | Splits by time into train, calibration and test, with a purge gap. |
| `train` | Cross-validates both models on training rows, picks one, fits it, calibrates quantiles on calibration rows. The test rows are never read here. |
| `predict_test` | Predicts the test origins exactly once; every later number comes from that table. |
| `metrics` | Accuracy, calibration, leakage proof and the business block. |
| `decide` | Applies the verdict rules to the metrics. |
| `report` | Renders the report and the chart. |
| `write` | Writes the metrics file last. |

Outputs:

- `reports/report.md`, `reports/metrics.json`, `reports/tradeoff.png`
- `artifacts/bundle.joblib`, `artifacts/bundle.json`, `artifacts/test_predictions.csv`

Running again with the same seed and inputs gives the same `metrics.json`.

## Reading the report

The report always has the same sections, in this order.

**Verdict.** One of `use_baseline`, `keep_status_quo`, `pilot_then_scale`, with reason codes. A headline sentence appears only when the claim is supported (verdict `pilot_then_scale`, equal service available, and the lower bound of the inventory difference above zero). Read the reasons before anything else.

**Accuracy.** The candidate against the baseline on the primary metric, with a bootstrap interval. A negative difference means the candidate is better. If the interval includes zero, the difference may be noise. When the baseline was selected in cross-validation, the candidate was never the model in use.

**Calibration.** For each service level, how often realised demand stayed at or below the forecast quantile, and whether that is within tolerance. If calibration misses, the stockout rate you plan for is not the one you will get. On a small test set a single miss is noisy; the report raises a flag instead of concluding the calibrator is broken.

**Stockout vs working capital.** Stockout rate and inventory for the status quo, the baseline and the model side by side. These rows are **not** a comparison and carry no claim. A policy can hold less stock simply by accepting more stockouts.

**Equal service (inside the business block).** The only place a claim lives. It compares inventory at the same stockout rate, with an interval, and the net value against the status quo. When the baseline was selected, the model and baseline policies are the same policy, so the equal-service difference is zero by identity; the report says so and it is not a finding.

**Flags.** Never change the verdict; they say how far to trust it. See the table in [Reference](reference.md#flags).

**What NOT to conclude.** Read it before quoting anything.

**Limitations.** Fixed list; see the README.

## What to do after each verdict

| Verdict | What it means | What to do |
|---|---|---|
| `use_baseline` | The candidate did not beat the baseline in cross-validation, or not on the test split. | Keep the baseline. Do not deploy the model. This is a legitimate result. Consider whether more history, more series or better features are worth a second attempt. |
| `keep_status_quo` | The model may be fine, but the value is not shown: too few test series or origin rows, no equal-service match, or net value not positive. | Do not change practice yet. The reason codes tell you which gate failed. Fix the data volume or revisit the cost inputs, then rerun. |
| `pilot_then_scale` | The candidate beats the baseline and the value checks passed under the stated assumptions. | Pilot on a limited scope first, never "scale now". Re-measure on live data against the same service level. |

Whatever the verdict, check the flags. Synthetic data always raises `F_SYNTHETIC_DATA`, and then nothing in the report supports a claim about real sales.

## Adapting the inputs

Change `config.yaml` only; code never holds decision parameters. The full key table is in [Reference](reference.md#configuration). The usual reasons to edit:

| You want to | Edit |
|---|---|
| Use your own economics | the `business` block |
| Change how often you order or how long delivery takes | `business.lead_time_periods`, `business.review_period_periods`, and `params.horizon_periods` to match |
| Match your demand rhythm | `params.season_length_periods` (for example weekly in daily data) |
| Make the model learn from more or less history | `params.domain.min_history_periods` |
| Be stricter about when a result counts | `params.report.min_test_series`, `params.report.min_test_obs` |
| Compare at more service levels | `params.tradeoff.taus` |
| Change the accuracy metric | `params.models.primary_metric` (`wape` or `rmsse`) |
| Reproduce or vary a run | `params.seed` |

Two rules to respect:

- Business values and the service target must never appear under `params`; the loader rejects them.
- The target service level is derived, never stored: it follows from the cost inputs.

If you change `config.yaml`, rerun the whole pipeline. The report carries a source hash that covers the config and the metrics, so a stale report can be detected.

## When a run stops

The CLI prints `error: CODE: message (step: name)` to stderr and exits with status one. Nothing is written after the failing step. The most common ones:

| Code | Usual cause | What to do |
|---|---|---|
| `E_CONFIG` | A config field is missing or inconsistent; the message names it. | Fix that field. |
| `E_DATA_FILE_MISSING` | No file at the real-data path. | Place the panel there or pass `--data`. |
| `E_DATA_COLUMNS_MISSING` | A required column is missing, or the panel or an identifier is empty. | Add the columns; check for blank ids. |
| `E_DATA_DUPLICATE_KEY` | The same series and period appears more than once. | Aggregate the panel first. |
| `E_DATA_DATE_INVALID` | Dates are not ISO dates on one regular grid. | Normalise the dates. |
| `E_DATA_GAP_TOO_LONG` | A gap inside a series is longer than the configured limit. | Fill or trim the series, or raise the limit knowingly. |
| `E_LEAKAGE` | A feature looks like the target, or the future-perturbation proof failed. | Treat as a real defect; do not loosen the threshold to pass. |
| `E_CV_TOO_SHORT` | Too little history for the configured folds. | Provide more periods or reduce folds. |

The full list is in [Reference](reference.md#error-codes).

## Questions people ask

**Why did my run end in `use_baseline`?** Whether the candidate wins is a measured outcome, recorded under `decision` in `reports/metrics.json`. A baseline win is allowed by design, so nothing is forced, and the reason codes say whether it lost in cross-validation or on the test split.

**The model is more accurate but the verdict is `keep_status_quo`. Why?** Accuracy is not value. The gates ask for enough test data, an available equal-service comparison and a positive net value. The reason codes name the one that failed.

**Can I quote the inventory reduction?** Only the equal-service figure, only with its interval and the label "conditional value", and only if the flags allow it. Never quote the side-by-side table as a saving.

**Can I use weekly data?** The period length is inferred from the panel, and config values count periods, not days. Adjust `season_length_periods`, lead time, review period and `periods_per_year` to your period.

**Where is the proof there is no look-ahead?** Features are computed as of the origin, and a check changes or deletes everything after each origin and requires identical features. Its result is in `reports/metrics.json` under the leakage block.

---

Next: [Methodology](methodology.md) to audit how a number was produced, or [Reference](reference.md) to look up a config key or error code. Back to the [README](../README.md).
