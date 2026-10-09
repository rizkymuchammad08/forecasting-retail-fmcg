# Methodology

How the numbers are produced and why each choice was made. Formulas are written with words for any constant; no value is copied from `config.yaml`. Key names point to where the value lives.

## Contents

- [Design guards](#design-guards)
- [Problem framing](#problem-framing)
- [Data preparation](#data-preparation)
- [Origins and features](#origins-and-features)
- [Leakage defence](#leakage-defence)
- [Time split and cross-validation](#time-split-and-cross-validation)
- [Models](#models)
- [Selection](#selection)
- [Accuracy and quantile metrics](#accuracy-and-quantile-metrics)
- [Calibration](#calibration)
- [Uncertainty: the bootstrap](#uncertainty-the-bootstrap)
- [Business layer: the newsvendor](#business-layer-the-newsvendor)
- [Equal service](#equal-service)
- [Verdict and flags](#verdict-and-flags)
- [Report integrity and reproducibility](#report-integrity-and-reproducibility)
- [The synthetic data](#the-synthetic-data)
- [Assumptions to keep in mind](#assumptions-to-keep-in-mind)

## Design guards

Six rules shape every part of the code. When something looks unusual, one of them is usually the reason.

1. **One adapter, synthetic as the default.** Input is an aggregate panel. Real data is a path in `config.yaml`; a missing file is a hard failure, never a silent fallback to synthetic data.
2. **Time is the only split axis.** The split has a purge gap at least as long as the horizon, and features are computed as of the origin.
3. **The model may lose.** A candidate that does not beat the baseline is a legitimate result; the verdict is then `use_baseline`.
4. **Realisation columns are a leakage risk.** Only columns in `schema.required_columns` are read; everything else is dropped.
5. **Business claims only at the same service level.** The efficiency claim is the inventory difference at the same stockout rate, always labelled "conditional value".
6. **Numbers live in two files only.** `config.yaml` for inputs, `reports/metrics.json` for outputs. The report template has no numeric literals and is rendered strictly; documentation carries no hand-typed result figures, and a test enforces that for the README, the notes, the config comments and the template.

## Problem framing

- **Series:** one SKU at one location. **Period:** the panel's own period (daily in the shipped setup). **Demand:** units sold per period.
- **Origin:** the last period whose demand is known.
- **Horizon:** lead time plus review period, in periods. `params.horizon_periods` is stored in the config and validated to equal that sum, so it cannot drift.
- **Target:** total demand in the window just after the origin, up to and including the origin plus the horizon.
- **Decision:** the order-up-to level for each series at each review cycle.

Demand is proxied by sales because the panel has no stockout column. True demand may be larger, and every report says so.

## Data preparation

- The unique key is the series and the period start. Duplicates raise `E_DATA_DUPLICATE_KEY`.
- The **active range** of a series runs from its first to its last row. Gaps inside it are filled with zero when no longer than `data.max_gap_periods`; a longer gap raises `E_DATA_GAP_TOO_LONG`.
- The **period length is inferred**, not configured: the most frequent positive spacing between consecutive rows of a series (the smaller on ties; daily if there is none). Dates must be ISO dates on one grid, otherwise `E_DATA_DATE_INVALID`. A panel that records only sales rows and is very sparse can be inferred wrongly or fail on gaps.
- Only the schema columns are read. Dtypes are cast explicitly so output does not depend on the pandas version.
- `validate` returns the list of data error codes and never raises for bad data. `load` and `clean` raise with the first code.
- An empty panel, or a blank identifier, maps to `E_DATA_COLUMNS_MISSING`.

## Origins and features

**Origins** live on one global grid: the first period of the panel, then every `params.origin_step_periods`. An origin is dropped when:

- its target window runs past the last row of the series (no data is not zero demand);
- the series has less history than `domain.min_history_periods` at that origin;
- the series is inactive: no demand in the last `domain.inactive_periods` periods up to the origin.

**Features** are all as of the origin and indexed by series and origin date:

| Group | Content |
|---|---|
| Lags | Demand a few periods back, plus lags of whole seasons. `lag_one` is the demand at the origin. |
| Rolling mean, standard deviation, zero share | Over windows of whole seasons and over the horizon; population standard deviation. |
| Age | Periods since the series' first row. |
| Calendar | Month, annual sine and cosine, and the number of weekend periods in the target window. |

Lags and windows are **derived, not chosen**: multiples of the season length, the horizon, or a short fixed lag, all capped by `domain.min_history_periods` so every origin has the history it needs and the matrix has no missing values. If the minimum history is below the season length, `E_CONFIG` names the key.

Identifiers (`series_id`, `sku_id`, `location_id`) are **not** model inputs on purpose: a model must not memorise series across the time split. Any series-level effect has to come from as-of history features.

## Leakage defence

Leakage is handled twice, and the two layers do different jobs.

- **Fences (cheap, can miss things).** `assert_no_forbidden` rejects any feature column whose name or prefix suggests the target (`y`, `target`, `label`, `future`, next-period names, date columns, schema columns). `check_correlation` raises `E_LEAKAGE` when a feature correlates with the target above `leakage.max_abs_corr`. Correlation is computed **within series** (each series' mean removed from feature and target), because pooled correlation is dominated by the spread of series sizes, not by information about the target. A copy of the target is still caught.
- **Proof.** `perturb_future_check` changes or deletes everything after each origin and requires the features to be identical. It runs the real feature builder end to end, so any hidden read of the future (for example through the period length or the grid start) would show. Failure raises `E_LEAKAGE` naming the columns that moved. The result is recorded in the metrics and a failed check raises a flag.

## Time split and cross-validation

- **Split.** Origins are cut by time into train, calibration and test shares (`split.train`, `split.calib`, `split.test`, which must sum to one). A purge gap of `split.purge_gap_periods`, validated to be at least the horizon, separates the parts so target windows do not overlap across the boundary. Too few origins for any side raises `E_SPLIT_NO_TIME_AXIS`.
- **Rolling-origin CV.** Run on training rows only, with `cv.n_folds` folds and at least `cv.min_train_origins` origins in each training part. Each validation origin lies later than the last training origin plus the horizon. If fewer folds fit than requested the effective count is reported; too short a history raises `E_CV_TOO_SHORT`.
- **The test rows are used once**, for the final measurement, never for any choice.

## Models

**Baseline: `seasonal_naive`.** Forecasts the target window with the demand of the same window shifted back by a whole number of seasons, chosen so the shift is at least the horizon. The shifted window therefore ends at or before the origin and is fully observed. If the shifted window starts before the first row of a series, the available part is scaled up to the horizon; nothing available gives zero. It reads only periods up to the origin.

**Candidate: `gbm_lag`.** A gradient-boosted regressor on the features above. Series differ in size by orders of magnitude, so the target and the demand-valued features are divided by a per-series **level**: the mean target on the training rows, floored at `calibration.scale_floor`. The level is learned from training rows only; a series unseen in training uses its own as-of rolling mean times the horizon. Forecasts are multiplied back and clipped at zero.

Hyper-parameters are fixed module constants (not config keys), with the random state from `params.seed`. They were not tuned on calibration or test rows. The model hash covers the name, parameters, seed and feature names, not the data.

## Selection

Both models are scored on every rolling-origin fold of the training rows with the primary metric (`models.primary_metric`, WAPE or RMSSE). The lower mean wins; **ties go to the baseline**. Calibration rows, test rows and their targets are never read. The selection verdict (`use_baseline` or the candidate) is stored and feeds the final verdict.

## Accuracy and quantile metrics

Throughout, `y` is the realised target, `yhat` the point forecast and `q` the quantile forecast at level tau.

| Metric | Definition | Reading |
|---|---|---|
| WAPE | Sum of absolute errors divided by the sum of actuals | Lower is better; a zero total raises `E_EVAL_ZERO_DENOMINATOR` |
| RMSSE | Per series: mean squared error divided by the series' naive scale; averaged over series (equal weight); square root | Scale-free across series of very different size |
| Bias | (sum of forecasts minus sum of actuals) divided by sum of actuals | Positive means over-forecast |
| Coverage | Share of rows with `y` at or below `q` | Should be close to tau |
| Coverage error | Coverage minus tau | Negative: the quantile is too low, which means stockouts |
| Pinball loss | Standard quantile loss at each tau | Lower is better |
| Delta on the primary metric | Candidate minus baseline | Negative: the candidate wins |

Malformed inputs (shape mismatch, non-finite values, bad tau, missing series scale) raise `E_PREDICT_BAD_INPUT`. Metrics never return NaN or infinity.

## Calibration

A point forecast is not enough for an order-up-to decision: the decision needs a quantile at the target service level. The selected model is calibrated by **conformal scaling**:

1. On the calibration rows, compute scaled residuals: (actual minus forecast) divided by the series scale.
2. For each tau, take the empirical quantile of those residuals at the finite-sample conformal level.
3. Quantile forecast = the larger of zero and (forecast plus scale times the offset).

Offsets are monotone in tau, so quantiles are monotone in tau, and the clip keeps them non-negative. The levels calibrated are the union of the target service level and `tradeoff.taus`. The baseline gets its own calibrator, fitted on its own forecasts over the same calibration rows.

**What it guarantees:** marginal coverage on average over exchangeable rows. **What it does not:** coverage per series, or coverage under a time shift between calibration and test. The report checks coverage against `evaluation.calibration_tolerance` and raises a flag when it misses; with a small test set a single miss is not proof of a broken calibrator.

## Uncertainty: the bootstrap

Differences (accuracy delta, net value, inventory difference) come with percentile intervals at `evaluation.ci_level` from `evaluation.n_boot` resamples. Resampling **draws whole series with replacement**, because rows of one series overlap in time, and both models are always scored on the same draw (paired). Correlation between series is ignored. The percentile interval is widened so it always contains the point estimate. The seed is `params.seed`.

## Business layer: the newsvendor

Each origin row is treated as one independent cycle that starts with no stock. Let S be the order-up-to level covering the horizon and D the realised demand in the target window.

**Cost rates per unit** (one implementation, `config.cost_rates`):

```
c_under = price - cost
c_over  = cost * holding_rate * review_period / periods_per_year
          + cost * (one - salvage_ratio) * writeoff_share
```

`c_under` is the margin lost per unit of unmet demand. `c_over` is one review period of holding cost plus the expected write-off of leftover units.

**Target service level (critical ratio):**

```
tau_star = c_under / (c_under + c_over)
```

It is derived, never stored, and validated to lie strictly between zero and one. A high margin against low holding cost pushes it up, meaning the policy buys service with more inventory.

**Simulation per origin:**

```
sold      = min(S, D)
leftover  = max(S - D, zero)
stockout  = D is strictly greater than S
net       = c_under * sold - c_over * leftover
avg_inv   = half of (S + leftover)       # linear depletion; the lead-time pipeline is ignored
dio       = horizon * sum(avg_inv) / sum(sold)     # null if nothing was sold
avg_inventory_value = unit_cost * mean(avg_inv)
```

**Three policies on the test origins:**

| Policy | Order-up-to level |
|---|---|
| `status_quo` | Baseline point forecast times (one plus `status_quo_buffer`) |
| `baseline_newsvendor` | Baseline quantile at tau_star |
| `model_newsvendor` | Selected model's quantile at tau_star |

When the verdict is `use_baseline`, the last two are the same policy.

The **trade-off curves** use every tau in the calibrated set, giving stockout rate against inventory for each policy. The status quo is a single point, marked "not at equal service".

## Equal service

"Better" is only meaningful at the same service level.

1. The target is the model's stockout rate at tau_star.
2. The baseline's inventory at that stockout rate is read off its curve by linear interpolation, never extrapolated. Points with the same stockout rate keep the smallest inventory.
3. The difference is the **relative inventory reduction at equal service**. Positive means the model needs less inventory for the same stockout rate.
4. If the target lies outside the baseline's stockout range, or baseline inventory is zero, the comparison is `available = false` with null values and a reason.

The net-value difference against the status quo is a total over the test origins; each resample total is rescaled by all origins over drawn origins. Resamples with no equal-service match are dropped and counted.

Two cautions built into the report: at a high tau_star the newsvendor policy can hold **more** inventory than the status quo (flagged), and when the baseline is selected the equal-service reduction is exactly zero by identity, not by measurement.

## Verdict and flags

`decide` is a pure function of `metrics.json` and `config.yaml`. A missing key raises `E_RENDER_MISSING_KEY`; nothing defaults silently. Priority, first match wins:

| Verdict | Condition |
|---|---|
| `use_baseline` | CV selection kept the baseline, or the test delta on the primary metric is not negative |
| `keep_status_quo` | Test series below `report.min_test_series`, test origin rows below `report.min_test_obs`, no equal-service match, or net value against the status quo not positive |
| `pilot_then_scale` | Everything else. Never "scale now" |

`headline_supported` is true only for `pilot_then_scale` with an available equal-service comparison whose lower bound is above zero. The gate on observations counts test **origin rows**, the unit the decisions are made on, so a small test split can fail it even for a winning candidate; that is a config decision (origin step, minimum observations), not a report change.

Flags never change the verdict. They record caveats: synthetic data, calibration out of tolerance, accuracy interval including zero, net-value interval including zero, more inventory than the status quo, and a failed leakage check.

## Report integrity and reproducibility

- **Strict rendering.** The template contains no numeric literal beyond the zero used for a sign test. Numbers come from the metrics through named formatters. A missing key, a null where none is allowed, a wrong type, or any undefined name raises `E_RENDER_MISSING_KEY` naming the path.
- **Source hash.** SHA-256 of the bytes of `config.yaml` plus the canonical JSON of the metrics (sorted keys, compact). It is printed in the report footer, so a report that no longer matches its metrics or config is detectable.
- **Atomic writes.** The metrics file is written last, from the same dictionary the report used, through a temporary file; NaN or infinity raises instead of being written.
- **Determinism.** All randomness uses `params.seed`; the same seed and inputs give an identical `metrics.json`. The data hash and model hash identify the run.
- **Model bundle.** `bundle.joblib` is a pickle, so `bundle.json` stores its SHA-256 and the loader refuses a file whose hash differs. This guards against corruption and mix-ups, not against someone who can write both files.
- **One prediction pass.** The test set is predicted once; every later number comes from that table, saved as `artifacts/test_predictions.csv`.

## The synthetic data

Demand is a gamma-Poisson (negative binomial) count around a trend, weekly and annual seasonality and promotions. The panel mixes high-volume, low-volume and intermittent series, plus outlier spikes, new series (late start) and delisted series (early end). A share of isolated zero-sales rows is left out so `clean` has gaps to fill. Size comes only from `params.data.synthetic`. It exists to demonstrate and test the method; it supports no claim about real sales and always carries a banner.

## Assumptions to keep in mind

- Demand equals sales; stockouts hide true demand.
- Each review cycle is independent and starts with no stock (no carry-over).
- No supplier constraints (minimum order, pack size, capacity) and no substitution between products or locations.
- Delisted series fade out and are under-represented in train and test.
- Cost inputs are assumptions; every business figure is conditional on them.
- Neighbouring origins have overlapping target windows, and the bootstrap ignores correlation between series.
- Coverage on a small test set is noisy.

---

Next: [Reference](reference.md) for config keys, formats and error codes. Back to the [User Guide](user-guide.md) or the [README](../README.md).
