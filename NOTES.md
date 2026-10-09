# NOTES

Working notes, filled in during P1 to P6. They feed the "What changed from the previous prompt" section after one full run. Rule for this file: it holds no result figures and no copied config values. Results live in `reports/metrics.json`, inputs in `config.yaml`; when a note needs a value it names the key or the file instead of repeating the number. A test enforces this.

## Six design guards

1. **One adapter, synthetic as default.** Input is an aggregate panel (one row per series and period). Real data is a path in `config.yaml`; a missing file is a hard failure and never falls back silently to synthetic data.
2. **Time is the only split axis.** Temporal split with a purge gap at least as long as the horizon. Features are computed as-of the origin.
3. **The model may lose.** A model that does not beat the baseline is a legitimate result; its verdict is `use_baseline`.
4. **Realization columns are a leakage risk.** Only columns in `schema.required_columns` are read; all other columns are dropped during clean.
5. **Business claims only at the same service level.** The efficiency claim is the inventory difference at the same stockout rate, always labeled "conditional value".
6. **Numbers live in two files only.** `config.yaml` and `reports/metrics.json`. The report template has no numeric literals and is rendered strictly. README, this file, the `config.yaml` comments and the template carry no hand-typed result figures.

## Fixed rules (from P0)

- Unique key is `(series_id, period_start)`. Active range of a series = first to last row.
- Gaps inside the active range are filled with zero when the gap is at most `data.max_gap_periods`; otherwise `E_DATA_GAP_TOO_LONG`.
- `data_source` is `synthetic` when run with `--sample`, otherwise `real`. Synthetic data always shows a banner in the CLI, report and chart.
- Synthetic generator size comes only from `params.data.synthetic`.
- Real data is read from `params.data.path`; missing file gives `E_DATA_FILE_MISSING`.
- Horizon `H = lead_time_periods + review_period_periods`. Target = total demand in `(origin, origin + H]`.

## File ownership per part

| Part | Files created |
|---|---|
| P1 | README.md (stub), NOTES.md (stub), Makefile, requirements.txt, .gitignore, config.yaml, config.py, tests/test_inputs.py (config part) |
| P2 | data.py, tests/test_inputs.py (data part) |
| P3 | features.py, tests/test_features.py |
| P4 | models.py, evaluate.py, tests/test_models.py |
| P5 | newsvendor.py, report.py, `templates/report.md.j2`, tests/test_business.py (newsvendor and report part) |
| P6 | pipeline.py, final README, full NOTES.md, final Makefile, end-to-end tests in tests/test_business.py |

`data/`, `artifacts/`, `reports/` are generated output and not counted as project files.

A part is done when its listed tests pass, `make lint` is clean, and `make test` is green.

## P1

- **Symptom:** `horizon_periods` is stored in `config.yaml` (not derived), so it can drift from lead time plus review period.
  **Cause:** P0 fixes the YAML keys, including `params.horizon_periods`.
  **Fix:** `validate_config` checks equality and names `params.horizon_periods` on mismatch. P0 change: no.
- **Symptom:** the three split shares can fail to sum to one on float rounding.
  **Cause:** binary floating point.
  **Fix:** compare with `math.isclose` and an absolute tolerance; covered by a test. P0 change: no.
- **Symptom:** the cost formulas exist in two places once P5 lands: `config.cost_rates` (used to validate that tau* is inside the open unit interval) and `newsvendor.costs`.
  **Cause:** P0 assigns `costs` and `tau_star` to `newsvendor.py`, but config validation needs them first.
  **Fix:** `config.cost_rates` is the single implementation; P5 delegates to it and a consistency test pins that. tau* is never stored. P0 change: no.
- **Symptom:** `pytest tests/test_inputs.py -k config` fails with an import error when run directly.
  **Cause:** `fcst` lives under `src/` and is importable only with that directory on the import path.
  **Fix:** `make test` exports `PYTHONPATH=src`; direct calls use `PYTHONPATH=src pytest ...`. Superseded in the packaging note under P6: `pyproject.toml` sets the pytest `pythonpath`, so plain `pytest` works too. P0 change: no.
- **Symptom:** with the illustrative cost inputs, tau* lands near the top of `tradeoff.taus`.
  **Cause:** the underage cost is much larger than the overage cost in `config.yaml`.
  **Fix:** none; watched in P5 (equal-service availability). The result is in `reports/metrics.json`. P0 change: no.

Decisions: the open items were resolved as channel `store`, period `day`, perishable with a shelf life longer than the horizon, data synthetic only. No dependencies were added beyond the P1 list.

## P2

- **Symptom:** an empty panel or a blank/null `sku_id`, `location_id` or `series_id` has no dedicated error code.
  **Cause:** the P0 code list covers missing columns, but not missing rows or missing identifier values.
  **Fix:** both map to `E_DATA_COLUMNS_MISSING` (tested). P0 change: no.
- **Symptom:** the period length is not a config key, yet gap filling needs it.
  **Cause:** P0 locks the config keys; only `periods_per_year` and `season_length_periods` exist.
  **Fix:** infer it as the most frequent positive spacing between consecutive rows of a series (the smaller one on ties; daily if there is no spacing). A real panel that records only sales rows (very sparse) would be inferred wrongly or fail with `E_DATA_GAP_TOO_LONG`. Dates must be ISO dates on one grid, else `E_DATA_DATE_INVALID`. P0 change: no; revisit if real data is sparse.
- **Symptom:** `ruff check` with its default rule set flagged `subprocess.run` without an explicit `check` argument in a test.
  **Cause:** the default rule set of recent ruff versions is wider than the basic error and pyflakes rules.
  **Fix:** pass `check=False`. P0 change: no.
- **Symptom:** two kinds of numbers live in code, not in `config.yaml`: the sample file size limit and the generator shape constants (group levels, noise shapes, promo lifts, share of isolated zero rows dropped).
  **Cause:** P0 fixes the config keys, and none of these is a decision parameter.
  **Fix:** kept as named module constants with comments; only the generator size comes from `params.data.synthetic`. P0 change: no.
- **Symptom:** string and datetime dtypes differ between pandas major versions.
  **Cause:** `requirements.txt` is unpinned, as P1 specified.
  **Fix:** `clean` casts dates and demand to explicit dtypes, so output does not depend on the pandas version. Generated values also depend on the numpy random stream version. P0 change: no.

Observations for later parts:

- The synthetic panel leaves out a share of isolated zero-sales rows (never the first or last row of a series), so `clean` has gaps to fill. The generator's `truth` table covers the full grid.
- The default sample mixes high-volume, low-volume and intermittent series, plus new series, delisted series and outlier spikes. No series is shorter than `domain.min_history_periods`, so that rule only removes early origins of each series, never a whole series.
- `make check-sample` regenerates twice, compares `data_hash`, checks the file size limit, and compares an existing `data/sample.csv` with the generator output (a stale file fails with a hint to run `make sample`).
- `data.py` exposes `to_csv_bytes` and `make_series_id` beyond the P0 list; `make_sample` expects an already cleaned panel.

## P3

- **Symptom:** the correlation fence nearly fired on honest features.
  **Cause:** pooled correlation is dominated by the spread of series sizes, not by anything the feature knows about the target. Spearman and log-scale pooled versions are no better.
  **Fix:** `check_correlation` removes each series' mean from the feature and from `y` before correlating (within-series), when the index has a `series_id` level; otherwise it falls back to plain pooled Pearson. A copy of the target stays near one, with or without per-series offsets (both are tested). The fence only catches copies; the proof is `perturb_future_check`. P0 change: no, the threshold key is unchanged; only the statistic is defined.
- **Symptom:** P0 says "drop origins whose window passes the last period" and "origins outside the domain rule", but not which last period or which rule.
  **Cause:** wording is ambiguous for series that end early (delisted) and for "inactive".
  **Fix:** the last period is the last row of the series (a window past it has no data, not zero demand). Min history means `origin - first_row + 1 >= domain.min_history_periods`. Inactive means no demand in the last `domain.inactive_periods` periods up to and including the origin (window clipped to the history). The origin grid starts at the first period of the whole panel and steps by `origin_step_periods`, so a refreshed panel with an earlier start moves the grid. P0 change: no.
- **Symptom:** the feature parameters (lags, windows) are not config keys.
  **Cause:** P0 locks the config keys.
  **Fix:** derived, not chosen: short lags plus lags of whole seasons; windows of whole seasons plus the horizon; all capped by `domain.min_history_periods`, so every origin has the history it needs and the matrix has no missing values. If `min_history_periods` is below `season_length_periods` the feature spec raises `E_CONFIG` naming `params.domain.min_history_periods`. P0 change: no.
- **Symptom:** a few situations have no dedicated error code.
  **Cause:** the P0 list is closed (and `test_config_error_codes_match_contract` pins it).
  **Fix:** too few origin dates for train, calib and test after purging (or an empty side), and a non-date or missing `origin_date`, all raise `E_SPLIT_NO_TIME_AXIS`. Origins that do not fit the panel (unknown series, off the period grid, outside the active range, too little history, label window past the end) reuse the matching `E_DATA_*` code. `X` and `y` with different indexes in `check_correlation` raise `E_PREDICT_BAD_INPUT`. P0 change: no.
- **Symptom:** the split gap is "at least H" but the CV gap is "more than H".
  **Cause:** P0 words them that way (split: distance at least H; CV: validation origin later than last train origin plus H).
  **Fix:** both are enforced as written, counted in origin steps on the sorted distinct origin dates. This relies on the table's origin dates being on the `build_origins` grid; gaps only get larger if dates are missing. P0 change: no.
- **Symptom:** a clean feature function could still read the future through global state (period length, grid start).
  **Cause:** `make_features` infers the period length from the whole panel.
  **Fix:** it uses the same rule as P2 (`data._infer_step`, imported although private) and the grid start is the earliest row of the panel, so deleting rows after an origin cannot move them on a cleaned panel. `perturb_future_check` runs the real `make_features` end to end on perturbed panels, so any such leak would show up.
- **Symptom:** identifiers (`series_id`, `sku_id`, `location_id`) are not usable as model inputs.
  **Cause:** they are in `forbidden_columns` (all schema columns plus `y`, `target`, `label`, date columns, and the prefixes `target`, `label`, `future`, `next_`, `y_`).
  **Fix:** deliberate: a model must not memorise series across the time split. A series-level effect has to come from as-of history features. P0 change: no.

Interface for later parts (all as-of the origin; `X` and `y` share the index `(series_id, origin_date)`):

- `build_origins(panel, cfg)`, `make_features(panel, origins, cfg)`, `build_labels(panel, origins, cfg)`, `feature_names(cfg)`; `table = X.join(y)` works as input to the split.
- `make_features` accepts the live origin (last row of a series, no target window yet); `build_labels` does not.
- `make_split(table, cfg)` returns `Split(train, calib, test, purged)` as row subsets in input order. `rolling_origin_folds(table, cfg)` returns a `CVPlan` (iterable of `Fold(train, valid)`) with requested and effective fold counts and `summary()`; fit it on `split.train` only.
- `perturb_future_check` returns `passed`, the number of origin dates, the number of rows and the variants for the report, or raises `E_LEAKAGE` naming the columns that moved.

Observations for later parts:

- Target windows of neighbouring origins overlap when `origin_step_periods` is below the horizon. Rows are therefore correlated, which matters for the bootstrap (resample by series, not by row).
- A series never contributes origins after its last row, so the end of a delisting (demand fading to zero, then no rows) is under-represented in train and test. It belongs in Limitations.
- `report.min_test_obs` is compared with the number of test origin rows, not with (series, period) observations inside the target windows. See the P5 report notes.

## P4

- **Symptom:** the P0 text that lists the `predict` output columns is not in this repository, so the column set had to be chosen.
  **Cause:** only P1 to P3 and their notes were available when P4 was written.
  **Fix:** `predict` returns `series_id`, `origin_date`, `yhat` and one `q_<tau>` column per calibration tau (`PREDICT_COLUMNS` and `evaluate.quantile_column` in one place each; the `q_` naming is the one `data.generate` already uses for its truth quantiles). If P0 says otherwise, change those two spots and the column test. P0 change: to be checked.
- **Symptom:** the seasonal-naive baseline cannot be computed from the P3 feature matrix.
  **Cause:** the shifted window is not a rolling window ending at the origin unless the shift equals the horizon, and P3 fixed the feature list.
  **Fix:** both models share `fit(X, y)` and `predict(X, panel)`; the baseline reads the panel through P3's dense matrix (`_dense`, `_resolve`, imported although private, as P3 did) and only touches periods up to the origin (tested by changing and deleting the future). If the shifted window starts before the first row of a series, the available part is scaled to the horizon; nothing available gives zero. With the default config this never happens because `min_history_periods` exceeds the shift.
- **Symptom:** "scale statistic" and "preprocessing state" are not defined in P0.
  **Cause:** the only P0 key is `calibration.scale_floor`.
  **Fix:** `fit_scale(y_train, cfg)` returns `ScaleStats`: `level` per series (mean target on train rows, floored at `scale_floor`) and `naive_mse` per series (mean squared difference of consecutive-origin targets, floored at the squared floor), plus medians as fallback. `GbmLag` divides the target and the lag, rolling-mean and rolling-std columns by the level, because series sizes span orders of magnitude; shares and calendar columns are untouched. A series that is not in train takes its own as-of rolling mean times the horizon as level (never a later value), so test-only series work. The same `level` is the `scale_calib` of the calibrator. The naive difference at origin spacing is a crude RMSSE denominator because neighbouring windows overlap; it is a constant of scale, not a claim about naive accuracy.
- **Symptom:** `coverage_error`, `bias` and `delta_primary` have no sign convention in P0.
  **Fix:** `coverage_error = coverage - tau` (negative: quantile too low); `bias = (sum yhat - sum y) / sum y` (positive: over-forecast); `delta_primary = candidate - baseline` (negative: the candidate wins). `coverage` is `mean(y <= q)`. "Coverage before calibration" in the tests means the raw point forecast used as the quantile.
- **Symptom:** the code list is closed and has no code for malformed metric or calibrator input, a missing or tampered bundle, or an unsupported model name.
  **Fix:** malformed arrays, shape mismatch, missing series scale, bad tau in a metric, and a missing or hash-mismatched bundle all raise `E_PREDICT_BAD_INPUT`; `E_DATA_*` raised while building features for `predict` is re-raised as `E_PREDICT_BAD_INPUT` (message kept); `params.models.baseline` or `candidate` other than `seasonal_naive` / `gbm_lag`, and invalid taus, raise `E_CONFIG`. A zero scale in RMSSE is `E_EVAL_ZERO_DENOMINATOR`. P0 change: no.
- **Symptom:** the percentile bootstrap interval can exclude the full-sample value when the resampled differences are skewed.
  **Cause:** the percentile method does not centre on the point estimate.
  **Fix:** `bootstrap_delta` widens the interval to contain `value`, so `ci_low <= value <= ci_high` always holds. Resampling is by series (whole series drawn with replacement, both models scored on the same draw), because rows of one series overlap in time; it ignores correlation between series. Seed is `params.seed` unless given.
- **Symptom:** a joblib bundle is a pickle.
  **Fix:** `save_bundle` writes `bundle.joblib` plus `bundle.json` (SHA-256 of the file, `model_hash`, per-model hashes, taus, offsets); `load_bundle` refuses a file whose hash differs and checks `model_hash` again after loading. This guards against corruption and mix-ups, not against someone who can write both files. The bundle keeps both fitted models so P5 can read the baseline forecast (`predict_point(..., which="seasonal_naive")`) for the status quo; quantiles are calibrated for the selected model only.
- **Symptom:** the GBM hyper-parameters are not config keys.
  **Fix:** module constant `GBM_PARAMS` (fixed learning rate, iteration count, leaf count, minimum samples per leaf, ridge penalty, no early stopping, `random_state` from `params.seed`); `model_hash` covers name, parameters, seed and feature names, not data. They were not tuned on calibration or test rows.

Interface for later parts:

- `models`: `train_bundle(table, panel, cfg, gbm_params=None) -> Bundle` (splits, selects on train folds, fits on train, calibrates on calib; never reads test), `select_model(train, panel, cfg)`, `predict(bundle, panel, origins, cfg)`, `predict_point(bundle, panel, origins, cfg, which=None)`, `fit_scale`, `fit_calibrator(yhat_calib, y_calib, scale_calib, taus, cfg)`, `calibration_taus(cfg)` (= tau* joined with `tradeoff.taus`, ascending), `tau_star(cfg)` (delegates to `config.cost_rates`), `model_hash`, `save_bundle`, `load_bundle`. `Bundle.selection` holds `primary_metric`, per-fold scores of both models, their means, `selected`, `verdict` (`use_model` / `use_baseline`) and the fold counts.
- `evaluate`: `wape`, `rmsse(y, yhat, series, scale)`, `bias`, `pinball_loss`, `coverage`, `coverage_error`, `calibration_report` (per tau coverage, error, `within_tolerance`, pinball; also `n`, never raises on tolerance), `bootstrap_delta`, `compare_to_baseline` (returns `delta_primary` and its interval), `primary_score`. For the RMSSE scale of test rows use `bundle.scale.naive_mse_for(series_ids)`.

Observations for later parts:

- A candidate that loses in cross-validation is a legitimate result (the third design guard). Whether it also loses on the test split is in `reports/metrics.json`; the test split was never used for any choice.
- Calibration offsets are scaled-residual quantiles fitted on the calibration rows only. With a small test set a coverage miss at one level is not evidence of a broken calibrator; `within_tolerance` reports it as it is and the report raises a flag.
- Tests may run with a stand-in for pytest where the real tools cannot be installed; the real `make lint` and `make test` must be run once, and the run record says so.

## P5

Done: `src/fcst/newsvendor.py` (kept under its line limit, which a test enforces), `src/fcst/report.py`, `templates/report.md.j2`, and the newsvendor and report parts of `tests/test_business.py`. Wiring in `pipeline.py` belongs to P6.

- **Symptom:** the P0 text for `simulate_policy`, `equal_service`, `dio_periods` and `avg_inventory_value` is not in this repository, so their exact definitions had to be chosen.
  **Cause:** only P1 to P4 and their notes were available.
  **Fix:** all in the `newsvendor.py` docstring and pinned by tests. `sold = min(S, D)`, `leftover = max(S - D, 0)`, stockout is strict (`D > S`), `net = c_under * sold - c_over * leftover`, average inventory per origin is `(S + leftover) / 2` (linear depletion; the lead-time pipeline is ignored), `dio_periods = H * sum(avg_inv) / sum(sold)` (null if nothing sold), `avg_inventory_value = avg_unit_cost * mean(avg_inv)`. `simulate_policy(S, D, costs, H, *, unit_cost)` takes the unit cost as a required keyword because `costs` is only `(c_under, c_over)`. If P0 says otherwise, change `_rows` / `_summary` and the hand-calculation tests. P0 change: to be checked.
- **Symptom:** the first draft of `newsvendor.py` went over the line limit.
  **Cause:** the bootstrap and the metrics block were written together with the policy code.
  **Fix:** removed the per-series output of `simulate_policy` (aggregation lives in `aggregate_policies`), dropped the stored cost fields of `PolicyAggregates`, replaced a hand-written JSON cleaner by a `default=` hook, shortened docstring and validation. A test fails above the limit.
- **Symptom:** the bundle calibrates quantiles for the selected model only, but the baseline needs its own curve.
  **Cause:** P4 stores one calibrator.
  **Fix:** the caller passes two quantile frames. The baseline frame is `fit_calibrator(baseline yhat on calib rows, y_calib, scale_calib, calibration_taus(cfg), cfg).quantiles(baseline yhat on test rows, scale_test)`, with the baseline forecast from `predict_point(..., which="seasonal_naive")`. P6 wires this. Test frame columns: `series_id` (column or index level), `y`, `yhat_baseline`; rows of the frame and of both quantile frames are matched by position.
- **Symptom:** three policies are named but not defined in the available text.
  **Fix:** `status_quo` = baseline point forecast times one plus `status_quo_buffer`; `baseline_newsvendor` and `model_newsvendor` = quantile at tau* of the baseline and of the selected model. When the verdict is `use_baseline` the last two are the same policy.
- **Symptom:** a quantile column for tau* is named from a rounded float.
  **Fix:** `aggregate_policies` needs the exact `q_<tau>` column of every tau (missing column: `E_PREDICT_BAD_INPUT`, no silent interpolation). `order_up_to(q, tau)` is the free-tau version: exact column if present, else linear interpolation between neighbouring tau columns (non-decreasing in tau), no extrapolation. It is what the "higher price never lowers `order_up_to`" test uses.
- **Symptom:** "same service level" needs a reference point.
  **Fix:** the target is the model's stockout rate at tau*. The baseline's inventory at that rate comes from its curve by linear interpolation; points with the same stockout rate keep the smallest inventory; a target outside the baseline stockout range, or a zero baseline inventory, gives `available = false` with null values and a `reason`. A positive `inventory_reduction_rel` means the model needs less inventory. Always labelled "conditional value".
- **Symptom:** `net_value_delta` depends on how many origins a resample holds.
  **Fix:** it is a total over the test origins; each resample total is rescaled by (all origins / drawn origins). Resampling is by whole series, as in P4; correlation between series is ignored. Intervals are percentile intervals widened to contain the point value. Resamples with no equal-service match are dropped and counted in `inventory_reduction_rel.n_boot_available` (the net-value interval keeps all resamples).
- **Symptom:** `write_business_metrics` has no P0 error code for a corrupt `metrics.json`.
  **Fix:** `E_CONFIG` naming the file; the file is left untouched. Other keys of `metrics.json` are kept, the write is atomic (temp file, then replace), and NaN or inf raises instead of being written.

Observations for later parts:

- When the verdict is `use_baseline`, model and baseline policies are identical, so the equal-service reduction is exactly zero with a zero-width interval. That is the honest result (the third design guard), not a bug.
- The newsvendor policy buys service with more inventory than the status quo when tau* is high; the report must not call that "less stock".

### Report, verdict, chart

- **Symptom:** the P0 verdict table and the exact `metrics.json` layout are not in this repository.
  **Cause:** only P1 to P4 notes and the task text were available.
  **Fix:** `decide` implements the rule below and `report.METRICS_SKELETON` fixes the shape; both are pinned by tests. If P0 says otherwise, change `decide` and the skeleton. P0 change: to be checked.
  - `use_baseline`: CV selection verdict is `use_baseline`, or `delta_primary >= 0` on test (`R_CV_CANDIDATE_NOT_BETTER`, `R_TEST_DELTA_NOT_NEGATIVE`).
  - `keep_status_quo`: test series below `report.min_test_series`, test origin rows below `report.min_test_obs`, equal service not available, or net value against the status quo not positive (`R_TEST_SERIES_BELOW_MIN`, `R_TEST_OBS_BELOW_MIN`, `R_EQUAL_SERVICE_UNAVAILABLE`, `R_NET_VALUE_NOT_POSITIVE`).
  - `pilot_then_scale`: otherwise. Priority `use_baseline` > `keep_status_quo` > `pilot_then_scale`; `reasons` hold only the codes of the chosen verdict.
  - `headline_supported` = verdict is `pilot_then_scale`, `equal_service.available`, and `inventory_reduction_rel.ci_low > 0`.
  - Flags (never change the verdict): `F_SYNTHETIC_DATA`, `F_CALIBRATION_OUT_OF_TOLERANCE`, `F_ACCURACY_CI_INCLUDES_ZERO`, `F_NET_VALUE_CI_INCLUDES_ZERO`, `F_MORE_INVENTORY_THAN_STATUS_QUO`, `F_LEAKAGE_CHECK_NOT_PASSED`. Their wording is in `report.FLAG_TEXT`, not in the codes.
- **Symptom:** `report.min_test_obs` can be larger than the number of test origin rows of a small config.
  **Fix:** the gate counts test origin rows (`test.n_obs`), the same unit the decisions are made on. With a small test split this fails the gate, so a candidate that wins would still get `keep_status_quo`. That is a config decision (`origin_step_periods`, `min_test_obs`), not a report change. P0 change: to be checked.
- **Symptom:** the code list is closed and has no code for a wrong type in `metrics.json` or a template error.
  **Fix:** a missing key, a null where none is allowed, a wrong type (`bool` is not a number, NaN or inf is rejected), and any undefined name in the template all raise `E_RENDER_MISSING_KEY` naming the path. A missing template file or unreadable `config.yaml` raises `E_CONFIG`. `decide` and `report.py` never use `.get` (a test greps for it).
- **Symptom:** several values the report needs are not in the P4 and P5 outputs as one block (hashes, test sizes, per-fold CV scores, calibration, leakage result, selection verdict, test accuracy with its interval).
  **Fix:** `METRICS_SKELETON` lists them; P6 builds `accuracy` (from `Bundle.selection` and `compare_to_baseline`), `calibration` (from `calibration_report`), `test`, `leakage` (from `perturb_future_check`), `data_source`, `data_hash`, `model_hash` and writes them next to the `business` block. Extra keys are allowed. `test.n_obs` is origin rows and `test.n_series` the series count.
- **Symptom:** `source_hash` must cover `metrics.json`, but `render_report` takes a dict.
  **Fix:** SHA-256 over the bytes of `config.yaml` plus the canonical JSON (sorted keys, compact) of the metrics, so formatting of the file does not matter but any value does. `config_path` defaults to the project `config.yaml`.
- **Symptom:** "no numeric literal in the template" while the report is full of numbers.
  **Fix:** numbers come only from the context; fixed formats are named filters (`ratio`, `pct`, `money`, `units`, `count`, `signed`; `n/a` for null). `TEMPLATE_NUMBER_WHITELIST` holds only the zero used for the sign of the accuracy difference. A test scans the template for any other digit, including digits inside names. Fixed texts (reasons, flags, limitations) live in `report.py`.
- **Symptom:** `render_report` writes only `report.md`.
  **Fix:** P6 calls `chart(metrics)` first so the image link resolves. `chart` marks tau* on both curves (the same point when the verdict is `use_baseline`) and the status quo as "not at equal service"; synthetic data adds a red banner.
- **Symptom:** the limitations list needs fixed keys.
  **Fix:** `no_carry_over`, `supplier_constraints`, `substitution`, `demand_proxied_by_sales` as specified, plus `delisting_tail` (from the P3 note).
- **Symptom:** when the CV selection is the baseline, the equal-service block reads as a real result (a zero reduction with a zero-width interval).
  **Cause:** model and baseline newsvendor policies are the same policy, so the difference is an identity, not a measurement.
  **Fix:** `render_report` passes `selected_is_baseline` (`accuracy.selection_verdict == "use_baseline"`) and `SELECTED_IS_BASELINE_NOTE` to the template, which prints a "Not an empirical result" note at the top of the equal-service paragraph. It is a rendering note, not a flag: `FLAGS`, the verdict, `metrics.json` and the hash are unchanged. When only the test delta is not negative (CV chose the candidate), the policies differ and no note is shown.
- **Symptom:** the "Stockout vs working capital" table shows stockout and inventory for status quo, baseline and model side by side, outside the equal-service block. Only the status quo row was labelled ("not at equal service"); a reader could take the baseline and model rows as "the model holds less stock".
  **Cause:** the "conditional value" label sat only in the general sentence above the section, not at the table.
  **Fix:** one sentence directly under the table says the rows are not a comparison and carry no claim, and that a claim exists only in the equal-service block (the label is printed from `value_label`). The sentence does not state why the rows differ, because when the baseline is selected the two rows are identical and there is no gap. No numbers added to the template. Test: `test_render_policy_table_states_that_rows_are_not_a_claim`, in all four render scenarios. The sentence avoids the claim words of the keyword check on purpose.

## P6

Done: `src/fcst/pipeline.py`, final `README.md` (five sections), this file, final `Makefile`, and the end-to-end tests in `tests/test_business.py` (sections "P6: pipeline, end to end", the temporary-copy run, and the README, Makefile and NOTES checks). No other source file was changed.

- **Symptom:** the P0 text for `pipeline.run`, the output layout and the CLI is not in this repository.
  **Cause:** only the P6 task text and the notes of P1 to P5 were available.
  **Fix:** `run(cfg, data_path, sample, out_dir)` as given, plus two keyword-only extras, `config_path` (so the report's source hash and a relative data path follow a non-default config) and `log` (a callable; silent by default). `out_dir` is the output root: `reports/` (`metrics.json`, `report.md`, `tradeoff.png`) and `artifacts/` (`bundle.joblib`, `bundle.json`, `test_predictions.csv`); default is the project root. The CLI is `python -m fcst.pipeline [--sample] [--data PATH] [--out DIR] [--config PATH]`. `sample` together with an explicit `data_path` is `E_CONFIG`. P0 change: to be checked.
- **Symptom:** the step list "CV and selection, fit, calibration" is one call in `models.train_bundle`.
  **Cause:** P4 builds the bundle in one function and it splits internally.
  **Fix:** one step `train`; the pipeline calls `make_split` once more for the test and calibration rows (deterministic, same rows). Steps are `load`, `validate`, `clean`, `origins`, `features`, `split`, `train`, `predict_test`, `metrics`, `decide`, `report`, `write` (`pipeline.STEPS`, pinned by a test).
- **Symptom:** "the test set is predicted once" while the business block needs a baseline forecast and a baseline calibrator as well.
  **Cause:** the bundle calibrates quantiles for the selected model only (P4 and P5 notes).
  **Fix:** each model predicts the test origins exactly once: `predict` for the selected model (point and quantiles), `predict_point(which=...)` for the other one. The baseline calibrator is fit on the calibration rows only. Everything goes into one table, `pred` (`series_id`, `origin_date`, `y`, `yhat_baseline`, `yhat_candidate`, `naive_mse`, selected `q_<tau>`, baseline `baseline_q_<tau>`), saved as `artifacts/test_predictions.csv`. Accuracy, interval, calibration and the business block are computed from `pred` only. Tests: a recorder on both `predict` methods sees the test rows in exactly one call per model and that call holds exactly the test rows; the metrics are recomputed from the CSV; the saved bundle reproduces the table.
- **Symptom:** `source_hash` covers the metrics, but the verdict is also wanted in `metrics.json`.
  **Fix:** `decision` (the `decide` output) is added to the metrics before rendering, so the hash covers it; `metrics.json` is written last, from the same dict (normalised through JSON first, so numpy numbers and the file agree). A test recomputes the hash from the file and finds it in the report footer. Extra keys beyond the skeleton: `decision`, `split` (row counts), `leakage.max_abs_corr`, `accuracy.n_boot`, `accuracy.n_folds_*`, `accuracy.selected`.
- **Symptom:** `FcstError` has no field for the failing step and the code list is closed.
  **Fix:** the pipeline sets `exc.step` on the exception it re-raises; the CLI prints `error: <code>: <message> (step: <name>)` to stderr and exits with status one. Nothing is written after the failing step; the leakage and training failures are tested with the later steps stubbed to prove they are not reached.
- **Symptom:** `--sample` could read `data/sample.csv` or generate in memory.
  **Fix:** it generates in memory from `params.seed` (same `data_hash` that `make check-sample` compares with the file), so a stale or missing `data/sample.csv` cannot change a run. `make all-sample` still runs `make sample` first, as specified. The real path uses a small `_read_raw` (same read options as `data.load_panel`) so that load, validate and clean are separate steps; a test pins that both give the same panel hash.
- **Symptom:** `make all` runs lint and tests before the pipeline, so a missing real file is reported only after them.
  **Fix:** unchanged on purpose (the P1 target definitions stay); `python -m fcst.pipeline` alone fails at the first step. The Makefile test uses `make -n` so that the test suite does not call itself.
- **Symptom:** `report.min_test_obs` against the test origin rows of the default sample.
  **Fix:** none, as decided in P5 (config decision). Whether the gate decides a run is in `reports/metrics.json` (`decision.reasons`).
- **Symptom:** newer ruff versions flag the short regex flag alias (a ruff rule about the short flag alias).
  **Fix:** spelled out as `re.MULTILINE`.
- **Symptom:** the end-to-end check "same seed gives identical output" needs a run that does not share a working directory, and the hand-typed-number rule needs a test, not a promise.
  **Fix:** the tests copy the project (without `data/`, `artifacts/`, `reports/` and caches) into a temporary directory and run the CLI there with `--sample`. They check that the three report files exist, that `metrics.json` passes `check_metrics_shape`, that the banner is in the CLI output, report and metrics, that the report has no placeholders, that two runs with the same seed give the same `metrics.json` hash, and that a missing real panel stops with `E_DATA_FILE_MISSING` and writes nothing. A separate test scans README, this file, the `config.yaml` comments and the template for digits (allowed: part labels, list markers, hash and date format names).
- **Symptom:** the code was importable only through `PYTHONPATH=src`, with no packaging metadata and no documentation of that requirement.
  **Cause:** P1 specified `PYTHONPATH` in the Makefile only; `fcst` was a namespace package without a marker file.
  **Fix:** added `pyproject.toml` (setuptools, `src` layout, console scripts `fcst-pipeline` and `fcst-data`, pytest `pythonpath` and `testpaths`), `src/fcst/__init__.py`, a `make install-editable` target, ignore rules for build output, and the README subsection on the import path. No module code changed. Runtime dependencies are listed in both `requirements.txt` and `pyproject.toml`; keep them in step. `joblib` is declared in `pyproject.toml` because `models.py` imports it directly (it also arrives through scikit-learn). The Makefile keeps exporting `PYTHONPATH`, so make targets behave as before. P0 change: no.
- **Symptom:** a regular (non-editable) install would break at run time.
  **Cause:** `DEFAULT_CONFIG_PATH` is resolved from the file location (two levels above `fcst/`), and `config.yaml` and `templates/` sit outside the package, so a wheel does not contain them.
  **Fix:** documented: install editable only. Making a wheel self-contained needs package data and a config lookup change, which is left out on purpose. P0 change: no.

## Run record

This section is text only. The figures of any run are in `reports/metrics.json` and `reports/report.md`, which carry the data hash, model hash and source hash that identify the run.

- Clean-state command chain: `make clean && make install && make lint && make test && make all-sample`. Expected result: every step exits zero, lint is clean, all tests pass, `check-sample` is ok, the pipeline writes `metrics.json`, `report.md` and `tradeoff.png`, and the synthetic banner is shown at the start and the end of the CLI output, in the report and in the chart.
- The sample run is synthetic. Its verdict and flags are in `reports/metrics.json` under `decision`; with the default config the candidate does not beat the baseline, so the verdict is `use_baseline`, which is a legitimate result (the third design guard) and not a failure.
- Verification status in this environment: `make lint` and `make test` have never been run here. `pytest` and `ruff` are not installed and the network is off, so `make install` could not run either. The tests were run only through a stand-in runner, which is not the real `pytest`, and `ruff` has not been run at all. The end-to-end tests that copy the project into a temporary directory through `tmp_path_factory` and run the CLI there with `--sample` have not been run either, so the same-seed check, the report and metrics shape checks and the missing-panel check are unconfirmed.
- Lint configuration: `pyproject.toml` has no ruff section, so the default rules and the default line length apply. Several lines in `tests/test_business.py` and a few in `src/fcst/` are longer than that default, and they predate the latest changes. Line length is not part of the default rule set as far as known, so these lines are not expected to fail `make lint`, but this is unconfirmed until ruff is run; if the real run flags them, either wrap them or add an explicit ruff section. Treat lint as unchecked and the tests as checked only by the stand-in until the clean-state command chain above is run once with the real tools in a connected environment; this record is to be updated with that result.
- Sample run, flags and equal service: the pipeline was run directly with `python -m fcst.pipeline --sample` in a temporary copy of the project (not through `make`, so lint and tests were not part of it). The verdict is `use_baseline` because the selected model is the seasonal naive baseline. Flags raised: `F_SYNTHETIC_DATA`, `F_CALIBRATION_OUT_OF_TOLERANCE`, `F_NET_VALUE_CI_INCLUDES_ZERO` and `F_MORE_INVENTORY_THAN_STATUS_QUO`. The equal-service difference is exactly zero, because the model and the baseline are the same forecast, so their stockout and inventory curves coincide; the report marks it as not an empirical result. None of this is a failure. Flags never change the verdict, a coverage miss at one level on a small test set is not evidence of a broken calibrator (see P4), and more inventory than the status quo is a reported trade-off, not an error.
- `make all` with no panel at `params.data.path`: lint and tests are expected to pass (not yet confirmed, see the status above), then the pipeline stops at step `load` with `E_DATA_FILE_MISSING`, exits with status one, and writes no report or artifact. `make all` is meant to be run only when the real panel exists.
- Known limit of the claim check: the test that keeps less-stock claims inside the labelled equal-service blocks is keyword based. It looks for a claim word on the same line as a subject word. The claim words cover English (less, fewer, lower, smaller, reduce, decrease, save, cut, down, drop, better, improve, gain, efficient, higher) and Indonesian (turun, menurun, kurang, berkurang, mengurangi, lebih sedikit/rendah/kecil/baik/hemat, hemat, efisien, naik, meningkat); the subject words are inventory, stock, working capital, service, fill rate and their Indonesian forms (persediaan, stok, modal kerja, layanan). The lists are tested on their own (`test_claim_detector_catches_english_and_indonesian_claim_phrases` and its neutral-line counterpart), so cutting a word breaks a test. It is still a keyword check: a phrase that uses none of these words, a language other than English or Indonesian, or a claim split over two lines passes unseen. If the template is edited often or translated, review the wording by hand as well.
- Not yet seen end to end: the `pilot_then_scale` verdict and the headline line that goes with it have been rendered only from test fixtures, never from a pipeline run. The only pipeline run so far is the synthetic sample, which ends in `use_baseline`, so the headline is absent there. The first real panel that produces this verdict needs its `reports/report.md` read by hand: the headline, the equal-service block with its label, and the flags.
- Update on the `pilot_then_scale` check (still open): the sample pipeline was also run with 80 and 120 series and seeds 42, 7 and 123 (four runs). In all of them CV kept the baseline, so none produced `pilot_then_scale`; a candidate win was not forced. The `pilot_then_scale` report rendered from the test fixture was read by hand: headline, equal-service block with its label, flags and the new note under the policy table read correctly. On the real sample run two identities hold and are worth checking on the first real panel: the model row's inventory equals `equal_service.inventory_model`, and the model minus status quo net values equal `net_value_delta.model_vs_status_quo.value`. The fixture itself breaks both (495 / 50 / 100 inventory, 15,848 minus 14,232 against +500) because it overrides single fields; that is a fixture artifact, not a template fault.
- Not done: no real panel was available, so there is no real-data result in this repository. Everything produced here comes from the synthetic sample and is not evidence about real sales.
