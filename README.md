# forecasting-retail-fmcg

Demand forecasting and a newsvendor order-up-to policy for retail / FMCG, with an honest verdict on whether the model is worth adopting.

Documentation, in reading order: [User Guide](docs/user-guide.md) (run it and read the report) · [Methodology](docs/methodology.md) (why each number is what it is) · [Reference](docs/reference.md) (config, formats, error codes, extending). Planners and buyers stop after the User Guide; analysts add the Methodology; developers go from here to the Reference.

> **In short.** For every SKU x location, the pipeline forecasts demand over one order cycle, turns the forecast into an order-up-to quantity, and tests whether that beats a simple baseline. What you get is a verdict you can act on, not only a forecast.

| | |
|---|---|
| **Decision supported** | How many units to order up to, per series, per review cycle. |
| **Who it is for** | Planners and buyers who must decide whether to adopt a model; analysts and developers who maintain the pipeline. |
| **You get, per run** | Calibrated forecast quantiles, an order-up-to policy, and one report with a verdict, its reason codes, flags and limitations. |
| **The one rule** | The model may lose. If it does not beat the baseline, the verdict is to keep the baseline. |
| **Status** | The method is shown on synthetic data only. No real-data result exists yet. |

```
panel (SKU x location x day)
  -> features computed as of the order date (no look-ahead)
  -> baseline vs candidate model, selected on training folds only
  -> calibrated quantiles
  -> order-up-to level at the target service level
  -> verdict + report (reports/)
```

**Where the money comes from.** Retail demand without contracts loses money in two ways: stockouts (margin lost on sales that did not happen) and overstock (cash tied up, expiry, markdown). The project prices both from the cost inputs and states the gain only as a conditional value.

| Lever | How the project handles it | Where you read it |
|---|---|---|
| Fewer lost sales | The target service level comes from the cost inputs: margin per unit against holding cost plus expected write-off. | `config.yaml`, report |
| Less cash tied up | Inventory is compared at the same stockout rate as the baseline ("equal service"). | equal-service block of the report |
| Not adopting a model that does not pay | Verdict gates: baseline wins, too little test data, or net value against the status quo not positive. | `decision` in `reports/metrics.json` |
| No false confidence | Bootstrap intervals, calibration check, flags, and a banner on synthetic data. | report |

**How to read a business figure.** It is a *conditional value*: it holds only under the cost inputs in `config.yaml` (illustrative assumptions) and only at the same service level. Holding less stock while missing more sales is not an improvement, so the only place that carries a claim is the equal-service block. The stockout versus working-capital table next to it is not a comparison.

## Problem

- **Domain.** Retail / FMCG, recurring retail demand without contracts. Channel: store.
- **Problem.** Stockout (lost sales) and overstock (tied-up working capital, expiry, markdown).
- **Forecast unit.** Series = SKU x location. Period = day. Demand = units sold per period.
- **Horizon and target.** H = lead_time_periods + review_period_periods. Target = total demand in (origin, origin + H].
- **Decision.** How many units to order-up-to per series per review cycle.
- **Value inputs.** Price, cost, holding cost, salvage, writeoff share: all in `config.yaml`, illustrative, assumption status.
- **Perishable?** yes, with a shelf life longer than the horizon (determines `overstock_writeoff_share`).
- **Data.** synthetic only (a real panel is read from `params.data.path` in `config.yaml`; a missing file is a hard error, never a silent fallback to synthetic data).
- **Note.** There is no stockout column: demand is proxied by sales, so true demand may be larger. This is always stated in Limitations.

## Results

This README holds no result figures. Every run writes its results to `reports/`:

- `reports/report.md`: the verdict with its reason codes, accuracy, calibration, the stockout versus working-capital trade-off, flags and limitations. It is rendered from `reports/metrics.json` and carries a source hash, so a stale report can be detected.
- `reports/metrics.json`: every number of the run.
- `reports/tradeoff.png`: the trade-off chart.

Numbers live in two places only: `config.yaml` (inputs, assumptions) and `reports/metrics.json` (outputs). Output from synthetic data is bannered and says nothing about real sales. Business figures are always a conditional value, compared at the same service level.

### The verdict

| Verdict | Meaning | Typical reasons |
|---|---|---|
| `use_baseline` | The candidate does not beat the baseline. Keep the baseline. This is a legitimate result, not a failure. | `R_CV_CANDIDATE_NOT_BETTER`, `R_TEST_DELTA_NOT_NEGATIVE` |
| `keep_status_quo` | The evidence is too thin or the value is not positive. Do not change the current practice. | `R_TEST_SERIES_BELOW_MIN`, `R_TEST_OBS_BELOW_MIN`, `R_EQUAL_SERVICE_UNAVAILABLE`, `R_NET_VALUE_NOT_POSITIVE` |
| `pilot_then_scale` | The candidate wins and the value is positive under the stated assumptions. Test it on a small scope first. | none (all gates passed) |

Priority is `use_baseline`, then `keep_status_quo`, then `pilot_then_scale`. The headline claim is printed only when the verdict is `pilot_then_scale`, equal service is available and the lower bound of the inventory difference is above zero. Flags (for example `F_SYNTHETIC_DATA`, `F_CALIBRATION_OUT_OF_TOLERANCE`, `F_NET_VALUE_CI_INCLUDES_ZERO`) never change the verdict; they tell you how far to trust it.

### Reading a report

- Start with the verdict and its reasons, then the flags.
- Check accuracy against the baseline and whether its interval includes zero.
- Check calibration: does the stated service level hold on the test set.
- Read the business block only through the equal-service paragraph.
- Read the limitations before quoting any figure.

## Structure

- `config.yaml`: all inputs and parameters (illustrative assumptions).
- `src/fcst/__init__.py`: package marker (docstring only).
- `src/fcst/config.py`: config loading, validation, the `FcstError` class and its error codes.
- `src/fcst/data.py`: data adapter (load, validate, clean, hash) and the synthetic generator.
- `src/fcst/features.py`: origins, as-of features, labels, leakage checks, temporal split, rolling-origin CV.
- `src/fcst/models.py`: baseline and candidate models, selection, calibration, bundle save and load, predict.
- `src/fcst/evaluate.py`: accuracy and quantile metrics, calibration report, paired bootstrap.
- `src/fcst/newsvendor.py`: cost rates, service target, policy simulation, equal-service comparison, business block.
- `src/fcst/report.py`: verdict rules, strict report rendering, trade-off chart.
- `src/fcst/pipeline.py`: the end-to-end run and the command line entry point.
- `templates/report.md.j2`: the report template (no numeric literals).
- `tests/test_inputs.py`, `tests/test_features.py`, `tests/test_models.py`, `tests/test_business.py`: tests per part; the last one also holds the end-to-end test.
- `docs/user-guide.md`, `docs/methodology.md`, `docs/reference.md`: the long-form documentation (workflow and reading the report, method and formulas, configuration and error codes).
- `Makefile`, `requirements.txt`, `NOTES.md` (design notes and the run record).
- `pyproject.toml`: packaging metadata (installable `fcst` package, console scripts, pytest `pythonpath`).
- Generated, not committed: `data/`, `artifacts/`, `reports/`, and packaging output (`build/`, `dist/`, `*.egg-info/`).

## Run

```
make install       # dependencies
make test          # lint is a separate target: make lint
make all-sample    # quick path on synthetic data: not a source of results
make all           # real data from params.data.path
```

`make all` stops with `E_DATA_FILE_MISSING` if the panel file does not exist. The panel is a CSV with one row per series and period and the columns `sku_id`, `location_id`, `series_id`, `period_start`, `demand`; other columns are dropped when the data is cleaned.

The pipeline can also be called directly:

```
python -m fcst.pipeline [--sample] [--data PATH] [--out DIR] [--config PATH]
python -m fcst.data --write-sample | --check-sample
```

The pipeline stops at the first step that fails and prints its error code and the step. The test set is predicted once and every later number comes from that prediction table.

### Import path: PYTHONPATH or editable install

The code lives in a `src/` layout, so `fcst` is importable only when `src` is on the import path. There are three ways to get it there; pick one.

| Way | When | Command |
|---|---|---|
| `make` targets | default; the Makefile exports `PYTHONPATH=src` | `make test`, `make all-sample` |
| `PYTHONPATH` by hand | direct calls outside `make` | see below |
| Editable install | you want no `PYTHONPATH` at all, or the console scripts | `make install-editable` |

By hand, from the project root (Linux and macOS):

```
export PYTHONPATH=src            # or per call: PYTHONPATH=src python -m fcst.pipeline --sample
python -m fcst.pipeline --sample
```

Windows PowerShell: `$env:PYTHONPATH = "src"`. Windows cmd: `set PYTHONPATH=src`.

Symptom of a missing path: `ModuleNotFoundError: No module named 'fcst'`. `pytest` does not need the variable, because `pyproject.toml` sets its `pythonpath` to `src`.

Editable install (`pip install -e ".[dev]"`, same as `make install-editable`) registers the package from `pyproject.toml` and adds two console scripts that mirror the module calls: `fcst-pipeline` is `python -m fcst.pipeline`, and `fcst-data` is `python -m fcst.data`, with the same options.

Install it editable, not as a regular wheel: `config.yaml` and `templates/` are read from the project root next to `src/`, and a regular install would copy only the code into site-packages and lose them. `requirements.txt` stays the plain dependency list for `make install`; `pyproject.toml` lists the same runtime packages (plus `joblib`, which the bundle code imports directly) and puts `pytest` and `ruff` in the `dev` extra.

## Limitations

- Demand is proxied by sales; there is no stockout column, so true demand may be larger.
- No carry-over: every review cycle starts with no stock and is independent.
- No supplier constraints: no minimum order, pack size or capacity.
- No substitution between products or locations.
- The fade-out of delisted series is under-represented in train and test.
- The cost inputs are illustrative assumptions; business figures hold only under them.
- Neighbouring origins have overlapping target windows, and the bootstrap resamples whole series and ignores correlation between series.
- Coverage on a small test set is noisy; a miss at one level is not by itself proof of a broken calibrator.
- Synthetic data demonstrates the method only. It does not support any claim about real sales.
