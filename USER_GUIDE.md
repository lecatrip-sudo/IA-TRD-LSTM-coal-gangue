# User guide

## 1. Choose the appropriate task

Use `code/examples/recompute_lab_metrics.py` to verify the existing laboratory
results from released observations and stored predictions. Use
`code/examples/synthetic_reference_example.py` to understand the reference
equations using manually specified, nonresearch inputs. Both tasks run offline
without retraining or third-party packages.

The original `verification/recompute_headline_numbers.py` can additionally
check prediction-file completeness and aggregate arithmetic. It requires
NumPy and pandas and has the narrower CFD scope documented in the original
README. Full CFD metric reconstruction requires the restricted targets.

## 2. Laboratory data and prediction schema

`data/laboratory/records_C.csv` has one row for a probe, trial and sample index.
It includes trial, probe, sample_index, time_s, temperature_C and source-workbook
provenance. The 6,936 rows comprise two trials, twelve probes and 289 samples
per probe at five-minute spacing. Temperatures are converted to kelvin by
adding 273.15; a temperature error has the same numerical value in kelvin and
degrees Celsius.

`data/laboratory/lab_predictions_v99.csv.gz` stores trial, method, seed, probe,
time_h, a_K, b_K, y_K and pred_K. Here `time_h` is the target time; `a_K` is
eight hours before that target and `b_K` is four hours before it. Each
trial-method-seed group contains 49 target times from 20 through 24 h, with
twelve probes at each time: 588 rows. Seed zero denotes deterministic
references. Neural forecasts retain seeds 42, 43 and 44.

The tutorial checks that all `a_K`, `b_K` and `y_K` values match the released
measured records, then recomputes MAE, RMSE, signed bias and R-squared for each
group. It compares those values with `lab_runs_v99.csv` rather than producing
a new evaluation split. The aggregation first averages seeds within each trial
and then weights the two trials equally, following the existing report.

The tutorial also reconstructs the deterministic reference forecasts. Saved
training-derived calibration coefficients come from `lab_calibration_v99.json`;
they are read unchanged and not fitted again. Own-history coefficients are
computed from the same earlier observed fields as in the retained code.

## 3. Save optional tutorial reports

```text
python code/examples/recompute_lab_metrics.py --output tutorial_output/laboratory_check.json
python code/examples/synthetic_reference_example.py --output tutorial_output/artificial_formula_example.json
```

The optional output paths are new files chosen by the reader. They do not
replace the research data or published summary tables. The laboratory JSON
includes input hashes, per-run metrics and aggregate checks. The artificial
JSON explicitly states `synthetic_tutorial_only: true` and
`not_manuscript_data: true`.

## 4. Interpret outcomes

An exit status of zero and `status: PASS` means that the checks implemented by
that tutorial succeeded. It does not certify the original physical solver,
prove the reported model generalizes, or recreate neural training.

A nonzero status means that a file, schema, record alignment, formula or metric
check failed. Keep the error message, verify `DISTRIBUTION_MANIFEST.json` and
confirm that the files are an unchanged copy of this distribution. Do not
silently loosen the numeric tolerance or replace targets to obtain a pass.

## 5. Historical source and licence boundaries

Original source snapshots, outputs and internal method labels are preserved
for provenance. All comments in the inspected original Python sources are
English; some original workstation path strings contain Chinese directory
names. Those strings are retained unchanged. The new tutorials and English
documentation do not depend on those paths.

MIT applies to code, including the added tutorials. CC BY 4.0 applies to the
released data/outputs and added distribution documentation. Restricted CFD
exports are excluded and require authorized access from their holder. No new
licence is asserted for those excluded records.
