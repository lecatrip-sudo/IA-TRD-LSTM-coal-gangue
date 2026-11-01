# IA-TRD-LSTM coal-gangue temperature forecasting: public release V101

This archive accompanies the manuscript

> Spatial agreement masks temporal failure in temperature forecasts: an instance-adaptive
> thermal-relaxation decomposition with CFD and laboratory evidence from coal-gangue covers

It contains source code, prediction-only outputs, aggregate metrics, inner-selection records
and laboratory records. The original CFD temperatures, input fields and coordinates belong to
the companion study (Li et al. 2026, doi 10.1016/j.jece.2026.123803) and are not redistributed.
CFD prediction files therefore contain only row identifiers and forecast values, not inputs or
targets. Row-wise CFD rescoring requires access to the restricted source records held by the
corresponding author; the release verifier checks prediction completeness and consistency of
the published aggregates, not an independent reconstruction of CFD errors.

## Layout

- `code/matched_hlstm/` frozen A1/A2 absolute-output and R/S increment-output training sources,
  preserved with their original workstation-layout and frozen-manifest requirements.
- `code/ia_trd/` arm-T implementation (`v99_t_worker.py`), the inner-fold worker (`ns_worker.py`),
  the nesting and selection orchestrator (`ns_suite.py`) and the shared network/scaler module
  (`legacy_core.py`). The first-execution worker is retained for provenance.
- `code/laboratory/` the cross-trial sensor benchmark in its delivered form
  (unified decomposition, own-history references, alpha sensitivity) and the protocol-variant
  sensitivity script; the first-execution sensor-specific form is retained for provenance.
- `code/analysis/` deterministic reference construction, post-compute analyses (alpha
  sensitivity, threshold assessment, spatial block bootstrap, relaxation timescales,
  probe-resolved summaries), aggregation of all CFD results and figure generation.
- `code/development_history/` the three exploratory scripts that preceded the first fixed
  configuration. They are included because the manuscript discloses them (Note S9.2) and
  Table S21 reproduces their grid.
- `data/predictions/primary/` every delivered A1/A2/R/S/T/T3/TMP run of the six folds and three
  seeds, stored as row identifiers and forecasts only; JSON sidecars retain run metadata.
- `data/predictions/references/` deterministic reference forecasts, also without source fields.
- `data/predictions/archived/` the archived aLSTM forecasts, stored without source fields.
- `data/predictions/diagnostic60/` per-epoch metrics for the 60-epoch diagnostics, not row-wise forecasts.
- `data/selection/` the complete 270-fit inner-validation surface (per-fit JSON, the aggregated
  score matrix `ns_inner_scores.csv`, the selected configuration per fold and the final plan).
- `data/summary/` machine-readable aggregate metrics and diagnostics used in the manuscript and Online Resource.
- `data/laboratory/` the two 24 h trials of twelve K-type probes (`records_C.csv`,
  `probe_coordinates.csv`), the delivered cross-trial predictions and the metric table.
- `verification/recompute_headline_numbers.py` checks prediction-file completeness and
  consistency between released per-run and aggregate metrics; it exits non-zero on mismatch.

File and method names that contain `v98` or `v99` (for example `v99_t_worker.py`,
`lab_predictions_v99.csv.gz` and the method key `IA-TRD-LSTM Eq. (10) form (V98)`) are internal
labels of the stage in which a file was produced. The method key denotes the sensor-specific
residual form of Online Resource Eq. (S8), which the manuscript reports as a post-hoc variant.

## Reproduction

    python verification/recompute_headline_numbers.py # checks the released files and aggregates

This verification command is self-contained: it reads only files inside this archive.
It cannot re-score CFD forecasts because the source targets are not redistributed.
The training and analysis scripts are historical source snapshots, not an installable
package or a turn-key retraining environment. They retain absolute Windows workstation
paths and cross-module imports. Retraining or rerunning the original analysis requires
adapting that project layout, obtaining the restricted CFD records and supplying the
saved fit directories/checkpoints used by the analysis. Their inclusion does not make
full independent CFD reproduction possible from this archive alone.

## Licences

Code in `code/` is released under the MIT licence (`code/LICENSE_MIT.txt`). Predictions,
aggregate tables, selection records and laboratory records are released under CC BY 4.0
(`LICENSE_CC_BY_4.0.txt`). The original CFD exports are not covered by either licence and are
not included. See `LICENSE_SCOPE.md`.

## Numerical environment

Python 3.11.9, PyTorch 2.10.0+cpu, NumPy 2.4.2, pandas 3.0.0, matplotlib 3.10.x, CPU float32
with deterministic operations, at most seven concurrent processes with two intra-operation
threads each.
