# IA-TRD-LSTM thermal forecasting code and retained outputs

This repository-ready tree accompanies the manuscript:

> Spatial agreement masks temporal failure in temperature forecasts: an
> instance-adaptive thermal-relaxation decomposition with CFD and laboratory
> evidence from coal-gangue covers

It contains the unchanged V101 release payload, with installation notes and two
small runnable tutorials added for distribution. It has been prepared locally
for author-managed release; this tree does not assert that a public repository
record or DOI already exists. The original payload README is retained as
`README_V101_original.md`.

## Quick start

Install Python 3.11 or later, open a terminal in this directory, and run:

```text
python code/examples/recompute_lab_metrics.py
python code/examples/synthetic_reference_example.py
```

Both commands use only the Python standard library. They do not install
packages, train networks, change source data, or create files by default.
The first re-scores the existing laboratory predictions against the released
measured records and verifies the released per-run metrics. The second uses
four manually chosen artificial locations to illustrate the paper's existing
reference formulas. Its values are tutorial inputs, not additional research
data or a new CFD validation case.

See [INSTALL.md](INSTALL.md), [USER_GUIDE.md](USER_GUIDE.md), and
[TUTORIAL.md](TUTORIAL.md) for commands, expected outputs, and scope.

## What is included

- `code/matched_hlstm/`: retained A1/A2 and R/S training sources.
- `code/ia_trd/`: IA-TRD-LSTM workers and inner-fold selection sources.
- `code/laboratory/`: retained cross-trial laboratory benchmark sources.
- `code/analysis/`: reference construction, analysis and figure sources.
- `code/development_history/`: disclosed exploratory development snapshots.
- `code/examples/`: new, standard-library-only distribution tutorials.
- `data/predictions/`: retained primary and reference CFD predictions, archived
  aLSTM predictions, and training-budget diagnostic metrics.
- `data/selection/`: retained inner-selection outputs and configuration records.
- `data/summary/`: reported aggregate metrics and diagnostic tables.
- `data/laboratory/`: two measured 24 h trials, probe coordinates, stored
  cross-trial predictions, calibration and per-run scores.
- `verification/`: the original release-verification source.

The historical numeric labels `v98` and `v99` in filenames or method keys are
retained for provenance. In particular, the laboratory key
`IA-TRD-LSTM Eq. (10) form (V98)` denotes the reported residual protocol variant;
it is distinct from the unified primary `IA-TRD-LSTM (T)` method.

## Reproduction scope

The tutorials reproduce released laboratory errors and deterministic references
without retraining. They also show a tiny artificial formula example. These
are executable entry points into the release, not replacements for the
historical research pipeline.

The original full-field CFD inputs, target temperatures and coordinates are
excluded from this public subset. Their access conditions and scientific
provenance are stated in the manuscript; requests go to corresponding author
Ang Li at `fcla@imau.edu.cn`. Full row-wise CFD rescoring and training require
access to those records. Prediction-only CFD files cannot independently
establish CFD error values without the corresponding targets.

The retained training and analysis scripts preserve their original Windows
workstation paths, local-module imports, saved-fit requirements and execution
layout. They are source snapshots, not an installable package or a turn-key
training environment. Installing the recorded dependencies does not restore
those missing resources. No training scripts were changed or rerun when these
distribution tutorials were added.

## Verification

`code/examples/recompute_lab_metrics.py` checks all 6,936 released temperature
records against the input/target fields in 30,576 stored forecast rows, verifies
52 trial-method-seed metric groups, and reconstructs 9,408 deterministic
reference rows. It checks MAE, RMSE, bias and R-squared against the existing
`lab_runs_v99.csv` at absolute tolerance `1e-9`.

The retained `verification/recompute_headline_numbers.py` is a different
verifier: it needs NumPy and pandas and checks public prediction completeness
and consistency of released aggregates. It cannot independently rescore
restricted CFD targets. It writes `verification_report.json`; the unchanged
payload already contains its earlier report.

`SOURCE_ARCHIVE.json` identifies the source archive and the original-to-current
README mapping. `DISTRIBUTION_MANIFEST.json` lists the repository payload with
SHA-256 values, excluding that manifest itself. Original V101 files are
byte-identical to the source archive, including the renamed original README.

## Licences

Research code and the added `code/examples/` scripts use the MIT licence in
`code/LICENSE_MIT.txt`. Released predictions, selection outputs, laboratory
records and aggregate tables use CC BY 4.0, as specified by the unchanged
`LICENSE_SCOPE.md` and `LICENSE_CC_BY_4.0.txt`. Added English distribution
documentation is also CC BY 4.0. The excluded original CFD exports are not
covered by these public-release licences. Dependencies retain their own
licences.

No submission, repository publication or account access is performed by these
files. The author should replace any repository-availability wording only
after the actual public record exists.
