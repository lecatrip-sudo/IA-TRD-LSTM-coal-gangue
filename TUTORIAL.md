# Tutorial

## A. Re-score stored laboratory forecasts

Run from the repository root:

```text
python code/examples/recompute_lab_metrics.py
```

Expected inventory checks are 6,936 measured records, 30,576 prediction rows,
52 trial-method-seed groups and 9,408 deterministic-reference row checks.
All four per-run metrics are compared to the retained scores at absolute
tolerance `1e-9`. These counts describe the existing release; no new windows,
forecasts or target observations are added.

The printed JSON includes `status: PASS`. Representative existing trial-equal
MAEs are approximately 0.3500789217 K for `Damped trend`, 0.3458003555 K for
`Relaxation component (no network)` and 0.5060316143 K for
`IA-TRD-LSTM (T)`. The different historical method key
`IA-TRD-LSTM Eq. (10) form (V98)` is the reported post-hoc residual variant and
must not be read as the primary unified method.

The script reads stored neural predictions rather than refitting them. Its
input/target alignment checks and metric recomputation are fully supported by
the released laboratory records. The same is not possible for the restricted
CFD targets from the public subset alone.

## B. Follow the own-history formulas on artificial input

```text
python code/examples/synthetic_reference_example.py
```

This is a manually specified arithmetic example at four artificial locations
and three equally spaced instants. It is neither measured data nor CFD output,
and it is not an added validation case. The inputs are:

```text
a0 = [300, 310, 290, 300] K
a  = [304, 312, 290, 302] K
b  = [306, 315, 290, 301] K
```

The earlier change is `d0 = a - a0` and the latest change is `d = b - a`.
The manuscript's own-history coefficient is
`q_star = sum(abs(d)) / sum(abs(d0)) = 6/8 = 0.75`.
The rowwise ratio is clipped to [0,1], with the retained fallback for zero
earlier change. Persistence, linear extrapolation, R1, R2 and R3 use the
existing manuscript formulas; the script does not fit an LSTM.

The hand-calculated R3 forecast is:

```text
[307.25, 317.625, 290.0, 300.625] K
```

The script checks that vector and verifies that shifting all three input
fields by a constant shifts the forecast by the same constant. These checks
exercise the formula implementation, not scientific accuracy. Both flags
`synthetic_tutorial_only` and `not_manuscript_data` are true in its output.

## C. Retain an execution record

Use the optional `--output` argument described in USER_GUIDE.md to save the
JSON. Record the Python version and the distribution manifest when sharing a
reproduction result. This tutorial does not publish anything to a repository
or contact an external service.
