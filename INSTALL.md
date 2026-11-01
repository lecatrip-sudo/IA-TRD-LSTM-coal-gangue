# Installation

## Tutorial environment

Use Python 3.11 or later. The two tutorials import only standard-library
modules. No third-party packages, GPU, network connection or account is needed.
The distribution check was executed with Python 3.12.14 on Windows; that check
is not a retraining replication of the recorded research environment.

From the repository root:

```text
python --version
python code/examples/recompute_lab_metrics.py
python code/examples/synthetic_reference_example.py
```

The paths are resolved relative to each tutorial file, so commands also work
from another working directory when the script is invoked by its full path.
If Python is exposed as `python3` on your operating system, use that executable
in place of `python`.

The root `requirements.txt` documents that the tutorial path has no external
requirements. The scripts exit with a nonzero status if input/metric checks
fail. By default they print JSON and do not write files.

## Retained research sources

The recorded numerical environment is Python 3.11.9, PyTorch 2.10.0+cpu,
NumPy 2.4.2, pandas 3.0.0 and matplotlib 3.10.x, using CPU float32. These values
are copied from the retained release documentation. The exact matplotlib
patch was not recorded.

`requirements_research_snapshot.txt` records the third-party research packages
and the unresolved matplotlib patch range. It is an environment snapshot, not
a complete lockfile or a promise of identical numerical results on another
platform. An optional isolated environment can be created with:

```text
python -m venv .venv
```

Select that environment's Python in your terminal or editor before installing
the snapshot dependencies. Select a PyTorch CPU build compatible with your
platform using the official PyTorch installation guidance.

The legacy verifier imports NumPy and pandas. Training and plotting sources
also import PyTorch and matplotlib. Local imports such as `legacy_core`,
`core`, `residual_core` and `run_experimental_benchmark` refer to historical
project files; they are not extra packages to find in a package index.

## Resources not supplied by installing dependencies

The training snapshots retain absolute original workstation paths and depend
on the original module layout, source exports, fit directories and checkpoints.
The public subset excludes the full CFD input fields, target temperatures and
coordinates. Readers must obtain authorized access and adapt the documented
project layout before attempting CFD training or full row-wise rescoring.

Do not run the historical training files expecting them to create a complete
project from this public subset. The runnable tutorials below the
`code/examples/` directory are the self-contained execution route provided by
this distribution wrapper.
