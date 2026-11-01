"""Recompute the released laboratory metrics without fitting any model.

This tutorial reads existing measured records and stored predictions. It does
not train networks, rescore restricted CFD records, or add manuscript results.
Python standard library only. Code licence: MIT, see code/LICENSE_MIT.txt.
"""

from argparse import ArgumentParser
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(rows):
    observed = [float(row["y_K"]) for row in rows]
    errors = [float(row["pred_K"]) - value for row, value in zip(rows, observed)]
    count = len(errors)
    mean_y = math.fsum(observed) / count
    sse = math.fsum(error * error for error in errors)
    sst = math.fsum((value - mean_y) ** 2 for value in observed)
    return {
        "MAE_K": math.fsum(abs(error) for error in errors) / count,
        "RMSE_K": math.sqrt(sse / count),
        "R2": 1.0 - sse / sst,
        "Bias_K": math.fsum(errors) / count,
    }


def close(actual, expected, label, tolerance=1e-9):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}; atol={tolerance}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, help="Optional JSON report; source data are never modified")
    args = parser.parse_args()
    lab = args.root.resolve() / "data" / "laboratory"
    paths = [lab / name for name in ["records_C.csv", "lab_predictions_v99.csv.gz", "lab_runs_v99.csv", "lab_calibration_v99.json"]]
    records, predictions, released_metrics = [read_csv(path) for path in paths[:3]]
    calibration = json.loads(paths[3].read_text(encoding="utf-8"))["calibration"]
    observations = {}
    for row in records:
        key = (int(row["trial"]), row["probe"], int(row["sample_index"]))
        if key in observations:
            raise AssertionError(f"Duplicate observation key: {key}")
        observations[key] = float(row["temperature_C"]) + 273.15
    if len(observations) != 6936:
        raise AssertionError(f"Expected the original 6936 records, found {len(observations)}")

    groups = defaultdict(list)
    unique_prediction_keys = set()
    for row in predictions:
        trial = int(row["trial"])
        sample = round(float(row["time_h"]) * 12)
        close(sample / 12, float(row["time_h"]), "Five-minute time grid")
        key = (trial, row["method"], int(row["seed"]))
        unique_key = (*key, row["probe"], sample)
        if unique_key in unique_prediction_keys:
            raise AssertionError(f"Duplicate prediction key: {unique_key}")
        unique_prediction_keys.add(unique_key)
        for column, index in [("a_K", sample - 96), ("b_K", sample - 48), ("y_K", sample)]:
            close(float(row[column]), observations[(trial, row["probe"], index)], f"Observation alignment {unique_key} {column}")
        if not math.isfinite(float(row["pred_K"])):
            raise AssertionError(f"Non-finite forecast: {unique_key}")
        groups[key].append(row)

    results = []
    max_metric_difference = 0.0
    expected_keys = set()
    for reference in released_metrics:
        key = (int(reference["trial"]), reference["method"], int(reference["seed"]))
        expected_keys.add(key)
        if key not in groups:
            raise AssertionError(f"No predictions for released metric: {key}")
        if len(groups[key]) != 588:
            raise AssertionError(f"Expected 49 target times x 12 probes for {key}")
        recomputed = metrics(groups[key])
        for metric, value in recomputed.items():
            expected = float(reference[metric])
            close(value, expected, f"Released metric {key} {metric}")
            max_metric_difference = max(max_metric_difference, abs(value - expected))
        results.append({"trial": key[0], "method": key[1], "seed": key[2], "rows": len(groups[key]), **recomputed})
    if expected_keys != set(groups):
        raise AssertionError("The released metrics and prediction-group inventories differ")

    reference_checks = 0
    reference_names = {"Persistence", "Linear extrapolation", "Damped trend", "Two-coefficient local", "Relaxation component (no network)", "Own-history mean ratio (R1)", "Own-history rowwise ratio (R2)", "Own-history blend (R3)"}
    for (trial, method, seed), rows in groups.items():
        if method not in reference_names:
            continue
        cal = calibration[str(trial)]
        d0 = []
        increments = []
        for row in rows:
            sample = round(float(row["time_h"]) * 12)
            previous = observations[(trial, row["probe"], sample - 144)]
            d0.append(float(row["a_K"]) - previous)
            increments.append(float(row["b_K"]) - float(row["a_K"]))
        q_own = math.fsum(abs(value) for value in increments) / max(math.fsum(abs(value) for value in d0), len(d0) * 1e-9)
        for row, previous_change, change in zip(rows, d0, increments):
            b = float(row["b_K"])
            ratio = change / previous_change if abs(previous_change) > 1e-9 else q_own
            clipped = min(max(ratio, 0.0), 1.0)
            if method == "Persistence":
                prediction = b
            elif method == "Linear extrapolation":
                prediction = b + change
            elif method == "Damped trend":
                prediction = b + float(cal["q_med"]) * change
            elif method == "Two-coefficient local":
                prediction = b + float(cal["q_med"]) * change + float(cal["beta"]) * (abs(change) > 0.1) * change
            elif method == "Relaxation component (no network)":
                prediction = b + float(cal["q_rel"]) * change
            elif method.endswith("(R1)"):
                prediction = b + q_own * change
            elif method.endswith("(R2)"):
                prediction = b + clipped * change
            else:
                prediction = b + 0.5 * (clipped + q_own) * change
            close(prediction, float(row["pred_K"]), f"Reference formula {trial} {method}")
            reference_checks += 1

    per_trial = defaultdict(list)
    for row in results:
        per_trial[(row["method"], row["trial"])].append(row["MAE_K"])
    per_method = defaultdict(list)
    for (method, trial), values in per_trial.items():
        per_method[method].append(math.fsum(values) / len(values))
    aggregates = {method: math.fsum(values) / len(values) for method, values in sorted(per_method.items())}
    report = {
        "status": "PASS",
        "purpose": "Tutorial reproduction of existing laboratory metrics and deterministic reference forecasts; no new training or research",
        "observation_count": len(observations),
        "prediction_rows": len(predictions),
        "trial_method_seed_groups": len(groups),
        "reference_formula_row_checks": reference_checks,
        "absolute_metric_tolerance": 1e-9,
        "maximum_absolute_metric_difference": max_metric_difference,
        "input_sha256": {path.relative_to(args.root.resolve()).as_posix(): sha256(path) for path in paths},
        "trial_equal_seed_mean_MAE_K": aggregates,
        "per_run_metrics": results,
        "scope": "The neural forecasts are read from the public archive, not retrained. CFD targets are not included and are not rescored.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"input_sha256", "per_run_metrics"}}, indent=2))


if __name__ == "__main__":
    main()
