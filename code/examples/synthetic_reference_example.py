"""Tiny artificial demonstration of the manuscript's existing references.

All temperatures are manually specified tutorial inputs. They are not measured
records, CFD output, an additional validation case, or evidence for the paper.
No model fitting occurs. Code licence: MIT, see code/LICENSE_MIT.txt.
"""

from argparse import ArgumentParser
import json
import math
from pathlib import Path


def forecast(a0, a, b):
    if not (len(a0) == len(a) == len(b)) or not b:
        raise ValueError("Provide three equally sized nonempty histories")
    d0 = [current - previous for current, previous in zip(a, a0)]
    d = [current - previous for current, previous in zip(b, a)]
    q = math.fsum(abs(value) for value in d) / max(math.fsum(abs(value) for value in d0), len(d0) * 1e-9)
    ratios = [min(max(change / old if abs(old) > 1e-9 else q, 0.0), 1.0) for change, old in zip(d, d0)]
    return {"q_star": q, "Persistence": list(b), "Linear extrapolation": [last + change for last, change in zip(b, d)],
            "R1": [last + q * change for last, change in zip(b, d)],
            "R2": [last + ratio * change for last, ratio, change in zip(b, ratios, d)],
            "R3": [last + 0.5 * (q + ratio) * change for last, ratio, change in zip(b, ratios, d)]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON tutorial output")
    args = parser.parse_args()
    inputs = {"a0_K": [300.0, 310.0, 290.0, 300.0], "a_K": [304.0, 312.0, 290.0, 302.0], "b_K": [306.0, 315.0, 290.0, 301.0]}
    result = forecast(inputs["a0_K"], inputs["a_K"], inputs["b_K"])
    expected = {"Persistence": [306.0, 315.0, 290.0, 301.0], "Linear extrapolation": [308.0, 318.0, 290.0, 300.0], "R3": [307.25, 317.625, 290.0, 300.625]}
    for method, values in expected.items():
        if result[method] != values:
            raise AssertionError(f"Hand-calculated tutorial mismatch for {method}")
    shifted = forecast(*[[value + 10.0 for value in inputs[key]] for key in ("a0_K", "a_K", "b_K")])
    for method in expected:
        if shifted[method] != [value + 10.0 for value in result[method]]:
            raise AssertionError(f"Temperature-offset covariance failed for {method}")
    report = {"status": "PASS", "synthetic_tutorial_only": True, "not_manuscript_data": True,
              "description": "Four manual locations at three equally spaced instants; existing persistence, linear and own-history formulas only",
              "formulas": "Main manuscript Eqs. (1), (3), (4); zero-increment fallback follows retained own_history_refs implementation",
              "inputs": inputs, "forecasts": result, "hand_calculated_checks": "PASS", "constant_offset_check": "PASS"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
