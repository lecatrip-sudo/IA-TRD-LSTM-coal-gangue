"""Verify the completed selection against the fully specified deterministic tie rule."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(r"E:\第三篇\outputs\ema_v99_revision")
CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
Q_ORDER = ["int", "med", "rms"]
LAM_ORDER = [0.005, 0.02, 0.05]
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
RULE = ("min mean inner-fold MAE; ties within 0.002 K broken by larger lambda, "
        "then q order int>med>rms, then alpha closest to 0.5, then lower alpha")


def main():
    scores = pd.read_csv(HERE / "ns_inner_scores.csv")
    info_path = HERE / "ns_selection.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for case in CASES:
        sub = scores[scores.outer == case]
        grid = []
        for q in Q_ORDER:
            for lam in LAM_ORDER:
                subset = sub[(sub.q == q) & (sub.lam == lam)]
                for alpha in ALPHAS:
                    grid.append({
                        "q": q, "lam": lam, "alpha": alpha,
                        "score": float(subset[f"mae_a{alpha:.1f}"].mean()),
                    })
        best = min(row["score"] for row in grid)
        candidates = [row for row in grid if row["score"] <= best + 0.002]
        candidates.sort(key=lambda row: (-row["lam"], Q_ORDER.index(row["q"]),
                                         abs(row["alpha"] - 0.5), row["alpha"]))
        selected = candidates[0]
        actual = info["selection"][case]
        assert actual["q"] == selected["q"], (case, actual, selected)
        assert abs(float(actual["lam"]) - selected["lam"]) < 1e-12, (case, actual, selected)
        assert abs(float(actual["alpha"]) - selected["alpha"]) < 1e-12, (case, actual, selected)
        assert abs(float(actual["score"]) - selected["score"]) < 1e-10, (case, actual, selected)
    info["rule"] = RULE
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("selection verified against explicit lower-alpha final tie-break")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
