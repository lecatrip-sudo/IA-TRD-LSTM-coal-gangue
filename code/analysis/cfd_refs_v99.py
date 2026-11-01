"""V99 deterministic CFD reference set: named references plus the own-history decay family.

Own-history decay references (no network, no training data, held-case inputs only):
  R1  case-mean ratio  q* = <|d|>/<|d0|>,            forecast b + q* d
  R2  rowwise ratio    clip(d/d0, 0, 1) per row,     forecast b + clip * d
  R3  blend            0.5*clip + 0.5*q* per row,     forecast b + blend * d
with d0 = T48 - T24, d = T72 - T48, b = T72, target T96. R2/R3 fall back to q*
wherever |d0| <= 1e-9. These use the held case's own three instants (24, 48, 72 h),
one more field than the two-input matched arms; the archived aLSTM also used
three input instants. They were constructed after the V98 arm-T results were
known; the full family is reported, not a selected member.

Everything is saved per case as arrays for downstream threshold, bootstrap and
table generation.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
OUT = Path(r"E:\第三篇\outputs\ema_v99_revision\cfd_refs")
OUT.mkdir(parents=True, exist_ok=True)
CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
CASE_LABELS = {"80mm": "80 mm", "150mm": "150 mm", "300mm": "300 mm",
               "0.25kxl": "Porosity 0.25", "0.5kxl": "Porosity 0.50", "0.75kxl": "Porosity 0.75"}


def metric(y, p):
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return dict(MAE_K=float(np.mean(np.abs(e))), RMSE_K=float(np.sqrt(np.mean(e ** 2))),
                Bias_K=float(e.mean()), P95AE_K=float(np.quantile(np.abs(e), 0.95, method="linear")),
                MaxAE_K=float(np.max(np.abs(e))), R2=float(1.0 - np.sum(e ** 2) / np.sum((y - np.mean(y)) ** 2)))


def main():
    arrays = {c: dict(np.load(BASE / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in CASES}
    rows = []
    with zipfile.ZipFile(BASE / "archived_aLSTM/archives/Supplementary_Data_1_v79_metadata_corrected.zip") as z_arch:
        for c in CASES:
            a = arrays[c]
            t24 = a["T24_K"].ravel().astype(np.float64)
            t48 = a["T48_K"].ravel().astype(np.float64)
            t72 = a["T72_K"].ravel().astype(np.float64)
            y = a["T96_K"].ravel().astype(np.float64)
            b = t72
            d = b - t48
            d0 = t48 - t24

            fit = json.loads((BASE / "local_control/frozen_parameters" / f"{c}_fold.json").read_text(encoding="utf-8-sig"))["methods"]
            q_fit = float(fit["SCALAR_MAE_U"]["final_fit"]["weights"][0])
            beta = float(fit["GATE_SLOPE_RESID"]["final_fit"]["beta"][0])
            f_arch = pd.read_csv(io.BytesIO(z_arch.read(f"data/{c}.csv")), float_precision="round_trip")
            p_arch = f_arch["T96_LSTM_K"].to_numpy(dtype=np.float64)

            q_star = float(np.mean(np.abs(d)) / max(np.mean(np.abs(d0)), 1e-9))
            ratio = np.where(np.abs(d0) > 1e-9, d / np.where(np.abs(d0) > 1e-9, d0, 1.0), q_star)
            clip = np.clip(ratio, 0.0, 1.0)
            blend = 0.5 * clip + 0.5 * q_star

            preds = {
                "Persistence": b,
                "Linear extrapolation": b + d,
                "Damped trend": b + q_fit * d,
                "Two-coefficient local": b + q_fit * d + beta * (np.abs(d) > 0.1).astype(np.float64) * d,
                "aLSTM": p_arch,
                "Own-history mean ratio (R1)": b + q_star * d,
                "Own-history rowwise ratio (R2)": b + clip * d,
                "Own-history blend (R3)": b + blend * d,
            }
            np.savez_compressed(
                OUT / f"{c}.npz",
                y=y, b=b, d=d, d0=d0, t24=t24, t48=t48, t72=t72,
                x=a["XYZ"][:, 0].astype(np.float64), z=a["XYZ"][:, 2].astype(np.float64),
                row_id=a["row_id"].ravel(),
                **{f"pred_{k.replace(' ', '_').replace('(', '').replace(')', '')}": v for k, v in preds.items()},
                q_fit=q_fit, beta=beta, q_star=q_star,
            )
            for name, pred in preds.items():
                m = metric(y, pred)
                rows.append(dict(case=c, case_label=CASE_LABELS[c], method=name, seed=0, **m))
            rows.append(dict(case=c, case_label=CASE_LABELS[c], method="q_star (own)", seed=0,
                             MAE_K=q_star, RMSE_K=np.nan, Bias_K=np.nan, P95AE_K=np.nan,
                             MaxAE_K=np.nan, R2=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cfd_reference_metrics.csv", index=False)
    piv = df[df.method.str.startswith("Own") | df.method.isin(["Damped trend", "Linear extrapolation"])].pivot_table(
        index="method", columns="case_label", values="MAE_K")
    order = [CASE_LABELS[c] for c in CASES]
    piv = piv[order]
    piv["case-equal"] = piv.mean(axis=1)
    print(piv.round(4).to_string())
    print("\nq* per case:")
    print(df[df.method == "q_star (own)"][["case_label", "MAE_K"]].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
