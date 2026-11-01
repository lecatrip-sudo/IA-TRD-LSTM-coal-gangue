"""Laboratory protocol sensitivity: deterministic references under alternative
lead/window choices that were explored during development (disclosed in Note S10).

Windows:
  P   primary, 4 h lead:   train targets 14-18 h (k=168..216), test targets 20-24 h (k=240..288)
  D   6 h lead, dense:     train targets 12-18 h (k=144..216), test targets 18-24 h (k=216..288)
  Q   6 h lead, quarter:   train target 18 h (k=216),         test target 24 h (k=288)

Methods (all deterministic, coefficients fitted on the training window only):
  Persistence, Linear extrapolation, Damped trend (weighted-median q),
  Relaxation (q_int = ratio of mean absolute increments).
The own-history reference needs a third instant at k - 3*lead; the 6 h-lead
dense variant would reach before 6 h for most rows, so it is reported only
where every input lies at or after 6 h (primary and quarter variants).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(r"E:\第三篇\outputs\ema_v99_revision")
EXP_DIR = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility\figure_sources\experimental_originals")
OUT = HERE / "analyses_v99"
OUT.mkdir(parents=True, exist_ok=True)
PROBES = [f"T{i}" for i in range(17, 29)]


def load():
    obs = pd.read_csv(EXP_DIR / "observations.csv")
    data = {}
    for trial in [1, 2]:
        sub = obs[obs["trial"] == trial]
        piv = sub.pivot(index="sample_index", columns="probe", values="temperature_C")[PROBES]
        data[trial] = piv.to_numpy(dtype=np.float64) + 273.15
    return data


def windows(arr, k_list, lead):
    rows = []
    for k in k_list:
        a = arr[k - 2 * lead]
        b = arr[k - lead]
        y = arr[k]
        for j in range(len(PROBES)):
            a0 = arr[k - 3 * lead][j] if k - 3 * lead >= 0 else np.nan
            rows.append((k, PROBES[j], a0, a[j], b[j], y[j]))
    return rows


def q_med_fit(d, v):
    m = np.abs(d) > 1e-9
    ratios = v[m] / d[m]
    mass = np.abs(d[m])
    order = np.argsort(ratios, kind="stable")
    return float(ratios[order[np.searchsorted(np.cumsum(mass[order]), mass.sum() / 2, side="left")]])


def main():
    data = load()
    variants = {
        "P (4 h lead, primary)": (list(range(168, 217)), list(range(240, 289)), 48),
        "D (6 h lead, dense)": (list(range(144, 217)), list(range(216, 289)), 72),
        "Q (6 h lead, quarter)": ([216], [288], 72),
    }
    rows = []
    for vname, (tr_k, te_k, lead) in variants.items():
        for held in [1, 2]:
            trn = 2 if held == 1 else 1
            tr = windows(data[trn], tr_k, lead)
            te = windows(data[held], te_k, lead)
            d_tr = np.array([r[4] - r[3] for r in tr])
            v_tr = np.array([r[5] - r[4] for r in tr])
            d_te = np.array([r[4] - r[3] for r in te])
            b_te = np.array([r[4] for r in te])
            y_te = np.array([r[5] for r in te])
            q = q_med_fit(d_tr, v_tr)
            q_int = float(np.mean(np.abs(v_tr)) / max(np.mean(np.abs(d_tr)), 1e-9))
            preds = {
                "Persistence": b_te,
                "Linear extrapolation": b_te + d_te,
                "Damped trend": b_te + q * d_te,
                "Relaxation (q_int)": b_te + q_int * d_te,
            }
            have_third = all(r[2] == r[2] for r in te) and (te[0][0] - 3 * lead) >= 72
            if have_third:
                d0_te = np.array([r[3] - r[2] for r in te])
                q_star = float(np.mean(np.abs(d_te)) / max(np.mean(np.abs(d0_te)), 1e-9))
                ratio = np.where(np.abs(d0_te) > 1e-9, d_te / np.where(np.abs(d0_te) > 1e-9, d0_te, 1.0), q_star)
                blend = 0.5 * np.clip(ratio, 0.0, 1.0) + 0.5 * q_star
                preds["Own-history blend (R3)"] = b_te + blend * d_te
            for m, p in preds.items():
                mae = float(np.mean(np.abs(p - y_te)))
                rows.append(dict(variant=vname, held_trial=held, method=m, n=len(y_te),
                                 q_med=q, q_int=q_int, MAE_K=mae))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lab_protocol_sensitivity.csv", index=False)
    piv = df.pivot_table(index=["variant", "method"], columns="held_trial", values="MAE_K")
    piv["mean"] = piv.mean(axis=1)
    print("=== Laboratory protocol sensitivity (deterministic references) ===")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
