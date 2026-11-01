from pathlib import Path
import numpy as np
import pandas as pd

base = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
cases = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
arrays = {c: dict(np.load(base / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in cases}

for name_q, use_int in [("q_med (V94 frozen q)", False), ("q_int (training integral q)", True), ("q_rms (training RMS ratio)", "rms")]:
    print(f"=== Using {name_q} ===")
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        case_maes = []
        case_biases = []
        for c in cases:
            train = [t for t in cases if t != c]
            b_te = arrays[c]["T72_K"].ravel()
            d_te = b_te - arrays[c]["T48_K"].ravel()
            y_te = arrays[c]["T96_K"].ravel()
            
            if use_int is False:
                ratios, mass = [], []
                for t in train:
                    dt = arrays[t]["T48_K"].ravel() - arrays[t]["T24_K"].ravel()
                    vt = arrays[t]["T72_K"].ravel() - arrays[t]["T48_K"].ravel()
                    m = dt != 0
                    ratios.append(vt[m] / dt[m])
                    mass.append(np.abs(dt[m]) / len(dt))
                ratios, mass = np.concatenate(ratios), np.concatenate(mass)
                order = np.argsort(ratios, kind="stable")
                q_val = float(ratios[order[np.searchsorted(np.cumsum(mass[order]), mass.sum()/2, side="left")]])
            elif use_int is True:
                q_val = float(np.mean([np.mean(np.abs(arrays[t]["T72_K"] - arrays[t]["T48_K"])) for t in train]) /
                              np.mean([np.mean(np.abs(arrays[t]["T48_K"] - arrays[t]["T24_K"])) for t in train]))
            else:
                q_val = float(np.sqrt(np.mean([np.mean((arrays[t]["T72_K"] - arrays[t]["T48_K"])**2) for t in train])) /
                              np.sqrt(np.mean([np.mean((arrays[t]["T48_K"] - arrays[t]["T24_K"])**2) for t in train])))
            
            seed_maes, seed_biases = [], []
            for seed in [42, 43, 44]:
                df = pd.read_csv(base / "new_runs" / f"{c}_s{seed}_S_a" / "predictions.csv.gz")
                v_pred = df.Tpred_K.to_numpy() - b_te
                # Normalize spatial neural increment to match instance enthalpy relaxation scale q_val * mean(|d_te|)
                v_cal = v_pred * ((q_val * np.mean(np.abs(d_te))) / np.mean(np.abs(v_pred)))
                v_hat = (1.0 - alpha) * (q_val * d_te) + alpha * v_cal
                seed_maes.append(np.mean(np.abs(b_te + v_hat - y_te)))
                seed_biases.append(np.mean(b_te + v_hat - y_te))
            case_maes.append(np.mean(seed_maes))
            case_biases.append(np.mean(seed_biases))
        print(f"  alpha={alpha:.1f}: 6-case MAE = {np.mean(case_maes):.4f} K | " + " ".join(f"{c}:{m:.4f}({b:+.2f})" for c, m, b in zip(cases, case_maes, case_biases)))
