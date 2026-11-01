"""Experimental Thermocouple Dual-Domain Validation Benchmark (Trials 1 & 2, Probes T17-T28)."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

BASE_REPRO = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
sys.path.insert(0, str(BASE_REPRO / "new_runs" / "80mm_s42_R_a"))
import legacy_core as lc

EXP_DIR = BASE_REPRO / "figure_sources" / "experimental_originals"
OUT_DIR = Path(r"E:\第三篇\outputs\ema_novelty_upgrade_v98\experimental_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBES = [f"T{i}" for i in range(17, 29)]
STEP = 48  # 4 hours = 48 * 5-min intervals in the post-ignition regime (t >= 6h, matching post-24h CFD protocol)
TR_INDICES = list(range(168, 217))  # Target time y in [14.0h .. 18.0h], inputs (a, b) in [6..10h, 10..14h] (all >= 6h!)
TE_INDICES = list(range(240, 289))  # Target time y in [20.0h .. 24.0h], inputs (a, b) in [12..16h, 16..20h]


def load_experimental_windows():
    obs = pd.read_csv(EXP_DIR / "observations.csv")
    sensors = pd.read_csv(EXP_DIR / "sensor_map.csv")
    sub_sensors = sensors[sensors["sensor_id"].isin(PROBES)].copy()
    coord_map = {
        row["sensor_id"]: np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=np.float32)
        for _, row in sub_sensors.iterrows()
    }
    data = {}
    for trial in [1, 2]:
        sub = obs[obs["trial"] == trial]
        piv = sub.pivot(index="sample_index", columns="probe", values="temperature_C")[PROBES]
        arr = piv.to_numpy(dtype=np.float64) + 273.15  # Kelvin
        data[trial] = {}
        for split_name, idx_list in [("train_window", TR_INDICES), ("test_window", TE_INDICES)]:
            xyz_l, a_l, b_l, y_l, t_l, p_l = [], [], [], [], [], []
            for k in idx_list:
                a_row = arr[k - 2 * STEP]
                b_row = arr[k - STEP]
                y_row = arr[k]
                for j, p in enumerate(PROBES):
                    xyz_l.append(coord_map[p])
                    a_l.append(a_row[j])
                    b_l.append(b_row[j])
                    y_l.append(y_row[j])
                    t_l.append(k * 300.0 / 3600.0)
                    p_l.append(p)
            data[trial][split_name] = dict(
                xyz=np.array(xyz_l, dtype=np.float32),
                a=np.array(a_l, dtype=np.float64),
                b=np.array(b_l, dtype=np.float64),
                y=np.array(y_l, dtype=np.float64),
                time_h=np.array(t_l, dtype=np.float64),
                probe=np.array(p_l),
            )
    return data


def fit_q_med(d_tr, v_tr):
    m = np.abs(d_tr) > 1e-9
    ratios = v_tr[m] / d_tr[m]
    mass = np.abs(d_tr[m])
    order = np.argsort(ratios, kind="stable")
    return float(ratios[order[np.searchsorted(np.cumsum(mass[order]), mass.sum() / 2.0, side="left")]])


def fit_beta_ridge(d_tr, v_tr, q_val, tau=0.1, lam=1e-4):
    r_tr = v_tr - q_val * d_tr
    gate = (np.abs(d_tr) > tau).astype(np.float64)
    x = gate * d_tr
    return float(np.dot(x, r_tr) / (np.dot(x, x) + lam))


def train_and_eval_arm(arm: str, tr: dict, te: dict, seed: int, epochs: int = 20, batch_size: int = 128):
    lc.seed_all(seed)
    torch.set_num_threads(2)

    cs = lc.Scaler().fit(tr["xyz"], ["exp_tr/xyz"])
    ts = lc.Scaler().fit(
        np.vstack([tr["a"].reshape(-1, 1), tr["b"].reshape(-1, 1), tr["y"].reshape(-1, 1)]).astype(np.float32),
        ["exp_tr/T"],
    )
    s_T = float(ts.span[0])
    d_tr = tr["b"] - tr["a"]
    v_tr = tr["y"] - tr["b"]
    s_v = float(np.sqrt(np.mean(d_tr ** 2)))
    s_d_tr = float(np.mean(np.abs(d_tr)))
    q_med = fit_q_med(d_tr, v_tr)
    q_int = float(np.mean(np.abs(v_tr)) / max(s_d_tr, 1e-6))
    # Use the unified enthalpy-conserving relaxation factor q_int
    q_tr = q_int
    r_tr = v_tr - q_tr * d_tr
    s_r_tr = float(np.sqrt(np.mean(r_tr ** 2)))

    coords_tr = torch.from_numpy(cs.transform(tr["xyz"]))
    seq_tr = torch.from_numpy(
        np.stack([ts.transform(tr["a"].reshape(-1, 1).astype(np.float32)), ts.transform(tr["b"].reshape(-1, 1).astype(np.float32))], axis=1)
    )

    if arm in ["A1", "A2"]:
        target_np = ts.transform(tr["y"].reshape(-1, 1).astype(np.float32))
        uloc_np = seq_tr[:, 1, 0:1].numpy()
    elif arm == "R":
        target_np = ((tr["y"] - tr["b"]) / s_T).reshape(-1, 1).astype(np.float32)
        uloc_np = np.zeros_like(target_np)
    elif arm == "S":
        target_np = ((tr["y"] - tr["b"]) / s_v).reshape(-1, 1).astype(np.float32)
        uloc_np = np.zeros_like(target_np)
    elif arm == "T":
        # Normalized thermal relaxation residual around q_tr * d_tr
        target_np = (r_tr / max(s_r_tr, 1e-6)).reshape(-1, 1).astype(np.float32)
        uloc_np = np.zeros_like(target_np)
    else:
        raise ValueError(arm)

    target_t = torch.from_numpy(target_np)
    uloc_t = torch.from_numpy(uloc_np)

    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(coords_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False, generator=gen, worker_init_fn=lc.seed_worker)
    model = lc.Model()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()
        _ = torch.empty((), dtype=torch.int64).random_(generator=gen).item()
        for idx in loader.batch_sampler:
            xb = coords_tr[idx]
            sb = seq_tr[idx]
            yb = target_t[idx]
            ulb = uloc_t[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb, sb)
            if arm == "A1":
                loss = torch.mean((pred - yb) ** 2) + 0.02 * torch.mean(torch.abs(pred - ulb))
            elif arm == "T":
                loss = torch.mean((pred - yb) ** 2) + 0.02 * torch.mean((pred - ulb) ** 2)
            else:
                loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()

    # Evaluate on held-out trial test window
    coords_te = torch.from_numpy(cs.transform(te["xyz"]))
    seq_te = torch.from_numpy(
        np.stack([ts.transform(te["a"].reshape(-1, 1).astype(np.float32)), ts.transform(te["b"].reshape(-1, 1).astype(np.float32))], axis=1)
    )
    model.eval()
    with torch.no_grad():
        raw_out = model(coords_te, seq_te).numpy().ravel().astype(np.float64)

    b_te = te["b"]
    d_te = te["b"] - te["a"]
    s_d_te = float(np.mean(np.abs(d_te)))

    if arm in ["A1", "A2"]:
        t_pred = ts.inverse(raw_out.reshape(-1, 1).astype(np.float32)).ravel().astype(np.float64)
    elif arm == "R":
        t_pred = b_te + s_T * raw_out
    elif arm == "S":
        t_pred = b_te + s_v * raw_out
    elif arm == "T":
        # Instance-adaptive local-activity-gated residual around physical relaxation q_tr * d_te
        gate_te = np.abs(d_te) / (np.abs(d_te) + s_d_te)
        sigma_te = s_r_tr * (s_d_te / max(s_d_tr, 1e-6)) * gate_te
        t_pred = b_te + q_tr * d_te + sigma_te * raw_out

    m = lc.metrics(te["y"], t_pred)
    return t_pred, m


def main():
    data = load_experimental_windows()
    summary_rows = []
    pred_records = []

    for held_trial in [1, 2]:
        train_trial = 2 if held_trial == 1 else 1
        tr = data[train_trial]["train_window"]
        te = data[held_trial]["test_window"]
        d_tr = tr["b"] - tr["a"]
        v_tr = tr["y"] - tr["b"]
        d_te = te["b"] - te["a"]

        q_med = fit_q_med(d_tr, v_tr)
        beta = fit_beta_ridge(d_tr, v_tr, q_med)

        baselines = {
            "Persistence": te["b"],
            "Linear extrapolation": te["b"] + d_te,
            "Damped trend": te["b"] + q_med * d_te,
            "Two-coefficient local": te["b"] + q_med * d_te + beta * (np.abs(d_te) > 0.1).astype(np.float64) * d_te,
        }
        for bname, bpred in baselines.items():
            m = lc.metrics(te["y"], bpred)
            summary_rows.append(
                dict(
                    trial=held_trial,
                    method=bname,
                    seed=0,
                    MAE_K=m["MAE_K"],
                    RMSE_K=m["RMSE_K"],
                    R2=m["R2"],
                    Bias_K=m["Bias_K"],
                )
            )
            for idx_row in range(len(bpred)):
                pred_records.append(
                    dict(
                        trial=held_trial,
                        method=bname,
                        seed=0,
                        probe=te["probe"][idx_row],
                        time_h=te["time_h"][idx_row],
                        a_K=te["a"][idx_row],
                        b_K=te["b"][idx_row],
                        y_K=te["y"][idx_row],
                        pred_K=bpred[idx_row],
                    )
                )

        for arm in ["A1", "A2", "R", "S", "T"]:
            mname = f"hLSTM {arm}" if arm != "T" else "IA-TRD-LSTM (T)"
            for seed in [42, 43, 44]:
                pred, m = train_and_eval_arm(arm, tr, te, seed=seed)
                summary_rows.append(
                    dict(
                        trial=held_trial,
                        method=mname,
                        seed=seed,
                        MAE_K=m["MAE_K"],
                        RMSE_K=m["RMSE_K"],
                        R2=m["R2"],
                        Bias_K=m["Bias_K"],
                    )
                )
                for idx_row in range(len(pred)):
                    pred_records.append(
                        dict(
                            trial=held_trial,
                            method=mname,
                            seed=seed,
                            probe=te["probe"][idx_row],
                            time_h=te["time_h"][idx_row],
                            a_K=te["a"][idx_row],
                            b_K=te["b"][idx_row],
                            y_K=te["y"][idx_row],
                            pred_K=pred[idx_row],
                        )
                    )

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "experimental_benchmark_runs.csv", index=False)
    pd.DataFrame(pred_records).to_csv(OUT_DIR / "experimental_predictions.csv.gz", index=False, compression="gzip")

    print("=== Experimental Thermocouple Benchmark (Trial 1 & Trial 2) ===")
    piv = df_sum.groupby(["method", "trial"])[["MAE_K", "RMSE_K", "R2", "Bias_K"]].mean().unstack("trial")
    for method in ["Persistence", "Linear extrapolation", "Damped trend", "Two-coefficient local", "hLSTM A1", "hLSTM A2", "hLSTM R", "hLSTM S", "IA-TRD-LSTM (T)"]:
        m1 = piv.loc[method, ("MAE_K", 1)]
        m2 = piv.loc[method, ("MAE_K", 2)]
        b1 = piv.loc[method, ("Bias_K", 1)]
        b2 = piv.loc[method, ("Bias_K", 2)]
        r2_mean = 0.5 * (piv.loc[method, ("R2", 1)] + piv.loc[method, ("R2", 2)])
        rmse_mean = 0.5 * (piv.loc[method, ("RMSE_K", 1)] + piv.loc[method, ("RMSE_K", 2)])
        print(f"{method:22s}: Mean MAE={0.5*(m1+m2):.4f} K, RMSE={rmse_mean:.4f} K, R2={r2_mean:.4f} | T1={m1:.4f}({b1:+.2f}), T2={m2:.4f}({b2:+.2f})")


if __name__ == "__main__":
    main()
