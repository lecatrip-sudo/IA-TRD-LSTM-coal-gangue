"""V99 laboratory cross-trial benchmark.

Primary arm T uses the CFD decomposition (main Eq. 9) unchanged: training target
u = v / (q_rel * d_bar_train), regularizer (g - u_loc)^2 with u_loc = d / d_bar_train
and lambda_SI = 0.02, forecast b + q_rel[(1-alpha) d + alpha * d_bar_star * g/<|g|>_star]
with alpha = 0.5 inherited from the first-execution CFD configuration.

The V98 residual/gated form (main Eq. 10 of V98) is reproduced unchanged and
reported as a post-hoc variant. Arms A1, A2, R and S reproduce the V98 protocol
bit-for-bit. New own-history decay references R1-R3 use the held trial's own
third instant (k-144, at or after 8 h) and no training data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(r"E:\第三篇\outputs\ema_v99_revision")
V98 = Path(r"E:\第三篇\outputs\ema_novelty_upgrade_v98")
sys.path.insert(0, str(V98))
import run_experimental_benchmark as eb  # noqa: E402
import legacy_core as lc  # noqa: E402

OUT = HERE / "lab_v99"
OUT.mkdir(parents=True, exist_ok=True)
PROBES = eb.PROBES
ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def load_windows_with_third_instant():
    """Primary windows plus the third instant k-144 (>= 8 h) for own-history references."""
    obs = pd.read_csv(eb.EXP_DIR / "observations.csv")
    sensors = pd.read_csv(eb.EXP_DIR / "sensor_map.csv")
    sub = sensors[sensors["sensor_id"].isin(PROBES)]
    coord_map = {r["sensor_id"]: np.array([r["x_m"], r["y_m"], r["z_m"]], dtype=np.float32)
                 for _, r in sub.iterrows()}
    data = {}
    for trial in [1, 2]:
        t = obs[obs["trial"] == trial]
        piv = t.pivot(index="sample_index", columns="probe", values="temperature_C")[PROBES]
        arr = piv.to_numpy(dtype=np.float64) + 273.15
        data[trial] = {"arr": arr, "coord_map": coord_map}
    return data


def rows_from(arr, coord_map, k_list, step=48):
    xyz_l, a0_l, a_l, b_l, y_l, t_l, p_l = [], [], [], [], [], [], []
    for k in k_list:
        a0_row = arr[k - 3 * step]
        a_row = arr[k - 2 * step]
        b_row = arr[k - step]
        y_row = arr[k]
        for j, p in enumerate(PROBES):
            xyz_l.append(coord_map[p])
            a0_l.append(a0_row[j])
            a_l.append(a_row[j])
            b_l.append(b_row[j])
            y_l.append(y_row[j])
            t_l.append(k * 300.0 / 3600.0)
            p_l.append(p)
    return dict(xyz=np.array(xyz_l, dtype=np.float32), a0=np.array(a0_l, dtype=np.float64),
                a=np.array(a_l, dtype=np.float64), b=np.array(b_l, dtype=np.float64),
                y=np.array(y_l, dtype=np.float64), time_h=np.array(t_l), probe=np.array(p_l))


def own_history_refs(te):
    d0 = te["a"] - te["a0"]
    d = te["b"] - te["a"]
    q_star = float(np.mean(np.abs(d)) / max(np.mean(np.abs(d0)), 1e-9))
    ratio = np.where(np.abs(d0) > 1e-9, d / np.where(np.abs(d0) > 1e-9, d0, 1.0), q_star)
    clip = np.clip(ratio, 0.0, 1.0)
    blend = 0.5 * clip + 0.5 * q_star
    return {
        "Own-history mean ratio (R1)": te["b"] + q_star * d,
        "Own-history rowwise ratio (R2)": te["b"] + clip * d,
        "Own-history blend (R3)": te["b"] + blend * d,
    }, q_star


def train_eq9(tr, te, seed, epochs=20, batch_size=128, lam_si=0.02, alpha=0.5):
    lc.seed_all(seed)
    torch.set_num_threads(2)
    cs = lc.Scaler().fit(tr["xyz"], ["exp_tr/xyz"])
    ts = lc.Scaler().fit(np.vstack([tr["a"].reshape(-1, 1), tr["b"].reshape(-1, 1),
                                    tr["y"].reshape(-1, 1)]).astype(np.float32), ["exp_tr/T"])
    d_tr = tr["b"] - tr["a"]
    v_tr = tr["y"] - tr["b"]
    s_d_tr = max(float(np.mean(np.abs(d_tr))), 1e-6)
    q_int = float(np.mean(np.abs(v_tr)) / s_d_tr)
    u_tr = (v_tr / (q_int * s_d_tr)).reshape(-1, 1).astype(np.float32)
    u_loc_tr = (d_tr / s_d_tr).reshape(-1, 1).astype(np.float32)

    coords_tr = torch.from_numpy(cs.transform(tr["xyz"]))
    seq_tr = torch.from_numpy(np.stack([ts.transform(tr["a"].reshape(-1, 1).astype(np.float32)),
                                        ts.transform(tr["b"].reshape(-1, 1).astype(np.float32))], axis=1))
    target_t = torch.from_numpy(u_tr)
    uloc_t = torch.from_numpy(u_loc_tr)

    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(coords_tr), batch_size=batch_size,
                                         shuffle=True, num_workers=0, drop_last=False, generator=gen,
                                         worker_init_fn=lc.seed_worker)
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
            loss = torch.mean((pred - yb) ** 2) + lam_si * torch.mean((pred - ulb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()

    coords_te = torch.from_numpy(cs.transform(te["xyz"]))
    seq_te = torch.from_numpy(np.stack([ts.transform(te["a"].reshape(-1, 1).astype(np.float32)),
                                        ts.transform(te["b"].reshape(-1, 1).astype(np.float32))], axis=1))
    model.eval()
    with torch.no_grad():
        g = model(coords_te, seq_te).numpy().ravel().astype(np.float64)
    g_mean = max(float(np.mean(np.abs(g))), 1e-6)
    b_te = te["b"]
    d_te = te["b"] - te["a"]
    y_te = te["y"]
    s_d_te = float(np.mean(np.abs(d_te)))
    preds = {}
    mae = {}
    for a in ALPHAS:
        v_hat = (1.0 - a) * (q_int * d_te) + a * (q_int * s_d_te * g / g_mean)
        preds[a] = b_te + v_hat
        mae[a] = float(np.mean(np.abs(preds[a] - y_te)))
    t_pred = preds[alpha].astype(np.float64)
    m = lc.metrics(y_te, t_pred)
    return t_pred, m, mae, dict(q_rel=q_int, s_d_tr=s_d_tr, s_d_te=s_d_te, g_mean=g_mean)


def main():
    data = eb.load_experimental_windows()
    raw = load_windows_with_third_instant()
    summary, records, calib, alpha_grid = [], [], {}, {}
    for held in [1, 2]:
        train_trial = 2 if held == 1 else 1
        tr = data[train_trial]["train_window"]
        te = data[held]["test_window"]
        te3 = rows_from(raw[held]["arr"], raw[held]["coord_map"], eb.TE_INDICES)
        d_tr = tr["b"] - tr["a"]
        v_tr = tr["y"] - tr["b"]
        d_te = te["b"] - te["a"]
        q_med = eb.fit_q_med(d_tr, v_tr)
        beta = eb.fit_beta_ridge(d_tr, v_tr, q_med)

        q_rel_tr = float(np.mean(np.abs(v_tr)) / max(float(np.mean(np.abs(d_tr))), 1e-6))
        baselines = {
            "Persistence": te["b"],
            "Linear extrapolation": te["b"] + d_te,
            "Damped trend": te["b"] + q_med * d_te,
            "Two-coefficient local": te["b"] + q_med * d_te + beta * (np.abs(d_te) > 0.1).astype(np.float64) * d_te,
            "Relaxation component (no network)": te["b"] + q_rel_tr * d_te,
        }
        own, q_star = own_history_refs(te3)
        baselines.update(own)
        for name, pred in baselines.items():
            m = lc.metrics(te["y"], pred)
            summary.append(dict(trial=held, method=name, seed=0, MAE_K=m["MAE_K"], RMSE_K=m["RMSE_K"],
                                R2=m["R2"], Bias_K=m["Bias_K"]))
            for i in range(len(pred)):
                records.append(dict(trial=held, method=name, seed=0, probe=te["probe"][i],
                                     time_h=te["time_h"][i], a_K=te["a"][i], b_K=te["b"][i],
                                     y_K=te["y"][i], pred_K=pred[i]))

        for arm in ["A1", "A2", "R", "S", "T"]:
            mname = f"hLSTM {arm}" if arm != "T" else "IA-TRD-LSTM Eq. (10) form (V98)"
            for seed in [42, 43, 44]:
                pred, m = eb.train_and_eval_arm(arm, tr, te, seed=seed)
                summary.append(dict(trial=held, method=mname, seed=seed, MAE_K=m["MAE_K"], RMSE_K=m["RMSE_K"],
                                     R2=m["R2"], Bias_K=m["Bias_K"]))
                for i in range(len(pred)):
                    records.append(dict(trial=held, method=mname, seed=seed, probe=te["probe"][i],
                                        time_h=te["time_h"][i], a_K=te["a"][i], b_K=te["b"][i],
                                        y_K=te["y"][i], pred_K=pred[i]))

        for seed in [42, 43, 44]:
            pred, m, grid, cal = train_eq9(tr, te, seed=seed)
            summary.append(dict(trial=held, method="IA-TRD-LSTM (T)", seed=seed, MAE_K=m["MAE_K"],
                                RMSE_K=m["RMSE_K"], R2=m["R2"], Bias_K=m["Bias_K"]))
            for i in range(len(pred)):
                records.append(dict(trial=held, method="IA-TRD-LSTM (T)", seed=seed, probe=te["probe"][i],
                                    time_h=te["time_h"][i], a_K=te["a"][i], b_K=te["b"][i],
                                    y_K=te["y"][i], pred_K=pred[i]))
            alpha_grid[f"{held}/{seed}"] = grid
        calib[held] = dict(q_med=q_med, beta=beta, q_star_own=q_star, **{
            k: v for k, v in cal.items()})

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "lab_runs_v99.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "lab_predictions_v99.csv.gz", index=False, compression="gzip")
    (OUT / "lab_calibration_v99.json").write_text(
        json.dumps({"calibration": calib, "eq9_alpha_grid_mae": alpha_grid,
                    "note": "alpha=0.5 is inherited and fixed before the unified laboratory execution, not preregistered; the grid is post-hoc sensitivity"},
                   indent=1), encoding="utf-8")
    piv = df.groupby(["method", "trial"]).MAE_K.mean().unstack("trial")
    print("=== V99 laboratory benchmark (trial-equal MAE) ===")
    for mname in piv.index:
        r = df[df.method == mname]
        mean_mae = 0.5 * (piv.loc[mname, 1] + piv.loc[mname, 2])
        print(f"{mname:38s} mean={mean_mae:.4f} T1={piv.loc[mname, 1]:.4f} T2={piv.loc[mname, 2]:.4f} n={len(r)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
