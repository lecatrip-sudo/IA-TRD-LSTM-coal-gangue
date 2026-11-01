"""V99 generalized outer-fold arm-T worker.

Modes:
  T   - two-input arm T (inputs 48, 72 h -> forecast 96 h; training [24, 48] -> 72 h),
        identical computation path to the V98 worker when q='int' and lam_si=0.02;
  T3  - three-input variant (inputs 24, 48, 72 h -> forecast 96 h on the held case,
        training [24, 48, 72] -> 96 h), constructed to share the inputs of the
        own-history decay references;
  TMP - two-input arm T with the LSTM replaced by an MLP backbone (ablation).

q variants: 'int' (ratio of mean absolute increments), 'med' (weighted median of
rowwise ratios), 'rms' (ratio of RMS increments), all fitted on training cases only.
alpha and lam_si are explicit arguments; the forecast follows main Eq. (9).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

BASE_REPRO = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
sys.path.insert(0, str(BASE_REPRO / "new_runs" / "80mm_s42_R_a"))
import legacy_core as lc  # noqa: E402

CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
OUT_ROOT = BASE_REPRO / "ia_trd_runs_v99"


class MLPModel(nn.Module):
    """Arm-T backbone ablation: the recurrent cell is replaced by a feed-forward net."""

    def __init__(self, seq_len=2):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.temporal = nn.Sequential(
            nn.Linear(seq_len * 129, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, coords, seq):
        spatial = self.spatial_encoder(coords)
        spatial_steps = spatial.unsqueeze(1).expand(-1, seq.shape[1], -1)
        features = torch.cat([spatial_steps, seq], dim=-1)
        temporal = self.temporal(features.reshape(seq.shape[0], -1))
        return self.decoder(torch.cat([spatial, temporal], dim=-1))


def support_maes(xyz, err):
    abs_err = np.abs(err)
    row_mae = float(np.mean(abs_err))
    _, inv_xyz = np.unique(np.round(xyz, 6), axis=0, return_inverse=True)
    sums_xyz = np.bincount(inv_xyz, weights=abs_err)
    cnts_xyz = np.bincount(inv_xyz)
    exact_xyz_mae = float(np.mean(sums_xyz / cnts_xyz))
    xz_bins = np.floor(xyz[:, [0, 2]] / 0.1 + 1e-9).astype(np.int64)
    _, inv_bin = np.unique(xz_bins, axis=0, return_inverse=True)
    sums_bin = np.bincount(inv_bin, weights=abs_err)
    cnts_bin = np.bincount(inv_bin)
    bin_xz_mae = float(np.mean(sums_bin / cnts_bin))
    return row_mae, exact_xyz_mae, bin_xz_mae


def fold_q(q_variant, arrays, train_cases):
    d = {c: (arrays[c]["T48_K"].ravel() - arrays[c]["T24_K"].ravel()).astype(np.float64) for c in train_cases}
    v = {c: (arrays[c]["T72_K"].ravel() - arrays[c]["T48_K"].ravel()).astype(np.float64) for c in train_cases}
    if q_variant == "int":
        return float(np.mean([np.mean(np.abs(v[c])) for c in train_cases]) /
                     np.mean([np.mean(np.abs(d[c])) for c in train_cases]))
    if q_variant == "rms":
        return float(np.sqrt(np.mean([np.mean(v[c] ** 2) for c in train_cases]) /
                             np.mean([np.mean(d[c] ** 2) for c in train_cases])))
    if q_variant == "med":
        ratios, mass = [], []
        for c in train_cases:
            m = d[c] != 0
            ratios.append(v[c][m] / d[c][m])
            mass.append(np.abs(d[c][m]) / len(d[c]))
        ratios = np.concatenate(ratios)
        mass = np.concatenate(mass)
        order = np.argsort(ratios, kind="stable")
        return float(ratios[order[np.searchsorted(np.cumsum(mass[order]), mass.sum() / 2, side="left")]])
    raise ValueError(q_variant)


def run_single(held_case, seed, mode="T", q_variant="int", lam_si=0.02, alpha=0.5,
               max_epochs=20, out_root=None):
    t_start = time.monotonic()
    assert mode in ("T", "T3", "TMP")
    if out_root is None:
        out_root = OUT_ROOT
    tag = mode if (q_variant == "int" and lam_si == 0.02) else f"{mode}_{q_variant}_l{lam_si}"
    run_id = f"{held_case}_s{seed}_{tag}_a" if max_epochs <= 20 else f"{held_case}_s{seed}_{tag}_e{max_epochs}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    arrays = {c: dict(np.load(BASE_REPRO / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in CASES}
    train_cases = [c for c in CASES if c != held_case]
    fit_ids = [f"{c}/{h}h" for c in train_cases for h in [24, 48, 72]]

    cs = lc.Scaler().fit(
        np.vstack([arrays[c]["XYZ"].astype(np.float32) for c in train_cases]),
        [f"{c}/24h" for c in train_cases],
    )
    ts = lc.Scaler().fit(
        np.vstack([arrays[c][f"T{h}_K"].reshape(-1, 1).astype(np.float32) for c in train_cases for h in [24, 48, 72]]),
        fit_ids,
    )

    q_val = fold_q(q_variant, arrays, train_cases)
    s_d_map = {}
    coords_list, seq_list, target_list, u_loc_list = [], [], [], []
    for c in train_cases:
        xyz = arrays[c]["XYZ"].astype(np.float32)
        t24 = arrays[c]["T24_K"].reshape(-1, 1).astype(np.float32)
        t48 = arrays[c]["T48_K"].reshape(-1, 1).astype(np.float32)
        t72 = arrays[c]["T72_K"].reshape(-1, 1).astype(np.float32)
        if mode == "T3":
            # Training transition [24, 48, 72] -> 96 h; d is the last within-window increment.
            d_c = (t72.ravel() - t48.ravel()).astype(np.float64)
            v_c = (arrays[c]["T96_K"].ravel() - t72.ravel()).astype(np.float64)
            s_d_c = max(float(np.mean(np.abs(d_c))), 1e-6)
            s_d_map[c] = s_d_c
            u_c = (v_c.reshape(-1, 1) / (np.float32(q_val) * np.float32(s_d_c))).astype(np.float32)
            u_loc_c = (d_c.reshape(-1, 1) / np.float32(s_d_c)).astype(np.float32)
            coords_list.append(cs.transform(xyz))
            seq_list.append(np.stack([ts.transform(t24), ts.transform(t48), ts.transform(t72)], axis=1))
        else:
            d_c = (t48.ravel() - t24.ravel()).astype(np.float64)
            s_d_c = max(float(np.mean(np.abs(d_c))), 1e-6)
            s_d_map[c] = s_d_c
            u_c = (t72 - t48) / (np.float32(q_val) * np.float32(s_d_c))
            u_loc_c = (t48 - t24) / np.float32(s_d_c)
            coords_list.append(cs.transform(xyz))
            seq_list.append(np.stack([ts.transform(t24), ts.transform(t48)], axis=1))
        target_list.append(u_c.reshape(-1, 1).astype(np.float32))
        u_loc_list.append(u_loc_c.reshape(-1, 1).astype(np.float32))

    coords_t = torch.from_numpy(np.vstack(coords_list))
    seq_t = torch.from_numpy(np.vstack(seq_list))
    target_t = torch.from_numpy(np.vstack(target_list))
    u_loc_t = torch.from_numpy(np.vstack(u_loc_list))

    xyz_raw = arrays[held_case]["XYZ"].astype(np.float32)
    xyz_te_t = torch.from_numpy(cs.transform(xyz_raw))
    t48_te = arrays[held_case]["T48_K"].reshape(-1, 1).astype(np.float32)
    t72_te = arrays[held_case]["T72_K"].reshape(-1, 1).astype(np.float32)
    t96_te = arrays[held_case]["T96_K"].ravel().astype(np.float64)
    if mode == "T3":
        t24_te = arrays[held_case]["T24_K"].reshape(-1, 1).astype(np.float32)
        seq_te_t = torch.from_numpy(
            np.stack([ts.transform(t24_te), ts.transform(t48_te), ts.transform(t72_te)], axis=1))
    else:
        seq_te_t = torch.from_numpy(np.stack([ts.transform(t48_te), ts.transform(t72_te)], axis=1))

    b_te = t72_te.ravel().astype(np.float64)
    d_te = (t72_te.ravel() - t48_te.ravel()).astype(np.float64)
    s_d_te = float(np.mean(np.abs(d_te)))

    lc.seed_all(seed)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(coords_t), batch_size=1024, shuffle=True, num_workers=0,
                        drop_last=False, generator=generator, worker_init_fn=lc.seed_worker)
    model = MLPModel(seq_len=seq_t.shape[1]) if mode == "TMP" else lc.Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    init_digest = hashlib.sha256(
        b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

    epoch_rows = []
    max_grad = 0.0
    n_train = len(coords_t)
    for ep in range(1, max_epochs + 1):
        ep_start = time.monotonic()
        model.train()
        tot_obj, tot_mse, tot_reg, n_seen = 0.0, 0.0, 0.0, 0
        _ = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
        for idx in loader.batch_sampler:
            xb = coords_t[idx]
            sb = seq_t[idx]
            yb = target_t[idx]
            ulb = u_loc_t[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, sb)
            mse_term = torch.mean((pred - yb) ** 2)
            reg_term = torch.mean((pred - ulb) ** 2)
            loss = mse_term + lam_si * reg_term
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            max_grad = max(max_grad, float(gnorm))
            optimizer.step()
            bs = len(idx)
            tot_obj += float(loss.detach()) * bs
            tot_mse += float(mse_term.detach()) * bs
            tot_reg += float(reg_term.detach()) * bs
            n_seen += bs

        wall_ep = time.monotonic() - ep_start
        rng_state = torch.get_rng_state()
        model.eval()
        with torch.no_grad():
            g_hat = model(xyz_te_t, seq_te_t).numpy().ravel().astype(np.float64)
        torch.set_rng_state(rng_state)

        g_mean_abs = max(float(np.mean(np.abs(g_hat))), 1e-6)
        g_norm = g_hat / g_mean_abs
        v_hat = (1.0 - alpha) * (q_val * d_te) + alpha * (q_val * s_d_te * g_norm)
        t_pred = (b_te + v_hat).astype(np.float32).astype(np.float64)

        m = lc.metrics(t96_te, t_pred)
        row_mae, exact_xyz_mae, bin_xz_mae = support_maes(xyz_raw.astype(np.float64), t_pred - t96_te)
        epoch_rows.append(dict(
            epoch=ep, n=n_seen, train_objective=tot_obj / n_seen, train_mse_normalized=tot_mse / n_seen,
            train_si_reg=tot_reg / n_seen, test_MAE_K=m["MAE_K"], test_RMSE_K=m["RMSE_K"], test_R2=m["R2"],
            test_bias_K=m["Bias_K"], row_MAE_K=row_mae, exact_XYZ_MAE_K=exact_xyz_mae,
            bin_0p1m_XZ_MAE_K=bin_xz_mae, g_mean_abs=g_mean_abs, wall_s=wall_ep))

        if ep == 20:
            torch.save(model.state_dict(), run_dir / "model.pt")
            pred_df = pd.DataFrame(dict(
                row_id=arrays[held_case]["row_id"].ravel(),
                x_m=xyz_raw[:, 0], y_m=xyz_raw[:, 1], z_m=xyz_raw[:, 2],
                T24_K=arrays[held_case]["T24_K"].ravel(),
                T48_K=t48_te.ravel(), T72_K=t72_te.ravel(), T96_K=t96_te,
                Tpred_K=t_pred.astype(np.float32)))
            pred_df.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
            pd.DataFrame(epoch_rows[:20]).to_csv(run_dir / "epochs.csv", index=False)
            (run_dir / "raw_metrics.json").write_text(json.dumps(dict(
                run_id=run_id, case=held_case, seed=seed, arm="T", mode=mode, method="IA-TRD-LSTM",
                epochs=20, alpha=alpha, lam_si=lam_si, q_variant=q_variant, q_value=q_val,
                s_d_te=s_d_te, s_d_train_map=s_d_map, initial_digest=init_digest, max_grad=max_grad,
                n_train=n_train, n_test=len(t96_te), metrics=m,
                support_metrics=dict(row_MAE_K=row_mae, exact_XYZ_MAE_K=exact_xyz_mae,
                                     bin_0p1m_XZ_MAE_K=bin_xz_mae),
                wall_s=time.monotonic() - t_start), indent=2), encoding="utf-8")

    if max_epochs > 20:
        pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_trajectory.csv", index=False)
        (run_dir / "trajectory_summary.json").write_text(json.dumps(dict(
            run_id=run_id, case=held_case, seed=seed, mode=mode, q_variant=q_variant, q_value=q_val,
            lam_si=lam_si, alpha=alpha, max_epochs=max_epochs, s_d_te=s_d_te,
            epochs=epoch_rows), indent=2), encoding="utf-8")

    idx20 = min(19, len(epoch_rows) - 1)
    print(f"COMPLETED {run_id}: ep20 MAE={epoch_rows[idx20]['test_MAE_K']:.4f} K, "
          f"bias={epoch_rows[idx20]['test_bias_K']:+.4f} K, last MAE={epoch_rows[-1]['test_MAE_K']:.4f} K "
          f"in {time.monotonic() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=CASES)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", default="T", choices=["T", "T3", "TMP"])
    p.add_argument("--q", default="int", choices=["int", "med", "rms"])
    p.add_argument("--lam-si", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    run_single(args.case, args.seed, mode=args.mode, q_variant=args.q, lam_si=args.lam_si,
               alpha=args.alpha, max_epochs=args.epochs)
