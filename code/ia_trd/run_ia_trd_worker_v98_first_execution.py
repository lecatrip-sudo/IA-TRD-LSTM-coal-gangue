"""Worker script for Instance-Adaptive Thermal Relaxation Decomposition LSTM (IA-TRD-LSTM, Arm T)."""
import argparse
import hashlib
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

CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]


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


def run_single(held_case: str, seed: int, max_epochs: int = 20, alpha: float = 0.5, lam_si: float = 0.02, out_root: Path = None):
    t_start = time.monotonic()
    if out_root is None:
        out_root = BASE_REPRO / "ia_trd_runs"
    run_id = f"{held_case}_s{seed}_T_a"
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

    # Calibrate training-only enthalpy-conserving relaxation ratio q_int and per-case input increment scales
    d_means, v_means, s_d_map = [], [], {}
    for c in train_cases:
        d_c = (arrays[c]["T48_K"].ravel() - arrays[c]["T24_K"].ravel()).astype(np.float64)
        v_c = (arrays[c]["T72_K"].ravel() - arrays[c]["T48_K"].ravel()).astype(np.float64)
        s_d_c = float(np.mean(np.abs(d_c)))
        s_d_map[c] = max(s_d_c, 1e-6)
        d_means.append(s_d_c)
        v_means.append(float(np.mean(np.abs(v_c))))
    q_int = float(np.mean(v_means) / np.mean(d_means))

    coords_list, seq_list, target_list, u_loc_list = [], [], [], []
    for c in train_cases:
        xyz = arrays[c]["XYZ"].astype(np.float32)
        t24 = arrays[c]["T24_K"].reshape(-1, 1).astype(np.float32)
        t48 = arrays[c]["T48_K"].reshape(-1, 1).astype(np.float32)
        t72 = arrays[c]["T72_K"].reshape(-1, 1).astype(np.float32)
        sd = np.float32(s_d_map[c])
        u_c = (t72 - t48) / (np.float32(q_int) * sd)
        u_loc_c = (t48 - t24) / sd
        coords_list.append(cs.transform(xyz))
        seq_list.append(np.stack([ts.transform(t24), ts.transform(t48)], axis=1))
        target_list.append(u_c.astype(np.float32))
        u_loc_list.append(u_loc_c.astype(np.float32))

    coords_t = torch.from_numpy(np.vstack(coords_list))
    seq_t = torch.from_numpy(np.vstack(seq_list))
    target_t = torch.from_numpy(np.vstack(target_list))
    u_loc_t = torch.from_numpy(np.vstack(u_loc_list))

    # Prepare held-out test tensors (input window 48h -> 72h only)
    xyz_raw = arrays[held_case]["XYZ"].astype(np.float32)
    xyz_te_t = torch.from_numpy(cs.transform(xyz_raw))
    t48_te = arrays[held_case]["T48_K"].reshape(-1, 1).astype(np.float32)
    t72_te = arrays[held_case]["T72_K"].reshape(-1, 1).astype(np.float32)
    t96_te = arrays[held_case]["T96_K"].ravel().astype(np.float64)
    seq_te_t = torch.from_numpy(np.stack([ts.transform(t48_te), ts.transform(t72_te)], axis=1))

    b_te = t72_te.ravel().astype(np.float64)
    d_te = (t72_te.ravel() - t48_te.ravel()).astype(np.float64)
    s_d_te = float(np.mean(np.abs(d_te)))

    lc.seed_all(seed)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(seed)
    dummy_ds = TensorDataset(coords_t)
    loader = DataLoader(
        dummy_ds,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
        worker_init_fn=lc.seed_worker,
    )
    model = lc.Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    init_digest = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

    epoch_rows = []
    max_grad = 0.0
    n_train = len(coords_t)

    for ep in range(1, max_epochs + 1):
        ep_start = time.monotonic()
        model.train()
        tot_obj, tot_mse, tot_reg, n_seen = 0.0, 0.0, 0.0, 0
        # Match DataLoader.__iter__ base_seed RNG draw for exact parity
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
        v_hat = (1.0 - alpha) * (q_int * d_te) + alpha * (q_int * s_d_te * g_norm)
        t_pred = (b_te + v_hat).astype(np.float32).astype(np.float64)

        m = lc.metrics(t96_te, t_pred)
        row_mae, exact_xyz_mae, bin_xz_mae = support_maes(xyz_raw.astype(np.float64), t_pred - t96_te)

        rec = dict(
            epoch=ep,
            n=n_seen,
            train_objective=tot_obj / n_seen,
            train_mse_normalized=tot_mse / n_seen,
            train_si_reg=tot_reg / n_seen,
            test_MAE_K=m["MAE_K"],
            test_RMSE_K=m["RMSE_K"],
            test_R2=m["R2"],
            test_bias_K=m["Bias_K"],
            row_MAE_K=row_mae,
            exact_XYZ_MAE_K=exact_xyz_mae,
            bin_0p1m_XZ_MAE_K=bin_xz_mae,
            g_mean_abs=g_mean_abs,
            wall_s=wall_ep,
        )
        epoch_rows.append(rec)

        if ep == 20:
            # Save standard 20-epoch checkpoint and predictions for the 18-run primary benchmark
            torch.save(model.state_dict(), run_dir / "model.pt")
            pred_df = pd.DataFrame(
                dict(
                    row_id=arrays[held_case]["row_id"].ravel(),
                    x_m=xyz_raw[:, 0],
                    y_m=xyz_raw[:, 1],
                    z_m=xyz_raw[:, 2],
                    T48_K=t48_te.ravel(),
                    T72_K=t72_te.ravel(),
                    T96_K=arrays[held_case]["T96_K"].ravel(),
                    Tpred_K=t_pred.astype(np.float32),
                )
            )
            pred_df.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
            pd.DataFrame(epoch_rows[:20]).to_csv(run_dir / "epochs.csv", index=False)
            raw_summary = dict(
                run_id=run_id,
                case=held_case,
                seed=seed,
                arm="T",
                method="IA-TRD-LSTM",
                epochs=20,
                alpha=alpha,
                lam_si=lam_si,
                q_int=q_int,
                s_d_te=s_d_te,
                s_d_train_map=s_d_map,
                initial_digest=init_digest,
                max_grad=max_grad,
                n_train=n_train,
                n_test=len(t96_te),
                metrics=m,
                support_metrics=dict(
                    row_MAE_K=row_mae,
                    exact_XYZ_MAE_K=exact_xyz_mae,
                    bin_0p1m_XZ_MAE_K=bin_xz_mae,
                ),
                wall_s=time.monotonic() - t_start,
            )
            (run_dir / "raw_metrics.json").write_text(json.dumps(raw_summary, indent=2), encoding="utf-8")

    if max_epochs > 20:
        diag_dir = out_root / f"{held_case}_s{seed}_T_e{max_epochs}"
        diag_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(epoch_rows).to_csv(diag_dir / "epoch_trajectory.csv", index=False)
        (diag_dir / "trajectory_summary.json").write_text(
            json.dumps(
                dict(
                    run_id=f"{held_case}_s{seed}_T_e{max_epochs}",
                    case=held_case,
                    seed=seed,
                    arm="T",
                    max_epochs=max_epochs,
                    q_int=q_int,
                    s_d_te=s_d_te,
                    epochs=epoch_rows,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    idx20 = min(19, len(epoch_rows) - 1)
    print(
        f"COMPLETED {run_id} (max_ep={max_epochs}): ep20 MAE={epoch_rows[idx20]['test_MAE_K']:.4f} K, "
        f"bias={epoch_rows[idx20]['test_bias_K']:+.4f} K, last_ep MAE={epoch_rows[-1]['test_MAE_K']:.4f} K "
        f"in {time.monotonic() - t_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=CASES)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lam-si", type=float, default=0.02)
    args = p.parse_args()
    run_single(args.case, args.seed, max_epochs=args.epochs, alpha=args.alpha, lam_si=args.lam_si)
