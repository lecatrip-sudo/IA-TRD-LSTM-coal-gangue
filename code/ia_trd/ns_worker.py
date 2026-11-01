"""V99 nested-selection inner-fold worker (pre-specified protocol).

For one outer fold (held outer case) and one inner fold (inner-held case drawn
from the outer fold's five training cases), this worker trains one arm-T
configuration (q variant in {int, med, rms}; lambda_SI in {0.005, 0.02, 0.05})
on the remaining four training cases using the matched transition
[24, 48] -> 72 h, then scores the trained network on the inner-held case at the
[48, 72] -> 96 h transition for the full alpha grid {0.0, 0.1, ..., 1.0}.

Provenance guarantees:
  * The outer held case is never read beyond what the curated loader loads for
    all cases; no outer-test target (96 h of the outer held case) is scored.
  * The inner-held case is one of the outer fold's training cases; its 96 h
    field is otherwise unused by the outer fit, so using it for configuration
    selection is training-set-internal validation.
  * Seed is fixed at 42 for the selection stage; final fits use seeds 42-44.

Selection rule (pre-specified in ns_suite.py):
  score(q, lam, alpha) = mean over the five inner folds of the inner-validation
  MAE; minimum score wins; candidates within 0.002 K of the minimum are broken
  by (larger lambda, then q order int > med > rms, then alpha closest to 0.5,
  then lower alpha if equally distant).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

BASE_REPRO = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
sys.path.insert(0, str(BASE_REPRO / "new_runs" / "80mm_s42_R_a"))
import legacy_core as lc  # noqa: E402

CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
OUT_DIR = Path(r"E:\第三篇\outputs\ema_v99_revision\ns_inner")


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


def run_single(outer, inner, q_variant, lam, seed=42, max_epochs=20):
    t0 = time.monotonic()
    arrays = {c: dict(np.load(BASE_REPRO / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in CASES}
    train_cases = [c for c in CASES if c != outer and c != inner]
    assert len(train_cases) == 4
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
    d_means, s_d_map = [], {}
    for c in train_cases:
        d_c = (arrays[c]["T48_K"].ravel() - arrays[c]["T24_K"].ravel()).astype(np.float64)
        s_d_c = float(np.mean(np.abs(d_c)))
        s_d_map[c] = max(s_d_c, 1e-6)
        d_means.append(s_d_c)

    coords_list, seq_list, target_list, u_loc_list = [], [], [], []
    for c in train_cases:
        xyz = arrays[c]["XYZ"].astype(np.float32)
        t24 = arrays[c]["T24_K"].reshape(-1, 1).astype(np.float32)
        t48 = arrays[c]["T48_K"].reshape(-1, 1).astype(np.float32)
        t72 = arrays[c]["T72_K"].reshape(-1, 1).astype(np.float32)
        sd = np.float32(s_d_map[c])
        u_c = (t72 - t48) / (np.float32(q_val) * sd)
        u_loc_c = (t48 - t24) / sd
        coords_list.append(cs.transform(xyz))
        seq_list.append(np.stack([ts.transform(t24), ts.transform(t48)], axis=1))
        target_list.append(u_c.astype(np.float32))
        u_loc_list.append(u_loc_c.astype(np.float32))

    coords_t = torch.from_numpy(np.vstack(coords_list))
    seq_t = torch.from_numpy(np.vstack(seq_list))
    target_t = torch.from_numpy(np.vstack(target_list))
    u_loc_t = torch.from_numpy(np.vstack(u_loc_list))

    # Inner-validation instance: the inner-held case at [48, 72] -> 96 h.
    xyz_in = arrays[inner]["XYZ"].astype(np.float32)
    t48_in = arrays[inner]["T48_K"].reshape(-1, 1).astype(np.float32)
    t72_in = arrays[inner]["T72_K"].reshape(-1, 1).astype(np.float32)
    seq_in = torch.from_numpy(np.stack([ts.transform(t48_in), ts.transform(t72_in)], axis=1))
    xyz_in_t = torch.from_numpy(cs.transform(xyz_in))

    b_in = t72_in.ravel().astype(np.float64)
    d_in = (t72_in.ravel() - t48_in.ravel()).astype(np.float64)
    y_in = arrays[inner]["T96_K"].ravel().astype(np.float64)
    s_d_in = float(np.mean(np.abs(d_in)))

    lc.seed_all(seed)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(coords_t), batch_size=1024, shuffle=True, num_workers=0,
                        drop_last=False, generator=generator, worker_init_fn=lc.seed_worker)
    model = lc.Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(max_epochs):
        model.train()
        _ = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
        for idx in loader.batch_sampler:
            xb = coords_t[idx]
            sb = seq_t[idx]
            yb = target_t[idx]
            ulb = u_loc_t[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, sb)
            loss = torch.mean((pred - yb) ** 2) + lam * torch.mean((pred - ulb) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        g = model(xyz_in_t, seq_in).numpy().ravel().astype(np.float64)
    g_mean_abs = max(float(np.mean(np.abs(g))), 1e-6)
    g_norm = g / g_mean_abs

    result = {
        "outer": outer,
        "inner": inner,
        "q_variant": q_variant,
        "lam_si": lam,
        "q_value": q_val,
        "s_d_inner": s_d_in,
        "g_mean_abs": g_mean_abs,
        "alpha_mae": {},
        "alpha_bias": {},
        "seed": seed,
        "epochs": max_epochs,
        "n_train": int(len(coords_t)),
        "n_val": int(len(y_in)),
        "wall_s": time.monotonic() - t0,
    }
    for a in ALPHAS:
        v_hat = (1.0 - a) * (q_val * d_in) + a * (q_val * s_d_in * g_norm)
        e = b_in + v_hat - y_in
        result["alpha_mae"][f"{a:.1f}"] = float(np.mean(np.abs(e)))
        result["alpha_bias"][f"{a:.1f}"] = float(np.mean(e))
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--outer", required=True, choices=CASES)
    p.add_argument("--inner", required=True, choices=CASES)
    p.add_argument("--q", required=True, choices=["int", "med", "rms"])
    p.add_argument("--lam", required=True, type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    assert args.outer != args.inner
    r = run_single(args.outer, args.inner, args.q, args.lam, seed=args.seed, max_epochs=args.epochs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.outer}__{args.inner}__{args.q}__{args.lam}.json"
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"INNER_DONE {args.outer} {args.inner} {args.q} {args.lam}: "
          f"mae(a=0)={r['alpha_mae']['0.0']:.4f} mae(a=0.5)={r['alpha_mae']['0.5']:.4f} "
          f"mae(a=1)={r['alpha_mae']['1.0']:.4f} in {r['wall_s']:.1f}s", flush=True)
