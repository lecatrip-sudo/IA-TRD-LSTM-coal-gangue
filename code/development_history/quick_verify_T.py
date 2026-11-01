import sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility\new_runs\80mm_s42_R_a")
import legacy_core as lc

base = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
cases = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
arrays = {c: dict(np.load(base / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in cases}

def test_quick(held_case, seed=42, epochs=5, alpha=0.5, lam_si=0.05):
    t0 = time.monotonic()
    train_cases = [c for c in cases if c != held_case]
    fit_ids = [f"{c}/{h}h" for c in train_cases for h in [24, 48, 72]]
    cs = lc.Scaler().fit(np.vstack([arrays[c]["XYZ"] for c in train_cases]), [f"{c}/24h" for c in train_cases])
    ts = lc.Scaler().fit(np.vstack([arrays[c][f"T{h}_K"].reshape(-1, 1) for c in train_cases for h in [24, 48, 72]]), fit_ids)
    
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
        
    coords = np.vstack(coords_list)
    seq = np.vstack(seq_list)
    target = np.vstack(target_list)
    u_loc = np.vstack(u_loc_list)
    
    # Held-out test inputs
    xyz_te = cs.transform(arrays[held_case]["XYZ"].astype(np.float32))
    t48_te = arrays[held_case]["T48_K"].reshape(-1, 1).astype(np.float32)
    t72_te = arrays[held_case]["T72_K"].reshape(-1, 1).astype(np.float32)
    t96_te = arrays[held_case]["T96_K"].ravel().astype(np.float64)
    seq_te = np.stack([ts.transform(t48_te), ts.transform(t72_te)], axis=1)
    b_te = t72_te.ravel().astype(np.float64)
    d_te = (t72_te.ravel() - t48_te.ravel()).astype(np.float64)
    s_d_te = float(np.mean(np.abs(d_te)))
    
    lc.seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(coords), torch.from_numpy(seq), torch.from_numpy(target), torch.from_numpy(u_loc))
    loader = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=0, drop_last=False, generator=gen, worker_init_fn=lc.seed_worker)
    model = lc.Model()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for xb, sb, yb, ulb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb, sb)
            loss = torch.mean((pred - yb) ** 2) + lam_si * torch.mean((pred - ulb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            tot_loss += float(loss.detach()) * len(xb)
            n += len(xb)
        rng_state = torch.get_rng_state()
        model.eval()
        with torch.no_grad():
            g_hat = model(torch.from_numpy(xyz_te), torch.from_numpy(seq_te)).numpy().ravel().astype(np.float64)
        torch.set_rng_state(rng_state)
        g_norm = g_hat / max(float(np.mean(np.abs(g_hat))), 1e-6)
        v_hat = (1.0 - alpha) * (q_int * d_te) + alpha * (q_int * s_d_te * g_norm)
        mae = float(np.mean(np.abs(b_te + v_hat - t96_te)))
        bias = float(np.mean(b_te + v_hat - t96_te))
        print(f"[{held_case}] ep={ep:2d} loss={tot_loss/n:.4f} test_MAE={mae:.4f} K, bias={bias:+.4f} K (elapsed={time.monotonic()-t0:.1f}s)")

if __name__ == "__main__":
    test_quick(sys.argv[1], epochs=5)
