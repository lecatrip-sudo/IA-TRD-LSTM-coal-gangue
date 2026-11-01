from pathlib import Path
import numpy as np
import pandas as pd

exp_dir = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility\figure_sources\experimental_originals")
obs = pd.read_csv(exp_dir / "observations.csv")
sensors = pd.read_csv(exp_dir / "sensor_map.csv")
sub_sensors = sensors[sensors["sensor_id"].isin([f"T{i}" for i in range(17, 29)])].copy()
coord_map = {row["sensor_id"]: np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=np.float32) for _, row in sub_sensors.iterrows()}
probes = [f"T{i}" for i in range(17, 29)]

# Build sliding 6-hour (72 samples of 300s) transitions for each trial:
# Training window: a in [2h, 6h], b = a + 6h in [8h, 12h], y = b + 6h in [14h, 18h] (completely <= 18h!)
# Forecast test window: a in [6h, 12h], b = a + 6h in [12h, 18h], y = b + 6h in [18h, 24h] (predicting the final 6h [18h..24h]!)
STEP = 72  # 72 * 300s = 21600s = 6 hours

def extract_windows(trial_num, end_idx_range):
    sub = obs[obs["trial"] == trial_num]
    piv = sub.pivot(index="sample_index", columns="probe", values="temperature_C")[probes]
    arr = piv.to_numpy(dtype=np.float64) + 273.15  # Convert to Kelvin for exact unit consistency with CFD!
    xyz_list, a_list, b_list, y_list, t_list, p_list = [], [], [], [], [], []
    for k in end_idx_range:
        a_row = arr[k - 2 * STEP]
        b_row = arr[k - STEP]
        y_row = arr[k]
        for j, p in enumerate(probes):
            xyz_list.append(coord_map[p])
            a_list.append(a_row[j])
            b_list.append(b_row[j])
            y_list.append(y_row[j])
            t_list.append(k * 300.0 / 3600.0)
            p_list.append(p)
    return dict(
        xyz=np.array(xyz_list, dtype=np.float32),
        a=np.array(a_list, dtype=np.float64),
        b=np.array(b_list, dtype=np.float64),
        y=np.array(y_list, dtype=np.float64),
        time_h=np.array(t_list, dtype=np.float64),
        probe=np.array(p_list),
    )

# Let's check the exact quarter checkpoints (6h, 12h, 18h, 24h -> sample_index 72, 144, 216, 288)
# AND dense 5-min trajectory evaluation across the final 6h horizon (y in [18h..24h], i.e. sample_index 216..288,
# with training transitions in the preceding window y in [12h..18h], i.e. sample_index 144..216)!
for mode_name, tr_idx, te_idx in [
    ("Dense 5-min Horizon (Train 12-18h -> Test 18-24h)", range(144, 217), range(216, 289)),
    ("Quarter Checkpoints (6,12->18h Train; 12,18->24h Test)", [216], [288]),
]:
    print(f"\n=== {mode_name} ===")
    for held_trial in [1, 2]:
        train_trial = 2 if held_trial == 1 else 1
        tr = extract_windows(train_trial, tr_idx)
        te = extract_windows(held_trial, te_idx)
        
        d_tr = tr["b"] - tr["a"]
        v_tr = tr["y"] - tr["b"]
        d_te = te["b"] - te["a"]
        v_te = te["y"] - te["b"]
        
        # Proposition 1 ratio
        r2_pers = 1.0 - np.sum((te["y"] - te["b"])**2) / np.sum((te["y"] - te["y"].mean())**2)
        mae_pers = np.mean(np.abs(te["b"] - te["y"]))
        mae_lin = np.mean(np.abs(te["b"] + d_te - te["y"]))
        
        # Weighted median q_med on training trial
        m = d_tr != 0
        ratios = v_tr[m] / d_tr[m]
        mass = np.abs(d_tr[m])
        order = np.argsort(ratios, kind="stable")
        q_med = float(ratios[order[np.searchsorted(np.cumsum(mass[order]), mass.sum()/2, side="left")]])
        q_int = float(np.mean(np.abs(v_tr)) / np.mean(np.abs(d_tr)))
        
        mae_damp = np.mean(np.abs(te["b"] + q_med * d_te - te["y"]))
        mae_int = np.mean(np.abs(te["b"] + q_int * d_te - te["y"]))
        print(f"  Held-out Trial {held_trial} (n={len(te['y'])}): R2_pers={r2_pers:.4f} | MAE_pers={mae_pers:.4f} K | MAE_lin={mae_lin:.4f} K | MAE_damp(q={q_med:.3f})={mae_damp:.4f} K | MAE_q_int(q={q_int:.3f})={mae_int:.4f} K")
