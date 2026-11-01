"""V99 post-compute analyses: threshold assessment, bootstrap intervals, alpha
sensitivity, own-q variant, development-grid disclosure, relaxation timescales,
and laboratory probe-resolved summaries. Reads the finalized run set produced by
ns_suite.py and writes CSV/JSON outputs used by the manuscript tables.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(r"E:\第三篇\outputs\ema_v99_revision")
BASE = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
V98_RUNS = BASE / "ia_trd_runs"
V99_RUNS = BASE / "ia_trd_runs_v99"
REFS = HERE / "cfd_refs"
OUT = HERE / "analyses_v99"
OUT.mkdir(parents=True, exist_ok=True)
CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
CASE_LABELS = {"80mm": "80 mm", "150mm": "150 mm", "300mm": "300 mm",
               "0.25kxl": "Porosity 0.25", "0.5kxl": "Porosity 0.50", "0.75kxl": "Porosity 0.75"}
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
SEEDS = [42, 43, 44]
BOOT_N = 2000
BOOT_SEED = 20261030

sys.path.insert(0, str(BASE / "new_runs" / "80mm_s42_R_a"))
import legacy_core as lc  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- run location
def plan():
    return json.loads((HERE / "ns_final_plan.json").read_text(encoding="utf-8"))


def tag_for(sel):
    return "T" if (sel["q"] == "int" and sel["lam"] == 0.02) else f"T_{sel['q']}_l{sel['lam']}"


def original_fit_dir(case, seed):
    """Directory holding model.pt of the network fit used by the primary T run."""
    p = plan()[case]
    if p["action"] in ("reuse", "recombine"):
        return V98_RUNS / f"{case}_s{seed}_T_a"
    return V99_RUNS / f"{case}_s{seed}_{tag_for(p['selected'])}_a"


def final_t_dir(case, seed):
    p = plan()[case]
    if p["action"] == "reuse":
        return V98_RUNS / f"{case}_s{seed}_T_a"
    if p["action"] == "recombine":
        return V99_RUNS / f"{case}_s{seed}_T_recomb_a{p['selected']['alpha']}"
    return V99_RUNS / f"{case}_s{seed}_{tag_for(p['selected'])}_a"


def variant_dir(case, seed, mode):
    sel = plan()[case]["selected"]
    tag = mode if (sel["q"] == "int" and sel["lam"] == 0.02) else f"{mode}_{sel['q']}_l{sel['lam']}"
    return V99_RUNS / f"{case}_s{seed}_{tag}_a"


def fold_arrays():
    return {c: dict(np.load(BASE / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in CASES}


def fold_scalers(arrays, case):
    train = [c for c in CASES if c != case]
    cs = lc.Scaler().fit(np.vstack([arrays[c]["XYZ"].astype(np.float32) for c in train]),
                         [f"{c}/24h" for c in train])
    ts = lc.Scaler().fit(np.vstack([arrays[c][f"T{h}_K"].reshape(-1, 1).astype(np.float32)
                                    for c in train for h in [24, 48, 72]]),
                         [f"{c}/{h}h" for c in train for h in [24, 48, 72]])
    return cs, ts


def held_g(arrays, case, seed):
    """Network output g on the held case for the primary T fit of this fold/seed."""
    cs, ts = fold_scalers(arrays, case)
    model = lc.Model()
    model.load_state_dict(torch.load(original_fit_dir(case, seed) / "model.pt", map_location="cpu"))
    model.eval()
    a = arrays[case]
    xyz = a["XYZ"].astype(np.float32)
    t48 = a["T48_K"].reshape(-1, 1).astype(np.float32)
    t72 = a["T72_K"].reshape(-1, 1).astype(np.float32)
    seq = torch.from_numpy(np.stack([ts.transform(t48), ts.transform(t72)], axis=1))
    with torch.no_grad():
        g = model(torch.from_numpy(cs.transform(xyz)), seq).numpy().ravel().astype(np.float64)
    return g


def eq9(b, d, s_d, g, q, alpha):
    g_mean = max(float(np.mean(np.abs(g))), 1e-6)
    v_hat = (1.0 - alpha) * (q * d) + alpha * (q * s_d * g / g_mean)
    return b + v_hat


# ---------------------------------------------------------------- A+B: alpha grid and own-q*
def alpha_and_ownq():
    arrays = fold_arrays()
    rows = []
    ownq_rows = []
    for case in CASES:
        p = plan()[case]
        sel = p["selected"]
        alpha_sel = sel["alpha"]
        a = arrays[case]
        t48 = a["T48_K"].ravel().astype(np.float64)
        t72 = a["T72_K"].ravel().astype(np.float64)
        y = a["T96_K"].ravel().astype(np.float64)
        b = t72
        d = b - t48
        s_d = float(np.mean(np.abs(d)))
        q_star = float(np.load(REFS / f"{case}.npz")["q_star"])
        ref = np.load(REFS / f"{case}.npz")
        for seed in SEEDS:
            g = held_g(arrays, case, seed)
            meta = json.loads((original_fit_dir(case, seed) / "raw_metrics.json").read_text(encoding="utf-8"))
            q_val = meta["q_value"] if "q_value" in meta else meta["q_int"]
            for al in ALPHAS:
                pred = eq9(b, d, s_d, g, q_val, al)
                rows.append(dict(case=case, case_label=CASE_LABELS[case], seed=seed, alpha=al,
                                 MAE_K=float(np.mean(np.abs(pred - y)))))
            pred_star = eq9(b, d, s_d, g, q_star, alpha_sel)
            ownq_rows.append(dict(case=case, case_label=CASE_LABELS[case], seed=seed,
                                  q_star=q_star, alpha=alpha_sel,
                                  MAE_K=float(np.mean(np.abs(pred_star - y))),
                                  Bias_K=float(np.mean(pred_star - y))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "alpha_sensitivity_primaryT.csv", index=False)
    piv = df.groupby(["case_label", "alpha"]).MAE_K.mean().unstack(0)
    piv["case-equal"] = piv.mean(axis=1)
    piv.round(4).to_csv(OUT / "alpha_sensitivity_primaryT_pivot.csv")
    print("=== Primary T: held-case MAE vs alpha (post-hoc sensitivity; selected alpha marked) ===")
    print(piv.round(4).to_string())
    dfo = pd.DataFrame(ownq_rows)
    dfo.to_csv(OUT / "own_q_variant_primaryT.csv", index=False)
    print("\n=== T with own-instance q* (post-hoc variant) ===")
    print(dfo.groupby("case_label")[["MAE_K", "Bias_K"]].mean().round(4).to_string())
    print("case-equal MAE:", round(float(dfo.MAE_K.mean()), 4))


# ---------------------------------------------------------------- C: development grid disclosure
def dev_grid():
    arrays = fold_arrays()
    rows = []
    for name_q, qv in [("med", "med"), ("int", "int"), ("rms", "rms")]:
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            case_maes = []
            for case in CASES:
                train = [c for c in CASES if c != case]
                b = arrays[case]["T72_K"].ravel().astype(np.float64)
                d = (b - arrays[case]["T48_K"].ravel().astype(np.float64))
                y = arrays[case]["T96_K"].ravel().astype(np.float64)
                dd = {c: (arrays[c]["T48_K"].ravel() - arrays[c]["T24_K"].ravel()).astype(np.float64) for c in train}
                vv = {c: (arrays[c]["T72_K"].ravel() - arrays[c]["T48_K"].ravel()).astype(np.float64) for c in train}
                if qv == "int":
                    q_val = float(np.mean([np.mean(np.abs(vv[c])) for c in train]) / np.mean([np.mean(np.abs(dd[c])) for c in train]))
                elif qv == "rms":
                    q_val = float(np.sqrt(np.mean([np.mean(vv[c] ** 2) for c in train]) / np.mean([np.mean(dd[c] ** 2) for c in train])))
                else:
                    ratios, mass = [], []
                    for c in train:
                        m = dd[c] != 0
                        ratios.append(vv[c][m] / dd[c][m])
                        mass.append(np.abs(dd[c][m]) / len(dd[c]))
                    ratios = np.concatenate(ratios)
                    mass = np.concatenate(mass)
                    order = np.argsort(ratios, kind="stable")
                    q_val = float(ratios[order[np.searchsorted(np.cumsum(mass[order]), mass.sum() / 2, side="left")]])
                seed_maes = []
                for seed in SEEDS:
                    dfp = pd.read_csv(BASE / "new_runs" / f"{case}_s{seed}_S_a" / "predictions.csv.gz",
                                     float_precision="round_trip")
                    v_pred = dfp["Tpred_K"].to_numpy(dtype=np.float64) - b
                    v_cal = v_pred * ((q_val * np.mean(np.abs(d))) / np.mean(np.abs(v_pred)))
                    v_hat = (1.0 - alpha) * (q_val * d) + alpha * v_cal
                    seed_maes.append(float(np.mean(np.abs(b + v_hat - y))))
                case_maes.append(float(np.mean(seed_maes)))
            rows.append(dict(q=qv, alpha=alpha, **{CASE_LABELS[c]: m for c, m in zip(CASES, case_maes)},
                             case_equal=float(np.mean(case_maes))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dev_time_grid_disclosure.csv", index=False)
    print("\n=== Development-stage grid (29 Oct 2026, post hoc on held cases; superseded by nested selection) ===")
    print(df.round(4).to_string())


# ---------------------------------------------------------------- D: thresholds
def classification_stats(y, pred, theta):
    exceed = y >= theta
    call = pred >= theta
    tp = int(np.sum(call & exceed))
    fn = int(np.sum(~call & exceed))
    fp = int(np.sum(call & ~exceed))
    tn = int(np.sum(~call & ~exceed))
    return dict(exceed_n=int(exceed.sum()), n=len(y),
                miss_rate=float((fn + fp) / len(y)),
                false_negative_rate=fn / max(exceed.sum(), 1),
                false_positive_rate=fp / max((~exceed).sum(), 1),
                tp=tp, fn=fn, fp=fp, tn=tn)


def thresholds():
    arrays = fold_arrays()
    rows = []
    for case in CASES:
        ref = np.load(REFS / f"{case}.npz")
        y = ref["y"]
        preds = {
            "Persistence": ref["pred_Persistence"],
            "Linear extrapolation": ref["pred_Linear_extrapolation"],
            "Damped trend": ref["pred_Damped_trend"],
            "aLSTM": ref["pred_aLSTM"],
            "Own-history blend (R3)": ref["pred_Own-history_blend_R3"],
        }
        for seed in SEEDS:
            dfp = pd.read_csv(final_t_dir(case, seed) / "predictions.csv.gz", float_precision="round_trip")
            preds[f"IA-TRD-LSTM (T) s{seed}"] = dfp["Tpred_K"].to_numpy(dtype=np.float64)
        # The tables, the ESM prose and the claim verification address one row per
        # method, so the seed-averaged forecast is reported under the bare name as
        # well; the per-seed rows above keep the full counts for the release.
        t_avg = np.vstack([preds[f"IA-TRD-LSTM (T) s{s}"] for s in SEEDS]).mean(axis=0)
        near = ref["z"] >= -0.5
        for theta in [318.15, 323.15]:
            for scope, mask in [("all rows", np.ones(len(y), bool)), ("near-surface z >= -0.5 m", near)]:
                for name, pred in list(preds.items()) + [("IA-TRD-LSTM (T)", t_avg)]:
                    st = classification_stats(y[mask], pred[mask], theta)
                    rows.append(dict(domain="CFD", case=CASE_LABELS[case], scope=scope, threshold_K=theta,
                                     method=name, **st))
    df = pd.DataFrame(rows)
    lab = pd.read_csv(HERE / "lab_v99" / "lab_predictions_v99.csv.gz")
    labp = {}
    for m in ["Persistence", "Linear extrapolation", "Damped trend", "Own-history blend (R3)"]:
        g = lab[(lab.method == m) & (lab.seed == 0)]
        labp[m] = g.sort_values(["trial", "probe", "time_h"]).pred_K.to_numpy()
    for m in ["IA-TRD-LSTM (T)", "IA-TRD-LSTM Eq. (10) form (V98)"]:
        gg = lab[lab.method == m].sort_values(["trial", "probe", "time_h"])
        # average seeds within rows
        gg = gg.groupby(["trial", "probe", "time_h"], as_index=False).pred_K.mean()
        labp[m] = gg.sort_values(["trial", "probe", "time_h"]).pred_K.to_numpy()
    base = lab[(lab.method == "Persistence") & (lab.seed == 0)].sort_values(["trial", "probe", "time_h"])
    y_lab = base.y_K.to_numpy()
    for theta in [318.15]:
        for name, pred in labp.items():
            assert len(pred) == len(y_lab), (name, len(pred), len(y_lab))
            st = classification_stats(y_lab, pred, theta)
            rows.append(dict(domain="Lab", case="both trials", scope="all rows", threshold_K=theta,
                             method=name, **st))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "threshold_assessment.csv", index=False)
    summ = df[df.domain == "CFD"].groupby(["method", "scope", "threshold_K"]).agg(
        miss_rate=("miss_rate", "mean"), fnr=("false_negative_rate", "mean")).round(4)
    print("\n=== Threshold assessment (CFD, case-equal mean miss/false-negative rates) ===")
    print(summ.to_string())
    print("\n=== Threshold assessment (Lab, 45 degC) ===")
    print(df[df.domain == "Lab"][["method", "exceed_n", "miss_rate", "false_negative_rate"]].round(4).to_string(index=False))


# ---------------------------------------------------------------- E: bootstrap
def block_keys(x, z, size=0.5):
    return np.stack([np.floor(x / size + 1e-9), np.floor((z + 6.5) / size + 1e-9)], axis=1)


def bootstrap_cfd():
    rng = np.random.default_rng(BOOT_SEED)
    blocks_by_case = {}
    for case in CASES:
        ref = np.load(REFS / f"{case}.npz")
        p = {"Damped trend": ref["pred_Damped_trend"], "Linear extrapolation": ref["pred_Linear_extrapolation"],
             "Persistence": ref["pred_Persistence"], "Own-history blend (R3)": ref["pred_Own-history_blend_R3"]}
        errors = {m: np.abs(pred - ref["y"]) for m, pred in p.items()}
        # The manuscript averages per-run errors, not ensemble predictions.
        errors["IA-TRD-LSTM (T)"] = np.mean([
            np.abs(pd.read_csv(final_t_dir(case, s) / "predictions.csv.gz",
                               float_precision="round_trip")["Tpred_K"].to_numpy(dtype=np.float64) - ref["y"])
            for s in SEEDS], axis=0)
        keys = block_keys(ref["x"], ref["z"])
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        blocks_by_case[case] = dict(
            n_blocks=len(uk), sizes=np.bincount(inv),
            error_sums={m: np.bincount(inv, weights=e) for m, e in errors.items()},
            point={m: float(e.mean()) for m, e in errors.items()})
    methods = ["Persistence", "Linear extrapolation", "Damped trend", "Own-history blend (R3)", "IA-TRD-LSTM (T)"]
    out_rows = []
    for rep in range(BOOT_N):
        maes = {m: [] for m in methods}
        for case in CASES:
            d = blocks_by_case[case]
            n_blocks = d["n_blocks"]
            pick = rng.integers(0, n_blocks, n_blocks)
            counts = np.bincount(pick, minlength=n_blocks)
            n_rows = float(np.dot(counts, d["sizes"]))
            for m in methods:
                maes[m].append(float(np.dot(counts, d["error_sums"][m]) / n_rows))
        for m in methods:
            out_rows.append(dict(rep=rep, method=m, mae=float(np.mean(maes[m]))))
        for a, bth in [("IA-TRD-LSTM (T)", "Damped trend"), ("Own-history blend (R3)", "IA-TRD-LSTM (T)"),
                       ("Own-history blend (R3)", "Damped trend"), ("IA-TRD-LSTM (T)", "Linear extrapolation")]:
            out_rows.append(dict(rep=rep, method=f"{a} - {bth}", mae=float(np.mean(maes[a]) - np.mean(maes[bth]))))
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT / "bootstrap_cfd_reps.csv.gz", index=False, compression="gzip")
    summ = df.groupby("method").mae.agg(["mean", lambda s: np.quantile(s, 0.025), lambda s: np.quantile(s, 0.975),
                                         lambda s: float((s < 0).mean())])
    summ.columns = ["mean", "ci2.5", "ci97.5", "frac_below0"]
    point = {m: float(np.mean([blocks_by_case[c]["point"][m] for c in CASES])) for m in methods}
    for a, b in [("IA-TRD-LSTM (T)", "Damped trend"), ("Own-history blend (R3)", "IA-TRD-LSTM (T)"),
                 ("Own-history blend (R3)", "Damped trend"), ("IA-TRD-LSTM (T)", "Linear extrapolation")]:
        point[f"{a} - {b}"] = point[a] - point[b]
    summ["point"] = pd.Series(point)
    summ.round(4).to_csv(OUT / "bootstrap_cfd_summary.csv")
    print("\n=== CFD spatial block bootstrap (0.5 m blocks, 2000 reps) ===")
    print(summ.round(4).to_string())


def bootstrap_lab():
    lab = pd.read_csv(HERE / "lab_v99" / "lab_predictions_v99.csv.gz")
    lab["ae"] = (lab.pred_K - lab.y_K).abs()
    pivot = {}
    probes = sorted(lab.probe.unique())
    for m in ["Damped trend", "Own-history blend (R3)", "IA-TRD-LSTM (T)", "IA-TRD-LSTM Eq. (10) form (V98)"]:
        g = lab[lab.method == m]
        # Each probe retains its time-mean, seed-mean absolute error.
        # Taking abs(mean(pred)-mean(target)) would measure bias instead.
        pv = g.pivot_table(index="probe", columns="trial", values="ae", aggfunc="mean")
        pv = pv.reindex(index=probes, columns=[1, 2])
        assert pv.notna().all().all(), (m, pv)
        pivot[m] = pv.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for rep in range(BOOT_N):
        pick = rng.integers(0, len(probes), len(probes))
        rec = {}
        for m, pv in pivot.items():
            rec[m] = float(pv[pick].mean())
        for m in rec:
            rows.append(dict(rep=rep, method=m, mae=rec[m]))
        rows.append(dict(rep=rep, method="T - damped", mae=rec["IA-TRD-LSTM (T)"] - rec["Damped trend"]))
        rows.append(dict(rep=rep, method="T - R3", mae=rec["IA-TRD-LSTM (T)"] - rec["Own-history blend (R3)"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap_lab_reps.csv.gz", index=False, compression="gzip")
    summ = df.groupby("method").mae.agg(["mean", lambda s: np.quantile(s, 0.025), lambda s: np.quantile(s, 0.975),
                                         lambda s: float((s < 0).mean())])
    summ.columns = ["mean", "ci2.5", "ci97.5", "frac_below0"]
    point = {m: float(pv.mean()) for m, pv in pivot.items()}
    point["T - damped"] = point["IA-TRD-LSTM (T)"] - point["Damped trend"]
    point["T - R3"] = point["IA-TRD-LSTM (T)"] - point["Own-history blend (R3)"]
    summ["point"] = pd.Series(point)
    summ.round(4).to_csv(OUT / "bootstrap_lab_summary.csv")
    print("\n=== Lab probe-cluster bootstrap (2000 reps) ===")
    print(summ.round(4).to_string())


# ---------------------------------------------------------------- F: timescales, G: probe split
def timescales():
    rows = []
    arrays = fold_arrays()
    for case in CASES:
        p = plan()[case]["selected"]
        meta = json.loads((original_fit_dir(case, 42) / "raw_metrics.json").read_text(encoding="utf-8"))
        q = meta["q_value"] if "q_value" in meta else meta["q_int"]
        ref = np.load(REFS / f"{case}.npz")
        rows.append(dict(case=CASE_LABELS[case], q=q, tau_h=float(-24.0 / np.log(q)),
                         q_star_own=float(ref["q_star"]), tau_star_h=float(-24.0 / np.log(ref["q_star"]))))
    cal = json.loads((HERE / "lab_v99" / "lab_calibration_v99.json").read_text(encoding="utf-8"))["calibration"]
    for trial, c in cal.items():
        rows.append(dict(case=f"Lab trial {trial}", q=c["q_rel"], tau_h=float(-4.0 / np.log(c["q_rel"])),
                         q_star_own=c.get("q_star_own"), tau_star_h=float(-4.0 / np.log(c["q_star_own"]))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "relaxation_timescales.csv", index=False)
    print("\n=== Implied relaxation timescales tau = -dt/ln(q) ===")
    print(df.round(3).to_string(index=False))


def probe_split():
    lab = pd.read_csv(HERE / "lab_v99" / "lab_predictions_v99.csv.gz")
    lab["ae"] = (lab.pred_K - lab.y_K).abs()
    hot = ["T18", "T19", "T22", "T23"]
    lab["group"] = np.where(lab.probe.isin(hot), "hot", "ambient")
    rows = []
    for m in sorted(lab.method.unique()):
        g = lab[lab.method == m].groupby(["group", "trial"]).ae.mean().unstack("trial")
        rows.append(dict(method=m, hot_all=float(g.loc["hot"].mean()), ambient_all=float(g.loc["ambient"].mean()),
                         hot_t1=float(g.loc["hot", 1]), hot_t2=float(g.loc["hot", 2]),
                         ambient_t1=float(g.loc["ambient", 1]), ambient_t2=float(g.loc["ambient", 2])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lab_hot_ambient_split.csv", index=False)
    print("\n=== Lab hot (T18,T19,T22,T23) vs ambient probe MAE ===")
    print(df.round(4).to_string(index=False))
    per_probe = lab.groupby(["method", "probe", "trial"]).ae.mean().reset_index()
    per_probe.to_csv(OUT / "lab_per_probe_mae.csv", index=False)


if __name__ == "__main__":
    torch.set_num_threads(2)
    alpha_and_ownq()
    dev_grid()
    thresholds()
    bootstrap_cfd()
    bootstrap_lab()
    timescales()
    probe_split()
    print("\nPOST_COMPUTE_ANALYSES_COMPLETE")
