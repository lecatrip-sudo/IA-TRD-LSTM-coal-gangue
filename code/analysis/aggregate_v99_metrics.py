"""V99 aggregation: all CFD methods on the 96 h target rows, own-history references,
nested-selected arm T, three-input and architecture ablations, paired contrasts,
loss decomposition and training-history diagnostics.

Parity: metrics, weighting and baseline definitions follow aggregate_v98_metrics.py
(which follows recompute_v94.py) so that the retained arms A1, A2, R, S and the
archived aLSTM reproduce the V98 values exactly.

Reads  : V95 reproduction archive, ia_trd_runs (V98 retained fits), ia_trd_runs_v99
         (nested-selected finals, T3/TMP, 60-epoch diagnostics), cfd_refs/,
         ns_selection.json, ns_inner_scores.csv, analyses_v99/ (post-compute)
Writes : analyses_v99/*.csv
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

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
SEEDS = [42, 43, 44]
sys.stdout.reconfigure(encoding="utf-8")

REF_METHODS = ["Persistence", "Linear extrapolation", "Damped trend", "Two-coefficient local",
               "aLSTM", "Own-history mean ratio (R1)", "Own-history rowwise ratio (R2)",
               "Own-history blend (R3)"]
ROW_ORDER = ["Persistence", "Linear extrapolation", "Damped trend", "Two-coefficient local",
             "Archived aLSTM",
             "hLSTM A1", "hLSTM A2", "hLSTM R", "hLSTM S",
             "Own-history mean ratio (R1)", "Own-history rowwise ratio (R2)", "Own-history blend (R3)",
             "Relaxation component of T (no network)", "IA-TRD-LSTM (T)",
             "IA-TRD-LSTM T3 (three inputs)", "IA-TRD-LSTM TMP (feed-forward)"]
ARM_OF = {"hLSTM A1": "A1", "hLSTM A2": "A2", "hLSTM R": "R", "hLSTM S": "S",
          "IA-TRD-LSTM (T)": "T", "IA-TRD-LSTM T3 (three inputs)": "T3",
          "IA-TRD-LSTM TMP (feed-forward)": "TMP"}


def metric(y, p):
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return dict(MAE_K=float(np.mean(np.abs(e))), RMSE_K=float(np.sqrt(np.mean(e ** 2))),
                Bias_K=float(e.mean()), P95AE_K=float(np.quantile(np.abs(e), 0.95, method="linear")),
                MaxAE_K=float(np.max(np.abs(e))),
                R2=float(1.0 - np.sum(e ** 2) / np.sum((y - np.mean(y)) ** 2)))


def weights(xyz, kind):
    keys = xyz if kind == "exact" else np.floor((xyz[:, [0, 2]] + np.array([0.0, 6.5])) / 0.1 + 1e-9).astype(np.int64)
    _, inverse, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return 1.0 / count[inverse]


def plan():
    return json.loads((HERE / "ns_final_plan.json").read_text(encoding="utf-8"))


def tag_for(sel, mode="T"):
    if sel["q"] == "int" and sel["lam"] == 0.02:
        return mode
    return f"{mode}_{sel['q']}_l{sel['lam']}"


def primary_t_dir(case, seed):
    p = plan()[case]
    if p["action"] == "reuse":
        return V98_RUNS / f"{case}_s{seed}_T_a"
    if p["action"] == "recombine":
        return V99_RUNS / f"{case}_s{seed}_T_recomb_a{p['selected']['alpha']}"
    return V99_RUNS / f"{case}_s{seed}_{tag_for(p['selected'])}_a"


def variant_dir(case, seed, mode):
    return V99_RUNS / f"{case}_s{seed}_{tag_for(plan()[case]['selected'], mode)}_a"


def q_of(meta):
    """Relaxation coefficient of a fit: new runs record q_value, V98 runs q_int."""
    return float(meta["q_value"] if "q_value" in meta else meta["q_int"])


def main():
    arrays = {c: dict(np.load(BASE / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in CASES}
    rows = []
    with zipfile.ZipFile(BASE / "archived_aLSTM/archives/Supplementary_Data_1_v79_metadata_corrected.zip") as z_arch:
        for c in CASES:
            xyz = arrays[c]["XYZ"].astype(np.float64)
            a = arrays[c]["T48_K"].ravel().astype(np.float64)
            b = arrays[c]["T72_K"].ravel().astype(np.float64)
            y = arrays[c]["T96_K"].ravel().astype(np.float64)
            d = b - a
            wx, wb = weights(xyz, "exact"), weights(xyz, "bins")
            ref = np.load(REFS / f"{c}.npz")
            assert np.allclose(ref["y"], y)

            def add(method, pred, seed, arm):
                m = metric(y, pred)
                ae = np.abs(np.asarray(pred, dtype=np.float64) - y)
                rows.append(dict(method=method, arm=arm, case=c, case_label=CASE_LABELS[c], seed=seed,
                                 row_MAE_K=m["MAE_K"], exact_XYZ_MAE_K=float(np.average(ae, weights=wx)),
                                 bin_0p1m_XZ_MAE_K=float(np.average(ae, weights=wb)), **m))

            for name in REF_METHODS:
                key = f"pred_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
                assert key in ref, key
                add("Archived aLSTM" if name == "aLSTM" else name, ref[key], 0,
                    "aLSTM" if name == "aLSTM" else name)

            meta42 = json.loads((V98_RUNS / f"{c}_s42_T_a" / "raw_metrics.json").read_text(encoding="utf-8"))
            q_used = json.loads((primary_t_dir(c, 42) / "raw_metrics.json").read_text(encoding="utf-8"))
            q_val = float(q_used.get("q_value", meta42.get("q_int")))
            add("Relaxation component of T (no network)", b + q_val * d, 0, "T_relax")

            for arm, mname in [("A1", "hLSTM A1"), ("A2", "hLSTM A2"), ("R", "hLSTM R"), ("S", "hLSTM S")]:
                for seed in SEEDS:
                    if arm in ["A1", "A2"]:
                        lam = "0p02" if arm == "A1" else "0p0"
                        p_file = BASE / "historical_runs" / f"{c}_s{seed}_l{lam}_a" / "predictions.csv.gz"
                    else:
                        p_file = BASE / "new_runs" / f"{c}_s{seed}_{arm}_a" / "predictions.csv.gz"
                    pred = pd.read_csv(p_file, float_precision="round_trip")["Tpred_K"].to_numpy(dtype=np.float64)
                    add(mname, pred, seed, arm)

            for seed in SEEDS:
                for mode, mname in [("T", "IA-TRD-LSTM (T)"), ("T3", "IA-TRD-LSTM T3 (three inputs)"),
                                    ("TMP", "IA-TRD-LSTM TMP (feed-forward)")]:
                    dpath = primary_t_dir(c, seed) if mode == "T" else variant_dir(c, seed, mode)
                    pred = pd.read_csv(dpath / "predictions.csv.gz", float_precision="round_trip")["Tpred_K"].to_numpy(dtype=np.float64)
                    add(mname, pred, seed, mode)

    run_df = pd.DataFrame(rows)
    run_df.to_csv(OUT / "per_run_support_metrics.csv", index=False)
    lin_mae = float(run_df[run_df.method == "Linear extrapolation"].MAE_K.mean())
    dmp_mae = float(run_df[run_df.method == "Damped trend"].MAE_K.mean())

    # ---------------- method summary (case-equal means) ----------------
    summ = []
    for mname in ROW_ORDER:
        sub = run_df[run_df.method == mname]
        by_case = sub.groupby("case")[["MAE_K", "RMSE_K", "Bias_K", "P95AE_K", "MaxAE_K", "R2",
                                      "row_MAE_K", "exact_XYZ_MAE_K", "bin_0p1m_XZ_MAE_K"]].mean()
        by_seed = sub.groupby("seed").MAE_K.mean()
        mae = float(by_case.MAE_K.mean())
        summ.append(dict(method=mname, arm=ARM_OF.get(mname, mname), n_runs=len(sub),
                         MAE_K=mae,
                         MAE_seed_SD_K=float(by_seed.std(ddof=1)) if len(by_seed) > 1 else 0.0,
                         MAE_seed_range_K=float(by_seed.max() - by_seed.min()) if len(by_seed) > 1 else 0.0,
                         RMSE_K=float(by_case.RMSE_K.mean()), Bias_K=float(by_case.Bias_K.mean()),
                         AbsBias_K=float(abs(by_case.Bias_K.mean())),
                         P95AE_K=float(by_case.P95AE_K.mean()), MaxAE_K=float(by_case.MaxAE_K.mean()),
                         R2=float(by_case.R2.mean()),
                         skill_vs_linear_pct=100.0 * (1.0 - mae / lin_mae),
                         skill_vs_damped_pct=100.0 * (1.0 - mae / dmp_mae),
                         row_MAE_K=float(by_case.row_MAE_K.mean()),
                         exact_XYZ_MAE_K=float(by_case.exact_XYZ_MAE_K.mean()),
                         bin_0p1m_XZ_MAE_K=float(by_case.bin_0p1m_XZ_MAE_K.mean())))
    msum = pd.DataFrame(summ)
    msum.to_csv(OUT / "method_support_summary.csv", index=False)

    # ---------------- case x method ----------------
    case_rows = []
    for c in CASES:
        for mname in ROW_ORDER:
            sub = run_df[(run_df.case == c) & (run_df.method == mname)]
            case_rows.append(dict(case=c, case_label=CASE_LABELS[c], method=mname, n_runs=len(sub),
                                  MAE_mean=float(sub.MAE_K.mean()),
                                  MAE_sd=float(sub.MAE_K.std(ddof=1)) if len(sub) > 1 else 0.0,
                                  RMSE_mean=float(sub.RMSE_K.mean()), Bias_mean=float(sub.Bias_K.mean()),
                                  P95AE_mean=float(sub.P95AE_K.mean()), MaxAE_mean=float(sub.MaxAE_K.mean()),
                                  R2_mean=float(sub.R2.mean())))
    pd.DataFrame(case_rows).to_csv(OUT / "case_by_method_summary.csv", index=False)

    # ---------------- paired contrasts (18 case-seed pairs) ----------------
    contrasts = [("A1", "A2"), ("R", "A2"), ("S", "R"), ("S", "A2"), ("T", "S"), ("T", "A2"),
                 ("T", "R"), ("T3", "T"), ("T3", "R3"), ("TMP", "T"), ("T3", "TMP")]
    rows = []
    for c in CASES:
        for s in SEEDS:
            for a1, a2 in contrasts:
                # Pair every neural seed with the same deterministic R3 forecast.
                # These are algorithmic repetitions, not independent physical cases.
                def get(arm):
                    if arm == "R3":
                        ref = np.load(REFS / f"{c}.npz")
                        return metric(ref["y"], ref["pred_Own-history_blend_R3"])
                    sub = run_df[(run_df.case == c) & (run_df.seed == s) & (run_df.arm == arm)]
                    return sub.iloc[0].to_dict() if len(sub) else None
                r1, r2 = get(a1), get(a2)
                if r1 is None or r2 is None:
                    continue
                rows.append(dict(contrast=f"{a1} - {a2}", case=c, case_label=CASE_LABELS[c], seed=s,
                                 MAE_K=float(r1["MAE_K"] - r2["MAE_K"]),
                                 delta_RMSE_K=float(r1["RMSE_K"] - r2["RMSE_K"]),
                                 delta_Bias_K=float(r1["Bias_K"] - r2["Bias_K"]),
                                 delta_P95AE_K=float(r1["P95AE_K"] - r2["P95AE_K"]),
                                 delta_MaxAE_K=float(r1["MaxAE_K"] - r2["MaxAE_K"])))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "paired_contrasts.csv", index=False)

    # contrast summary: case-mean and pair counts
    cs = []
    for cname, sub in pairs.groupby("contrast"):
        case_means = sub.groupby("case").MAE_K.mean()
        cs.append(dict(contrast=cname, mean_dMAE_K=float(sub.MAE_K.mean()),
                       min_dMAE_K=float(sub.MAE_K.min()), max_dMAE_K=float(sub.MAE_K.max()),
                       cases_lower=int((case_means < 0).sum()), n_cases=len(case_means),
                       pairs_lower=int((sub.MAE_K < 0).sum()), n_pairs=len(sub),
                       mean_dMaxAE_K=float(sub.delta_MaxAE_K.mean()),
                       pairs_lower_maxae=int((sub.delta_MaxAE_K < 0).sum()),
                       pairs_lower_mae_higher_maxae=int(((sub.MAE_K < 0) & (sub.delta_MaxAE_K > 0)).sum()),
                       mean_dP95AE_K=float(sub.delta_P95AE_K.mean()),
                       mean_dBias_K=float(sub.delta_Bias_K.mean())))
    pd.DataFrame(cs).to_csv(OUT / "contrast_summary.csv", index=False)

    # ---------------- case deletion across neural arms ----------------
    del_rows = []
    for label, subset in [("All 6 cases", CASES)] + [(f"Without {CASE_LABELS[c]}", [x for x in CASES if x != c]) for c in CASES]:
        rec = {"subset": label}
        for arm in ["A1", "A2", "R", "S", "T", "T3", "TMP"]:
            rec[arm] = float(run_df[(run_df.arm == arm) & (run_df.case.isin(subset))].groupby("case").MAE_K.mean().mean())
        rec["A1_minus_A2"] = rec["A1"] - rec["A2"]
        rec["R_minus_A2"] = rec["R"] - rec["A2"]
        rec["S_minus_R"] = rec["S"] - rec["R"]
        rec["T_minus_S"] = rec["T"] - rec["S"]
        rec["T_minus_A2"] = rec["T"] - rec["A2"]
        rec["T3_minus_T"] = rec["T3"] - rec["T"]
        rec["TMP_minus_T"] = rec["TMP"] - rec["T"]
        del_rows.append(rec)
    pd.DataFrame(del_rows).to_csv(OUT / "arm_case_deletion.csv", index=False)

    # ---------------- arm T loss decomposition (nested-selected fits) ----------------
    loss_rows = []
    for c in CASES:
        for seed in SEEDS:
            dpath = primary_t_dir(c, seed)
            meta = json.loads((dpath / "raw_metrics.json").read_text(encoding="utf-8"))
            ep = pd.read_csv(dpath / "epochs.csv")
            r20 = ep.iloc[-1]
            lam = float(meta["lam_si"])
            d_term = float(r20["train_mse_normalized"])
            p_weighted = lam * float(r20["train_si_reg"])
            tot = float(r20["train_objective"])
            loss_rows.append(dict(case=c, case_label=CASE_LABELS[c], seed=seed,
                                  q_variant=meta.get("q_variant", "int"), lam_si=lam, alpha=meta["alpha"],
                                  q_value=q_of(meta), s_d_te=float(meta["s_d_te"]),
                                  data_loss_D=d_term, weighted_reg_P=p_weighted, total_loss=tot,
                                  reg_share_pct=100.0 * p_weighted / tot,
                                  g_mean_abs=float(r20["g_mean_abs"])))
    pd.DataFrame(loss_rows).to_csv(OUT / "arm_T_loss_decomposition.csv", index=False)

    # ---------------- arm T bias identity ----------------
    sys.path.insert(0, str(HERE))
    import post_compute_analyses as pca
    pca_arrays = pca.fold_arrays()
    pattern_cache = {}
    id_rows = []
    for c in CASES:
        ref = np.load(REFS / f"{c}.npz")
        y, b, d = ref["y"], ref["b"], ref["d"]
        obs = float(np.mean(np.abs(y - b)))
        for seed in SEEDS:
            meta = json.loads((primary_t_dir(c, seed) / "raw_metrics.json").read_text(encoding="utf-8"))
            q = q_of(meta)
            alpha = float(meta["alpha"])
            g = pca.held_g(pca_arrays, c, seed)
            pattern_cache[(c, seed)] = g
            sd = float(np.mean(np.abs(d)))
            signed_value = (q * ((1 - alpha) * float(np.mean(d)) +
                               alpha * sd * float(np.mean(g)) / max(float(np.mean(np.abs(g))), 1e-6))
                            - float(np.mean(y - b)))
            pred = pd.read_csv(primary_t_dir(c, seed) / "predictions.csv.gz", float_precision="round_trip")["Tpred_K"].to_numpy(dtype=np.float64)
            m = metric(y, pred)
            id_rows.append(dict(case=c, case_label=CASE_LABELS[c], seed=seed, alpha=meta["alpha"],
                                q_rel=q, q_rel_dbar_star=float(q * np.mean(np.abs(d))),
                                observed_mean_abs_change=obs,
                                identity_value=float(q * np.mean(np.abs(d)) - obs),
                                bias_K=m["Bias_K"],
                                identity_error_K=float(m["Bias_K"] - (q * np.mean(np.abs(d)) - obs)),
                                signed_identity_value_K=signed_value,
                                signed_identity_error_K=float(m["Bias_K"] - signed_value)))
    pd.DataFrame(id_rows).to_csv(OUT / "arm_T_bias_identity.csv", index=False)

    # ---------------- spatial pattern correlation ----------------
    pat_rows = []
    for c in CASES:
        ref = np.load(REFS / f"{c}.npz")
        true_inc = ref["y"] - ref["b"]
        input_inc = ref["d"]
        pat_rows.append(dict(case=c, case_label=CASE_LABELS[c], seed=0, source="input increment",
                             pearson_r=float(np.corrcoef(input_inc, true_inc)[0, 1])))
        for seed in SEEDS:
            g = pattern_cache[(c, seed)]
            pat_rows.append(dict(case=c, case_label=CASE_LABELS[c], seed=seed, source="network pattern",
                                 pearson_r=float(np.corrcoef(g, true_inc)[0, 1]),
                                 pearson_r_abs=float(np.corrcoef(np.abs(g), np.abs(true_inc))[0, 1])))
    pd.DataFrame(pat_rows).to_csv(OUT / "pattern_correlations.csv", index=False)

    # ---------------- nested selection table ----------------
    sel = json.loads((HERE / "ns_selection.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(HERE / "ns_inner_scores.csv")
    alpha_sens = pd.read_csv(OUT / "alpha_sensitivity_primaryT.csv")
    nested = []
    for c in CASES:
        s = sel["selection"][c]
        sub = scores[(scores.outer == c) & (scores.q == s["q"]) & (scores.lam == s["lam"])]
        per_alpha = {f"{a:.1f}": float(sub[f"mae_a{a:.1f}"].mean()) for a in [round(0.1 * i, 1) for i in range(11)]}
        # full candidate grid of this fold, for the tie count of the documented rule
        grid = {}
        for q_v in ["int", "med", "rms"]:
            for lam_v in sorted(scores.lam.unique()):
                s2 = scores[(scores.outer == c) & (scores.q == q_v) & (scores.lam == lam_v)]
                if not len(s2):
                    continue
                for a in [round(0.1 * i, 1) for i in range(11)]:
                    grid[(q_v, float(lam_v), a)] = float(s2[f"mae_a{a:.1f}"].mean())
        best_all = min(grid.values())
        held = run_df[(run_df.arm == "T") & (run_df.case == c)].MAE_K.mean()
        a_s = alpha_sens[alpha_sens.case == c].groupby("alpha").MAE_K.mean()
        nested.append(dict(case=c, case_label=CASE_LABELS[c], q=s["q"], lam_si=s["lam"], alpha=s["alpha"],
                           inner_score_K=s["score"], best_grid_score_K=float(best_all),
                           grid_candidates=int(len(grid)),
                           grid_ties_within_0p002=int(sum(1 for v in grid.values() if v <= best_all + 0.002)),
                           inner_score_alpha_min_K=float(min(per_alpha.values())),
                           inner_score_alpha_max_K=float(max(per_alpha.values())),
                           action=plan()[c]["action"], held_MAE_K=float(held),
                           held_MAE_at_selected_alpha_K=float(a_s.loc[s["alpha"]]),
                           held_MAE_alpha_min_K=float(a_s.min()), held_MAE_alpha_max_K=float(a_s.max()),
                           held_MAE_best_alpha=float(a_s.idxmin())))
    pd.DataFrame(nested).to_csv(OUT / "nested_selection_summary.csv", index=False)

    # ---------------- 60-epoch diagnostics ----------------
    traj_rows = []
    for c in ["0.25kxl", "80mm"]:
        for arm in ["A1", "A2", "R", "S"]:
            v94_ep = pd.read_csv(BASE / "v94_diagnostics/analysis/all480_diagnostic_epochs.csv")
            sub = v94_ep[(v94_ep.case == c) & (v94_ep.arm == arm)].sort_values("epoch")
            for _, r in sub.iterrows():
                traj_rows.append(dict(case=c, case_label=CASE_LABELS[c], arm=arm, seed=42, epoch=int(r["epoch"]),
                                      held_MAE_K=float(r["held_MAE_K"]), held_RMSE_K=float(r["held_RMSE_K"]),
                                      held_Bias_K=float(r["held_Bias_K"]),
                                      train_mse_normalized=float(r["train_mse_normalized"]),
                                      train_weighted_penalty=float(r["train_weighted_penalty"]),
                                      train_objective=float(r["train_mse_normalized"]) + float(r["train_weighted_penalty"])))
    v99_meta = {}
    for c in ["0.25kxl", "80mm"]:
        selc = sel["selection"][c]
        for seed in SEEDS:
            dpath = V99_RUNS / f"{c}_s{seed}_{tag_for(selc)}_e60"
            assert (dpath / "epoch_trajectory.csv").exists(), dpath
            ep = pd.read_csv(dpath / "epoch_trajectory.csv").sort_values("epoch")
            lam = float(selc["lam"])
            v99_meta[f"{c}_s{seed}"] = dict(q=selc["q"], lam=lam, alpha=selc["alpha"])
            for _, r in ep.iterrows():
                traj_rows.append(dict(case=c, case_label=CASE_LABELS[c], arm="T", seed=seed, epoch=int(r["epoch"]),
                                      held_MAE_K=float(r["test_MAE_K"]), held_RMSE_K=float(r["test_RMSE_K"]),
                                      held_Bias_K=float(r["test_bias_K"]),
                                      train_mse_normalized=float(r["train_mse_normalized"]),
                                      train_weighted_penalty=lam * float(r["train_si_reg"]),
                                      train_objective=float(r["train_objective"])))
    traj = pd.DataFrame(traj_rows)
    traj.to_csv(OUT / "trajectory60_epochs.csv", index=False)
    tsum = []
    for c in ["0.25kxl", "80mm"]:
        for arm in ["A1", "A2", "R", "S"]:
            sub = traj[(traj.case == c) & (traj.arm == arm) & (traj.seed == 42)].set_index("epoch")
            mn = int(sub.held_MAE_K.idxmin())
            tsum.append(dict(case=c, case_label=CASE_LABELS[c], arm=arm, seed=42,
                             mae_ep20=float(sub.loc[20, "held_MAE_K"]), mae_ep60=float(sub.loc[60, "held_MAE_K"]),
                             change=float(sub.loc[60, "held_MAE_K"] - sub.loc[20, "held_MAE_K"]),
                             min_mae=float(sub.loc[mn, "held_MAE_K"]), min_epoch=mn,
                             bias_ep20=float(sub.loc[20, "held_Bias_K"]), bias_ep60=float(sub.loc[60, "held_Bias_K"])))
        for seed in SEEDS:
            sub = traj[(traj.case == c) & (traj.arm == "T") & (traj.seed == seed)].set_index("epoch")
            mn = int(sub.held_MAE_K.idxmin())
            tsum.append(dict(case=c, case_label=CASE_LABELS[c], arm="T", seed=seed,
                             mae_ep20=float(sub.loc[20, "held_MAE_K"]), mae_ep60=float(sub.loc[60, "held_MAE_K"]),
                             change=float(sub.loc[60, "held_MAE_K"] - sub.loc[20, "held_MAE_K"]),
                             min_mae=float(sub.loc[mn, "held_MAE_K"]), min_epoch=mn,
                             bias_ep20=float(sub.loc[20, "held_Bias_K"]), bias_ep60=float(sub.loc[60, "held_Bias_K"])))
    pd.DataFrame(tsum).to_csv(OUT / "trajectory60_summary.csv", index=False)

    # ---------------- training-history MSE records (extended to V99 arm T) ----------------
    old = pd.read_csv(BASE / "figure_sources/data/all1440_epoch_training_MSE.csv")
    t_rows = []
    for c in CASES:
        for seed in SEEDS:
            dpath = primary_t_dir(c, seed)
            meta = json.loads((dpath / "raw_metrics.json").read_text(encoding="utf-8"))
            q_int = q_of(meta)
            mean_sd_tr = float(np.mean(list(meta["s_d_train_map"].values())))
            scale = q_int * mean_sd_tr
            ep = pd.read_csv(dpath / "epochs.csv").sort_values("epoch")
            for _, r in ep.iterrows():
                norm = float(r["train_mse_normalized"])
                t_rows.append(dict(case=c, seed=seed, arm="T", run_id=f"{c}_s{seed}_T_ns",
                                   epoch=int(r["epoch"]), n_train=int(r["n"]), normalized_MSE=norm,
                                   output_scale_K=scale, training_MSE_K2=norm * scale ** 2))
    pd.concat([old, pd.DataFrame(t_rows)], ignore_index=True).to_csv(OUT / "all1800_epoch_training_MSE.csv", index=False)

    # ---------------- console summary ----------------
    print("=== V99 CFD method summary (case-equal) ===")
    print(msum[["method", "MAE_K", "MAE_seed_SD_K", "RMSE_K", "Bias_K", "MaxAE_K", "skill_vs_damped_pct"]]
          .round(4).to_string(index=False))
    print("\n=== Contrast summary ===")
    print(pd.DataFrame(cs).round(4).to_string(index=False))
    print("\n=== Nested selection ===")
    print(pd.DataFrame(nested).round(4).to_string(index=False))
    print("\nAGGREGATE_V99_COMPLETE")


if __name__ == "__main__":
    main()
