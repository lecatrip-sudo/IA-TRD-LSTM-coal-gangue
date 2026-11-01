"""Build the V101 figures: F02, F08, F09 and F10 are redrawn for V101 (kelvin axis and
highlighted warm probes in F02; tick labels on all four F08 panels; seed markers without
connecting lines in F09; a single-column opaque legend in F10). F11, FS3 and FS4 are
regenerated unchanged from the retained V99 analyses, and the seven unchanged V98 figures are
copied. 600 DPI, Times New Roman, 162.99 mm width.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

WIDTH = 162.99 / 25.4
CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
LABELS = ["80 mm", "150 mm", "300 mm", "Porosity 0.25", "Porosity 0.50", "Porosity 0.75"]
SHORT = ["80 mm", "150 mm", "300 mm", "Por. 0.25", "Por. 0.50", "Por. 0.75"]
ARMS5 = ["A1", "A2", "R", "S", "T"]
SEEDS = [42, 43, 44]
COL = ["#1F4E79", "#B55D00", "#1B7837", "#6A3D9A", "#A61C1C", "#4D4D4D"]
ARM_COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#B2182B"]
MARK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", "-.", ":", "-"]
R3_COL = "#7B3294"

BASE = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
HERE = Path(r"E:\第三篇\outputs\ema_v101_revision")
DIAG = HERE / "analyses_v99"
LAB = HERE / "lab_v99"
FIG_OUT = HERE / "figures_v101"
V98_FIG = Path(r"E:\第三篇\outputs\ema_novelty_upgrade_v98\figures_v98")
V98_DELIVERED_FIG = Path(
    r"E:\第三篇\deliverables\Environmental_Modeling_Assessment_submission_V98\02_Editable_sources\Figures")
DEST_MEDIA = Path(r"E:\第三篇\deliverables\Environmental_Modeling_Assessment_submission_V101\02_Editable_sources\assets\media")
DEST_FIGS = Path(r"E:\第三篇\deliverables\Environmental_Modeling_Assessment_submission_V101\02_Editable_sources\Figures")

UNCHANGED = ["F01_CFD_and_experimental_context", "F03_CFD_temperature_histories",
             "F04_CFD_archived_temperature_fields", "F06_thickness_residual_fields",
             "F05_archived_errors_and_skill", "F07_porosity_residual_fields", "FS2_native_96h_fields"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 9.0, "axes.titlesize": 9.0, "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "axes.linewidth": 0.7,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65, "xtick.major.size": 3.0,
        "ytick.major.size": 3.0, "lines.linewidth": 1.0, "lines.markersize": 3.8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def fig(nr: int = 1, nc: int = 2, height: float = 82.0):
    f, aa = plt.subplots(nr, nc, figsize=(WIDTH, height / 25.4), squeeze=False)
    for ax in aa.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
    return f, aa


def panel(ax, letter: str, title: str = "") -> None:
    ax.set_title(title, loc="left", pad=6.5, fontweight="normal")
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=10.0, fontweight="bold", va="bottom")


def categorical(ax) -> None:
    ax.set_xticks(range(6))
    ax.set_xticklabels(SHORT, rotation=20, ha="right")
    ax.set_xlim(-0.45, 5.45)


def export(f, stem: str) -> dict:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    DEST_MEDIA.mkdir(parents=True, exist_ok=True)
    DEST_FIGS.mkdir(parents=True, exist_ok=True)
    f.canvas.draw()
    r = f.canvas.get_renderer()
    inactive = set()
    for ax in f.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            lo, hi = sorted(axis.get_view_interval())
            for minor in [False, True]:
                locs = axis.get_minorticklocs() if minor else axis.get_majorticklocs()
                labels = axis.get_minorticklabels() if minor else axis.get_majorticklabels()
                for loc, label in zip(locs, labels):
                    if not ax.axison or loc < lo - 1e-10 or loc > hi + 1e-10:
                        inactive.add(id(label))
    texts = [t for t in f.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in inactive]
    min_font = min(float(t.get_fontsize()) for t in texts)
    outside = []
    for t in texts:
        b = t.get_window_extent(renderer=r)
        if b.x0 < -0.8 or b.y0 < -0.8 or b.x1 > f.bbox.width + 0.8 or b.y1 > f.bbox.height + 0.8:
            outside.append(t.get_text())
    assert min_font >= 7.5, f"{stem} font below 7.5 pt: {min_font}"
    assert not outside, f"{stem} has text outside canvas: {outside}"
    for ext in ("png", "pdf", "svg"):
        f.savefig(FIG_OUT / f"{stem}.{ext}", dpi=600 if ext == "png" else None)
        shutil.copy2(FIG_OUT / f"{stem}.{ext}", DEST_FIGS / f"{stem}.{ext}")
    shutil.copy2(FIG_OUT / f"{stem}.png", DEST_MEDIA / f"{stem}.png")
    plt.close(f)
    return {"stem": stem, "min_font_pt": min_font, "outside_texts": len(outside)}


def build_paired_differences(runs_df: pd.DataFrame) -> pd.DataFrame:
    neural = runs_df[runs_df.arm.isin(ARMS5)].copy()
    contrasts = [("A1_minus_A2", "A1", "A2"), ("R_minus_A2", "R", "A2"), ("S_minus_R", "S", "R"),
                 ("T_minus_S", "T", "S"), ("T_minus_A2", "T", "A2")]
    rows = []
    for cname, arm1, arm2 in contrasts:
        for c in CASES:
            for s in SEEDS:
                r1 = neural[(neural.case == c) & (neural.seed == s) & (neural.arm == arm1)].iloc[0]
                r2 = neural[(neural.case == c) & (neural.seed == s) & (neural.arm == arm2)].iloc[0]
                rows.append(dict(contrast=cname, case=c, seed=s,
                                 delta_MAE_K=float(r1.MAE_K - r2.MAE_K),
                                 delta_RMSE_K=float(r1.RMSE_K - r2.RMSE_K),
                                 delta_Bias_K=float(r1.Bias_K - r2.Bias_K),
                                 delta_P95AE_K=float(r1.P95AE_K - r2.P95AE_K),
                                 delta_MaxAE_K=float(r1.MaxAE_K - r2.MaxAE_K)))
    df = pd.DataFrame(rows)
    df.to_csv(DIAG / "all90_paired_metric_differences.csv", index=False)
    return df


def main() -> None:
    style()
    manifest = {}
    runs_df = pd.read_csv(DIAG / "per_run_support_metrics.csv")
    pairs_df = build_paired_differences(runs_df)
    refs_df = pd.read_csv(HERE / "cfd_refs" / "cfd_reference_metrics.csv")
    ref_line = {c: {"Linear": float(refs_df[(refs_df.case == c) & (refs_df.method == "Linear extrapolation")].MAE_K.iloc[0]),
                    "Damped trend": float(refs_df[(refs_df.case == c) & (refs_df.method == "Damped trend")].MAE_K.iloc[0]),
                    "Own-history blend (R3)": float(refs_df[(refs_df.case == c) & (refs_df.method == "Own-history blend (R3)")].MAE_K.iloc[0])}
                for c in CASES}
    epochs60 = pd.read_csv(DIAG / "trajectory60_epochs.csv")
    epochs20 = pd.read_csv(DIAG / "all1800_epoch_training_MSE.csv")

    # ------------------------------------------------ F02 laboratory records and forecasts
    measured = pd.read_csv(BASE / "figure_sources/data/measured_records.csv")
    lab_runs = pd.read_csv(LAB / "lab_runs_v99.csv")
    lab_preds = pd.read_csv(LAB / "lab_predictions_v99.csv.gz")

    f, aa = fig(2, 2, height=122)
    f.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.93, wspace=0.28, hspace=0.50)
    probes = [f"T{i}" for i in range(17, 29)]
    warm = {"T18": ("#B2182B", "-"), "T19": ("#E08214", "-"),
            "T22": ("#2166AC", "-"), "T23": ("#1B7837", (0, (4, 1, 1, 1)))}
    ymax_meas = float(measured["temperature_C"].max() + 273.15)
    for t_idx in (1, 2):
        ax = aa[0, t_idx - 1]
        sub = measured[measured["trial"] == t_idx]
        for p in probes:
            if p in warm:
                continue
            s = sub[sub["probe"] == p].sort_values("sample_index")
            ax.plot(s["time_h"], s["temperature_C"] + 273.15, lw=0.7, color="0.68", zorder=1,
                    label="Other eight probes" if (t_idx == 1 and p == "T17") else "_nolegend_")
        for p, (col, ls_) in warm.items():
            s = sub[sub["probe"] == p].sort_values("sample_index")
            ax.plot(s["time_h"], s["temperature_C"] + 273.15, lw=1.15, color=col, ls=ls_, zorder=3,
                    label=p if t_idx == 1 else "_nolegend_")
        ax.axvline(6.0, color=".45", ls=":", lw=0.75)
        ax.set(xlim=(0, 24), ylim=(288, ymax_meas * 1.13), xticks=[0, 6, 12, 18, 24],
               xlabel="Elapsed time (h)")
        ax.set_ylabel("Temperature (K)" if t_idx == 1 else "")
        panel(ax, "a" if t_idx == 1 else "b", f"Trial {t_idx} measured records")
    handles, labels = aa[0, 0].get_legend_handles_labels()
    order = [labels.index(n) for n in ["T18", "T19", "T22", "T23", "Other eight probes"]]
    aa[0, 0].legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", ncol=3,
                    handlelength=1.5, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.3)

    ax_c = aa[1, 0]
    exp_methods = [("Persistence", "Persist."), ("Linear extrapolation", "Linear"),
                   ("Damped trend", "Damped"), ("hLSTM A1", "A1"), ("hLSTM A2", "A2"),
                   ("hLSTM R", "R"), ("hLSTM S", "S"),
                   ("Own-history blend (R3)", "R3"),
                   ("IA-TRD-LSTM Eq. (10) form (V98)", "T resid."), ("IA-TRD-LSTM (T)", "T")]
    x_pos = np.arange(len(exp_methods))
    t1, t2, mean_v = [], [], []
    for m_key, _ in exp_methods:
        sub_m = lab_runs[lab_runs["method"] == m_key]
        m1 = float(sub_m[sub_m["trial"] == 1]["MAE_K"].mean())
        m2 = float(sub_m[sub_m["trial"] == 2]["MAE_K"].mean())
        t1.append(m1)
        t2.append(m2)
        mean_v.append(0.5 * (m1 + m2))
    ax_c.plot(x_pos - 0.14, t1, ls="none", marker="o", c=COL[0], mfc="white", label="Trial 1")
    ax_c.plot(x_pos, t2, ls="none", marker="s", c=COL[1], mfc="white", label="Trial 2")
    ax_c.plot(x_pos + 0.14, mean_v, ls="none", marker="^", c=COL[4], mfc=COL[4], label="Mean")
    ax_c.axhline(mean_v[2], ls=":", lw=0.75, c=".45")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([lab for _, lab in exp_methods], rotation=40, ha="right", rotation_mode="anchor")
    ax_c.set(xlim=(-0.5, len(exp_methods) - 0.5), ylim=(0, 1.45 * max(max(t1), max(t2))),
             ylabel="4 h forecast MAE (K)")
    ax_c.legend(loc="upper left", ncol=2, handlelength=1.1, columnspacing=0.6, labelspacing=0.25)
    panel(ax_c, "c", "Cross-trial sensor errors")

    ax_d = aa[1, 1]
    lab_preds = lab_preds.copy()
    lab_preds["err_K"] = lab_preds["pred_K"] - lab_preds["y_K"]
    probe_x = np.arange(len(probes))
    d_methods = [("Persistence", "Persist.", COL[5], "o", ":"),
                 ("hLSTM S", "hLSTM S", COL[3], "s", "--"),
                 ("Damped trend", "Damped", COL[0], "^", "-."),
                 ("Own-history blend (R3)", "R3", R3_COL, "v", (0, (3, 1, 1, 1))),
                 ("IA-TRD-LSTM (T)", "IA-TRD (T)", COL[4], "D", "-")]
    for m_key, m_lab, m_col, m_mk, m_ls in d_methods:
        sub_p = lab_preds[lab_preds["method"] == m_key].groupby("probe")["err_K"].mean().reindex(probes)
        ax_d.plot(probe_x, sub_p.to_numpy(), ls=m_ls, lw=0.95, marker=m_mk, ms=3.3, c=m_col,
                  mfc="white", label=m_lab)
    ax_d.axhline(0.0, c=".5", lw=0.7)
    ax_d.set_xticks(probe_x)
    ax_d.set_xticklabels(probes, rotation=35, ha="right")
    ax_d.set(xlim=(-0.45, 11.45), ylim=(-5.6, 5.4), ylabel="Mean forecast − obs. (K)")
    ax_d.legend(loc="lower right", ncol=2, handlelength=1.25, columnspacing=0.55, labelspacing=0.25)
    panel(ax_d, "d", "Channel-wise 4 h bias (T17–T28)")
    manifest["F02_measured_temperature_records"] = export(f, "F02_measured_temperature_records")

    # ------------------------------------------------ F08 paired penalty changes
    f, aa = fig(2, 2, height=124)
    f.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.93, wspace=0.30, hspace=0.62)
    for row_idx, cname in enumerate(["A1_minus_A2", "T_minus_S"]):
        sub_p = pairs_df[pairs_df["contrast"] == cname]
        for k, seed in enumerate(SEEDS):
            v = sub_p[sub_p["seed"] == seed].set_index("case").loc[CASES]
            for col_idx, field in enumerate(["delta_MAE_K", "delta_Bias_K"]):
                ax = aa[row_idx, col_idx]
                ax.plot(np.arange(6) + (k - 1) * 0.14, v[field], ls="none", marker=MARK[k], c=COL[k],
                        mfc="white", label=f"Seed {seed}")
        for col_idx in (0, 1):
            ax = aa[row_idx, col_idx]
            categorical(ax)
            ax.axhline(0, c=".5", lw=0.7)
    for r_ in (0, 1):
        for c_ in (0, 1):
            aa[r_, c_].set_ylabel("Paired ΔMAE (K)" if c_ == 0 else "Paired Δbias (K)")
    panel(aa[0, 0], "a", "A1 − A2 error change")
    panel(aa[0, 1], "b", "A1 − A2 bias change")
    panel(aa[1, 0], "c", "T − S error change")
    panel(aa[1, 1], "d", "T − S bias change")
    aa[0, 0].legend(loc="upper left", handlelength=1.4, labelspacing=0.3)
    aa[1, 0].legend(loc="lower left", handlelength=1.4, labelspacing=0.3)
    manifest["F08_paired_penalty_changes"] = export(f, "F08_paired_penalty_changes")

    # ------------------------------------------------ F09 case comparison of five arms
    f, aa = fig(2, 3, height=132)
    f.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.94, wspace=0.37, hspace=0.48)
    neural = runs_df[runs_df.arm.isin(ARMS5)]
    for n, (c, ax) in enumerate(zip(CASES, aa.flat)):
        s = neural[neural.case == c]
        for k, seed in enumerate(SEEDS):
            yy = s[s.seed == seed].set_index("arm").loc[ARMS5, "MAE_K"]
            ax.plot(np.arange(5) + (k - 1) * 0.12, yy, color=COL[k], marker=MARK[k], ls="none", mfc="white",
                    label=f"Seed {seed}")
        ax.axhline(ref_line[c]["Linear"], lw=0.8, ls="--", c=".3", label="Linear")
        ax.axhline(ref_line[c]["Damped trend"], lw=0.8, ls=":", c=".55", label="Damped")
        ax.axhline(ref_line[c]["Own-history blend (R3)"], lw=0.9, ls=(0, (4, 1, 1, 1)), c=R3_COL, label="R3")
        ymax = max(s.MAE_K.max(), ref_line[c]["Linear"], ref_line[c]["Damped trend"], ref_line[c]["Own-history blend (R3)"])
        ax.set(xticks=range(5), xticklabels=ARMS5, xlim=(-0.3, 4.3), ylim=(0, ymax * 1.65))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel("MAE (K)" if n % 3 == 0 else "")
        panel(ax, chr(97 + n), LABELS[n])
        ax.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.5, labelspacing=0.28,
                  borderaxespad=0.35)
    manifest["F09_four_arm_case_comparison"] = export(f, "F09_four_arm_case_comparison")

    # ------------------------------------------------ F10 mean and tail changes
    f, aa = fig(2, 3, height=118)
    f.subplots_adjust(left=0.095, right=0.985, bottom=0.12, top=0.92, wspace=0.34, hspace=0.46)
    for col_idx, cname in enumerate(["R_minus_A2", "S_minus_R", "T_minus_S"]):
        pp = pairs_df[pairs_df["contrast"] == cname]
        for row_idx, yfield in enumerate(["delta_P95AE_K", "delta_MaxAE_K"]):
            ax = aa[row_idx, col_idx]
            for c, col in zip(CASES, COL):
                for seed, m in zip(SEEDS, MARK[:3]):
                    v = pp[(pp.case == c) & (pp.seed == seed)].iloc[0]
                    ax.scatter(v["delta_MAE_K"], v[yfield], s=20, marker=m, facecolors="white",
                               edgecolors=col, lw=0.9, zorder=3)
            ax.axhline(0, lw=0.7, c=".6")
            ax.axvline(0, lw=0.7, c=".6")
            if row_idx == 1:
                ax.set_xlabel("ΔMAE (K)")
            if col_idx == 0:
                ax.set_ylabel("ΔP95AE (K)" if row_idx == 0 else "ΔMaxAE (K)")
            title = {"R_minus_A2": "R − A2", "S_minus_R": "S − R", "T_minus_S": "T − S"}[cname]
            panel(ax, chr(97 + row_idx * 3 + col_idx), f"{title} ({'P95' if row_idx == 0 else 'Max'})")
    aa[0, 0].legend(handles=[Line2D([], [], ls="-", c=col, label=lab) for lab, col in zip(SHORT, COL)],
                    loc="lower right", bbox_to_anchor=(0.98, 0.29), ncol=1, handlelength=0.95,
                    labelspacing=0.16, borderaxespad=0.0, handletextpad=0.5)
    aa[0, 2].legend(handles=[Line2D([], [], ls="none", marker=m, c=".25", mfc="white", label=f"Seed {s}")
                             for s, m in zip(SEEDS, MARK[:3])],
                    loc="lower right", bbox_to_anchor=(0.88, 0.03), ncol=1, handlelength=0.9,
                    labelspacing=0.28, borderaxespad=0.0)
    manifest["F10_mean_and_tail_changes"] = export(f, "F10_mean_and_tail_changes")

    # ------------------------------------------------ F11 fixed budget diagnostic
    f, aa = fig(1, 2, height=96)
    f.subplots_adjust(left=0.10, bottom=0.18, top=0.89, right=0.985, wspace=0.29)
    styles5 = ["-", "--", "-.", ":", "-"]
    marks5 = ["o", "s", "^", "D", "v"]
    for n, c in enumerate(["0.25kxl", "80mm"]):
        ax = aa[0, n]
        selected = epochs60[epochs60.case == c]
        for arm, col, ls, mark in zip(ARMS5, ARM_COL, styles5, marks5):
            v = selected[selected.arm == arm].sort_values("epoch")
            ax.plot(v.epoch, v.held_MAE_K, c=col, ls=ls, marker=mark, markevery=5, ms=2.8, mfc="white",
                    lw=1.0, label=arm)
            endpoints = v[v.epoch.isin([20, 60])]
            ax.plot(endpoints.epoch, endpoints.held_MAE_K, c=col, marker=mark, ms=4.0, mfc="white",
                    ls="none", label="_fixed_endpoints")
        ax.axhline(ref_line[c]["Linear"], ls="--", c=".3", lw=0.8, label="Linear")
        ax.axhline(ref_line[c]["Damped trend"], ls=":", c=".55", lw=0.8, label="Damped")
        ax.axhline(ref_line[c]["Own-history blend (R3)"], ls=(0, (4, 1, 1, 1)), c=R3_COL, lw=0.9, label="R3")
        ymax = max(selected.held_MAE_K.max(), max(ref_line[c].values()))
        ymin = min(selected.held_MAE_K.min(), min(ref_line[c].values()))
        lower = 10.0 ** np.floor(np.log10(ymin))
        ax.set(xlim=(0, 62), yscale="log", ylim=(lower, ymax * 2.4), xlabel="Training epoch",
               ylabel="Held-case MAE (K)" if n == 0 else "")
        ax.set_xticks([0, 20, 40, 60])
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, pos: f"{value:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axvline(20, c=".7", ls=":", lw=0.65, label="_primary_budget")
        panel(ax, chr(97 + n), "Porosity 0.25" if n == 0 else "80 mm")
        ax.legend(loc="upper right", ncol=3, handlelength=1.35, columnspacing=0.5, labelspacing=0.28,
                  borderaxespad=0.4)
    manifest["F11_fixed_budget_diagnostic"] = export(f, "F11_fixed_budget_diagnostic")

    # ------------------------------------------------ FS3 increment and penalty cost
    f, aa = fig(1, 2, height=76)
    f.subplots_adjust(left=0.10, bottom=0.21, top=0.89, right=0.985, wspace=0.31)
    p_persist = refs_df[refs_df.method == "Persistence"].set_index("case").MAE_K
    for j, (cname, ylab, title, ylims, offsets) in enumerate([
        ("A1_minus_A2", "Seed-mean ΔMAE (A1 − A2, K)", "Unscaled penalty contrast",
         (-1.55, 2.15), [(6, 4), (6, -3), (6, -10), (6, 4), (-38, 6), (-42, -13)]),
        ("T_minus_S", "Seed-mean ΔMAE (T − S, K)", "Activity-normalized contrast",
         (-6.2, 0.85), [(5, -15), (-38, 4), (-38, -11), (6, 4), (-16, 6), (-32, -13)]),
    ]):
        ax = aa[0, j]
        c_means = pairs_df[pairs_df.contrast == cname].groupby("case").delta_MAE_K.mean()
        for i, c in enumerate(CASES):
            xval = float(p_persist.loc[c])
            yval = float(c_means.loc[c])
            ax.scatter(xval, yval, marker=MARK[i % 3], s=25, facecolors="white",
                       edgecolors=COL[0 if i < 3 else 1], lw=0.95)
            ax.annotate(SHORT[i], (xval, yval), xytext=offsets[i], textcoords="offset points", fontsize=8.0)
        ax.axhline(0, lw=0.7, c=".6")
        ax.set(xlim=(0.8, 7.8), ylim=ylims, xlabel="Persistence MAE at 96 h (K)", ylabel=ylab)
        panel(ax, chr(97 + j), title)
    manifest["FS3_increment_and_penalty_cost"] = export(f, "FS3_increment_and_penalty_cost")

    # ------------------------------------------------ FS4 training MSE across arms
    f, aa = fig(2, 3, height=106)
    f.subplots_adjust(left=0.095, right=0.985, bottom=0.12, top=0.93, wspace=0.38, hspace=0.54)
    for n, (c, ax) in enumerate(zip(CASES, aa.flat)):
        for k, arm in enumerate(ARMS5):
            v = epochs20[(epochs20.case == c) & (epochs20.arm == arm)].groupby("epoch")["training_MSE_K2"].agg(["mean", "min", "max"])
            ax.fill_between(v.index, v["min"], v["max"], color=ARM_COL[k], alpha=0.11, lw=0)
            ax.plot(v.index, v["mean"], color=ARM_COL[k], ls=styles5[k], marker=marks5[k],
                    markevery=[0, 4, 9, 14, 19], mfc="white", label=arm)
        ax.set(yscale="log", xlim=(1, 20), xticks=[1, 5, 10, 15, 20], xlabel="Epoch")
        ax.set_ylabel("Batch MSE (K²)" if n % 3 == 0 else "")
        panel(ax, chr(97 + n), LABELS[n])
        ax.legend(loc="upper right", ncol=2, handlelength=1.35, columnspacing=0.55, labelspacing=0.25)
    manifest["FS4_four_arm_training_MSE"] = export(f, "FS4_four_arm_training_MSE")

    # ------------------------------------------------ unchanged figures
    for stem in UNCHANGED:
        for ext in ("png", "pdf", "svg"):
            src = V98_DELIVERED_FIG / f"{stem}.{ext}"
            assert src.exists(), f"retained V98 figure missing: {src}"
            shutil.copy2(src, FIG_OUT / f"{stem}.{ext}")
            shutil.copy2(src, DEST_FIGS / f"{stem}.{ext}")
        shutil.copy2(V98_DELIVERED_FIG / f"{stem}.png", DEST_MEDIA / f"{stem}.png")
        manifest[stem] = {"stem": stem, "source": "V98 retained"}

    (FIG_OUT / "V101_FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("FIGURES_V101_COMPLETE")


if __name__ == "__main__":
    main()
