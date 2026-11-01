"""Independent verification of the V101 numbers.

Two modes, selected automatically:

* work-directory mode (this file in outputs/ema_v101_revision): recomputes each headline
  number from the retained run directories and checks that the delivered Markdown prints
  the same value.
* archive mode (this file copied to verification/ of the public release): reads only
  data/ inside the archive, verifies prediction-only file completeness, and checks the
  published per-run to aggregate metric arithmetic. It cannot rescore CFD errors because
  the source targets are not redistributed.

Exits non-zero on any mismatch.
"""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
CL = {"80mm": "80 mm", "150mm": "150 mm", "300mm": "300 mm", "0.25kxl": "Porosity 0.25",
      "0.5kxl": "Porosity 0.50", "0.75kxl": "Porosity 0.75"}
SEEDS = [42, 43, 44]
sys.stdout.reconfigure(encoding="utf-8")

FAILS = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    if not ok:
        FAILS.append(f"{name}: {detail}")


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def _number(cell):
    """Leading number of a markdown cell, tolerating U+2212 minus and a '± sd' suffix."""
    text = re.sub(r"\u2212", "-", cell).strip()
    m = re.match(r"[-+]?\d+(?:\.\d+)?", text)
    if m is None:
        return None
    return float(m.group(0))


def md_tables(text):
    """Yield (label, number, body) for every caption-delimited markdown table."""
    blocks, label, num, cur = [], None, None, []
    for line in text.splitlines():
        m = re.match(r"^\*\*Table (S?)(\d+)\.", line)
        if m:
            if num is not None:
                blocks.append((label, num, "\n".join(cur)))
            label, num, cur = ("TableS" if m.group(1) else "Table"), int(m.group(2)), []
            continue
        if num is None:
            continue
        if line.startswith("|"):
            cur.append(line)
        elif line.strip():
            blocks.append((label, num, "\n".join(cur)))
            label, num, cur = None, None, []
    if num is not None:
        blocks.append((label, num, "\n".join(cur)))
    return blocks


def md_value(text, row_label, column_index=1, table_number=None, supplementary=False):
    """Read a numeric cell from the row labelled row_label.

    Method names label rows in several tables of the manuscript, so the search is
    restricted to one table whenever table_number is given. Supplementary tables
    are addressed with supplementary=True.
    """
    want = "TableS" if supplementary else "Table"
    for label, num, body in md_tables(text):
        if table_number is not None and (num != int(table_number) or label != want):
            continue
        for line in body.splitlines():
            if line.startswith("|") and line.split("|")[1].strip().strip("*") == row_label:
                cells = [c.strip().strip("*") for c in line.split("|")[1:-1]]
                if column_index >= len(cells):
                    return None
                return _number(cells[column_index])
    return None


def citation_order(text):
    body = text.split("## References", 1)[0]
    body = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", body, flags=re.S)
    order = []
    for match in re.finditer(r"\[([0-9,\s]+)\]", body):
        if re.match(r"\s*(?:→|h\b)", body[match.end():]):
            continue
        for number in re.findall(r"\d+", match.group(1)):
            n = int(number)
            if n not in order:
                order.append(n)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--manuscript", type=Path, default=None)
    ap.add_argument("--esm", type=Path, default=None)
    a = ap.parse_args()
    root = a.root.resolve()
    # In the public release this script lives in verification/, while summary/
    # and predictions/ live one directory above it.
    if not (root / "data" / "summary").exists() and (root.parent / "data" / "summary").exists():
        root = root.parent
    archive = (root / "data" / "summary").exists()
    print(f"mode={'archive' if archive else 'workdir'} root={root}")

    if archive:
        summary = root / "data" / "summary"
        preds = root / "data" / "predictions" / "primary"
        lab = root / "data" / "laboratory"
        seldir = root / "data" / "selection"
    else:
        summary = root / "analyses_v99"
        preds = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility\ia_trd_runs_v99")
        lab = root / "lab_v99"
        seldir = root

    msum = pd.read_csv(summary / "method_support_summary.csv").set_index("method")
    cbm = pd.read_csv(summary / "case_by_method_summary.csv")
    cm = cbm.pivot_table(index="case", columns="method", values="MAE_mean")
    ident = pd.read_csv(summary / "arm_T_bias_identity.csv")
    loss = pd.read_csv(summary / "arm_T_loss_decomposition.csv")
    nested = pd.read_csv(summary / "nested_selection_summary.csv")
    thr = pd.read_csv(summary / "threshold_assessment.csv")
    plan = json.loads((seldir / "ns_final_plan.json").read_text(encoding="utf-8"))

    # ---- 1. recompute the primary T aggregate from the primary run files
    def t_pred_file(case, seed):
        p = plan[case]
        sel = p["selected"]
        if p["action"] == "reuse":
            name = f"{case}_s{seed}_T_a"
        elif p["action"] == "recombine":
            name = f"{case}_s{seed}_T_recomb_a{sel['alpha']}"
        else:
            tag = "T" if (sel["q"] == "int" and sel["lam"] == 0.02) else f"T_{sel['q']}_l{sel['lam']}"
            name = f"{case}_s{seed}_{tag}_a"
        if archive:
            return preds / f"{name}.csv.gz"
        for base in [preds, Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility\ia_trd_runs")]:
            if (base / name / "predictions.csv.gz").exists():
                return base / name / "predictions.csv.gz"
        raise FileNotFoundError(name)

    if archive:
        per_run = pd.read_csv(summary / "per_run_support_metrics.csv")
        t_runs = per_run[per_run.arm == "T"].copy()
        case_equal = float(t_runs.groupby("case").MAE_K.mean().mean())
        check("T case-equal MAE agrees with released per-run summaries",
              close(case_equal, msum.loc["IA-TRD-LSTM (T)", "MAE_K"], 5e-4),
              f"{case_equal:.4f} vs {msum.loc['IA-TRD-LSTM (T)', 'MAE_K']:.4f}")
        method_errors = []
        for method, rec in per_run.groupby("method"):
            reported = msum.loc[method, "MAE_K"]
            recombined = float(rec.groupby("case").MAE_K.mean().mean())
            method_errors.append(abs(recombined - float(reported)))
        check("released per-run to method-summary MAE arithmetic",
              len(method_errors) == len(msum) and max(method_errors, default=float("inf")) <= 5e-4,
              f"{len(method_errors)} methods; max error {max(method_errors, default=float('inf')):.6f} K")
        check("18 per-run T metric records are present", len(t_runs) == len(CASES) * len(SEEDS),
              f"{len(t_runs)} rows")
        prediction_checks = []
        for case in CASES:
            for seed in SEEDS:
                path = t_pred_file(case, seed)
                frame = pd.read_csv(path)
                sidecar = Path(str(path).replace(".csv.gz", ".json"))
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                row = t_runs[(t_runs.case == case) & (t_runs.seed == seed)]
                values_ok = (list(frame.columns) == ["row_id", "Tpred_K"] and
                             len(frame) == meta.get("n_test") and
                             np.isfinite(frame.Tpred_K.to_numpy(dtype=np.float64)).all() and
                             len(row) == 1 and
                             close(meta["metrics"]["MAE_K"], row.iloc[0].MAE_K, 5e-4))
                prediction_checks.append(bool(values_ok))
        check("prediction-only T files are complete, target-free and match run metadata",
              all(prediction_checks), f"{sum(prediction_checks)}/{len(prediction_checks)} files")
        t_recomputed = case_equal
    else:
        per_case = []
        for case in CASES:
            maes = []
            for seed in SEEDS:
                df = pd.read_csv(t_pred_file(case, seed))
                maes.append(float(np.mean(np.abs(df.Tpred_K.to_numpy(dtype=np.float64) -
                                              df.T96_K.to_numpy(dtype=np.float64)))))
            per_case.append(float(np.mean(maes)))
        t_recomputed = float(np.mean(per_case))
        check("T case-equal MAE recomputed from author-held predictions and targets",
              close(t_recomputed, msum.loc["IA-TRD-LSTM (T)", "MAE_K"], 5e-4),
              f"{t_recomputed:.4f} vs {msum.loc['IA-TRD-LSTM (T)', 'MAE_K']:.4f}")

    # ---- 2. bias identity
    ident_max = float(ident.identity_error_K.abs().max())
    signed_identity_max = float(ident.signed_identity_error_K.abs().max())
    check("exact signed bias identity agrees up to float32 forecast rounding",
          signed_identity_max < 6e-5, f"{signed_identity_max:.8f} K")
    check("positive-increment approximation residual is finite",
          np.isfinite(ident_max), f"{ident_max:.4f} K (reported, not assumed zero)")

    # ---- 3. selection rule respected
    for _, r in nested.iterrows():
        ok = (r.held_MAE_at_selected_alpha_K <= r.held_MAE_alpha_max_K + 1e-9)
        check(f"alpha sensitivity bracket {r.case}", ok, f"selected {r.held_MAE_at_selected_alpha_K:.4f}")

    # ---- 4. loss shares
    share = 100.0 * loss.weighted_reg_P / loss.total_loss
    check("arm-T regularizer share within [0,100]", bool(((share >= 0) & (share <= 100)).all()),
          f"{share.min():.1f}-{share.max():.1f}%")
    check("arm-T loss tables complete", len(loss) == 18, f"{len(loss)} rows")

    # ---- 5. thresholds monotone sanity: persistence misses more than T at 318.15 K all rows
    def miss(m, scope="all rows"):
        return float(thr[(thr.domain == "CFD") & (thr.method == m) & (thr.scope == scope) &
                         (np.isclose(thr.threshold_K, 318.15))].miss_rate.mean())
    check("threshold assessment present", len(thr) > 0, f"{len(thr)} rows")
    check("T miss rate below persistence miss rate", miss("IA-TRD-LSTM (T)") < miss("Persistence"),
          f"{miss('IA-TRD-LSTM (T)'):.4f} < {miss('Persistence'):.4f}")

    # ---- 6. bootstrap interval ordering
    bc = pd.read_csv(summary / "bootstrap_cfd_summary.csv", index_col=0)
    check("bootstrap T interval brackets its point estimate",
          bc.loc["IA-TRD-LSTM (T)", "ci2.5"] <= msum.loc["IA-TRD-LSTM (T)", "MAE_K"] <=
          bc.loc["IA-TRD-LSTM (T)", "ci97.5"],
          f"[{bc.loc['IA-TRD-LSTM (T)', 'ci2.5']:.4f}, {bc.loc['IA-TRD-LSTM (T)', 'ci97.5']:.4f}]")
    for method in ["Persistence", "Linear extrapolation", "Damped trend",
                   "Own-history blend (R3)", "IA-TRD-LSTM (T)"]:
        check(f"CFD bootstrap estimand matches reported per-run MAE: {method}",
              close(bc.loc[method, "point"], msum.loc[method, "MAE_K"], 6e-4),
              f"{bc.loc[method, 'point']:.4f} vs {msum.loc[method, 'MAE_K']:.4f}")

    # ---- 7. laboratory: trial-equal T and damped from the released tables
    labr = pd.read_csv(lab / "lab_runs_v99.csv")
    lab_t = labr[labr.method == "IA-TRD-LSTM (T)"].groupby("trial").MAE_K.mean().mean()
    lab_d = labr[labr.method == "Damped trend"].groupby("trial").MAE_K.mean().mean()
    lab_predictions = pd.read_csv(lab / "lab_predictions_v99.csv.gz")
    lab_predictions["ae"] = (lab_predictions.pred_K - lab_predictions.y_K).abs()
    for method, reported in [("IA-TRD-LSTM (T)", lab_t), ("Damped trend", lab_d)]:
        recomputed = (lab_predictions[lab_predictions.method == method]
                      .groupby(["trial", "seed"]).ae.mean().groupby("trial").mean().mean())
        check(f"laboratory {method} summary matches row-level rescoring",
              close(recomputed, reported, 1e-8), f"{recomputed:.6f} vs {reported:.6f}")
    check("reported laboratory failure against damped trend agrees with results",
          lab_t > lab_d, f"{lab_t:.4f} vs {lab_d:.4f}")
    bl = pd.read_csv(summary / "bootstrap_lab_summary.csv", index_col=0)
    for method in ["Damped trend", "Own-history blend (R3)", "IA-TRD-LSTM (T)",
                   "IA-TRD-LSTM Eq. (10) form (V98)"]:
        reported = labr[labr.method == method].groupby("trial").MAE_K.mean().mean()
        check(f"laboratory bootstrap estimand matches per-run MAE: {method}",
              close(bl.loc[method, "point"], reported, 6e-4),
              f"{bl.loc[method, 'point']:.4f} vs {reported:.4f}")

    # ---- 8. delivered-text agreement (only when the Markdown is provided or present)
    ms = a.manuscript
    if ms is None:
        for cand in [root.parent.parent / "deliverables/Environmental_Modeling_Assessment_submission_V101/02_Editable_sources/01_Manuscript_EMA_V101.md",
                     root.parent / "Environmental_Modeling_Assessment_submission_V101/02_Editable_sources/01_Manuscript_EMA_V101.md",
                     root / "deliverables/Environmental_Modeling_Assessment_submission_V101/02_Editable_sources/01_Manuscript_EMA_V101.md"]:
            if cand.exists():
                ms = cand
                break
    if ms is not None and Path(ms).exists():
        text = Path(ms).read_text(encoding="utf-8")
        check("no unresolved template token in manuscript", "{{" not in text and "@@" not in text)
        check("no overstatement phrase 'corrects'", "corrects" not in text.lower())
        check("no unqualified held-case-selection claim",
              "not selected by held-case results" not in text.lower())
        printed = md_value(text, "IA-TRD-LSTM (T)", 1, table_number=3)
        check("Table 3 T MAE matches recomputation", printed is not None and close(printed, t_recomputed, 6e-4),
              f"printed {printed} vs {t_recomputed:.4f}")
        printed_r3 = md_value(text, "Own-history blend (R3)", 1, table_number=3)
        check("Table 3 R3 MAE matches summary", printed_r3 is not None and
              close(printed_r3, msum.loc["Own-history blend (R3)", "MAE_K"], 6e-4),
              f"printed {printed_r3}")
        refs = re.findall(r"^\d+\\\.\s+", text[text.index("## References"):], re.M)
        check("reference list has 28 entries", len(refs) == 28, f"{len(refs)}")
        cited = citation_order(text)
        check("every cited reference number exists", all(1 <= c <= 28 for c in cited), str(cited))
        check("references numbered by actual first citation", cited == list(range(1, 29)), str(cited))
        check("own-history earliest field identity stated correctly", "$b-d-d_0$" in text)
        figs = re.findall(r"^!\[Figure (\d+)\]\(([^)]+)\)", text, re.M)
        check("ten main figures embedded", len(figs) == 10, f"{len(figs)}")
        check("figure numbers are 1..10", [int(f[0]) for f in figs] == list(range(1, 11)), str([f[0] for f in figs]))
        for _, path in figs:
            check(f"figure file {path}", (Path(ms).parent / path).exists())
        tabs = re.findall(r"^\*\*Table (\d+)\.", text, re.M)
        check("main tables 1..13 present", [int(t) for t in tabs] == list(range(1, 14)), str(tabs))
        lab_printed = md_value(text, "IA-TRD-LSTM (T)", 3, table_number=11)
        check("lab number printed in Table 11 matches recomputation",
              lab_printed is not None and close(lab_printed, lab_t, 6e-4),
              f"printed {lab_printed}; recomputed {lab_t:.4f}")

    es = a.esm
    if es is not None and Path(es).exists():
        etext = Path(es).read_text(encoding="utf-8")
        check("no unresolved template token in ESM", "{{" not in etext and "@@" not in etext)
        check("no 'corrects' overstatement in ESM", "corrects" not in etext.lower())
        etabs = [int(t) for t in re.findall(r"^\*\*Table S(\d+)\.", etext, re.M)]
        check("ESM tables S1..S32 present", etabs == list(range(1, 33)), str(etabs))
        enotes = [int(t) for t in re.findall(r"^## Note S(\d+) ", etext, re.M)]
        check("ESM notes S1..S14 present", enotes == list(range(1, 15)), str(enotes))

    report = {"mode": "archive" if archive else "workdir", "checks": CHECKS,
              "failures": FAILS, "n_checks": len(CHECKS), "all_pass": not FAILS}
    out = (root / "verification_report.json") if archive else (root / "checks_v101" / "VERIFY_CLAIMS.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for c in CHECKS:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
    print(json.dumps({"n_checks": len(CHECKS), "failures": FAILS}))
    if FAILS:
        sys.exit(1)
    print("VERIFY_CLAIMS_V101_COMPLETE")


if __name__ == "__main__":
    main()
