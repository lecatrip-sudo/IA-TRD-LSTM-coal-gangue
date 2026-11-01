"""V99 compute orchestrator: nested selection, final fits, variants, diagnostics.

Pre-specified selection rule (fixed before any phase-3 result was seen):
  For each outer fold, score(q, lam, alpha) is the mean over the five inner
  folds of the inner-validation MAE. The minimum score wins; configurations
  within 0.002 K of the minimum are broken in favour of (1) larger lambda_SI,
  (2) q order int > med > rms, (3) alpha closest to 0.5, then lower alpha.

Phases (restartable; state in ns_state.json):
  1 inner     270 inner-fold runs (6 outer x 5 inner x 3 q variants x 3 lambda), seed 42
  2 selection apply the rule -> ns_selection.json
  3 final     primary arm T per outer fold: reuse the V98 fits when the selected
              configuration is (int, 0.02, 0.5); recombine from stored checkpoints
              when only alpha differs; otherwise refit with v99_t_worker (seeds 42-44)
  4 variants  three-input T3 and MLP-backed TMP fits with the per-fold selected
              configuration (seeds 42-44)
  5 diag60    60-epoch trajectories for seeds 43/44 in the two diagnostic cases
  6 lab       full laboratory benchmark (Eq. 9 form primary, Eq. 10 form variant,
              own-history references) via run_lab_v99.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(r"E:\第三篇\outputs\ema_v99_revision")
BASE_REPRO = Path(r"E:\第三篇\deliverables\CFD_LSTM_revised_V95\05_Reproducibility")
V98_RUNS = BASE_REPRO / "ia_trd_runs"
V99_RUNS = BASE_REPRO / "ia_trd_runs_v99"
PYTHON = r"C:\Users\Andrew LI\AppData\Local\Programs\Python\Python311\python.exe"
CASES = ["80mm", "150mm", "300mm", "0.25kxl", "0.5kxl", "0.75kxl"]
Q_VARIANTS = ["int", "med", "rms"]
LAMS = [0.005, 0.02, 0.05]
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
TIE_EPS = 0.002
CONCURRENCY = 7
STATE = HERE / "ns_state.json"
sys.path.insert(0, str(BASE_REPRO / "new_runs" / "80mm_s42_R_a"))

sys.stdout.reconfigure(encoding="utf-8")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"phases_done": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


def run_pool(jobs, log_dir):
    """jobs: list of (name, [argv...]). Runs with bounded concurrency, restart-safe by caller."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running = []
    failures = []
    while pending or running:
        while pending and len(running) < CONCURRENCY:
            name, argv = pending.pop(0)
            log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(f"[pool] start {name} ({len(pending)} queued)", flush=True)
        time.sleep(5)
        for item in list(running):
            name, proc, log = item
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            running.remove(item)
            if rc != 0:
                failures.append(name)
                print(f"[pool] FAILED {name} rc={rc}", flush=True)
            else:
                print(f"[pool] done {name}", flush=True)
    return failures


def phase1(st):
    jobs = []
    for outer in CASES:
        for inner in [c for c in CASES if c != outer]:
            for q in Q_VARIANTS:
                for lam in LAMS:
                    name = f"{outer}__{inner}__{q}__{lam}"
                    out = HERE / "ns_inner" / f"{name}.json"
                    if out.exists():
                        try:
                            json.loads(out.read_text(encoding="utf-8"))
                            continue
                        except Exception:
                            out.unlink()
                    jobs.append((name, [PYTHON, str(HERE / "ns_worker.py"),
                                         "--outer", outer, "--inner", inner, "--q", q, "--lam", str(lam)]))
    print(f"[phase1] {len(jobs)} inner runs to do", flush=True)
    if jobs:
        failures = run_pool(jobs, HERE / "logs" / "inner")
        assert not failures, f"phase1 failures: {failures}"
    st["phases_done"].append("inner")
    save_state(st)


def phase2(st):
    records = []
    for p in sorted((HERE / "ns_inner").glob("*.json")):
        records.append(json.loads(p.read_text(encoding="utf-8")))
    assert len(records) == 270, len(records)
    df = pd.DataFrame([{
        "outer": r["outer"], "inner": r["inner"], "q": r["q_variant"], "lam": r["lam_si"],
        **{f"mae_a{a}": r["alpha_mae"][f"{a:.1f}"] for a in ALPHAS}} for r in records])
    df.to_csv(HERE / "ns_inner_scores.csv", index=False)
    selection = {}
    for outer in CASES:
        sub = df[df.outer == outer]
        scores = []
        for q in Q_VARIANTS:
            for lam in LAMS:
                for a in ALPHAS:
                    mae = float(sub[(sub.q == q) & (sub.lam == lam)][f"mae_a{a}"].mean())
                    scores.append(dict(q=q, lam=lam, alpha=a, score=mae))
        best = min(s["score"] for s in scores)
        cands = [s for s in scores if s["score"] <= best + TIE_EPS]
        cands.sort(key=lambda s: (-s["lam"], Q_VARIANTS.index(s["q"]),
                                  abs(s["alpha"] - 0.5), s["alpha"]))
        sel = cands[0]
        selection[outer] = sel
        ranked = sorted(scores, key=lambda s: s["score"])[:12]
        print(f"[phase2] {outer}: selected q={sel['q']} lam={sel['lam']} alpha={sel['alpha']} "
              f"score={sel['score']:.4f} (best={best:.4f}, ties={len(cands)})", flush=True)
        print("   top:", "; ".join(f"{r['q']}/l{r['lam']}/a{r['alpha']}={r['score']:.4f}" for r in ranked[:6]), flush=True)
    (HERE / "ns_selection.json").write_text(
        json.dumps({"rule": "min mean inner-fold MAE; ties within 0.002 K broken by larger lambda, then q order int>med>rms, then alpha closest to 0.5, then lower alpha",
                    "seed": 42, "alpha_grid": ALPHAS, "q_variants": Q_VARIANTS, "lams": LAMS,
                    "selection": selection}, indent=2), encoding="utf-8")
    st["phases_done"].append("selection")
    st["selection"] = selection
    save_state(st)


def recombine_v98(case, seed, q_val, alpha, out_dir):
    """Rebuild the Eq. (9) forecast of an existing V98 fit at a new alpha (no retraining)."""
    sys.path.insert(0, str(BASE_REPRO / "new_runs" / "80mm_s42_R_a"))
    import legacy_core as lc
    arrays = {c: dict(np.load(BASE_REPRO / "inputs/curated_inputs" / f"{c}.npz", allow_pickle=False)) for c in CASES}
    train_cases = [c for c in CASES if c != case]
    cs = lc.Scaler().fit(np.vstack([arrays[c]["XYZ"].astype(np.float32) for c in train_cases]),
                         [f"{c}/24h" for c in train_cases])
    ts = lc.Scaler().fit(np.vstack([arrays[c][f"T{h}_K"].reshape(-1, 1).astype(np.float32)
                                    for c in train_cases for h in [24, 48, 72]]),
                         [f"{c}/{h}h" for c in train_cases for h in [24, 48, 72]])
    src = V98_RUNS / f"{case}_s{seed}_T_a"
    meta = json.loads((src / "raw_metrics.json").read_text(encoding="utf-8"))
    if q_val is None:
        q_val = meta["q_int"]
    model = lc.Model()
    model.load_state_dict(torch.load(src / "model.pt", map_location="cpu"))
    model.eval()
    xyz = arrays[case]["XYZ"].astype(np.float32)
    t48 = arrays[case]["T48_K"].reshape(-1, 1).astype(np.float32)
    t72 = arrays[case]["T72_K"].reshape(-1, 1).astype(np.float32)
    seq = torch.from_numpy(np.stack([ts.transform(t48), ts.transform(t72)], axis=1))
    with torch.no_grad():
        g = model(torch.from_numpy(cs.transform(xyz)), seq).numpy().ravel().astype(np.float64)
    g_mean = max(float(np.mean(np.abs(g))), 1e-6)
    b = t72.ravel().astype(np.float64)
    d = b - t48.ravel().astype(np.float64)
    y = arrays[case]["T96_K"].ravel().astype(np.float64)
    s_d_te = float(np.mean(np.abs(d)))
    v_hat = (1.0 - alpha) * (q_val * d) + alpha * (q_val * s_d_te * g / g_mean)
    t_pred = (b + v_hat).astype(np.float32).astype(np.float64)
    m = lc.metrics(y, t_pred)
    run_id = f"{case}_s{seed}_T_recomb_a{alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(row_id=arrays[case]["row_id"].ravel(), x_m=xyz[:, 0], y_m=xyz[:, 1], z_m=xyz[:, 2],
                      T24_K=arrays[case]["T24_K"].ravel(), T48_K=t48.ravel(), T72_K=t72.ravel(),
                      T96_K=y, Tpred_K=t_pred.astype(np.float32))
                 ).to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    meta = json.loads((src / "raw_metrics.json").read_text(encoding="utf-8"))
    (out_dir / "raw_metrics.json").write_text(json.dumps(dict(
        run_id=run_id, case=case, seed=seed, arm="T", mode="T", method="IA-TRD-LSTM",
        epochs=20, alpha=alpha, lam_si=meta["lam_si"], q_variant="int", q_value=q_val,
        s_d_te=s_d_te, s_d_train_map=meta["s_d_train_map"], initial_digest=meta["initial_digest"],
        max_grad=meta["max_grad"], n_train=meta["n_train"], n_test=len(y), metrics=m,
        support_metrics=None, source="recombined from V98 checkpoint, identical network fit",
        wall_s=None), indent=2), encoding="utf-8")
    import shutil
    shutil.copy2(src / "epochs.csv", out_dir / "epochs.csv")
    return dict(run_id=run_id, metrics=m)


def phase3(st):
    selection = st["selection"]
    jobs = []
    plan = {}
    for case in CASES:
        sel = selection[case]
        reuse = (sel["q"] == "int" and sel["lam"] == 0.02 and sel["alpha"] == 0.5)
        recomb = (sel["q"] == "int" and sel["lam"] == 0.02 and sel["alpha"] != 0.5)
        plan[case] = dict(selected=sel, action="reuse" if reuse else ("recombine" if recomb else "refit"))
        if not reuse and not recomb:
            for seed in [42, 43, 44]:
                tag = "T" if (sel["q"] == "int" and sel["lam"] == 0.02) else f"T_{sel['q']}_l{sel['lam']}"
                name = f"final_{case}_s{seed}_{tag}"
                out_dir = V99_RUNS / f"{case}_s{seed}_{tag}_a"
                if not (out_dir / "raw_metrics.json").exists():
                    jobs.append((name, [PYTHON, str(HERE / "v99_t_worker.py"), "--case", case,
                                         "--seed", str(seed), "--mode", "T", "--q", sel["q"],
                                         "--lam-si", str(sel["lam"]), "--alpha", str(sel["alpha"])]))
    print(f"[phase3] plan {json.dumps(plan)}", flush=True)
    if jobs:
        failures = run_pool(jobs, HERE / "logs" / "final")
        assert not failures, f"phase3 failures: {failures}"
    # Recombine path (fast, in-process).
    for case in CASES:
        sel = selection[case]
        if plan[case]["action"] == "recombine":
            for seed in [42, 43, 44]:
                out_dir = V99_RUNS / f"{case}_s{seed}_T_recomb_a{sel['alpha']}"
                if not (out_dir / "raw_metrics.json").exists():
                    r = recombine_v98(case, seed, q_val=None, alpha=sel["alpha"], out_dir=out_dir)
                    print(f"[phase3] recombined {case} s{seed} MAE={r['metrics']['MAE_K']:.4f}", flush=True)
    (HERE / "ns_final_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    st["phases_done"].append("final")
    st["final_plan"] = plan
    save_state(st)


def phase4(st):
    selection = st["selection"]
    jobs = []
    for mode in ["T3", "TMP"]:
        for case in CASES:
            sel = selection[case]
            for seed in [42, 43, 44]:
                tag = mode if (sel["q"] == "int" and sel["lam"] == 0.02) else f"{mode}_{sel['q']}_l{sel['lam']}"
                name = f"{mode}_{case}_s{seed}"
                out_dir = V99_RUNS / f"{case}_s{seed}_{tag}_a"
                if (out_dir / "raw_metrics.json").exists():
                    continue
                jobs.append((name, [PYTHON, str(HERE / "v99_t_worker.py"), "--case", case,
                                     "--seed", str(seed), "--mode", mode, "--q", sel["q"],
                                     "--lam-si", str(sel["lam"]), "--alpha", str(sel["alpha"])]))
    print(f"[phase4] {len(jobs)} variant fits (T3/TMP)", flush=True)
    if jobs:
        failures = run_pool(jobs, HERE / "logs" / "variants")
        assert not failures, f"phase4 failures: {failures}"
    st["phases_done"].append("variants")
    save_state(st)


def phase5(st):
    selection = st["selection"]
    jobs = []
    import shutil
    for case in ["0.25kxl", "80mm"]:
        sel = selection[case]
        # Run-name tag used by v99_t_worker.py: bare mode when q=int and lam=0.02.
        tag60 = "T" if (sel["q"] == "int" and sel["lam"] == 0.02) else f"T_{sel['q']}_l{sel['lam']}"
        for seed in [42, 43, 44]:
            name = f"diag60_{case}_s{seed}"
            out_dir = V99_RUNS / f"{case}_s{seed}_{tag60}_e60"
            if (out_dir / "epoch_trajectory.csv").exists():
                continue
            if seed == 42 and sel["q"] == "int" and sel["lam"] == 0.02 and sel["alpha"] == 0.5:
                # The retained V98 60-epoch diagnostic used exactly this configuration; reuse it.
                src = V98_RUNS / f"{case}_s42_T_e60"
                if src.exists() and (src / "epoch_trajectory.csv").exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for f in ["epoch_trajectory.csv", "trajectory_summary.json"]:
                        if (src / f).exists():
                            shutil.copy2(src / f, out_dir / f)
                    print(f"[phase5] reused V98 60-epoch diagnostic for {case} seed 42", flush=True)
                    continue
            jobs.append((name, [PYTHON, str(HERE / "v99_t_worker.py"), "--case", case, "--seed", str(seed),
                                 "--mode", "T", "--q", sel["q"], "--lam-si", str(sel["lam"]),
                                 "--alpha", str(sel["alpha"]), "--epochs", "60"]))
            # record the target directory name for the aggregate
            st.setdefault("diag60_dirs", {})[f"{case}_s{seed}"] = str(out_dir)
    save_state(st)
    print(f"[phase5] {len(jobs)} diagnostic 60-epoch fits", flush=True)
    if jobs:
        failures = run_pool(jobs, HERE / "logs" / "diag60")
        assert not failures, f"phase5 failures: {failures}"
    st["phases_done"].append("diag60")
    save_state(st)


def phase6(st):
    out = HERE / "lab_v99" / "lab_predictions_v99.csv.gz"
    if not out.exists():
        rc = subprocess.run([PYTHON, str(HERE / "run_lab_v99.py")]).returncode
        assert rc == 0, "lab benchmark failed"
    st["phases_done"].append("lab")
    save_state(st)


if __name__ == "__main__":
    st = load_state()
    t0 = time.time()
    for phase, fn in [("inner", phase1), ("selection", phase2), ("final", phase3),
                      ("variants", phase4), ("diag60", phase5), ("lab", phase6)]:
        if phase in st["phases_done"]:
            print(f"[skip] {phase} already done", flush=True)
            continue
        print(f"[phase] {phase} starting at {time.time()-t0:.0f}s", flush=True)
        fn(st)
        print(f"[phase] {phase} done at {time.time()-t0:.0f}s", flush=True)
    print("ALL_PHASES_COMPLETE", flush=True)
