"""
Phase 8 - ablation summary and final results package.
Every number is copied from the result files written by the earlier phases; nothing is typed in by hand.

  python run_phase.py 8            or            python phases/phase8_final_package.py
Writes: results/final/README.md, final_tables.md, ablation_auc.png, detector_comparison.png
"""
import csv, json, os, re, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OUT = os.path.join(R, "final")
LINE = "=" * 112
THIN = "-" * 112
LABELS = {"calib99": "99th pct (normal data)", "calib95": "95th pct (normal data)", "tuned_on_tuning_half": "Tuned (uses attacks)"}

def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def num(s):
    m = re.match(r"\s*(-?\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else float("nan")

def ms(rows, key):
    v = [float(r[key]) for r in rows]
    return str(round(float(np.mean(v)), 3)) + " +/- " + str(round(float(np.std(v)), 3))

def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

def show(header, rows, widths):
    print(LINE)
    print(" | ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(header, widths))))
    print(LINE)
    for r in rows:
        print(" | ".join(str(x).ljust(w) if i == 0 else str(x).rjust(w) for i, (x, w) in enumerate(zip(r, widths))))
    print(LINE)

def mode_of(path):
    if not os.path.exists(path):
        return "missing"
    txt = open(path, errors="ignore").read()
    return "full" if "mode: full" in txt else ("demo" if "mode: demo" in txt else "not recorded")

print("\n" + LINE + "\n  PHASE 8: ABLATION AND FINAL RESULTS PACKAGE\n" + LINE)
print("Concept Analogy : Binding every experiment into one report folder, like a lab notebook where each number is copied from the instrument, never typed by hand.")
print("Inputs          : result files from phases 3 to 7 and the improvement experiments (6b, 6c)")
print("Outputs         : results/final/README.md, final_tables.md, ablation_auc.png, detector_comparison.png")
print(THIN)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- sanity checks
print("\n[Sanity Check 1/2] Verifying result files from earlier phases...")
need = [("Phase 3 baselines", "phase3/baseline_metrics.csv", None), ("Phase 4 autoencoder", "phase4/autoencoder_metrics.csv", "phase4/summary.md"),
        ("Phase 5 LSTM", "phase5/lstm_metrics.csv", "phase5/summary.md"), ("Phase 6 evaluation", "phase6/comparison_table.csv", "phase6/summary.md"),
        ("Phase 6b ablation", "phase6b/experiments.csv", None), ("Phase 6b final (ADFA-LD)", "phase6b/final.csv", None),
        ("Phase 6c second look", "phase6c/rules.csv", None), ("Phase 7 LID-DS", "phase7/lidds_metrics.csv", "phase7/summary.md")]
problems = 0
for label, f, summ in need:
    rows = read(os.path.join(R, f))
    if not rows:
        optional = "6c" in label
        print("  [" + ("OPTIONAL" if optional else "MISSING ") + "] " + label.ljust(26) + f)
        problems += 0 if optional else 1
        continue
    mode = mode_of(os.path.join(R, summ)) if summ else "-"
    flag = "WARNING" if mode in ("demo", "missing", "not recorded") else "PASSED "
    problems += 1 if flag == "WARNING" else 0
    print("  [" + flag + "] " + label.ljust(26) + str(len(rows)).rjust(4) + " rows" + ("   mode: " + mode if summ else ""))
if problems:
    print("  Warning: " + str(problems) + " item(s) above need attention; the package is still built from what exists.")

exp_rows = read(os.path.join(R, "phase6b", "experiments.csv"))
last = {}
for r in exp_rows:
    last[(r["exp"], r["hidden"], r["layers"], r["L"], r["extra_train"], r["ens"], r["rule"], r["pct"])] = r
print("[Sanity Check 2/2] Ablation file: " + str(len(exp_rows)) + " rows, " + str(len(last)) + " distinct settings, " + str(len(exp_rows) - len(last)) + " repeated runs (the latest run of each setting is used).")

# ---------------------------------------------------------------- Step 1: detector comparison
print("\n[Step 1/4] ADFA-LD detector comparison (Phase 6, 5 seeds, mean +/- std)")
p6 = read(os.path.join(R, "phase6", "comparison_table.csv"))
t_detectors = md(["Detector", "Threshold pct", "Precision", "Recall", "F1", "False alarm rate", "AUC", "ms/window", "Parameters", "Size"],
                 [[r["model"], r["pct"], r["precision"], r["recall"], r["f1"], r["far"], r["auc"], r["ms_per_window"], r["params"], r["size_kb"]] for r in p6])
show(["Detector", "Pct", "Precision", "Recall", "F1 Score", "FAR", "ROC AUC", "ms/win", "Params", "Size"],
     [[r["model"], str(r["pct"]).replace(".0", ""), r["precision"], r["recall"], r["f1"], r["far"], r["auc"], r["ms_per_window"], r["params"], r["size_kb"]] for r in p6],
     [24, 4, 15, 15, 15, 15, 15, 7, 7, 10])

# ---------------------------------------------------------------- Step 2: ablation
print("\n[Step 2/4] Ablation on the tuning half (mean-window scoring, 95th percentile threshold)")
abl = [r for k, r in last.items() if k[6] == "mean" and k[7] == "95"]
abl.sort(key=lambda r: (r["exp"], int(r["hidden"]), int(r["layers"]), int(r["L"]), int(r["extra_train"]), int(r["ens"])))
names = {"A": "A model size", "B": "B window", "C": "C extra data", "E": "E ensemble"}
t_abl = md(["Exp", "Hidden", "Layers", "Window", "Extra train traces", "Ensemble", "AUC", "Precision@95", "Recall@95", "F1@95", "Train s"],
           [[r["exp"], r["hidden"], r["layers"], r["L"], r["extra_train"], r["ens"], r["auc"], r["precision"], r["recall"], r["f1"], r["train_s"]] for r in abl])
show(["Experiment", "Hidden", "Layers", "Window", "Extra", "Ens", "AUC", "F1@95", "Train s"],
     [[names.get(r["exp"], r["exp"]), r["hidden"], r["layers"], r["L"], r["extra_train"], r["ens"], r["auc"], r["f1"], r["train_s"]] for r in abl], [14, 6, 6, 6, 6, 4, 7, 7, 8])
rule_rows = sorted([r for k, r in last.items() if k[0] == "C" and k[4] == "2000" and k[7] == "95"], key=lambda r: r["rule"])
t_rules = md(["Trace scoring rule", "AUC", "F1@95"], [[r["rule"], r["auc"], r["f1"]] for r in rule_rows])

# ---------------------------------------------------------------- Step 3: final results
print("\n[Step 3/4] Final results on the held-out test halves")
fin = read(os.path.join(R, "phase6b", "final.csv"))
labels = ["calib99", "calib95", "tuned_on_tuning_half"]
fin_rows = [[LABELS[lab]] + [ms([r for r in fin if r["threshold"] == lab], k) for k in ("precision", "recall", "f1", "far", "auc")] for lab in labels if any(r["threshold"] == lab for r in fin)]
t_final = md(["Threshold", "Precision", "Recall", "F1", "False alarm rate", "AUC"], fin_rows)
print("  ADFA-LD (Phase 6b FINAL, " + str(len({r["seed"] for r in fin})) + " model seeds)")
show(["Threshold", "Precision", "Recall", "F1 Score", "FAR", "ROC AUC"], fin_rows, [24, 15, 15, 15, 15, 15])

rules6c = read(os.path.join(R, "phase6c", "rules.csv"))
t_6c = ""
if rules6c:
    seen = {}
    for r in rules6c:
        seen[(r["extra_train"], r["method"], r["pct"])] = r
    sel = [r for r in seen.values() if str(r["pct"]) in ("95", "90")]
    t_6c = md(["Extra train", "Method", "Pct", "Precision", "Recall", "F1", "False alarm rate", "AUC"], [[r["extra_train"], r["method"], r["pct"], r["precision"], r["recall"], r["f1"], r["far"], r["auc"]] for r in sel])
    print("  ADFA-LD second look (Phase 6c, 1 seed, same test half used before)")
    show(["Method", "Pct", "Precision", "Recall", "F1 Score", "FAR", "ROC AUC"], [[r["method"], r["pct"], r["precision"], r["recall"], r["f1"], r["far"], r["auc"]] for r in sel], [30, 4, 10, 10, 10, 10, 10])

lid = read(os.path.join(R, "phase7", "lidds_metrics.csv"))
info = json.load(open(os.path.join(R, "phase7", "scenario_info.json"))) if os.path.exists(os.path.join(R, "phase7", "scenario_info.json")) else {}
t_lid = ""
scen = sorted({r["scenario"] for r in lid})
for s in scen:
    sub = [r for r in lid if r["scenario"] == s]
    rws = [[LABELS[lab]] + [ms([r for r in sub if r["threshold"] == lab], k) for k in ("precision", "recall", "f1", "far", "auc")] for lab in labels]
    t_lid += "\n**" + s + "** (scoring rule " + sub[0]["rule"] + ", " + str(len({r["seed"] for r in sub})) + " model seeds, " + sub[0]["n_test_recordings"] + " test recordings, " + sub[0]["n_test_attacks"] + " attacks)\n\n"
    t_lid += md(["Threshold", "Precision", "Recall", "F1", "False alarm rate", "AUC"], rws) + "\n\nModel: " + sub[0]["params"] + " parameters, " + sub[0]["size_kb"] + " KB, " + sub[0]["ms_per_window"] + " ms/window (1 thread)\n"
    if info.get(s, {}).get("baselines"):
        t_lid += "\nTrivial baselines (AUC): " + ", ".join(k + " " + str(v) for k, v in info[s]["baselines"].items()) + "\n"
    print("  LID-DS " + s + " (" + str(len({r["seed"] for r in sub})) + " model seeds)")
    show(["Threshold", "Precision", "Recall", "F1 Score", "FAR", "ROC AUC"], rws, [24, 15, 15, 15, 15, 15])

# ---------------------------------------------------------------- honest notes (computed from the data)
def mean_of(rows, lab, key):
    sub = [float(r[key]) for r in rows if r["threshold"] == lab]
    return float(np.mean(sub)) if sub else float("nan")
notes = []
if fin:
    f95, far95, ft, farT = mean_of(fin, "calib95", "f1"), mean_of(fin, "calib95", "far"), mean_of(fin, "tuned_on_tuning_half", "f1"), mean_of(fin, "tuned_on_tuning_half", "far")
    notes.append("ADFA-LD (test half): F1 at the 95th-percentile threshold is " + str(round(f95, 3)) + " with a false alarm rate of " + str(round(far95, 3)) + (". The F1 target of 0.90 was NOT met." if f95 < 0.9 else ". The F1 target of 0.90 was met."))
    notes.append("ADFA-LD: a threshold tuned for the best F1 gives F1 " + str(round(ft, 3)) + " but a false alarm rate of " + str(round(farT, 3)) + ", which is too high for practical use, so it is not the headline result.")
for s in scen:
    sub = [r for r in lid if r["scenario"] == s]
    notes.append(s + " (test half): F1 " + str(round(mean_of(sub, "calib99", "f1"), 3)) + " at the 99th-percentile threshold (false alarm rate " + str(round(mean_of(sub, "calib99", "far"), 3)) + ") and " + str(round(mean_of(sub, "calib95", "f1"), 3)) + " at the 95th (false alarm rate " + str(round(mean_of(sub, "calib95", "far"), 3)) + "), using thresholds learned from normal data only.")
    for n in info.get(s, {}).get("notes", []):
        notes.append(s + ": " + n)
if rules6c and fin:
    delta = max(float(r["auc"]) for r in rules6c) - mean_of(fin, "calib95", "auc")
    notes.append("Phase 6c reused the same ADFA-LD test half as a second look; its best AUC differs from the 5-seed final by " + str(round(delta, 3)) + ", so the earlier 5-seed result stays the headline.")
notes.append("CPU overhead on a live host was not measured. Resource numbers are single-threaded replay measurements of inference time, memory during scoring, and serialized model size.")
notes.append("The Isolation Forest, One-Class SVM and autoencoder use frequency and hashed n-gram features, so they do not see full call order. The LSTM does.")

# ---------------------------------------------------------------- Step 4: files
print("\n[Step 4/4] Writing the report files...")
if abl:
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(abl) + 1.5))
    ax.barh(range(len(abl)), [float(r["auc"]) for r in abl], color="#1F3864")
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels([r["exp"] + ": h" + r["hidden"] + " l" + r["layers"] + " L" + r["L"] + " +" + r["extra_train"] + " ens" + r["ens"] for r in abl], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC on the tuning half (mean-window scoring)")
    ax.set_title("Ablation: model size, window, extra data, ensemble")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_auc.png"), dpi=200)
    plt.close(fig)
p95 = [r for r in p6 if str(r["pct"]).startswith("95")]
if p95:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(p95))
    ax.bar(x - 0.2, [num(r["auc"]) for r in p95], 0.4, label="AUC", color="#1F3864")
    ax.bar(x + 0.2, [num(r["f1"]) for r in p95], 0.4, label="F1 at 95th pct", color="#60A5FA")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in p95], fontsize=8, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("ADFA-LD detectors (5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detector_comparison.png"), dpi=200)
    plt.close(fig)

tables = ("# Final tables (generated from results/*.csv)\n\n## ADFA-LD detector comparison (Phase 6, 5 seeds)\n\n" + t_detectors + "\n\n## Ablation on the tuning half (Phase 6b, mean-window scoring, 95th percentile)\n\n" + t_abl
          + "\n\nScoring rules for the final configuration:\n\n" + t_rules + "\n\n## ADFA-LD final result on the test half (Phase 6b FINAL)\n\n" + t_final
          + ("\n\n## ADFA-LD second look (Phase 6c, 1 seed)\n\n" + t_6c if t_6c else "") + "\n\n## LID-DS 2021 test half (Phase 7)\n" + t_lid + "\n")
open(os.path.join(OUT, "final_tables.md"), "w").write(tables)
commands = """```
python run_phase.py 1            # data loading and verification
python run_phase.py 2            # preprocessing and features
python run_phase.py 3            # Isolation Forest and One-Class SVM
python run_phase.py 4            # lightweight autoencoder
python run_phase.py 5            # LSTM next-call baseline
python run_phase.py 6            # 5-seed evaluation and resource measurement
python phase6b_improve.py --data . --exp A
python phase6b_improve.py --data . --exp B --hidden 64 --layers 1
python phase6b_improve.py --data . --exp C --hidden 64 --layers 1 --L 20
python phase6b_improve.py --data . --exp E --hidden 64 --layers 1 --L 20 --extra_train 2000
python phase6b_improve.py --data . --exp FINAL --hidden 64 --layers 1 --L 20 --extra_train 2000 --ens 1 --rule mean --seeds 5
python phase6c_rules.py --data . --extra_train 3000
python run_phase.py 7            # LID-DS 2021, every scenario under .\\LID-DS (use --demo for a quick test)
python run_phase.py 8            # builds this folder
```"""
readme = ("# Host-Based Intrusion Detection Using System Call Analysis\n\nOperating Systems project. An anomaly-based host intrusion detector that learns normal Linux system call behaviour and flags deviations. "
          "Compared detectors: Isolation Forest, One-Class SVM, a lightweight autoencoder and an LSTM next-call model, on ADFA-LD and LID-DS 2021.\n\n"
          "## Protocol\n- Training uses normal data only. The vocabulary is built from training data only.\n- ADFA-LD: the 746 attack traces and an equal number of normal validation traces are split into a tuning half and a test half. "
          "All settings and the scoring rule were chosen on the tuning half. The test half was used for the final result.\n- LID-DS: train on normal training recordings, threshold from normal validation recordings, test recordings split into a tuning half and a test half.\n"
          "- Thresholds: 99th and 95th percentile of normal calibration scores, and one tuned for best F1 on the tuning half (this one uses labelled attacks).\n\n## How to run\n" + commands
          + "\n\n## Results\nFull tables: final_tables.md. Plots: ablation_auc.png, detector_comparison.png.\n\n### ADFA-LD final (test half)\n\n" + t_final + "\n\n### LID-DS 2021 (test half)\n" + t_lid + "\n## Honest notes\n" + "\n".join("- " + n for n in notes) + "\n")
open(os.path.join(OUT, "README.md"), "w").write(readme)

print("\n" + LINE + "\nHonest notes (computed from the data):")
for n in notes:
    print("  - " + n)
print("\n" + LINE + "\nPHASE 8 COMPLETE\nFiles written:")
for f in ("README.md", "final_tables.md", "ablation_auc.png", "detector_comparison.png"):
    if os.path.exists(os.path.join(OUT, f)):
        print("  - " + os.path.abspath(os.path.join(OUT, f)))
print(LINE)
if os.path.exists("status.py"):
    try:
        subprocess.run([sys.executable, "status.py"])
    except Exception:
        pass
