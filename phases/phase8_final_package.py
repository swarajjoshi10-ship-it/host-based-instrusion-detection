"""
Phase 8 - builds the final results package from the CSV files that the earlier phases really wrote.
Nothing here is typed in by hand: every number comes from results/*.csv.
Run:  python phase8_final_package.py
Writes: results/final/final_tables.md, ablation_auc.png, detector_comparison.png, README.md
"""
import csv, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OUT = os.path.join(R, "final")
os.makedirs(OUT, exist_ok=True)

def read(path):
    if not os.path.exists(path):
        print("warning: missing " + path)
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def num(s):
    m = re.match(r"\s*(-?\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else float("nan")

def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

def mean_std(rows, key):
    v = [float(r[key]) for r in rows]
    return str(round(float(np.mean(v)), 3)) + " +/- " + str(round(float(np.std(v)), 3))

# ---------------------------------------------------------------- Phase 6: detector comparison (ADFA-LD, 5 seeds)
p6 = read(os.path.join(R, "phase6", "comparison_table.csv"))
t_detectors = md(["Detector", "Threshold pct", "Precision", "Recall", "F1", "False alarm rate", "AUC", "ms/window", "Parameters", "Size"],
                 [[r["model"], r["pct"], r["precision"], r["recall"], r["f1"], r["far"], r["auc"], r["ms_per_window"], r["params"], r["size_kb"]] for r in p6])

# ---------------------------------------------------------------- Phase 6b: ablation (tuning half, last run of each setting wins)
exp_rows = read(os.path.join(R, "phase6b", "experiments.csv"))
last = {}
for r in exp_rows:
    last[(r["exp"], r["hidden"], r["layers"], r["L"], r["extra_train"], r["ens"], r["rule"], r["pct"])] = r
abl = []
for k, r in last.items():
    if k[6] == "mean" and k[7] == "95":
        abl.append(r)
abl.sort(key=lambda r: (r["exp"], int(r["hidden"]), int(r["layers"]), int(r["L"]), int(r["extra_train"]), int(r["ens"])))
t_abl = md(["Exp", "Hidden", "Layers", "Window", "Extra train traces", "Ensemble", "AUC", "Precision@95", "Recall@95", "F1@95", "Train s"],
           [[r["exp"], r["hidden"], r["layers"], r["L"], r["extra_train"], r["ens"], r["auc"], r["precision"], r["recall"], r["f1"], r["train_s"]] for r in abl])
rule_rows = [r for k, r in last.items() if k[0] == "C" and k[4] == "2000" and k[7] == "95"]
t_rules = md(["Trace scoring rule", "AUC", "F1@95"], [[r["rule"], r["auc"], r["f1"]] for r in sorted(rule_rows, key=lambda r: r["rule"])])

# ---------------------------------------------------------------- Phase 6b FINAL (ADFA-LD test half) and Phase 7 (LID-DS test half)
fin = read(os.path.join(R, "phase6b", "final.csv"))
labels = ["calib99", "calib95", "tuned_on_tuning_half"]
t_final = md(["Threshold", "Precision", "Recall", "F1", "False alarm rate", "AUC"],
             [[lab] + [mean_std([r for r in fin if r["threshold"] == lab], k) for k in ("precision", "recall", "f1", "far", "auc")]
              for lab in labels if any(r["threshold"] == lab for r in fin)])
lid = read(os.path.join(R, "phase7", "lidds_metrics.csv"))
t_lid = ""
scen = sorted({r["scenario"] for r in lid})
for s in scen:
    sub = [r for r in lid if r["scenario"] == s]
    t_lid += "\n**" + s + "** (scoring rule " + sub[0]["rule"] + ", " + str(len({r["seed"] for r in sub})) + " model seeds, " + sub[0]["n_test_recordings"] + " test recordings, " + sub[0]["n_test_attacks"] + " attacks)\n\n"
    t_lid += md(["Threshold", "Precision", "Recall", "F1", "False alarm rate", "AUC"],
                [[lab] + [mean_std([r for r in sub if r["threshold"] == lab], k) for k in ("precision", "recall", "f1", "far", "auc")] for lab in labels]) + "\n"
    t_lid += "\nModel: " + sub[0]["params"] + " parameters, " + sub[0]["size_kb"] + " KB, " + sub[0]["ms_per_window"] + " ms/window (1 thread)\n"

# ---------------------------------------------------------------- target check, computed from the data
def best(rows, lab):
    sub = [r for r in rows if r["threshold"] == lab]
    return float(np.mean([float(r["f1"]) for r in sub])) if sub else float("nan"), float(np.mean([float(r["far"]) for r in sub])) if sub else float("nan")
notes = []
if fin:
    f95, far95 = best(fin, "calib95")
    notes.append("ADFA-LD (test half): F1 at the 95th-percentile threshold is " + str(round(f95, 3)) + " with a false alarm rate of " + str(round(far95, 3))
                 + (". The F1 target of 0.90 was NOT met." if f95 < 0.9 else ". The F1 target of 0.90 was met."))
    ft, farT = best(fin, "tuned_on_tuning_half")
    notes.append("ADFA-LD: a threshold tuned for the best F1 gives F1 " + str(round(ft, 3)) + " but a false alarm rate of " + str(round(farT, 3))
                 + ", which is too high for practical use, so it is not used as the headline result.")
for s in scen:
    sub = [r for r in lid if r["scenario"] == s]
    f95, far95 = best(sub, "calib95")
    notes.append(s + " (test half): F1 " + str(round(f95, 3)) + " at the 95th-percentile threshold, false alarm rate " + str(round(far95, 3))
                 + ". Check results/phase7 and the sanity check (phase7_sanity.py) before quoting this.")
notes.append("CPU overhead on a live host was not measured. Resource numbers are single-threaded replay measurements of inference time, memory during scoring, and serialized model size.")
notes.append("The Isolation Forest, One-Class SVM and autoencoder use frequency and hashed n-gram features, so they do not see full call order. The LSTM does.")

# ---------------------------------------------------------------- plots
if abl:
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(abl) + 1.5))
    names = [r["exp"] + ": h" + r["hidden"] + " l" + r["layers"] + " L" + r["L"] + " +" + r["extra_train"] + " ens" + r["ens"] for r in abl]
    ax.barh(range(len(abl)), [float(r["auc"]) for r in abl], color="#1F3864")
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC on the tuning half (mean-window scoring)")
    ax.set_title("Ablation: effect of model size, window, extra data, ensemble")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_auc.png"), dpi=200)
    plt.close(fig)
if p6:
    p95 = [r for r in p6 if str(r["pct"]).startswith("95")]
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

# ---------------------------------------------------------------- write tables + README
tables = ("# Final tables (generated from results/*.csv)\n\n## ADFA-LD detector comparison (Phase 6, 5 seeds)\n\n" + t_detectors
          + "\n\n## Ablation on the tuning half (Phase 6b, mean-window scoring, 95th percentile)\n\n" + t_abl + "\n\nScoring rules for the final configuration:\n\n" + t_rules
          + "\n\n## ADFA-LD final result on the test half (Phase 6b FINAL)\n\n" + t_final + "\n\n## LID-DS 2021 test half (Phase 7)\n" + t_lid + "\n")
with open(os.path.join(OUT, "final_tables.md"), "w") as fh:
    fh.write(tables)

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
python phase7_lidds.py --scenario ".\\LID-DS\\Bruteforce_CWE-307\\Bruteforce_CWE-307" --step prepare
python phase7_lidds.py --scenario ".\\LID-DS\\Bruteforce_CWE-307\\Bruteforce_CWE-307" --step run --seeds 3
python phase7_sanity.py Bruteforce_CWE-307
python phase8_final_package.py   # builds this folder
```"""
readme = ("# Host-Based Intrusion Detection Using System Call Analysis\n\nOperating Systems project. An anomaly-based host intrusion detector that learns normal Linux system call behaviour "
          "and flags deviations. Compared detectors: Isolation Forest, One-Class SVM, a lightweight autoencoder and an LSTM next-call model, on ADFA-LD and LID-DS 2021.\n\n"
          "## Protocol\n- Training uses normal data only. The vocabulary is built from training data only.\n- ADFA-LD: the 746 attack traces and an equal number of normal validation traces are split into a tuning half and a test half. "
          "All settings and the scoring rule were chosen on the tuning half. The test half was used once for the final result.\n- LID-DS: train on normal training recordings, threshold from normal validation recordings, "
          "test recordings split into a tuning half and a test half.\n- Thresholds: 99th and 95th percentile of normal calibration scores, and one tuned for best F1 on the tuning half.\n\n"
          "## How to run\n" + commands + "\n\n## Results\nFull tables: final_tables.md. Plots: ablation_auc.png, detector_comparison.png.\n\n### ADFA-LD final (test half)\n\n" + t_final
          + "\n\n### LID-DS 2021 (test half)\n" + t_lid + "\n## Honest notes\n" + "\n".join("- " + n for n in notes) + "\n")
with open(os.path.join(OUT, "README.md"), "w") as fh:
    fh.write(readme)
print("Wrote results/final/README.md, final_tables.md, ablation_auc.png, detector_comparison.png")
for n in notes:
    print("- " + n)