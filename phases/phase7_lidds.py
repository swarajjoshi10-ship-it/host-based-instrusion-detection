"""
Phase 7 - LID-DS 2021 evaluation (every scenario found under .\\LID-DS\\ ).

  python run_phase.py 7            or            python phases/phase7_lidds.py
  options: --demo (quick test, writes to results/phase7_demo)   --only brute   --seeds 3
Recordings are read once and cached, so a second run starts straight away.
"""
import argparse, contextlib, csv, glob, io, json, os, pickle, random, sys, time, zipfile
import numpy as np
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

LINE = "=" * 112
THIN = "-" * 112
LABELS = {"calib99": "99th pct (normal data)", "calib95": "95th pct (normal data)", "tuned_on_tuning_half": "Tuned (uses attacks)"}

def banner(demo):
    d = "phase7_demo" if demo else "phase7"
    print("\n" + LINE + "\n  PHASE 7: LID-DS 2021 EVALUATION" + (" [DEMO MODE]" if demo else "") + "\n" + LINE)
    print("Concept Analogy : A second, modern building to guard: server recordings with real attacks, to check the idea works beyond the old ADFA-LD data.")
    print("Inputs          : LID-DS 2021 recordings (zip files) for each scenario found under the LID-DS folder")
    print("Outputs         : results/" + d + "/lidds_metrics.csv, results/" + d + "/summary.md, results/" + d + "/scenario_info.json")
    print(THIN)

def ms(rows, key):
    v = [float(r[key]) for r in rows]
    return str(round(float(np.mean(v)), 3)) + "+/-" + str(round(float(np.std(v)), 3))

# ------------------------------------------------------------------ reading recordings
def find_scenarios(root):
    found = []
    for dp, dirs, _ in os.walk(root):
        if all(x in dirs for x in ("training", "validation", "test")):
            found.append(dp)
            dirs[:] = []
    return sorted(found)

def parse_recording(zf, sc_name, vocab_ids, stats):
    threads = {}
    with zf.open(sc_name) as fh:
        for raw in fh:
            p = raw.decode(errors="replace").split(None, 7)
            if len(p) < 7:
                stats["bad"] += 1
                continue
            if p[6] != ">":
                continue
            if p[5] not in vocab_ids:
                vocab_ids[p[5]] = len(vocab_ids)
            threads.setdefault(p[4], []).append(vocab_ids[p[5]])
    return [np.array(v, dtype=np.int32) for v in threads.values()]

def load_or_prepare(spath, name):
    pkl = os.path.join("results", "phase7", "lidds_" + name + ".pkl")
    os.makedirs(os.path.dirname(pkl), exist_ok=True)
    if os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            blob = pickle.load(fh)
        print("  Using cached recordings: " + pkl)
        return blob
    vocab_ids, data, stats, t0 = {}, {}, {"bad": 0}, time.perf_counter()
    for split in ("training", "validation", "test"):
        files = sorted(glob.glob(os.path.join(spath, split, "**", "*.zip"), recursive=True))
        recs = []
        for i, f in enumerate(files):
            with zipfile.ZipFile(f) as zf:
                names = zf.namelist()
                sc = [n for n in names if n.endswith(".sc")][0]
                js = [n for n in names if n.endswith(".json")][0]
                meta = json.loads(zf.read(js))
                threads = parse_recording(zf, sc, vocab_ids, stats)
            recs.append(dict(name=os.path.basename(f), exploit=bool(meta.get("exploit")), threads=threads))
            print("  reading " + split + ": " + str(i + 1) + "/" + str(len(files)) + "  (" + str(int(time.perf_counter() - t0)) + " s)", end="\r", flush=True)
        data[split] = recs
        print(" " * 70, end="\r")
    blob = dict(data=data, vocab_ids=vocab_ids, bad_lines=stats["bad"])
    with open(pkl, "wb") as fh:
        pickle.dump(blob, fh)
    print("  Read and cached " + str(sum(len(v) for v in data.values())) + " recordings in " + str(int(time.perf_counter() - t0)) + " s")
    return blob

# ------------------------------------------------------------------ windows, scores, split
def windows_for(recs, lut, V, L, S, min_len, torch):
    PAD = V + 1
    chunks, owner = [], []
    for ri, r in enumerate(recs):
        for t in r["threads"]:
            if len(t) < min_len:
                continue
            ids = lut[t]
            if len(ids) < L:
                w = np.full((1, L), PAD, dtype=np.int64)
                w[0, :len(ids)] = ids
            else:
                w = np.lib.stride_tricks.sliding_window_view(ids, L)[::S].astype(np.int64)
            chunks.append(w)
            owner.extend([ri] * len(w))
    X = np.concatenate(chunks) if chunks else np.zeros((0, L), dtype=np.int64)
    return torch.from_numpy(X), np.asarray(owner)

def rec_scores(ws, owner, n, rule):
    out = np.full(n, np.nan)
    if len(owner) == 0:
        return np.zeros(n)
    idx = np.flatnonzero(np.diff(owner)) + 1
    for g, o in zip(np.split(np.asarray(ws), idx), owner[np.r_[0, idx]]):
        if rule == "mean":
            out[o] = g.mean()
        elif rule == "max":
            out[o] = g.max()
        else:
            out[o] = np.sort(g)[-max(int(np.ceil(0.1 * len(g))), 1):].mean()
    return np.nan_to_num(out, nan=float(np.nanmin(out)))

def split_test(y, seed):
    rng = random.Random(seed)
    pos = [i for i, v in enumerate(y) if v]
    neg = [i for i, v in enumerate(y) if not v]
    rng.shuffle(pos)
    rng.shuffle(neg)
    return sorted(pos[:len(pos) // 2] + neg[:len(neg) // 2]), sorted(pos[len(pos) // 2:] + neg[len(neg) // 2:])

def trivial_baselines(train, test, y, V):
    def feats(recs):
        calls, thr, Hh = [], [], np.zeros((len(recs), V))
        for i, r in enumerate(recs):
            c = np.zeros(V)
            for t in r["threads"]:
                c += np.bincount(t, minlength=V)
            calls.append(c.sum())
            thr.append(len(r["threads"]))
            Hh[i] = c / max(c.sum(), 1)
        return np.array(calls), np.array(thr), Hh
    _, _, Ht = feats(train)
    ce, te, He = feats(test)
    sep = lambda a: max(a, 1 - a)
    m = IsolationForest(n_estimators=200, random_state=0).fit(Ht)
    return dict(length=sep(roc_auc_score(y, ce)), threads=sep(roc_auc_score(y, te)), freq_if=roc_auc_score(y, -m.decision_function(He)))

# ------------------------------------------------------------------ one scenario
def run_scenario(a, spath, P, torch):
    name = os.path.basename(spath)
    t_start = time.perf_counter()
    print("\n" + LINE + "\n  SCENARIO: " + name + "\n" + LINE)
    print("\n[Step 1/5] Reading recordings...")
    blob = load_or_prepare(spath, name)
    data = blob["data"]
    train = [r for r in data["training"] if not r["exploit"]]
    calib = [r for r in data["validation"] if not r["exploit"]]
    test = data["test"]
    y_all = np.array([1 if r["exploit"] else 0 for r in test])
    for split in ("training", "validation", "test"):
        recs = data[split]
        print("  " + split.ljust(11) + str(len(recs)).rjust(5) + " recordings | attacks: " + str(sum(r["exploit"] for r in recs)).rjust(4)
              + " | system calls (entry events): " + str(sum(len(t) for r in recs for t in r["threads"])))
    print("  distinct system call names: " + str(len(blob["vocab_ids"])))

    print("\n[Step 2/5] Protocol and windows (settings carried over from the ADFA-LD experiments, not tuned here)...")
    tune_idx, test_idx = split_test(y_all, a.split_seed)
    tune = [test[i] for i in tune_idx]
    final = [test[i] for i in test_idx]
    y_tune, y_final = y_all[tune_idx], y_all[test_idx]
    gids = np.unique(np.concatenate([t for r in train for t in r["threads"]]))
    V = len(gids)
    lut = np.full(max(blob["vocab_ids"].values()) + 1, V, dtype=np.int64)
    lut[gids] = np.arange(V)
    PAD, L, S = V + 1, a.L, max(a.L // 2, 1)
    Xtr, _ = windows_for(train, lut, V, L, S, a.min_len, torch)
    n_before = len(Xtr)
    if len(Xtr) > a.max_train_windows:
        Xtr = Xtr[torch.from_numpy(np.random.RandomState(0).choice(len(Xtr), a.max_train_windows, replace=False))]
    Xc, oc = windows_for(calib, lut, V, L, S, a.min_len, torch)
    Xt, ot = windows_for(tune, lut, V, L, S, a.min_len, torch)
    Xf, of = windows_for(final, lut, V, L, S, a.min_len, torch)
    print("  Train on " + str(len(train)) + " normal recordings | threshold from " + str(len(calib)) + " normal validation recordings")
    print("  Tuning half : " + str(len(tune)) + " recordings (" + str(int(y_tune.sum())) + " attacks) | Test half: " + str(len(final)) + " recordings (" + str(int(y_final.sum())) + " attacks)")
    print("  Windows     : vocabulary " + str(V) + " | train " + str(len(Xtr)) + (" (sampled from " + str(n_before) + ")" if n_before > len(Xtr) else "")
          + " | calibration " + str(len(Xc)) + " | tuning " + str(len(Xt)) + " | test " + str(len(Xf)))

    print("\n[Step 3/5] Sanity baselines (no neural network): how easy is this scenario?")
    sb = trivial_baselines(train, test, y_all, len(blob["vocab_ids"]))
    print("  Recording length alone         AUC " + str(round(sb["length"], 3)))
    print("  Number of threads alone        AUC " + str(round(sb["threads"], 3)))
    print("  Isolation Forest, call counts  AUC " + str(round(sb["freq_if"], 3)) + "  (no call-order information)")

    print("\n[Step 4/5] Training the LSTM (hidden " + str(a.hidden) + ", " + str(a.layers) + " layer, window " + str(L) + ", up to " + str(a.epochs) + " epochs)...")
    rules = ("mean", "max", "top10")
    store = []
    for s in range(a.seeds):
        t0 = time.perf_counter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model, ep = P.train_lstm(Xtr, V, a.hidden, a.layers, 100 * s, max_epochs=a.epochs)
        train_s = time.perf_counter() - t0
        nll = [l.split()[-1] for l in buf.getvalue().splitlines() if "NLL" in l]
        sc, st = P.window_nll(model, Xc, PAD), P.window_nll(model, Xt, PAD)
        n_thr = torch.get_num_threads()
        torch.set_num_threads(1)
        t1 = time.perf_counter()
        sf = P.window_nll(model, Xf, PAD)
        ms_win = 1000 * (time.perf_counter() - t1) / max(len(Xf), 1)
        torch.set_num_threads(n_thr)
        try:
            params = sum(p.numel() for p in model.parameters())
            bio = io.BytesIO()
            torch.save(model.state_dict(), bio)
            size_kb = len(bio.getvalue()) / 1024
        except Exception as e:
            print("  warning: could not measure model size: " + str(e))
            params, size_kb = 0, 0.0
        print("  Seed " + str(s) + ": " + str(ep) + " epochs in " + str(int(train_s)) + " s | held-out normal NLL " + (nll[-1] if nll else "n/a"))
        store.append(dict(seed=s, epochs=ep, train_s=train_s, ms=ms_win, params=params, size_kb=size_kb,
                          rec={r: dict(calib=rec_scores(sc, oc, len(calib), r), tune=rec_scores(st, ot, len(tune), r), final=rec_scores(sf, of, len(final), r)) for r in rules}))

    print("\n[Step 5/5] Evaluating on the held-out TEST half...")
    tune_auc = {r: float(np.mean([roc_auc_score(y_tune, d["rec"][r]["tune"]) for d in store])) for r in rules}
    rule = a.rule if a.rule else max(tune_auc, key=tune_auc.get)
    print("  Scoring rule chosen on the TUNING half only (AUC): " + "   ".join(r + " " + str(round(tune_auc[r], 3)) for r in rules) + "   -> using: " + rule)
    rows = []
    for d in store:
        R = d["rec"][rule]
        auc = roc_auc_score(y_final, R["final"])
        thr_tuned = P.best_f1_threshold(y_tune, R["tune"])
        for label, thr in (("calib99", np.percentile(R["calib"], 99)), ("calib95", np.percentile(R["calib"], 95)), ("tuned_on_tuning_half", thr_tuned)):
            m = P.metrics(y_final, R["final"], thr)
            rows.append(dict(scenario=name, seed=d["seed"], threshold=label, rule=rule, auc=round(auc, 4), epochs=d["epochs"], train_s=round(d["train_s"], 1),
                             ms_per_window=round(d["ms"], 4), params=d["params"], size_kb=round(d["size_kb"], 1), n_train_windows=len(Xtr),
                             n_test_recordings=len(final), n_test_attacks=int(y_final.sum()), **{k: round(v, 4) for k, v in m.items()}))
    print_results(name, rows, a.seeds)
    notes = []
    if sb["freq_if"] > 0.95:
        notes.append("This scenario is easy: a frequency-only Isolation Forest already reaches AUC " + str(round(sb["freq_if"], 3)) + ", so a high LSTM score here is not strong evidence on its own.")
    f99 = [r for r in rows if r["threshold"] == "calib99"]
    if f99 and float(np.mean([r["far"] for r in f99])) > 0.03:
        notes.append("Thresholds learned from validation recordings gave about " + str(round(100 * float(np.mean([r["far"] for r in f99])), 1)) + "% false alarms at the 99th percentile, so validation and test normals differ somewhat.")
    notes.append("The tuned threshold uses labelled attack recordings from the tuning half, so it is not a normal-data-only setting.")
    for n in notes:
        print("  Note: " + n)
    print("  Scenario time: " + str(int(time.perf_counter() - t_start)) + " s")
    info = dict(rule=rule, tune_auc={k: round(v, 4) for k, v in tune_auc.items()}, baselines={k: round(float(v), 4) for k, v in sb.items()}, notes=notes,
                protocol="train " + str(len(train)) + " | calibration " + str(len(calib)) + " | tuning half " + str(len(tune)) + " (" + str(int(y_tune.sum())) + " attacks) | test half " + str(len(final)) + " (" + str(int(y_final.sum())) + " attacks)",
                windows="vocabulary " + str(V) + ", train windows " + str(len(Xtr)) + " (cap " + str(a.max_train_windows) + ")")
    return rows, info

def print_results(name, rows, seeds):
    print("\n" + LINE)
    print("Threshold                | " + "Precision".rjust(14) + " | " + "Recall".rjust(14) + " | " + "F1 Score".rjust(14) + " | " + "FAR".rjust(14) + " | " + "ROC AUC".rjust(14) + " |  ms/win |  Params |    Size")
    print(LINE)
    for lab in ("calib99", "calib95", "tuned_on_tuning_half"):
        sub = [r for r in rows if r["threshold"] == lab]
        if not sub:
            continue
        print(LABELS[lab].ljust(24) + " | " + " | ".join(ms(sub, k).rjust(14) for k in ("precision", "recall", "f1", "far", "auc"))
              + " | " + str(round(float(np.mean([r["ms_per_window"] for r in sub])), 4)).rjust(7) + " | " + str(sub[0]["params"]).rjust(7) + " | " + (str(sub[0]["size_kb"]) + " KB").rjust(8))
    print(LINE)
    print("Scenario " + name + " | " + str(seeds) + " model seeds | mean +/- std | FAR = false alarm rate on normal recordings")

# ------------------------------------------------------------------ files
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def write_outputs(out, new_rows, new_info, mode):
    path = os.path.join(out, "lidds_metrics.csv")
    names = {r["scenario"] for r in new_rows}
    rows = [r for r in read_csv(path) if r["scenario"] not in names] + new_rows
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new_rows[0].keys()), extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    ipath = os.path.join(out, "scenario_info.json")
    info = json.load(open(ipath)) if os.path.exists(ipath) else {}
    info.update(new_info)
    json.dump(info, open(ipath, "w"), indent=1)
    lines = ["mode: " + mode, "", "# Phase 7 - LID-DS 2021 (test half, mean +/- std over model seeds)", ""]
    for name in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == name]
        inf = info.get(name, {})
        lines += ["## " + name, "", "- Scoring rule: " + str(sub[0]["rule"]) + " (chosen on the tuning half)", "- Protocol: " + inf.get("protocol", ""), "- Windows: " + inf.get("windows", ""),
                  "- Trivial baselines (AUC): " + ", ".join(k + " " + str(v) for k, v in inf.get("baselines", {}).items()), "",
                  "| Threshold | Precision | Recall | F1 | False alarm rate | AUC |", "|---|---|---|---|---|---|"]
        for lab in ("calib99", "calib95", "tuned_on_tuning_half"):
            s2 = [r for r in sub if r["threshold"] == lab]
            if s2:
                lines.append("| " + LABELS[lab] + " | " + " | ".join(ms(s2, k) for k in ("precision", "recall", "f1", "far", "auc")) + " |")
        lines += ["", "Model: " + str(sub[0]["params"]) + " parameters, " + str(sub[0]["size_kb"]) + " KB, " + str(sub[0]["ms_per_window"]) + " ms/window (1 thread)", ""]
        lines += ["- Note: " + n for n in inf.get("notes", [])] + [""]
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    return [path, os.path.join(out, "summary.md"), ipath]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--root", default="LID-DS")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--max_train_windows", type=int, default=150000)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--rule", default="", choices=["", "mean", "max", "top10"])
    a, _ = ap.parse_known_args()
    if a.demo:
        a.seeds, a.epochs, a.max_train_windows = 1, 3, 20000
    mode = "demo" if a.demo else "full"
    out = os.path.join("results", "phase7_demo" if a.demo else "phase7")
    os.makedirs(out, exist_ok=True)
    banner(a.demo)
    print("[Mode] " + mode.upper())
    print("[Sanity Check 1/2] Checking PyTorch and the Phase 6b code...")
    try:
        import torch
        import phase6b_improve as P
    except Exception as e:
        print("  [FAILED] " + str(e))
        sys.exit(1)
    print("  [PASSED] PyTorch " + str(getattr(torch, "__version__", "?")) + " detected. LSTM code from Phase 6b loaded.")
    print("[Sanity Check 2/2] Locating LID-DS scenarios under " + a.root + "...")
    scen = [s for s in find_scenarios(a.root) if a.only.lower() in os.path.basename(s).lower()]
    if not scen:
        print("  [FAILED] No scenario folders (with training / validation / test) found under " + a.root)
        sys.exit(1)
    print("  [PASSED] Found " + str(len(scen)) + " scenario(s): " + ", ".join(os.path.basename(s) for s in scen))
    all_rows, all_info = [], {}
    for s in scen:
        rows, info = run_scenario(a, s, P, torch)
        all_rows += rows
        all_info[os.path.basename(s)] = info
    files = write_outputs(out, all_rows, all_info, mode)
    print("\n" + LINE + "\nPHASE 7 COMPLETE  [mode: " + mode + "]\nFiles written:")
    for f in files:
        print("  - " + os.path.abspath(f))
    print(LINE)

if __name__ == "__main__":
    main()
