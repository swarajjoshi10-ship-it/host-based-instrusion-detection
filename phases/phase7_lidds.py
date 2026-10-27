"""
Phase 7 - LID-DS 2021 (one scenario at a time), reading the official zip recordings directly.

  Step 1 (once per scenario): python phase7_lidds.py --scenario "LID-DS\\Bruteforce_CWE-307\\Bruteforce_CWE-307" --step prepare
  Step 2:                     python phase7_lidds.py --scenario "LID-DS\\Bruteforce_CWE-307\\Bruteforce_CWE-307" --step run --seeds 3

Design (fixed in advance, not tuned on the test half):
  - only system call ENTER events (direction >) are used, so each call counts once
  - windows are built per thread (L=20, stride 10), settings carried over from ADFA-LD experiments
  - train on normal training recordings only; threshold from normal validation recordings
  - the test recordings are split 50/50 (balanced by attack/normal) into a TUNING half and a TEST half
  - the recording-scoring rule (mean / max / top10) is picked on the tuning half only; results are reported on the test half
"""
import argparse, csv, glob, io, json, os, pickle, random, time, zipfile
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = os.path.join("results", "phase7")

# ------------------------------------------------------------------ step 1: prepare
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
            name = p[5]
            if name not in vocab_ids:
                vocab_ids[name] = len(vocab_ids)
            threads.setdefault(p[4], []).append(vocab_ids[name])
    return [np.array(v, dtype=np.int32) for v in threads.values()]

def prepare(a):
    name = os.path.basename(os.path.normpath(a.scenario))
    os.makedirs(OUT, exist_ok=True)
    vocab_ids = {}
    data = {}
    stats = {"bad": 0}
    t0 = time.perf_counter()
    for split in ("training", "validation", "test"):
        files = sorted(glob.glob(os.path.join(a.scenario, split, "**", "*.zip"), recursive=True))
        recs = []
        for i, f in enumerate(files):
            with zipfile.ZipFile(f) as zf:
                names = zf.namelist()
                sc = [n for n in names if n.endswith(".sc")][0]
                js = [n for n in names if n.endswith(".json")][0]
                meta = json.loads(zf.read(js))
                threads = parse_recording(zf, sc, vocab_ids, stats)
            recs.append(dict(name=os.path.basename(f), exploit=bool(meta.get("exploit")), threads=threads))
            if (i + 1) % 50 == 0:
                print("  " + split + ": " + str(i + 1) + "/" + str(len(files)) + "  (" + str(int(time.perf_counter() - t0)) + " s)")
        data[split] = recs
        n_ex = sum(r["exploit"] for r in recs)
        calls = sum(len(t) for r in recs for t in r["threads"])
        n_thr = [len(r["threads"]) for r in recs]
        print("[" + split + "] recordings " + str(len(recs)) + ", exploit recordings " + str(n_ex) + ", entry calls " + str(calls)
              + ", threads per recording (mean) " + str(round(float(np.mean(n_thr)), 1)))
    path = os.path.join(OUT, "lidds_" + name + ".pkl")
    with open(path, "wb") as fh:
        pickle.dump(dict(data=data, vocab_ids=vocab_ids), fh)
    print("distinct system call names: " + str(len(vocab_ids)) + " | unparsable lines skipped: " + str(stats["bad"]))
    print("Saved " + path)

# ------------------------------------------------------------------ step 2: run
def windows_for(recs, lut, V, L, S, min_len, torch):
    PAD = V + 1
    chunks = []
    owner = []
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
    tune = pos[:len(pos) // 2] + neg[:len(neg) // 2]
    test = pos[len(pos) // 2:] + neg[len(neg) // 2:]
    return sorted(tune), sorted(test)

def run(a):
    import torch
    import phase6b_improve as P
    name = os.path.basename(os.path.normpath(a.scenario))
    with open(os.path.join(OUT, "lidds_" + name + ".pkl"), "rb") as fh:
        blob = pickle.load(fh)
    data = blob["data"]
    train = [r for r in data["training"] if not r["exploit"]]
    calib = [r for r in data["validation"] if not r["exploit"]]
    test = data["test"]
    y_all = np.array([1 if r["exploit"] else 0 for r in test])
    tune_idx, test_idx = split_test(y_all, a.split_seed)
    tune = [test[i] for i in tune_idx]
    final = [test[i] for i in test_idx]
    y_tune = y_all[tune_idx]
    y_final = y_all[test_idx]
    print("[protocol] train " + str(len(train)) + " normal recordings | calibration " + str(len(calib)) + " normal validation recordings")
    print("[protocol] tuning half: " + str(len(tune)) + " recordings (" + str(int(y_tune.sum())) + " attacks) | test half: "
          + str(len(final)) + " recordings (" + str(int(y_final.sum())) + " attacks)")

    gids = np.unique(np.concatenate([t for r in train for t in r["threads"]]))
    V = len(gids)
    lut = np.full(max(blob["vocab_ids"].values()) + 1, V, dtype=np.int64)
    lut[gids] = np.arange(V)
    PAD = V + 1
    L, S = a.L, max(a.L // 2, 1)
    Xtr, _ = windows_for(train, lut, V, L, S, a.min_len, torch)
    if len(Xtr) > a.max_train_windows:
        sel = np.random.RandomState(0).choice(len(Xtr), a.max_train_windows, replace=False)
        Xtr = Xtr[torch.from_numpy(sel)]
    Xc, oc = windows_for(calib, lut, V, L, S, a.min_len, torch)
    Xt, ot = windows_for(tune, lut, V, L, S, a.min_len, torch)
    Xf, of = windows_for(final, lut, V, L, S, a.min_len, torch)
    print("[windows] vocabulary V=" + str(V) + " | train " + str(len(Xtr)) + " (cap " + str(a.max_train_windows) + ") | calib " + str(len(Xc))
          + " | tuning " + str(len(Xt)) + " | test " + str(len(Xf)))

    rules = ("mean", "max", "top10")
    store = []
    for s in range(a.seeds):
        print("\n--- model seed " + str(s))
        t0 = time.perf_counter()
        model, ep = P.train_lstm(Xtr, V, a.hidden, a.layers, 100 * s, max_epochs=a.epochs)
        train_s = time.perf_counter() - t0
        sc = P.window_nll(model, Xc, PAD)
        st = P.window_nll(model, Xt, PAD)
        n_thr = torch.get_num_threads()
        torch.set_num_threads(1)
        t1 = time.perf_counter()
        sf = P.window_nll(model, Xf, PAD)
        ms = 1000 * (time.perf_counter() - t1) / max(len(Xf), 1)
        torch.set_num_threads(n_thr)
        try:
            params = sum(p.numel() for p in model.parameters())
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            size_kb = len(buf.getvalue()) / 1024
        except Exception as e:
            print("  warning: could not measure model size: " + str(e))
            params, size_kb = 0, 0.0
        store.append(dict(seed=s, epochs=ep, train_s=train_s, ms=ms, params=params, size_kb=size_kb,
                          rec={r: dict(calib=rec_scores(sc, oc, len(calib), r), tune=rec_scores(st, ot, len(tune), r),
                                       final=rec_scores(sf, of, len(final), r)) for r in rules}))

    print("\n[rule choice on the TUNING half only] mean AUC over seeds")
    tune_auc = {}
    for r in rules:
        tune_auc[r] = float(np.mean([roc_auc_score(y_tune, d["rec"][r]["tune"]) for d in store]))
        print("  " + r + ": " + str(round(tune_auc[r], 3)))
    rule = a.rule if a.rule else max(tune_auc, key=tune_auc.get)
    print("  -> rule used: " + rule)

    rows = []
    for d in store:
        R = d["rec"][rule]
        auc = roc_auc_score(y_final, R["final"])
        thr_tuned = P.best_f1_threshold(y_tune, R["tune"])
        for label, thr in (("calib99", np.percentile(R["calib"], 99)), ("calib95", np.percentile(R["calib"], 95)), ("tuned_on_tuning_half", thr_tuned)):
            m = P.metrics(y_final, R["final"], thr)
            rows.append(dict(scenario=name, seed=d["seed"], threshold=label, rule=rule, auc=round(auc, 4), epochs=d["epochs"],
                             train_s=round(d["train_s"], 1), ms_per_window=round(d["ms"], 4), params=d["params"], size_kb=round(d["size_kb"], 1),
                             n_train_windows=len(Xtr), n_test_recordings=len(final), n_test_attacks=int(y_final.sum()),
                             **{k: round(v, 4) for k, v in m.items()}))
    with open(os.path.join(OUT, "lidds_metrics.csv"), "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if fh.tell() == 0:
            w.writeheader()
        w.writerows(rows)

    lines = ["[LID-DS " + name + ": TEST half] mean +/- std over " + str(a.seeds) + " model seeds",
             "threshold              Prec          Recall        F1            FAR           AUC"]
    for label in ("calib99", "calib95", "tuned_on_tuning_half"):
        Rr = [r for r in rows if r["threshold"] == label]
        def f(k):
            return str(round(float(np.mean([r[k] for r in Rr])), 3)) + "+/-" + str(round(float(np.std([r[k] for r in Rr])), 3))
        lines.append(label.ljust(22) + f("precision").ljust(14) + f("recall").ljust(14) + f("f1").ljust(14) + f("far").ljust(14) + f("auc"))
    lines.append("ms/window (1 thread): " + str(round(float(np.mean([d["ms"] for d in store])), 4)) + " | params: " + str(store[0]["params"])
                 + " | size KB: " + str(round(store[0]["size_kb"], 1)))
    print("\n" + "\n".join(lines))
    with open(os.path.join(OUT, "summary.md"), "a") as fh:
        fh.write("mode: full\n\n## " + name + "\n- rule: " + rule + " (chosen on tuning half)\n- train windows: " + str(len(Xtr)) + " (cap " + str(a.max_train_windows)
                 + ")\n- LSTM hidden " + str(a.hidden) + ", layers " + str(a.layers) + ", L " + str(L) + ", max epochs " + str(a.epochs)
                 + "\n- model seeds: " + str(a.seeds) + "\n```\n" + "\n".join(lines) + "\n```\n")
    print("Saved " + os.path.join(OUT, "lidds_metrics.csv") + " and summary.md")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--step", required=True, choices=["prepare", "run"])
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--max_train_windows", type=int, default=150000)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--rule", default="", choices=["", "mean", "max", "top10"])
    a = ap.parse_args()
    prepare(a) if a.step == "prepare" else run(a)

if __name__ == "__main__":
    main()