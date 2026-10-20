"""
Shared utilities and functions for all HIDS phases.
Reuses verified implementations from hids_m1_m2.py, run_all.py, and diagnose.py.
"""
# ── Threading must be pinned BEFORE numpy/sklearn import ───────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS",  "1")
os.environ.setdefault("MKL_NUM_THREADS",  "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# ───────────────────────────────────────────────────────────────────────────────
import io, pickle, time, random, tracemalloc
from pathlib import Path
import numpy as np
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ── Banner ─────────────────────────────────────────────────────────────────────
def print_banner(phase_num, title, analogy, inputs, outputs):
    print("\n" + "=" * 80)
    print(f"  PHASE {phase_num}: {title.upper()}")
    print("=" * 80)
    print(f"Concept Analogy : {analogy}")
    print(f"Inputs          : {inputs}")
    print(f"Outputs         : {outputs}")
    print("-" * 80 + "\n")


# ── Model size ─────────────────────────────────────────────────────────────────
def sklearn_serialized_kb(model):
    """Serialized size of any sklearn/MLP model via pickle (bytes -> KB)."""
    return len(pickle.dumps(model, protocol=4)) / 1024.0


def torch_serialized_kb(model):
    """Serialized size of a PyTorch model's state_dict via BytesIO (bytes -> KB)."""
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


def torch_param_count(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


# ── Inference profiling ────────────────────────────────────────────────────────
def profile_inference(score_fn, n_windows):
    """
    Run score_fn() and return:
      wall_s          – wall-clock seconds
      cpu_time_s      – process CPU time consumed (user + sys, single-threaded)
      rss_before_mb   – RSS of whole process before scoring
      inference_mem_mb– RSS delta (after - before), i.e. memory taken by the
                        scoring operation; also capped at 0 if RSS shrank.
      scores          – whatever score_fn returned

    cpu_time_per_1000_windows_s = cpu_time_s / n_windows * 1000
    """
    proc = psutil.Process()
    # Warm the CPU-time counter so the interval read is accurate
    _ = proc.cpu_times()

    rss_before_mb = proc.memory_info().rss / 1e6

    t_cpu_before = sum(proc.cpu_times()[:2])   # user + sys
    t_wall_before = time.perf_counter()

    scores = score_fn()

    t_wall_after = time.perf_counter()
    t_cpu_after = sum(proc.cpu_times()[:2])
    rss_after_mb = proc.memory_info().rss / 1e6

    wall_s = t_wall_after - t_wall_before
    cpu_time_s = t_cpu_after - t_cpu_before
    inference_mem_mb = max(0.0, rss_after_mb - rss_before_mb)
    cpu_time_per_1000 = (cpu_time_s / max(n_windows, 1)) * 1000.0

    return {
        "wall_s": wall_s,
        "cpu_time_s": cpu_time_s,
        "cpu_time_per_1000_windows_s": cpu_time_per_1000,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "inference_mem_mb": inference_mem_mb,
        "scores": scores,
    }


# ── Dataset discovery ──────────────────────────────────────────────────────────
def find_dirs(root):
    """Find dataset subdirectories matching train, valid, attack."""
    found = {}
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        low = name.lower()
        if "train" in low:
            found["train"] = p
        elif "valid" in low:
            found["val"] = p
        elif "attack" in low:
            found["attack"] = p
    missing = {"train", "val", "attack"} - set(found)
    if missing:
        raise SystemExit(f"Could not find folders for: {missing} inside {root}")
    return found


def read_traces(folder):
    """Read space-separated syscall number sequences from .txt files."""
    traces = []
    for dp, _, files in os.walk(folder):
        for f in sorted(files):
            if not f.lower().endswith(".txt"):
                continue
            with open(os.path.join(dp, f), "r", errors="ignore") as fh:
                toks = fh.read().split()
            seq = [int(t) for t in toks if t.isdigit()]
            if seq:
                traces.append(seq)
    return traces


def get_trace_stats(traces):
    """Compute basic distribution statistics on trace lengths."""
    lengths = np.array([len(t) for t in traces])
    return {
        "count": len(traces),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p10": float(np.percentile(lengths, 10)),
        "p90": float(np.percentile(lengths, 90)),
    }


# ── Windowing ──────────────────────────────────────────────────────────────────
def windows(seq, L, S):
    """Slice a sequence into fixed-size windows with given stride."""
    if len(seq) <= L:
        return [seq]
    return [seq[i : i + L] for i in range(0, len(seq) - L + 1, S)]


def trace_windows(traces, L, S):
    """Window all traces and retain index mapping back to original trace."""
    wins, owner = [], []
    for i, t in enumerate(traces):
        w = windows(t, L, S)
        wins += w
        owner += [i] * len(w)
    return wins, np.array(owner)


def build_vocabulary(train_traces):
    """Build vocabulary of unique system calls from training data only."""
    unique_calls = sorted({c for t in train_traces for c in t})
    vocab = {c: k for k, c in enumerate(unique_calls)}
    return vocab, len(vocab)


# ── Featurization ──────────────────────────────────────────────────────────────
def featurize(wins, vocab, V, B=256, B_tri=256, use_trigrams=False, normalize=True):
    """
    Featurize windows:
      - Syscall frequency histogram (V+1 bins; index V = unknown syscall)
      - Hashed bigram counts (B bins)
      - Optional hashed trigram counts (B_tri bins)
    If normalize=True, each block is L1-normalised (sums to 1.0).
    If normalize=False, raw integer counts (float32) are preserved.
    """
    D = V + 1 + B + (B_tri if use_trigrams else 0)
    X = np.zeros((len(wins), D), dtype=np.float32)
    tri_offset = V + 1 + B
    for r, w in enumerate(wins):
        ids = [vocab.get(c, V) for c in w]
        for c in ids:
            X[r, c] += 1
        for a, b in zip(ids[:-1], ids[1:]):
            X[r, V + 1 + ((a * 31 + b) % B)] += 1
        if use_trigrams and len(ids) >= 3:
            for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
                X[r, tri_offset + (((a * 31 + b) * 31 + c) % B_tri)] += 1
        if normalize:
            s1 = X[r, :V + 1].sum()
            s2 = X[r, V + 1 : tri_offset].sum()
            if s1:
                X[r, :V + 1] /= s1
            if s2:
                X[r, V + 1 : tri_offset] /= s2
            if use_trigrams:
                s3 = X[r, tri_offset:].sum()
                if s3:
                    X[r, tri_offset:] /= s3
    return X


# ── Protocol split ─────────────────────────────────────────────────────────────
def split_adfa_ld(val_traces, attack_traces, seed=0):
    """
    Standard ADFA-LD split protocol:
    - Test set: all attack traces + equal number of normal validation traces.
    - Calibration set: remaining normal validation traces.
    """
    rng = random.Random(seed)
    idx = list(range(len(val_traces)))
    rng.shuffle(idx)
    n_test_norm = min(len(attack_traces), len(val_traces) // 2)
    test_norm = [val_traces[i] for i in idx[:n_test_norm]]
    calib = [val_traces[i] for i in idx[n_test_norm:]]
    test_traces = test_norm + attack_traces
    y_test = np.array([0] * len(test_norm) + [1] * len(attack_traces))
    return calib, test_traces, y_test


# ── Score aggregation ──────────────────────────────────────────────────────────
def trace_scores(win_scores, owner, n_traces, method="mean"):
    """Aggregate window anomaly scores to trace level (default: mean rule)."""
    if method == "mean":
        sums = np.zeros(n_traces)
        cnt = np.zeros(n_traces)
        for s, o in zip(win_scores, owner):
            sums[o] += s
            cnt[o] += 1
        return sums / np.maximum(cnt, 1)
    elif method == "max":
        scores = np.full(n_traces, -np.inf)
        for s, o in zip(win_scores, owner):
            if s > scores[o]:
                scores[o] = s
        scores[scores == -np.inf] = 0.0
        return scores
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, scores, calib_scores, percentiles=[99.0, 95.0]):
    """Compute precision, recall, F1, false alarm rate, and AUC at threshold percentiles."""
    auc = roc_auc_score(y_true, scores)
    out = {"auc": auc}
    for pct in percentiles:
        thr = np.percentile(calib_scores, pct)
        pred = (scores > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred)
        f = f1_score(y_true, pred)
        far = fp / max(fp + tn, 1)
        out[pct] = {
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "far": float(far),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
    return out


# ── File utilities ─────────────────────────────────────────────────────────────
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
