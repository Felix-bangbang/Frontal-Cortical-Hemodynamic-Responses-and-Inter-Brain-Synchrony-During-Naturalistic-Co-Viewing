"""Window-level WTC and phase for every A x B pair, per film, at 5 Hz (motion.DECIMATE).

For HbO or HbR (first command-line argument, default "hbo"):
  windows    channel-averaged Fisher z WTC in each 50-s window (1-s steps), for every A x B pair;
             the boundary-trimming analyses (stats_boundary.py) average only the windows that lie
             fully inside the retained part of the film
  phase      channel-averaged phase measures per pair, as in run_wtc.py (coherence.phase_summary)

Writes results/windows_<chromophore>.npz (per film: A x B x windows, rows and columns in dyad
order, so the diagonal is the real dyads) and results/phase_pairs_<chromophore>.csv.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker; parallelism is across jobs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from coherence import phase_summary, wavelet_coherence, window_z
from data import load_films
from motion import DECIMATE

MEASURES = ["resultant_x", "resultant_y", "in_phase", "anti_phase", "a_leads", "b_leads", "lag_s"]


def pair_windows(x, y, good_x, good_y, fs):
    """Channel-averaged window WTC and phase summary for one pair (None if no shared channel)."""
    use = np.flatnonzero(good_x & good_y)
    zs, phases = [], []
    for k in use:
        R2, phase, period = wavelet_coherence(x[k], y[k], fs)
        zs.append(window_z(R2, fs))
        phases.append(phase_summary(R2, phase, period))
    mean = {m: np.mean([p[m] for p in phases]) for m in MEASURES}
    rx, ry = mean.pop("resultant_x"), mean.pop("resultant_y")
    return np.nanmean(zs, axis=0), dict(phase_consistency=np.hypot(rx, ry), mean_phase=np.arctan2(ry, rx), **mean)


def one_person_vs_all(film, a, x_a, good_a, b_people, x_b, good_b, fs):
    rows, wins = [], []
    for j, b in enumerate(b_people):
        w, ph = pair_windows(x_a, x_b[j], good_a, good_b[j], fs)
        wins.append(w.astype(np.float32))
        real = a["dyad"] == b["dyad"]
        rows.append(dict(film=film, a_dyad=a["dyad"], b_dyad=b["dyad"], kind="real" if real else "pseudo",
                         group=a["group"] if real else "pseudo", wtc=float(np.nanmean(w)), **ph))
    return rows, np.array(wins)


if __name__ == "__main__":
    chromophore = sys.argv[1] if len(sys.argv) > 1 else "hbo"
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    films = load_films(chromophore)
    first = films[C.FILMS[0]]
    people, good, fs = first["people"], first["good"], first["sfreq"] / DECIMATE
    order = sorted(range(len(people)), key=lambda i: (people[i]["dyad"], people[i]["role"]))
    a_idx = [i for i in order if people[i]["role"] == "A"]           # dyad order
    b_idx = [i for i in order if people[i]["role"] == "B"]           # same dyad order
    jobs = []
    for film in C.FILMS:
        x = films[film]["data"][:, :, ::DECIMATE]
        jobs += [(film, people[i], x[i], good[i], [people[j] for j in b_idx], x[b_idx], good[b_idx], fs) for i in a_idx]
    print(f"{chromophore}: {len(jobs)} jobs ({time.time() - start:.0f} s)", flush=True)
    out = Parallel(n_jobs=-1, verbose=5)(delayed(one_person_vs_all)(*job) for job in jobs)
    n_a = len(a_idx)
    arrays = {film: np.stack([w for _, w in out[k * n_a:(k + 1) * n_a]]) for k, film in enumerate(C.FILMS)}
    np.savez_compressed(C.RESULTS / f"windows_{chromophore}.npz", dyads=np.array([people[i]["dyad"] for i in a_idx]),
                        groups=np.array([people[i]["group"] for i in a_idx]), fs=fs, **arrays)
    pd.DataFrame([r for rows, _ in out for r in rows]).to_csv(C.RESULTS / f"phase_pairs_{chromophore}.csv",
                                                              index=False, float_format="%.6g")
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)
