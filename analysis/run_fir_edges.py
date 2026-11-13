"""FIR-edge sensitivity: Comedy WTC for every A x B pair with the filter edges treated differently.

The HbO inside the Comedy block is kept and band-passed with the main analysis filter, with the
samples outside the block replaced by a reflected or a constant extension (boundary.py). WTC is
then computed per 50-s window at 5 Hz as in run_windows.py.

Writes results/windows_fir_<edge>.npz (Comedy: A x B x windows, dyad order) for edge in
reflect, constant.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker; parallelism is across jobs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import numpy as np
from joblib import Parallel, delayed

import config as C
from boundary import refilter, unfiltered
from data import list_participants
from motion import DECIMATE
from run_windows import pair_windows

FILM = "Comedy"
EDGES = ["reflect", "constant"]


def a_row(xa, good_a, xb, good_b, fs):
    return np.array([pair_windows(xa, xb[j], good_a, good_b[j], fs)[0].astype(np.float32) for j in range(len(xb))])


if __name__ == "__main__":
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    people = Parallel(n_jobs=-1)(delayed(unfiltered)(p) for p in list_participants())
    people.sort(key=lambda p: (p["dyad"], p["role"]))
    n = min(int(round(p["onsets"][FILM][1] * p["sfreq"])) for p in people)     # common length, as in data.py
    fs = people[0]["sfreq"] / DECIMATE
    good = np.array([p["good"] for p in people])
    a_idx = [i for i, p in enumerate(people) if p["role"] == "A"]
    b_idx = [i for i, p in enumerate(people) if p["role"] == "B"]
    print(f"loaded ({time.time() - start:.0f} s)", flush=True)
    for edge in EDGES:
        x = np.array([refilter(p["data"], int(round(p["onsets"][FILM][0] * p["sfreq"])), n, p["sfreq"], edge)
                      for p in people])[:, :, ::DECIMATE]
        x[~good] = np.nan
        rows = Parallel(n_jobs=-1)(delayed(a_row)(x[i], good[i], x[b_idx], good[b_idx], fs) for i in a_idx)
        np.savez_compressed(C.RESULTS / f"windows_fir_{edge}.npz", dyads=np.array([people[i]["dyad"] for i in a_idx]),
                            fs=fs, **{FILM: np.stack(rows)})
        print(f"{edge} done ({(time.time() - start) / 60:.1f} min)", flush=True)
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)
