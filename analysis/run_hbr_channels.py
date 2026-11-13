"""HbR WTC of each channel for the 46 real dyads (secondary Q3 channel tests), at 5 Hz (motion.DECIMATE).

Same coherence and channel rules as run_hbr.py, without averaging over channels.

Writes results/hbr_channels.csv: film x real dyad x channel.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker; parallelism is across jobs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from data import load_films
from motion import DECIMATE
from run_motion import block_z


def one_dyad(film, dyad, group, a, b, good_a, good_b, channels, fs):
    """Per-channel WTC of one real dyad in one film."""
    return [dict(film=film, dyad=dyad, group=group, channel=f"S{channels[k][0]}_D{channels[k][1]}",
                 wtc=block_z(a[k], b[k], fs)) for k in np.flatnonzero(good_a & good_b)]


if __name__ == "__main__":
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    films = load_films("hbr")
    first = films[C.FILMS[0]]
    people, good, channels = first["people"], first["good"], first["channels"]
    fs = first["sfreq"] / DECIMATE
    where = {(p["dyad"], p["role"]): i for i, p in enumerate(people)}
    jobs = []
    for film in C.FILMS:
        x = films[film]["data"][:, :, ::DECIMATE]
        for p in people:
            if p["role"] == "A":
                a, b = where[(p["dyad"], "A")], where[(p["dyad"], "B")]
                jobs.append((film, p["dyad"], p["group"], x[a], x[b], good[a], good[b], channels, fs))
    print(f"{len(jobs)} jobs ({time.time() - start:.0f} s)", flush=True)
    out = Parallel(n_jobs=-1)(delayed(one_dyad)(*job) for job in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(C.RESULTS / "hbr_channels.csv", index=False, float_format="%.6g")
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)
