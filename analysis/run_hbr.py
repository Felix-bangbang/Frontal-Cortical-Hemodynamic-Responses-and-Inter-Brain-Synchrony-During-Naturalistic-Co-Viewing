"""Real vs pseudo WTC for HbR, unchanged and with each participant's own motion regressed out.

Same as run_motion.py but for deoxygenated haemoglobin (HbR), at 5 Hz (motion.DECIMATE):
  hbr      WTC of the HbR signals, unchanged, for all 46 x 46 A x B pairs
  cleaned  WTC after regressing each participant's own motion regressors out of each HbR channel;
           missing (NaN) for pairs involving a dyad with a flat accelerometer

Writes results/hbr_pairs.csv: film x A x B.
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
from motion import DECIMATE, prepare_motion, remove_motion
from run_motion import block_z


def one_person_vs_all(film, a, raw_a, clean_a, good_a, b_people, raw_b, clean_b, good_b, fs):
    """All pairs of one A participant in one film: WTC of HbR, unchanged and motion-cleaned."""
    rows = []
    for j, b in enumerate(b_people):
        use = np.flatnonzero(good_a & good_b[j])
        cleanable = clean_a is not None and clean_b[j] is not None
        real = a["dyad"] == b["dyad"]
        rows.append(dict(film=film, a_dyad=a["dyad"], b_dyad=b["dyad"], kind="real" if real else "pseudo",
                         group=a["group"] if real else "pseudo",
                         hbr=np.mean([block_z(raw_a[k], raw_b[j, k], fs) for k in use]),
                         cleaned=np.mean([block_z(clean_a[k], clean_b[j][k], fs) for k in use]) if cleanable else np.nan))
    return rows


if __name__ == "__main__":
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    films = load_films("hbr")
    m = prepare_motion(films)
    first = films[C.FILMS[0]]
    people, good, fs = first["people"], first["good"], first["sfreq"] / DECIMATE
    kept = {p["id"]: k for k, p in enumerate(m["info"])}                # participants with motion regressors
    print("flat sensors:", m["flat"], flush=True)
    is_a = np.array([p["role"] == "A" for p in people])
    a_idx, b_idx = np.flatnonzero(is_a), np.flatnonzero(~is_a)
    jobs = []
    for film in C.FILMS:
        raw = films[film]["data"][:, :, ::DECIMATE]
        clean = [remove_motion(raw[i], m["reg"][film][kept[p["id"]]])[0] if p["id"] in kept else None
                 for i, p in enumerate(people)]
        b_people = [people[j] for j in b_idx]
        jobs += [(film, people[i], raw[i], clean[i], good[i], b_people, raw[b_idx], [clean[j] for j in b_idx],
                  good[b_idx], fs) for i in a_idx]
    print(f"{len(jobs)} jobs ({time.time() - start:.0f} s)", flush=True)
    out = Parallel(n_jobs=-1, verbose=5)(delayed(one_person_vs_all)(*job) for job in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(C.RESULTS / "hbr_pairs.csv", index=False, float_format="%.6g")
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)
