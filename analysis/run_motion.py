"""Motion control: do the synchrony results survive when head motion is accounted for?

Works on the data at 5 Hz (motion.DECIMATE). Dyads containing a flat accelerometer are left out.

For every A x B pair and film (as in run_wtc.py) computes wavelet coherence of:
  hbo         the HbO signals, unchanged
  cleaned     HbO after regressing each participant's own motion regressors out of each channel
  motion      each of the five motion signals between the two people, averaged (plus each signal)

Per participant and film: variance of HbO explained by their OWN motion, and by the motion of
other participants watching the same film (chance level for the same number of regressors).

Writes to results/:
  motion_pairs.csv          film x A x B : WTC of hbo, cleaned, motion
  motion_explained.csv      film x participant : variance explained by own / others' motion
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
from coherence import wavelet_coherence, window_z
from data import load_films
from motion import DECIMATE, SIGNALS, prepare_motion, remove_motion


def block_z(x, y, fs):
    """Film-block WTC of two signals: mean Fisher z over the 50-s windows."""
    return np.nanmean(window_z(wavelet_coherence(x, y, fs)[0], fs))


def one_person_vs_all(film, a, hbo_a, clean_a, motion_a, good_a, b_people, hbo_b, clean_b, motion_b, good_b, fs):
    """All pairs of one A participant in one film: WTC of HbO, cleaned HbO and motion."""
    rows = []
    for j, b in enumerate(b_people):
        use = np.flatnonzero(good_a & good_b[j])
        raw = [block_z(hbo_a[k], hbo_b[j, k], fs) for k in use]
        cleaned = [block_z(clean_a[k], clean_b[j, k], fs) for k in use]
        moves = [block_z(motion_a[s], motion_b[j, s], fs) for s in range(len(SIGNALS))]
        real = a["dyad"] == b["dyad"]
        rows.append(dict(film=film, a_dyad=a["dyad"], b_dyad=b["dyad"], kind="real" if real else "pseudo",
                         group=a["group"] if real else "pseudo", hbo=np.mean(raw), cleaned=np.mean(cleaned),
                         motion=np.mean(moves), **dict(zip(SIGNALS, moves))))
    return rows


if __name__ == "__main__":
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    films = load_films()
    m = prepare_motion(films)
    info, keep, good, partner, fs = m["info"], m["keep"], m["good"], m["partner"], m["fs"]
    print("flat sensors:", m["flat"], "| participants kept:", len(info))
    rng = np.random.default_rng(C.PERM_SEED)

    jobs, explained = [], []
    is_a = np.array([p["role"] == "A" for p in info])
    for film in C.FILMS:
        hbo = films[film]["data"][keep][:, :, ::DECIMATE]
        reg = m["reg"][film]                                               # participants x regressors x samples
        clean = np.empty_like(hbo)
        for i in range(len(info)):
            clean[i], r2 = remove_motion(hbo[i], reg[i])
            others = rng.choice([j for j in range(len(info)) if j not in (i, partner[i])], 20, replace=False)
            chance = np.nanmean([remove_motion(hbo[i], reg[j])[1] for j in others])
            explained.append(dict(film=film, id=info[i]["id"], group=info[i]["group"], own=np.nanmean(r2), others=chance))
        a_idx, b_idx = np.flatnonzero(is_a), np.flatnonzero(~is_a)
        b_people = [info[j] for j in b_idx]
        jobs += [(film, info[i], hbo[i], clean[i], reg[i], good[i], b_people, hbo[b_idx], clean[b_idx], reg[b_idx],
                  good[b_idx], fs) for i in a_idx]
        print(f"{film}: cleaned ({time.time() - start:.0f} s)", flush=True)
    pd.DataFrame(explained).to_csv(C.RESULTS / "motion_explained.csv", index=False)

    print(f"{len(jobs)} WTC jobs", flush=True)
    out = Parallel(n_jobs=-1, verbose=5)(delayed(one_person_vs_all)(*job) for job in jobs)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(C.RESULTS / "motion_pairs.csv", index=False, float_format="%.6g")
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)
