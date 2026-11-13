"""Q2-Q4: wavelet coherence (magnitude and phase) for every possible A x B pair, per film.

Every participant who wore headset A is paired with every participant who wore headset B, for
each film. One of those pairs is the real dyad; all the others are stimulus-matched pseudo-dyads:
two people who never interacted, watching the same film, aligned to film onset (data.py). This
all-pairs matrix is what the permutation tests in stats.py draw from.

For each pair, film and channel that passed SCI in both partners:
  wtc         Fisher z of the coherence magnitude in 50-s windows (coherence.window_z), averaged
              over the film ("absolute" WTC, no baseline subtracted)
  phase       coherence-weighted summary of the phase of A relative to B (coherence.phase_summary)
Pair-level values are the average over the pair's channels.

Writes to results/:
  wtc_pairs.csv         film x A x B : channel-averaged WTC and phase measures
  wtc_channels.csv      film x A x B x channel : WTC per channel
  wtc_windows_real.csv  film x real dyad x window : channel-averaged WTC time course (real dyads only)
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
from coherence import phase_summary, wavelet_coherence, window_z
from data import load_films


def one_person_vs_all(film, a, a_data, a_good, b_people, b_data, b_good, channels, fs):
    """Coherence of one A participant with every B participant in one film."""
    pairs, chans, windows = [], [], []
    for b, y_all, yg in zip(b_people, b_data, b_good):
        real = a["dyad"] == b["dyad"]
        use = np.flatnonzero(a_good & yg)
        if len(use) == 0:                                   # no channel usable in both: pair is missing
            continue
        z_by_channel, phases, time_courses = [], [], []
        for k in use:
            R2, phase, period = wavelet_coherence(a_data[k], y_all[k], fs)
            z = window_z(R2, fs)
            z_by_channel.append(np.nanmean(z))
            phases.append(phase_summary(R2, phase, period))
            if real:
                time_courses.append(z)
            chans.append(dict(film=film, a=a["id"], b=b["id"], kind="real" if real else "pseudo",
                              channel=f"S{channels[k][0]}_D{channels[k][1]}", wtc=np.nanmean(z)))
        # channel average: mean WTC; phase measures averaged with equal weight per channel
        mean = lambda key: np.mean([p[key] for p in phases])
        rx, ry = mean("resultant_x"), mean("resultant_y")
        hist = np.mean([p["hist"] for p in phases], axis=0)
        pairs.append(dict(film=film, a=a["id"], b=b["id"], a_dyad=a["dyad"], b_dyad=b["dyad"],
                          kind="real" if real else "pseudo", group=a["group"] if real else "pseudo",
                          n_channels=len(use), wtc=np.mean(z_by_channel),
                          phase_consistency=np.hypot(rx, ry), mean_phase=np.arctan2(ry, rx),
                          in_phase=mean("in_phase"), anti_phase=mean("anti_phase"),
                          a_leads=mean("a_leads"), b_leads=mean("b_leads"), lag_s=mean("lag_s"),
                          **{f"hist_{i:02d}": h for i, h in enumerate(hist)}))
        if real:
            tc = np.nanmean(np.array(time_courses), axis=0)
            windows += [dict(film=film, dyad=a["dyad"], group=a["group"], window=w, wtc=v) for w, v in enumerate(tc)]
    return pairs, chans, windows


if __name__ == "__main__":
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    films = load_films()
    print(f"preprocessed ({time.time() - start:.0f} s)")

    jobs = []
    for film in C.FILMS:
        d = films[film]
        is_a = np.array([p["role"] == "A" for p in d["people"]])
        a_idx, b_idx = np.flatnonzero(is_a), np.flatnonzero(~is_a)
        b_people = [d["people"][j] for j in b_idx]
        b_data, b_good = d["data"][b_idx], d["good"][b_idx]      # large arrays are shared with workers
        jobs += [(film, d["people"][i], d["data"][i], d["good"][i], b_people, b_data, b_good,
                  d["channels"], d["sfreq"]) for i in a_idx]
    print(f"{len(jobs)} jobs (one per A participant and film)")

    out = Parallel(n_jobs=-1, verbose=5)(delayed(one_person_vs_all)(*job) for job in jobs)
    pd.DataFrame([r for p, _, _ in out for r in p]).to_csv(C.RESULTS / "wtc_pairs.csv", index=False, float_format="%.6g")
    pd.DataFrame([r for _, c, _ in out for r in c]).to_csv(C.RESULTS / "wtc_channels.csv", index=False, float_format="%.6g")
    pd.DataFrame([r for _, _, w in out for r in w]).to_csv(C.RESULTS / "wtc_windows_real.csv", index=False, float_format="%.6g")
    print(f"done in {(time.time() - start) / 60:.1f} min")
