"""Q1: do the films elicit reliable stimulus-locked responses across participants? (ISC)

Pairwise inter-subject correlation (ISC) of preprocessed HbO, computed separately for each film
and each channel, with every participant's film block starting at that participant's film onset
(data.py). For each pair of participants: Pearson r between their signals, Fisher z. Pairs of real
partners (who watched together) are left out, and a channel enters a pair only if it passed SCI
in both participants.

Optional command-line argument: "hbo" (default) or "hbr".

Writes to results/:
  isc_matrices.npz      for each film, channels x participants x participants of Fisher z
                        (NaN for left-out pairs), plus the channel names and participant ids
                        (isc_matrices_hbr.npz for HbR)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker; parallelism is across jobs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time

import numpy as np

import config as C
from data import load_films, partner_index


def pairwise_isc(data, good, partner):
    """Fisher z of the Pearson correlation for every pair of participants and channel.

    data: participants x channels x samples; good: participants x channels.
    Returns channels x participants x participants, NaN on the diagonal, for real partners and
    where a channel is missing in either participant.
    """
    n_people, n_chan, n_time = data.shape
    z = (data - data.mean(axis=2, keepdims=True)) / data.std(axis=2, keepdims=True)
    allowed = ~np.eye(n_people, dtype=bool)
    allowed[np.arange(n_people), partner] = False
    out = np.full((n_chan, n_people, n_people), np.nan)
    for k in range(n_chan):
        g = good[:, k]
        x = np.where(g[:, None], z[:, k, :], 0.0)                  # missing participants contribute 0
        r = x @ x.T / n_time
        valid = allowed & g[:, None] & g[None, :]
        out[k][valid] = np.arctanh(np.clip(r[valid], -0.999999, 0.999999))
    return out


if __name__ == "__main__":
    start = time.time()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    chromophore = sys.argv[1] if len(sys.argv) > 1 else "hbo"
    films = load_films(chromophore)
    print(f"preprocessed ({time.time() - start:.0f} s)")
    first = films[C.FILMS[0]]
    matrices = {film: pairwise_isc(films[film]["data"], films[film]["good"], partner_index(films[film]["people"]))
                for film in C.FILMS}
    name = "isc_matrices.npz" if chromophore == "hbo" else f"isc_matrices_{chromophore}.npz"
    np.savez_compressed(C.RESULTS / name, channels=np.array([f"S{s}_D{d}" for s, d in first["channels"]]),
                        ids=np.array([p["id"] for p in first["people"]]), **matrices)
    print(f"done in {(time.time() - start) / 60:.1f} min")
