"""Motion control for Q1: pairwise ISC of unchanged HbO, of HbO with each participant's own motion
regressed out, and of the motion signals themselves. Same 44-dyad, 5 Hz data as run_motion.py and
the same pairwise ISC as run_isc.py.

Writes results/isc_motion_matrices.npz: for each film and data set (hbo, cleaned, each motion
signal), channels x participants x participants of Fisher z.

Optional command-line argument "hbr": the same for HbR (isc_motion_matrices_hbr.npz with hbr and
cleaned only), plus the variance of HbR explained by each participant's own motion and by the
motion of 20 other non-partner participants (motion_explained_hbr.csv, as run_motion.py does for HbO).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker; parallelism is across jobs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

import numpy as np
import pandas as pd

import config as C
from data import load_films
from motion import DECIMATE, SIGNALS, prepare_motion, remove_motion
from run_isc import pairwise_isc

if __name__ == "__main__":
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    chromophore = sys.argv[1] if len(sys.argv) > 1 else "hbo"
    films = load_films(chromophore)
    m = prepare_motion(films)
    good, partner, info, out, explained = m["good"], m["partner"], m["info"], {}, []
    rng = np.random.default_rng([C.PERM_SEED, 4])
    for film in C.FILMS:
        x = films[film]["data"][m["keep"]][:, :, ::DECIMATE]
        reg = m["reg"][film]
        clean = np.empty_like(x)
        for i in range(len(x)):
            clean[i], r2 = remove_motion(x[i], reg[i])
            if chromophore != "hbo":
                others = rng.choice([j for j in range(len(x)) if j not in (i, partner[i])], 20, replace=False)
                chance = np.nanmean([remove_motion(x[i], reg[j])[1] for j in others])
                explained.append(dict(film=film, id=info[i]["id"], group=info[i]["group"], own=np.nanmean(r2), others=chance))
        out[f"{film}_{chromophore}"] = pairwise_isc(x, good, partner)
        out[f"{film}_cleaned"] = pairwise_isc(clean, good, partner)
        if chromophore == "hbo":
            for s, signal in enumerate(SIGNALS):
                out[f"{film}_{signal}"] = pairwise_isc(reg[:, [s], :], np.ones((len(x), 1), bool), partner)
    out["ids"] = np.array([p["id"] for p in info])
    name = "isc_motion_matrices.npz" if chromophore == "hbo" else f"isc_motion_matrices_{chromophore}.npz"
    np.savez_compressed(C.RESULTS / name, **out)
    if explained:
        pd.DataFrame(explained).to_csv(C.RESULTS / f"motion_explained_{chromophore}.csv", index=False)
    print("done")
