"""Stimulus-boundary and FIR-edge sensitivity (exploratory).

Trimming: the first and last TRIM seconds of every film are left out (0, 30, 60, 120 s).
  ISC   pairwise ISC recomputed on the central part of each film (run_isc.pairwise_isc), tested as
        in stats.py: Wilcoxon of participant means and the participant sign-flip test, each with
        FDR over the 3 films within a trim
  WTC   a pair's score is the mean of the 50-s windows lying fully inside the central part
        (windows_hbo.npz from run_windows.py); real vs pseudo by the re-matching test (stats.py),
        FDR over the 3 films within each trim; friends vs strangers by Mann-Whitney U
  Order-matched null: the same WTC test with re-matchings restricted to dyads that saw the films
        in the same order (Latin-square code), nobody paired with their own partner
FIR edges: Comedy WTC with reflected or constant extensions outside the block (windows_fir_*.npz),
        at each trim, re-matching test (uncorrected; only Comedy was computed).
Check: the 5 Hz block scores (mean of all windows in windows_hbo.npz) against the 25 Hz scores of
        run_wtc.py (wtc_pairs.csv), over all A x B pairs and films.

Writes results/boundary_isc.csv, boundary_wtc.csv, boundary_fir.csv, boundary_check.csv.
"""
import warnings

import numpy as np
import pandas as pd

import config as C
from data import list_participants, load_films, partner_index
from run_isc import pairwise_isc
from stats import derangements, fdr, group_test, isc_row, rematching_test, sign_flips

R = C.RESULTS
TRIMS = [0, 30, 60, 120]
rng = np.random.default_rng([C.PERM_SEED, 3])


def trimmed(windows, trim):
    """Mean over the windows (last axis, 1-s steps, 50 s long) lying fully inside the central part."""
    n = windows.shape[-1]
    length = (n - 1) * C.STEP_SECONDS + C.WINDOW_SECONDS
    starts = np.arange(n) * C.STEP_SECONDS
    keep = (starts >= trim) & (starts + C.WINDOW_SECONDS <= length - trim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(windows[..., keep], axis=-1)


def order_matched(codes, n_perm):
    """Re-matchings within groups of dyads sharing a film order, no dyad keeping its own partner."""
    codes = np.asarray(codes)
    out = np.empty((n_perm, len(codes)), int)
    for c in np.unique(codes):
        g = np.flatnonzero(codes == c)
        k = 0
        while k < n_perm:
            cand = np.argsort(rng.random((4 * n_perm, len(g))), axis=1)
            cand = cand[(cand != np.arange(len(g))).all(axis=1)][: n_perm - k]
            out[k:k + len(cand)][:, g] = g[cand]
            k += len(cand)
    return out


if __name__ == "__main__":
    # ---- ISC ----
    films = load_films("hbo")
    first = films[C.FILMS[0]]
    flips = sign_flips(len(first["people"]))
    partner = partner_index(first["people"])
    rows = []
    for trim in TRIMS:
        part = []
        for film in C.FILMS:
            d = films[film]
            cut = int(round(trim * d["sfreq"]))
            data = d["data"][:, :, cut: d["data"].shape[2] - cut]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                M = np.nanmean(pairwise_isc(data, d["good"], partner), axis=0)
            part.append(dict(trim_s=trim, film=film, **isc_row(M, flips)))
        for row, p, q in zip(part, fdr([r["p"] for r in part]), fdr([r["p_signflip"] for r in part])):
            row["p_fdr"], row["p_signflip_fdr"] = p, q
        rows += part
    pd.DataFrame(rows).to_csv(R / "boundary_isc.csv", index=False)

    # ---- WTC ----
    w = np.load(R / "windows_hbo.npz")
    dyads, groups = list(w["dyads"]), list(w["groups"])
    latin = {p["dyad"]: p["latin"] for p in list_participants()}
    perms = derangements(len(dyads), C.N_PERMUTATIONS)
    perms_order = order_matched([latin[d] for d in dyads], C.N_PERMUTATIONS)
    is_friend = np.array([g == "friend" for g in groups])
    rows = []
    for trim in TRIMS:
        part = {"unrestricted": [], "order-matched": []}
        for film in C.FILMS:
            M = trimmed(w[film], trim)
            for null, pm in [("unrestricted", perms), ("order-matched", perms_order)]:
                t = rematching_test(M, pm)
                part[null].append(dict(trim_s=trim, film=film, null=null, real=t["real"], pseudo=t["pseudo"],
                                       difference=t["difference"], null_mean=t["null_mean"], z=t["z"], p=t["p_greater"]))
            g = group_test(np.diag(M), is_friend)
            part.setdefault("friends", []).append(dict(trim_s=trim, film=film, null="friends - strangers",
                                                      difference=g["difference"], z=g["rank_biserial"], p=g["p"]))
        for family in part.values():
            for row, p in zip(family, fdr([r["p"] for r in family])):
                row["p_fdr"] = p
            rows += family
    pd.DataFrame(rows).to_csv(R / "boundary_wtc.csv", index=False)

    # ---- FIR edges (Comedy, every trim) ----
    rows = []
    for edge in ["continuous", "reflect", "constant"]:
        name = "windows_hbo.npz" if edge == "continuous" else f"windows_fir_{edge}.npz"
        if not (R / name).exists():
            continue
        for trim in TRIMS:
            M = trimmed(np.load(R / name)["Comedy"], trim)
            t = rematching_test(M, perms)
            rows.append(dict(edge=edge, trim_s=trim, film="Comedy", real=t["real"], pseudo=t["pseudo"],
                             difference=t["difference"], z=t["z"], p=t["p_greater"]))
    pd.DataFrame(rows).to_csv(R / "boundary_fir.csv", index=False)

    # ---- 5 Hz vs 25 Hz block scores ----
    full = pd.read_csv(R / "wtc_pairs.csv")
    diffs = []
    for film in C.FILMS:
        f = full[full.film == film].pivot(index="a_dyad", columns="b_dyad", values="wtc").loc[dyads, dyads].to_numpy()
        diffs.append((trimmed(w[film], 0) - f).ravel())
    diffs = np.concatenate(diffs)
    pd.DataFrame([dict(n_estimates=diffs.size, mean_abs_difference=np.abs(diffs).mean(), max_abs_difference=np.abs(diffs).max(),
                       mean_difference=diffs.mean())]).to_csv(R / "boundary_check.csv", index=False)

    pd.set_option("display.width", 200)
    for name in ["boundary_isc", "boundary_wtc", "boundary_fir", "boundary_check"]:
        t = pd.read_csv(R / f"{name}.csv")
        print(f"\n{name}\n{t.round(5).to_string(index=False)}")
