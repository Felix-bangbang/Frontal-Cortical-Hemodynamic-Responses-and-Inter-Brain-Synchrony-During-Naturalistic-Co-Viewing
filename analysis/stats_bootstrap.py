"""95% confidence intervals from resampling intact dyads (stratified by friend / stranger).

Each resample draws the friend dyads with replacement from the friend dyads, and the stranger dyads
from the stranger dyads, so every resample keeps the 23 / 23 (or 22 / 22) balance. Partners stay
together. Two-sided percentile intervals, not adjusted for multiple comparisons.

  Q1  mean ISC: participants of the drawn dyads; their pairwise ISC matrix is rebuilt from the
      drawn rows and columns (self and partner pairs stay excluded) and averaged as in stats.py
  Q2  real - pseudo: mean of the drawn dyads' own pairs minus the mean of every drawn A x B pair
      that does not come from the same original dyad (pseudo pairs are dependent and are never
      treated as independent observations). For "motion coupling regressed out", the slope of brain
      WTC on motion WTC is refitted over the drawn A x B pairs in every resample.
  Q3  friends - strangers difference in means (and a 98.33% Bonferroni interval); in the 44 motion
      dyads for each motion version (slope refitted per resample as in Q2); after trimming 30, 60
      and 120 s from each end (windows_hbo.npz, as stats_boundary.py)
  Q4  circular mean phase angle (deviation from the observed angle, wrapped) and R-bar; real -
      pseudo difference of each pair-level phase measure (as Q2)
  Motion  own - other participants' explained variance (percentage points; participants of the
      drawn dyads); mean ISC after own-motion regression (as Q1), for HbO and HbR

Writes results/bootstrap_ci.csv.
"""
import warnings

import numpy as np
import pandas as pd

import config as C
from stats import pair_matrix
from stats_motion import residual

R = C.RESULTS
N_BOOT = 20000
rng = np.random.default_rng([C.PERM_SEED, 2])


def draws(groups):
    """N_BOOT x n index arrays: resample within each group, preserving group sizes."""
    groups = np.asarray(groups)
    out = np.empty((N_BOOT, len(groups)), int)
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        out[:, where] = rng.choice(where, size=(N_BOOT, len(where)))
    return out


def interval(values, level=95):
    lo = (100 - level) / 2
    return np.nanpercentile(values, [lo, 100 - lo])


def isc_ci(M, ids):
    dyad = np.array([i.rsplit("_", 1)[0] for i in ids])
    dyads = np.unique(dyad)
    members = np.array([np.flatnonzero(dyad == d) for d in dyads])        # dyads x 2
    stat = lambda S: np.nanmean(np.nanmean(S, axis=1))
    boot = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in draws([x.split("_")[0] for x in dyads]):
            idx = members[d].ravel()
            boot.append(stat(M[np.ix_(idx, idx)]))
        return stat(M), interval(boot)


def contrast_ci(M, dyads):
    same = np.equal.outer(np.arange(len(dyads)), np.arange(len(dyads)))
    observed = np.nanmean(np.diag(M)) - np.nanmean(M[~same])
    boot = []
    for d in draws([x.split("_")[0] for x in dyads]):
        S = M[np.ix_(d, d)]
        other = ~np.equal.outer(d, d)                                     # different original dyads
        boot.append(np.nanmean(M[d, d]) - np.nanmean(S[other]))
    return observed, interval(boot)


def contrast_refit_ci(Ys, Xs, dyads):
    """As contrast_ci for Y with the across-pairs linear effect of X removed, the slope refitted in
    every resample. Ys, Xs: lists of matrices (one per film; the adjusted matrices are averaged)."""
    def adjust(SY, SX):
        slope = np.polyfit(SX.ravel(), SY.ravel(), 1)[0]
        return SY - slope * (SX - SX.mean())
    def stat(d):
        adj = np.mean([adjust(Y[np.ix_(d, d)], X[np.ix_(d, d)]) for Y, X in zip(Ys, Xs)], axis=0)
        return np.mean(np.diag(adj)) - np.mean(adj[~np.equal.outer(d, d)])
    observed = stat(np.arange(len(dyads)))
    return observed, interval([stat(d) for d in draws([x.split("_")[0] for x in dyads])])


def group_refit_ci(Ys, Xs, dyads):
    """Friends - strangers of real-dyad Y with the across-pairs effect of X removed, the slope refitted
    in every resample. Ys, Xs: lists of matrices (one per film; the adjusted real-dyad values are averaged)."""
    friend = np.array([x.startswith("friend") for x in dyads])
    def stat(d):
        adj = []
        for Y, X in zip(Ys, Xs):
            SY, SX = Y[np.ix_(d, d)], X[np.ix_(d, d)]
            slope = np.polyfit(SX.ravel(), SY.ravel(), 1)[0]
            adj.append(np.diag(SY - slope * (SX - SX.mean())))
        adj = np.mean(adj, axis=0)
        return adj[friend[d]].mean() - adj[~friend[d]].mean()
    observed = stat(np.arange(len(dyads)))
    return observed, interval([stat(d) for d in draws([x.split("_")[0] for x in dyads])])


def explained_ci(ex):
    """Mean own - others explained variance (percentage points) over participants; dyads resampled."""
    ex = ex.assign(dyad=ex.id.str.rsplit("_", n=1).str[0], diff=100 * (ex.own - ex.others))
    per = ex.groupby("dyad")["diff"].mean()
    values, dyads = per.to_numpy(), list(per.index)
    boot = [values[d].mean() for d in draws([x.split("_")[0] for x in dyads])]
    return values.mean(), interval(boot)


def group_ci(values, is_friend, level=95):
    values, is_friend = np.asarray(values, float), np.asarray(is_friend, bool)
    f, s = values[is_friend], values[~is_friend]
    boot = [rng.choice(f, len(f)).mean() - rng.choice(s, len(s)).mean() for _ in range(N_BOOT)]
    return f.mean() - s.mean(), interval(boot, level)


def circular_ci(angles, groups):
    angles = np.asarray(angles)
    mean = lambda a: np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    rbar = lambda a: np.hypot(np.sin(a).mean(), np.cos(a).mean())
    m0 = mean(angles)
    dev, rb = [], []
    for d in draws(groups):
        a = angles[d]
        dev.append(np.angle(np.exp(1j * (mean(a) - m0))))
        rb.append(rbar(a))
    lo, hi = interval(dev)
    return np.degrees(m0), np.degrees([m0 + lo, m0 + hi]), rbar(angles), interval(rb)


if __name__ == "__main__":
    rows = []
    # ---- Q1 ----
    for chrom, name in [("HbO", "isc_matrices.npz"), ("HbR", "isc_matrices_hbr.npz")]:
        data = np.load(R / name)
        ids = list(data["ids"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = {f: np.nanmean(data[f], axis=0) for f in C.FILMS}
        for film, M in list(avg.items()) + [("all films", np.mean(list(avg.values()), axis=0))]:
            est, (lo, hi) = isc_ci(M, ids)
            rows.append(dict(question="Q1 ISC", chromophore=chrom, version="all participants", film=film, estimate=est, lo=lo, hi=hi))
    # ---- Q2 ----
    hbo46 = pd.read_csv(R / "wtc_pairs.csv")
    hbr46 = pd.read_csv(R / "hbr_pairs.csv")
    motion = pd.read_csv(R / "motion_pairs.csv")
    keep = set(motion.a_dyad)
    hbr44 = hbr46[hbr46.a_dyad.isin(keep) & hbr46.b_dyad.isin(keep)].merge(
        motion[["film", "a_dyad", "b_dyad", "motion"]], on=["film", "a_dyad", "b_dyad"])
    for d, y in [(hbr44, "hbr"), (motion, "hbo")]:
        d["adjusted"] = d.groupby("film", group_keys=False).apply(lambda f: residual(f, y=y), include_groups=False)
    versions = [("HbO", "46 dyads, unchanged", hbo46, "wtc"), ("HbR", "46 dyads, unchanged", hbr46, "hbr"),
                ("HbO", "44 dyads, unchanged", motion, "hbo"), ("HbO", "44 dyads, own motion regressed out", motion, "cleaned"),
                ("HbO", "44 dyads, motion coupling regressed out", motion, "adjusted"),
                ("HbR", "44 dyads, unchanged", hbr44, "hbr"), ("HbR", "44 dyads, own motion regressed out", hbr44, "cleaned"),
                ("HbR", "44 dyads, motion coupling regressed out", hbr44, "adjusted")]
    raw_of = {"HbO": "hbo", "HbR": "hbr"}
    for chrom, label, df, col in versions:
        mats = []
        for film in C.FILMS:
            M, dyads = pair_matrix(df[df.film == film], col)
            mats.append(M)
            if col == "adjusted":
                est, (lo, hi) = contrast_refit_ci([pair_matrix(df[df.film == film], raw_of[chrom])[0]],
                                                  [pair_matrix(df[df.film == film], "motion")[0]], dyads)
            else:
                est, (lo, hi) = contrast_ci(M, dyads)
            rows.append(dict(question="Q2 real - pseudo", chromophore=chrom, version=label, film=film, estimate=est, lo=lo, hi=hi))
        if col == "adjusted":           # pooled: mean over films of each film's adjusted matrix
            est, (lo, hi) = contrast_refit_ci([pair_matrix(df[df.film == f], raw_of[chrom])[0] for f in C.FILMS],
                                              [pair_matrix(df[df.film == f], "motion")[0] for f in C.FILMS], dyads)
        else:
            est, (lo, hi) = contrast_ci(np.nanmean(mats, axis=0), dyads)
        rows.append(dict(question="Q2 real - pseudo", chromophore=chrom, version=label, film="all films", estimate=est, lo=lo, hi=hi))
    # ---- Q3 ----
    for chrom, df, col in [("HbO", hbo46, "wtc"), ("HbR", hbr46, "hbr")]:
        real = df[df.kind == "real"]
        for film in C.FILMS + ["all films"]:
            r = real.groupby(["a_dyad", "group"], as_index=False)[col].mean() if film == "all films" else real[real.film == film]
            est, (lo, hi) = group_ci(r[col], r.group == "friend")
            row = dict(question="Q3 friends - strangers", chromophore=chrom, version="46 dyads", film=film, estimate=est, lo=lo, hi=hi)
            if film != "all films":
                row["lo_bonferroni"], row["hi_bonferroni"] = group_ci(r[col], r.group == "friend", level=100 - 5 / 3)[1]
            rows.append(row)
    # ---- Q4 ----
    phase_files = [("HbO", "wtc_pairs.csv"), ("HbR", "phase_pairs_hbr.csv")]
    for chrom, name in phase_files:
        if not (R / name).exists():
            continue
        real = pd.read_csv(R / name).query("kind == 'real'")
        for film in C.FILMS:
            r = real[real.film == film]
            m, (lo, hi), rb, (rlo, rhi) = circular_ci(r.mean_phase.values, r.group.values)
            rows.append(dict(question="Q4 mean phase angle (deg)", chromophore=chrom, version="real dyads", film=film, estimate=m, lo=lo, hi=hi))
            rows.append(dict(question="Q4 R-bar", chromophore=chrom, version="real dyads", film=film, estimate=rb, lo=rlo, hi=rhi))
    # ---- additions: 44-dyad Q3 by motion version, trimmed Q3, phase contrasts, motion ----
    for chrom, label, df, col in versions[2:]:
        for film in C.FILMS + ["all films"]:
            films_used = C.FILMS if film == "all films" else [film]
            if col == "adjusted":
                dyads = pair_matrix(df[df.film == films_used[0]], raw_of[chrom])[1]
                est, (lo, hi) = group_refit_ci([pair_matrix(df[df.film == f], raw_of[chrom])[0] for f in films_used],
                                               [pair_matrix(df[df.film == f], "motion")[0] for f in films_used], dyads)
            else:
                r = df[(df.kind == "real") & df.film.isin(films_used)].groupby(["a_dyad", "group"], as_index=False)[col].mean()
                est, (lo, hi) = group_ci(r[col], r.group == "friend")
            rows.append(dict(question="Q3 friends - strangers", chromophore=chrom, version=label, film=film, estimate=est, lo=lo, hi=hi))
    from stats_boundary import trimmed
    win = np.load(R / "windows_hbo.npz")
    friend = np.array([g == "friend" for g in win["groups"]])
    for trim in [0, 30, 60, 120]:
        for film in C.FILMS:
            est, (lo, hi) = group_ci(np.diag(trimmed(win[film], trim)), friend)
            rows.append(dict(question="Q3 friends - strangers", chromophore="HbO", version=f"46 dyads, 5 Hz, trim {trim} s",
                             film=film, estimate=est, lo=lo, hi=hi))
    measures = {"in_phase": "in_phase", "anti_phase": "anti_phase", "lagged": None, "phase_consistency": "phase_consistency",
                "lag_s": "lag_s"}
    for chrom, name in phase_files:
        ph = pd.read_csv(R / name)
        ph["lagged"] = ph.a_leads + ph.b_leads
        for film in C.FILMS:
            for m in measures:
                M, dyads = pair_matrix(ph[ph.film == film], m)
                est, (lo, hi) = contrast_ci(M, dyads)
                rows.append(dict(question=f"Q4 real - pseudo {m}", chromophore=chrom, version="46 dyads", film=film,
                                 estimate=est, lo=lo, hi=hi))
    for chrom, ex_name, isc_name, label in [("HbO", "motion_explained.csv", "isc_motion_matrices.npz", "hbo"),
                                            ("HbR", "motion_explained_hbr.csv", "isc_motion_matrices_hbr.npz", "hbr")]:
        ex = pd.read_csv(R / ex_name)
        mats = np.load(R / isc_name)
        ids = list(mats["ids"]) if "ids" in mats.files else list(ex[ex.film == C.FILMS[0]].id)
        for film in C.FILMS:
            est, (lo, hi) = explained_ci(ex[ex.film == film])
            rows.append(dict(question="Motion own - other explained variance (pp)", chromophore=chrom, version="44 dyads",
                             film=film, estimate=est, lo=lo, hi=hi))
            for data in [label, "cleaned"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    M = np.nanmean(mats[f"{film}_{data}"], axis=0)
                est, (lo, hi) = isc_ci(M, ids)
                rows.append(dict(question="Q1 ISC", chromophore=chrom,
                                 version="44 dyads, unchanged (5 Hz)" if data == label else "44 dyads, own motion regressed out",
                                 film=film, estimate=est, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(R / "bootstrap_ci.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
