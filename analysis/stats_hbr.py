"""HbR versions of Q1-Q4, beside HbO where useful.

Q1  ISC of HbR (isc_matrices_hbr.npz): Wilcoxon of participant means and sign-flip test, as stats.py;
    channel average (FDR over 3 films) and each channel (FDR over 3 films x 24 channels).
Q2  real vs stimulus-matched pseudo for HbR, beside the same test for HbO (below).
    Also a six-test FDR over the HbO and HbR films together, for each version.
Q3  friends vs strangers for HbR (Mann-Whitney U): 46 dyads unchanged; 44 dyads unchanged; 44 dyads
    with own motion regressed out; 44 dyads with motion coupling regressed out. Secondary: each
    channel (hbr_channels.csv; FDR over 3 films x 24 channels) and each 50-s window (windows_hbr.npz;
    FDR over all windows of the 3 films), as stats.py does for HbO.
Q4  phase tests for HbR (phase_pairs_hbr.csv from run_windows.py), as stats.phase_tests.

Q2 details:

Re-matching test as in stats.py (one-tailed, real > pseudo; FDR over the 3 films). Versions:
  all 46 dyads, unchanged                      HbR: hbr_pairs.csv     HbO: wtc_pairs.csv (25 Hz)
  44 dyads with IMU, unchanged                 HbR: hbr_pairs.csv     HbO: motion_pairs.csv (5 Hz)
  44 dyads, own motion regressed out           HbR: hbr_pairs.csv     HbO: motion_pairs.csv
  44 dyads, motion coupling regressed out      each chromophore's WTC with the across-pairs linear
                                               effect of motion WTC (motion_pairs.csv) removed

Motion (44 dyads, from run_motion_isc.py hbr): variance of HbR explained by a participant's own
    motion vs other participants' motion (paired t-test, as stats_motion.py m1); ISC of HbR unchanged
    and with own motion regressed out (Q1 tests, FDR over 3 films within each).

Writes results/hbr_q1.csv, hbr_q1_channels.csv, hbr_q2.csv (with p_fdr6), hbr_q3.csv,
hbr_q3_channels.csv, hbr_q3_time.csv, hbr_q4.csv, hbr_motion_explained.csv, hbr_motion_isc.csv.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

import config as C
from stats import derangements, fdr, group_test, isc_row, pair_matrix, phase_tests, rematching_test, sign_flips
from stats_motion import residual

R = C.RESULTS


def q2(pairs, column, label, chromophore, perms):
    rows, mats = [], []
    for film in C.FILMS:
        M, _ = pair_matrix(pairs[pairs.film == film], column)
        mats.append(M)
        rows.append(dict(chromophore=chromophore, version=label, film=film, **rematching_test(M, perms)))
    for row, p in zip(rows, fdr([r["p_greater"] for r in rows])):
        row["p_fdr"] = p
    rows.append(dict(chromophore=chromophore, version=label, film="all films",
                     **rematching_test(np.nanmean(mats, axis=0), perms), p_fdr=np.nan))
    return rows


if __name__ == "__main__":
    hbr = pd.read_csv(R / "hbr_pairs.csv")
    hbo46 = pd.read_csv(R / "wtc_pairs.csv").rename(columns={"wtc": "hbo"})
    motion = pd.read_csv(R / "motion_pairs.csv")                 # 44 dyads: hbo, cleaned, motion
    keep = set(motion.a_dyad)
    hbr44 = hbr[hbr.a_dyad.isin(keep) & hbr.b_dyad.isin(keep)].merge(
        motion[["film", "a_dyad", "b_dyad", "motion"]], on=["film", "a_dyad", "b_dyad"])
    for d, y in [(hbr44, "hbr"), (motion, "hbo")]:
        d["adjusted"] = d.groupby("film", group_keys=False).apply(lambda f: residual(f, y=y), include_groups=False)
    perms46, perms44 = derangements(46, C.N_PERMUTATIONS), derangements(44, C.N_PERMUTATIONS)
    rows = []
    for chromophore, all46, sub, raw in [("HbR", hbr, hbr44, "hbr"), ("HbO", hbo46, motion, "hbo")]:
        rows += q2(all46, raw, "46 dyads, unchanged", chromophore, perms46)
        rows += q2(sub, raw, "44 dyads, unchanged", chromophore, perms44)
        rows += q2(sub, "cleaned", "44 dyads, own motion regressed out", chromophore, perms44)
        rows += q2(sub, "adjusted", "44 dyads, motion coupling regressed out", chromophore, perms44)
    out = pd.DataFrame(rows)
    # six-test FDR: HbO and HbR films together, within each version
    films = out.film != "all films"
    for _, part in out[films].groupby("version"):
        out.loc[part.index, "p_fdr6"] = fdr(part.p_greater)
    out.to_csv(R / "hbr_q2.csv", index=False)

    # ---- Q1: HbR ISC -------------------------------------------------------------------------------
    data = np.load(R / "isc_matrices_hbr.npz")
    flips = sign_flips(len(data["ids"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN cells are expected
        averages = {film: np.nanmean(data[film], axis=0) for film in C.FILMS}
    q1 = pd.DataFrame([dict(film=film, **isc_row(averages[film], flips)) for film in C.FILMS])
    q1["p_fdr"] = fdr(q1.p)
    q1.loc[len(q1)] = dict(film="all films", **isc_row(np.mean([averages[f] for f in C.FILMS], axis=0), flips))
    q1.to_csv(R / "hbr_q1.csv", index=False)
    q1c = pd.DataFrame([dict(film=film, channel=str(ch), **isc_row(data[film][k], flips))
                        for film in C.FILMS for k, ch in enumerate(data["channels"])])
    q1c["p_fdr"], q1c["p_signflip_fdr"] = fdr(q1c.p), fdr(q1c.p_signflip)
    q1c.to_csv(R / "hbr_q1_channels.csv", index=False)

    # ---- Q3: HbR friends vs strangers ----------------------------------------------------------------
    rows = []
    for label, df, col in [("46 dyads, unchanged", hbr, "hbr"), ("44 dyads, unchanged", hbr44, "hbr"),
                           ("44 dyads, own motion regressed out", hbr44, "cleaned"),
                           ("44 dyads, motion coupling regressed out", hbr44, "adjusted")]:
        real = df[df.kind == "real"]
        part = [dict(version=label, film=f, **group_test(real[real.film == f][col], real[real.film == f].group == "friend"))
                for f in C.FILMS]
        for row, p in zip(part, fdr([r["p"] for r in part])):
            row["p_fdr"] = p
        avg = real.groupby(["a_dyad", "group"], as_index=False)[col].mean()
        rows += part + [dict(version=label, film="all films", **group_test(avg[col], avg.group == "friend"))]
    q3 = pd.DataFrame(rows)
    q3.to_csv(R / "hbr_q3.csv", index=False)
    ch = pd.read_csv(R / "hbr_channels.csv")
    q3c = pd.DataFrame([dict(film=film, channel=c, **group_test(d.wtc, d.group == "friend"))
                        for (film, c), d in ch.groupby(["film", "channel"])])
    q3c["p_fdr"] = fdr(q3c.p)
    q3c.to_csv(R / "hbr_q3_channels.csv", index=False)
    win = np.load(R / "windows_hbr.npz")
    is_friend = np.array([g == "friend" for g in win["groups"]])
    rows = []
    for film in C.FILMS:
        real_w = win[film][np.arange(46), np.arange(46)]              # real dyads x windows
        rows += [dict(film=film, start_s=w * C.STEP_SECONDS, end_s=w * C.STEP_SECONDS + C.WINDOW_SECONDS,
                      **group_test(real_w[:, w], is_friend)) for w in range(real_w.shape[1])]
    q3t = pd.DataFrame(rows)
    q3t["p_fdr"] = fdr(q3t.p)
    q3t.to_csv(R / "hbr_q3_time.csv", index=False)

    # ---- HbR motion: explained variance and ISC after own-motion regression -------------------------
    ex = pd.read_csv(R / "motion_explained_hbr.csv")
    rows = []
    for film, d in ex.groupby("film"):
        t, p = st.ttest_rel(d.own, d.others)
        rows.append(dict(film=film, n=len(d), own=d.own.mean(), others=d.others.mean(),
                         difference=(d.own - d.others).mean(), t=t, p=p))
    mex = pd.DataFrame(rows)
    mex["p_fdr"] = fdr(mex.p)
    mex.to_csv(R / "hbr_motion_explained.csv", index=False)
    mats = np.load(R / "isc_motion_matrices_hbr.npz")
    flips88 = sign_flips(len(mats["ids"]))
    rows = []
    for label in ["hbr", "cleaned"]:
        part = []
        for film in C.FILMS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                part.append(dict(film=film, data=label, **isc_row(np.nanmean(mats[f"{film}_{label}"], axis=0), flips88)))
        for row, p in zip(part, fdr([r["p"] for r in part])):
            row["p_fdr"] = p
        rows += part
    pd.DataFrame(rows).to_csv(R / "hbr_motion_isc.csv", index=False)

    # ---- Q4: HbR phase -------------------------------------------------------------------------------
    if (R / "phase_pairs_hbr.csv").exists():
        q4 = phase_tests(pd.read_csv(R / "phase_pairs_hbr.csv"), perms46)
        q4.to_csv(R / "hbr_q4.csv", index=False)
        print("\nQ4 HbR")
        print(q4[["film", "comparison", "measure", "test", "value", "statistic", "p", "p_fdr"]].round(4).to_string(index=False))
    print("\nQ1 HbR")
    print(q1[["film", "n", "isc", "n_positive", "rank_biserial", "p", "p_fdr", "p_signflip"]].round(5).to_string(index=False))
    print("\nQ3 HbR")
    print(q3[["version", "film", "friends", "strangers", "difference", "rank_biserial", "p", "p_fdr"]].round(4).to_string(index=False))
    pd.set_option("display.width", 200)
    print(out[["chromophore", "version", "film", "real", "pseudo", "difference", "z", "p_greater", "p_fdr", "p_fdr6"]]
          .round(4).to_string(index=False))
