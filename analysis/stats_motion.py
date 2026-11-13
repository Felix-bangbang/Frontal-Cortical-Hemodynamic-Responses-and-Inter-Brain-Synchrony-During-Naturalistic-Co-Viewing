"""Statistics for the motion control. Reads the outputs of run_motion.py (and wtc_pairs.csv).

All tests are per film, on the dyads with working IMUs, with FDR over the 3 films in each family.

m1_explained.csv     how much HbO variance a participant's own motion explains, against the motion
                     of other participants watching the same film (paired t-test)
m2_amount.csv        amount of movement per dyad (mean of the partners): friends vs strangers
                     (Mann-Whitney U), and its Spearman correlation with real-dyad WTC
m3_motion_coupling.csv  WTC of the partners' MOTION: real vs pseudo (re-matching test, z) and friends
                     vs strangers (Mann-Whitney U, rank-biserial r)
m4_motion_vs_brain.csv  across all A x B pairs: does motion coupling predict brain coherence?
                     Slope of HbO WTC on motion WTC; p from re-matching the B participants
m5_control.csv       Q2 and Q3 repeated three ways on the same dyads: HbO unchanged; HbO with each
                     person's motion regressed out; and HbO WTC with motion WTC regressed out
                     across pairs
m6_isc.csv           pairwise ISC of HbO unchanged, HbO cleaned and of the motion signals: Wilcoxon
                     signed-rank test of participant means against 0, with the sign-flip check
m0_check.csv         the 5 Hz WTC used here against the 25 Hz WTC of run_wtc.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

import config as C
from data import list_participants
from motion import SIGNALS, movement_amount
from stats import derangements, fdr, group_test, isc_row, pair_matrix, rematching_test, sign_flips

R = C.RESULTS


def residual(pairs, y="hbo", x="motion"):
    """y with the across-pairs linear effect of x removed (mean of y kept)."""
    slope, intercept = np.polyfit(pairs[x], pairs[y], 1)
    return pairs[y] - slope * (pairs[x] - pairs[x].mean())


def mantel_slope(f, perms, x="motion", y="hbo"):
    """Slope of y on x over all pairs of one film; null: B participants re-matched in x only."""
    Y, _ = pair_matrix(f, y)
    X, _ = pair_matrix(f, x)
    slope = lambda a, b: np.polyfit(a.ravel(), b.ravel(), 1)[0]
    observed = slope(X, Y)
    null = np.array([slope(X[:, p], Y) for p in perms[:2000]])
    return dict(slope=observed, r=np.corrcoef(X.ravel(), Y.ravel())[0, 1],
                p=(1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))


if __name__ == "__main__":
    pairs = pd.read_csv(R / "motion_pairs.csv")
    dyads = sorted(pairs[pairs.kind == "real"].a_dyad.unique())
    perms = derangements(len(dyads), C.N_PERMUTATIONS)
    real = pairs[pairs.kind == "real"]

    # ---- m0: does 5 Hz reproduce the 25 Hz WTC? ------------------------------------------------
    full = pd.read_csv(R / "wtc_pairs.csv")[["film", "a_dyad", "b_dyad", "wtc"]]
    both = pairs.merge(full, on=["film", "a_dyad", "b_dyad"])
    check = [dict(film=f, r=np.corrcoef(d.hbo, d.wtc)[0, 1], mean_difference=(d.hbo - d.wtc).mean(),
                  max_abs_difference=(d.hbo - d.wtc).abs().max(), n_pairs=len(d)) for f, d in both.groupby("film")]
    pd.DataFrame(check).to_csv(R / "m0_check.csv", index=False)

    # ---- m1: variance of HbO explained by own motion -------------------------------------------
    ex = pd.read_csv(R / "motion_explained.csv")
    rows = []
    for film, d in ex.groupby("film"):
        t, p = st.ttest_rel(d.own, d.others)
        rows.append(dict(film=film, n=len(d), own=d.own.mean(), others=d.others.mean(),
                         difference=(d.own - d.others).mean(), t=t, p=p))
    m1 = pd.DataFrame(rows)
    m1["p_fdr"] = fdr(m1.p)
    m1.to_csv(R / "m1_explained.csv", index=False)

    # ---- m2: amount of movement -------------------------------------------------------------------
    amounts = []
    for person in list_participants():
        by_film = movement_amount(person)
        if by_film is not None:
            amounts += [dict(dyad=person["dyad"], group=person["group"], film=f, amount=v) for f, v in by_film.items()]
    amount = pd.DataFrame(amounts).groupby(["dyad", "group", "film"], as_index=False).amount.mean()
    amount = amount[amount.dyad.isin(dyads)]
    rows = []
    for film in C.FILMS:
        d = amount[amount.film == film].merge(real[real.film == film][["a_dyad", "hbo"]], left_on="dyad", right_on="a_dyad")
        t = group_test(d.amount, d.group == "friend")
        rho, p_rho = st.spearmanr(d.amount, d.hbo)
        rows.append(dict(film=film, friends=t["friends"], strangers=t["strangers"], rank_biserial=t["rank_biserial"], p=t["p"],
                         rho_with_wtc=rho, p_rho=p_rho))
    m2 = pd.DataFrame(rows)
    m2["p_fdr"], m2["p_rho_fdr"] = fdr(m2.p), fdr(m2.p_rho)
    m2.to_csv(R / "m2_amount.csv", index=False)

    # ---- m3: coupling of the partners' motion ------------------------------------------------------
    rows = []
    for film in C.FILMS:
        f = pairs[pairs.film == film]
        for signal in ["motion"] + SIGNALS:
            M, _ = pair_matrix(f, signal)
            t = rematching_test(M, perms)
            rows.append(dict(film=film, signal=signal, comparison="real vs pseudo", real=t["real"], pseudo=t["pseudo"],
                             difference=t["difference"], z=t["z"], p=t["p_two"]))
            r = f[f.kind == "real"]
            t = group_test(r[signal], r.group == "friend")
            rows.append(dict(film=film, signal=signal, comparison="friends vs strangers", real=t["friends"],
                             pseudo=t["strangers"], difference=t["difference"], z=t["rank_biserial"], p=t["p"]))
    m3 = pd.DataFrame(rows)
    m3["p_fdr"] = np.nan
    for _, part in m3.groupby(["comparison", m3.signal == "motion"]):
        m3.loc[part.index, "p_fdr"] = fdr(part.p)
    m3.to_csv(R / "m3_motion_coupling.csv", index=False)

    # ---- m4: does motion coupling predict brain coherence across pairs? ---------------------------
    rows = [dict(film=film, **mantel_slope(pairs[pairs.film == film], perms)) for film in C.FILMS]
    m4 = pd.DataFrame(rows)
    m4["p_fdr"] = fdr(m4.p)
    m4.to_csv(R / "m4_motion_vs_brain.csv", index=False)

    # ---- m5: Q2 and Q3 with and without motion control ------------------------------------------
    pairs["adjusted"] = pairs.groupby("film", group_keys=False).apply(residual, include_groups=False)
    rows = []
    for version, column in [("HbO unchanged", "hbo"), ("own motion regressed out of HbO", "cleaned"),
                            ("motion coupling regressed out across pairs", "adjusted")]:
        q2, q3 = [], []
        for film in C.FILMS:
            f = pairs[pairs.film == film]
            t = rematching_test(pair_matrix(f, column)[0], perms)
            q2.append(dict(version=version, test="Q2 real vs pseudo", film=film, a=t["real"], b=t["pseudo"],
                           difference=t["difference"], effect=t["z"], p=t["p_greater"]))
            r = f[f.kind == "real"]
            t = group_test(r[column], r.group == "friend")
            q3.append(dict(version=version, test="Q3 friends vs strangers", film=film, a=t["friends"], b=t["strangers"],
                           difference=t["difference"], effect=t["rank_biserial"], p=t["p"]))
        for family in (q2, q3):
            for row, p in zip(family, fdr([r["p"] for r in family])):
                row["p_fdr"] = p
            rows += family
        M = np.nanmean([pair_matrix(pairs[pairs.film == f], column)[0] for f in C.FILMS], axis=0)
        t = rematching_test(M, perms)
        rows.append(dict(version=version, test="Q2 real vs pseudo", film="all films", a=t["real"], b=t["pseudo"],
                         difference=t["difference"], effect=t["z"], p=t["p_greater"]))
        avg = pairs[pairs.kind == "real"].groupby(["a_dyad", "group"], as_index=False)[column].mean()
        t = group_test(avg[column], avg.group == "friend")
        rows.append(dict(version=version, test="Q3 friends vs strangers", film="all films", a=t["friends"],
                         b=t["strangers"], difference=t["difference"], effect=t["rank_biserial"], p=t["p"]))
    pd.DataFrame(rows).to_csv(R / "m5_control.csv", index=False)

    # ---- m6: ISC with and without motion, and ISC of motion itself --------------------------------
    matrices = np.load(R / "isc_motion_matrices.npz")
    n_people = matrices[f"{C.FILMS[0]}_hbo"].shape[1]
    flips = sign_flips(n_people)
    rows = []
    for film in C.FILMS:
        for label in ["hbo", "cleaned"] + SIGNALS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN cells are expected
                average = np.nanmean(matrices[f"{film}_{label}"], axis=0)
            rows.append(dict(film=film, data=label, **isc_row(average, flips)))
    m6 = pd.DataFrame(rows)
    m6["p_fdr"] = np.nan
    for _, part in m6.groupby("data"):
        m6.loc[part.index, "p_fdr"] = fdr(part.p)
    m6.to_csv(R / "m6_isc.csv", index=False)

    pd.set_option("display.width", 220)
    for name in ["m0_check", "m1_explained", "m2_amount", "m3_motion_coupling", "m4_motion_vs_brain", "m5_control", "m6_isc"]:
        print(f"\n{name}\n{pd.read_csv(R / (name + '.csv')).round(4).to_string(index=False)}")
