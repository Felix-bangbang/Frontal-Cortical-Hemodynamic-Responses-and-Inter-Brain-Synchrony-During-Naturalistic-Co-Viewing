"""Statistics for the four research questions. Reads the outputs of run_isc.py and run_wtc.py.

Every analysis is run separately for each film (Audio, Comedy, Horror). Rows labelled
"all films" average each pair (or participant) over the three films first and are reported as a
summary, outside the correction families. Multiple comparisons: Benjamini-Hochberg FDR within each
family stated below; Q1 channel-level tests also get a max-statistic family-wise p.

Q1  ISC (q1_isc.csv)
    The unit is the participant: each participant's mean pairwise ISC (Fisher z) with everyone
    except their partner. Two-tailed Wilcoxon signed-rank test of the participant means against 0.
    Participant means share pairs, so a sign-flip test is reported alongside: each participant's
    pairs are multiplied by a random +1 or -1, so pair (i, j) is multiplied by s_i * s_j.
    Families: channel average over 3 films; each channel over 3 films x channels.
Q2  Real vs pseudo WTC (q2_real_vs_pseudo.csv)
    Statistic: mean WTC of the 46 real dyads. Null: the same mean for random re-matchings of the A
    and B participants in which nobody is paired with their real partner (each participant used
    exactly once, like the real set). One-tailed (real > pseudo) and two-tailed p. Family: 3 films.
Q3  Friends vs strangers absolute WTC (q3_friends_vs_strangers.csv, q3_channels.csv, q3_time.csv)
    Real dyads only, one value per dyad. Two-tailed Mann-Whitney U test. Families: 3 films (channel
    average, primary); 3 films x channels (secondary); all windows of the 3 films (secondary).
Q4  Phase (q4_phase.csv, q4_phase_histograms.csv)
    Real dyads: Rayleigh test (are dyads' mean phase angles concentrated?) and V-test toward 0
    (concentrated in phase?). Real vs pseudo (re-matching test, two-tailed) and friends vs
    strangers (Mann-Whitney U, two-tailed) on: share of in-phase / anti-phase / lagged coherence,
    phase consistency and mean lag. Families: Rayleigh and V-test each over 3 films; each
    comparison over 3 films x 5 measures.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config as C

R = C.RESULTS
rng = np.random.default_rng(C.PERM_SEED)


def fdr(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def rank_p(observed, null, tail="greater"):
    """Monte-Carlo p: (1 + number of null values at least as extreme) / (N + 1)."""
    null = np.asarray(null)
    n = len(null)
    greater = (1 + np.sum(null >= observed)) / (n + 1)
    less = (1 + np.sum(null <= observed)) / (n + 1)
    if tail == "greater":
        return greater
    return min(1.0, 2 * min(greater, less))


# ---------------------------------------------------------------------------------------------
# Q1 helpers
# ---------------------------------------------------------------------------------------------
def sign_flips(n_people):
    """Random +1 / -1 for every participant, N_PERMUTATIONS times (own generator, so the
    re-matchings drawn for Q2 and Q4 do not depend on it)."""
    return np.random.default_rng([C.PERM_SEED, 1]).choice([-1.0, 1.0], size=(C.N_PERMUTATIONS, n_people))


def signflip_p(M, flips):
    """Two-tailed sign-flip p for the mean of a symmetric participants x participants matrix of
    pair values (NaN = left out). Each participant gets a sign; pair (i, j) is multiplied by both."""
    valid = ~np.isnan(M)
    filled = np.where(valid, M, 0.0)
    observed = filled.sum()
    null = np.einsum("ni,ni->n", flips @ filled, flips)
    return (1 + np.sum(np.abs(null) >= abs(observed))) / (len(flips) + 1)


def isc_row(M, flips):
    """Group statistics for one participants x participants matrix of pairwise Fisher z."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN rows are expected
        means = np.nanmean(M, axis=1)
    means = means[~np.isnan(means)]
    w = stats.wilcoxon(means)
    ranks = stats.rankdata(np.abs(means))
    effect = (ranks[means > 0].sum() - ranks[means < 0].sum()) / ranks.sum()   # matched rank-biserial r
    return dict(n=len(means), isc=means.mean(), isc_median=np.median(means),
                isc_r=np.tanh(means.mean()), n_positive=int((means > 0).sum()), w=w.statistic, rank_biserial=effect,
                p=w.pvalue, p_signflip=signflip_p(M, flips))


def isc_tests():
    data = np.load(R / "isc_matrices.npz")
    channels = list(data["channels"])
    flips = sign_flips(len(data["ids"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN cells are expected
        averages = {film: np.nanmean(data[film], axis=0) for film in C.FILMS}
    rows = [dict(film=film, channel="average", **isc_row(averages[film], flips)) for film in C.FILMS]
    rows += [dict(film="all films", channel="average", **isc_row(np.mean([averages[f] for f in C.FILMS], axis=0), flips))]
    rows += [dict(film=film, channel=ch, **isc_row(data[film][k], flips))
             for film in C.FILMS for k, ch in enumerate(channels)]
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    avg = (out.channel == "average") & (out.film != "all films")
    chan = out.channel != "average"
    out.loc[avg, "p_fdr"] = fdr(out.p[avg])
    out.loc[chan, "p_fdr"] = fdr(out.p[chan])
    return out


# ---------------------------------------------------------------------------------------------
# Q2 / Q4 helpers: the A x B pair matrix and the re-matching test
# ---------------------------------------------------------------------------------------------
def pair_matrix(pairs, value):
    """A x B matrix of `value`, rows and columns in the same dyad order, so the diagonal is the
    real dyads and everything off the diagonal is a stimulus-matched pseudo pair."""
    m = pairs.pivot_table(index="a_dyad", columns="b_dyad", values=value)
    dyads = sorted(set(m.index) & set(m.columns))
    return m.loc[dyads, dyads].to_numpy(), dyads


def derangements(n, n_perm):
    """n_perm random permutations of 0..n-1 with no fixed point (nobody keeps their real partner)."""
    out = []
    while len(out) < n_perm:
        batch = np.argsort(rng.random((4 * n_perm, n)), axis=1)
        out += list(batch[(batch != np.arange(n)).all(axis=1)])
    return np.array(out[:n_perm])


def rematching_test(M, perms):
    """Mean of the real dyads (diagonal) vs the same mean over random re-matchings."""
    n = len(M)
    observed = np.nanmean(np.diag(M))
    null = np.nanmean(M[np.arange(n)[None, :], perms], axis=1)
    off = M[~np.eye(n, dtype=bool)]
    return dict(n=n, real=observed, pseudo=np.nanmean(off), difference=observed - np.nanmean(off),
                null_mean=null.mean(), null_sd=null.std(), z=(observed - null.mean()) / null.std(),
                p_greater=rank_p(observed, null, "greater"), p_two=rank_p(observed, null, "two"))


# ---------------------------------------------------------------------------------------------
# Friends vs strangers helpers
# ---------------------------------------------------------------------------------------------
def group_test(values, is_friend):
    """Friends vs strangers: two-tailed Mann-Whitney U test, with means, medians and rank-biserial r."""
    values, is_friend = np.asarray(values, float), np.asarray(is_friend, bool)
    ok = ~np.isnan(values)
    f, s = values[ok & is_friend], values[ok & ~is_friend]
    u, p = stats.mannwhitneyu(f, s, alternative="two-sided")
    return dict(n_friends=len(f), n_strangers=len(s), friends=f.mean(), strangers=s.mean(), difference=f.mean() - s.mean(),
                friends_median=np.median(f), strangers_median=np.median(s), u=u,
                rank_biserial=2 * u / (len(f) * len(s)) - 1, p=p)


# ---------------------------------------------------------------------------------------------
# Q4 helpers: circular statistics
# ---------------------------------------------------------------------------------------------
def rayleigh(angles):
    """Rayleigh test of non-uniformity (Zar, 2010, eq. 27.4). Returns mean angle, R-bar, p."""
    n = len(angles)
    c, s = np.cos(angles).sum(), np.sin(angles).sum()
    r_n = np.hypot(c, s)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - r_n ** 2)) - (1 + 2 * n))
    return np.arctan2(s, c), r_n / n, min(1.0, p)


def v_test(angles, direction=0.0):
    """V-test of concentration around `direction` (Zar, 2010, eq. 27.5); one-tailed normal approx."""
    n = len(angles)
    v = np.sum(np.cos(angles - direction))
    u = v * np.sqrt(2 / n)
    return u, stats.norm.sf(u)


def phase_tests(pairs, perms):
    """Q4 tests on pair-level phase measures (one row per film x A x B; columns as in run_wtc.py).

    Real dyads: Rayleigh and V-test (toward 0) on the dyads' mean phase angles. Real vs pseudo:
    re-matching test (two-tailed). Friends vs strangers: Mann-Whitney U. FDR within each
    comparison and test.
    """
    pairs = pairs.assign(lagged=pairs.a_leads + pairs.b_leads)
    measures = ["in_phase", "anti_phase", "lagged", "phase_consistency", "lag_s"]
    rows = []
    for film in C.FILMS:
        f = pairs[pairs.film == film]
        r = f[f.kind == "real"]
        mean_angle, r_bar, p_ray = rayleigh(r.mean_phase.values)
        u, p_v = v_test(r.mean_phase.values, 0.0)
        rows.append(dict(film=film, comparison="real dyads: direction", measure="mean phase angle",
                         value=np.degrees(mean_angle), statistic=r_bar, p=p_ray, test="Rayleigh (R-bar)"))
        rows.append(dict(film=film, comparison="real dyads: direction", measure="mean phase angle",
                         value=np.degrees(mean_angle), statistic=u, p=p_v, test="V-test toward 0 deg (u)"))
        for m in measures:
            M, _ = pair_matrix(f, m)
            t = rematching_test(M, perms)
            rows.append(dict(film=film, comparison="real vs pseudo", measure=m, value=t["difference"],
                             real=t["real"], pseudo=t["pseudo"], statistic=t["z"], p=t["p_two"],
                             test="re-matching permutation (z)"))
            t = group_test(r[m], r.group == "friend")
            rows.append(dict(film=film, comparison="friends vs strangers", measure=m, value=t["difference"],
                             friends=t["friends"], strangers=t["strangers"], statistic=t["rank_biserial"], p=t["p"],
                             test="Mann-Whitney U (rank-biserial r)"))
    q4 = pd.DataFrame(rows)
    q4["p_fdr"] = np.nan
    for _, rows_ in q4.groupby(["comparison", "test"]):           # one family per comparison and test
        q4.loc[rows_.index, "p_fdr"] = fdr(rows_.p)
    return q4


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Q1: ISC --------------------------------------------------------------------------------
    q1 = isc_tests()
    q1.to_csv(R / "q1_isc.csv", index=False)

    pairs = pd.read_csv(R / "wtc_pairs.csv")
    n_dyads = pairs[pairs.kind == "real"].a_dyad.nunique()
    perms = derangements(n_dyads, C.N_PERMUTATIONS)        # one set of re-matchings for every test

    # ---- Q2: real vs stimulus-matched pseudo, absolute WTC ----------------------------------------
    rows, mats = [], []
    for film in C.FILMS:
        M, dyads = pair_matrix(pairs[pairs.film == film], "wtc")
        mats.append((M, dyads))
        rows.append(dict(film=film, **rematching_test(M, perms)))
    q2 = pd.DataFrame(rows)
    q2["p_fdr"] = fdr(q2.p_greater)
    q2.loc[len(q2)] = dict(film="all films", **rematching_test(np.nanmean([M for M, _ in mats], axis=0), perms), p_fdr=np.nan)
    q2.to_csv(R / "q2_real_vs_pseudo.csv", index=False)

    # ---- Q3: friends vs strangers, absolute WTC -----------------------------------------------------
    real = pairs[pairs.kind == "real"]
    rows = []
    for film in C.FILMS:
        d = real[real.film == film]
        rows.append(dict(film=film, **group_test(d.wtc, d.group == "friend")))
    q3 = pd.DataFrame(rows)
    q3["p_fdr"] = fdr(q3.p)
    avg = real.groupby(["a_dyad", "group"], as_index=False).wtc.mean()
    q3.loc[len(q3)] = dict(film="all films", **group_test(avg.wtc, avg.group == "friend"), p_fdr=np.nan)
    q3.to_csv(R / "q3_friends_vs_strangers.csv", index=False)

    # secondary: each channel
    ch = pd.read_csv(R / "wtc_channels.csv")
    ch = ch[ch.kind == "real"].merge(real[["film", "a", "group"]].drop_duplicates(), on=["film", "a"])
    rows = [dict(film=film, channel=c, **group_test(d.wtc, d.group == "friend"))
            for (film, c), d in ch.groupby(["film", "channel"])]
    q3c = pd.DataFrame(rows)
    q3c["p_fdr"] = fdr(q3c.p)
    q3c.to_csv(R / "q3_channels.csv", index=False)

    # secondary: over time (50-s windows, 1-s steps), one test per window
    win = pd.read_csv(R / "wtc_windows_real.csv")
    rows = []
    for film in C.FILMS:
        ts = win[win.film == film].pivot_table(index=["dyad", "group"], columns="window", values="wtc").dropna(axis=1)
        is_friend = ts.index.get_level_values("group").values == "friend"
        # windows are 1 s apart, so window index = start time (s); a window covers 50 s
        rows += [dict(film=film, start_s=w * C.STEP_SECONDS, end_s=w * C.STEP_SECONDS + C.WINDOW_SECONDS,
                      **group_test(ts[w].values, is_friend)) for w in ts.columns]
    q3t = pd.DataFrame(rows)
    q3t["p_fdr"] = fdr(q3t.p)
    q3t.to_csv(R / "q3_time.csv", index=False)

    # ---- Q4: phase ------------------------------------------------------------------------------
    q4 = phase_tests(pairs, perms)
    q4.to_csv(R / "q4_phase.csv", index=False)

    # phase histograms (coherence-weighted share of cells per 30-degree bin), for plotting
    hist_cols = [c for c in pairs.columns if c.startswith("hist_")]
    centres = np.degrees(np.linspace(-np.pi, np.pi, C.N_PHASE_BINS + 1)[:-1] + np.pi / C.N_PHASE_BINS)
    h = pairs.assign(set=np.where(pairs.kind == "real", pairs.group, "pseudo")).groupby(["film", "set"])[hist_cols].mean()
    h.columns = [f"{c:.0f}" for c in centres]
    h.reset_index().to_csv(R / "q4_phase_histograms.csv", index=False)

    pd.set_option("display.width", 200)
    print(q1[q1.channel == "average"].round(4).to_string(index=False))
    chan = q1[q1.channel != "average"]
    print(chan.groupby("film").agg(channels=("p", "size"), fdr_sig=("p_fdr", lambda v: (v < .05).sum()),
                                   signflip_sig=("p_signflip", lambda v: (v < .05).sum()), lowest=("isc", "min"), highest=("isc", "max")).round(4))
    for name, t in [("Q2", q2), ("Q3", q3)]:
        print(f"\n{name}\n{t.round(4).to_string(index=False)}")
    print(f"\nQ3 channels: {(q3c.p < .05).sum()} of {len(q3c)} p < .05; lowest p_fdr {q3c.p_fdr.min():.3f}")
    print(q3c.nsmallest(3, "p")[["film", "channel", "difference", "rank_biserial", "p", "p_fdr"]].round(4).to_string(index=False))
    print(f"\nQ3 time: {(q3t.p < .05).sum()} of {len(q3t)} windows p < .05; lowest p_fdr {q3t.p_fdr.min():.3f}")
    print(f"\nQ4\n{q4.round(4).to_string(index=False)}")
