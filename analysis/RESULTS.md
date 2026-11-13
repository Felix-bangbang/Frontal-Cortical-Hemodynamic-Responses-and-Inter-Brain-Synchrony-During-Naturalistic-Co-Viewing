# Results (46 dyads, 92 participants)

All tests are run separately for each film. "All films" rows are a summary and sit outside the correction families. Full tables are in `results/`.

## Q1. Inter-subject correlation (`q1_isc.csv`)

Pairwise ISC, real partners left out. Each participant's mean ISC is tested against 0 with a two-tailed Wilcoxon signed-rank test (n = 92).

| Film | Mean ISC (z) | Median | Participants above 0 | Rank-biserial r | Wilcoxon p | p_FDR | Sign-flip p |
|---|---|---|---|---|---|---|---|
| Audio | .013 | .014 | 78 / 92 | .82 | 1.0 × 10⁻¹¹ | 1.0 × 10⁻¹¹ | .0001 |
| Comedy | .031 | .029 | 87 / 92 | .96 | 9.5 × 10⁻¹⁶ | 1.7 × 10⁻¹⁵ | .0001 |
| Horror | .044 | .046 | 84 / 92 | .96 | 1.1 × 10⁻¹⁵ | 1.7 × 10⁻¹⁵ | .0001 |
| All films | .029 | .031 | 92 / 92 | 1.00 | 8.1 × 10⁻¹⁷ | — | .0001 |

- **Channel level:** every channel is significant in every film (72 of 72 after FDR; largest p_FDR = .017). All 72 also pass the sign-flip check at p < .05 (largest p = .021).
- **Size:** Horror > Comedy > Audio.
- **Caveat:** a shared response to a film starting would count here along with responses to film content.

## Q2. Real vs stimulus-matched pseudo-dyads, absolute WTC (`q2_real_vs_pseudo.csv`)

| Film | Real | Pseudo (all 2,070 pairs) | Difference | z (re-matching null) | p (one-tailed) | p_FDR |
|---|---|---|---|---|---|---|
| Audio | .7007 | .6990 | +.0017 | 0.69 | .246 | .246 |
| Comedy | .7082 | .7021 | **+.0062** | 2.28 | **.012** | **.036** |
| Horror | .7051 | .7008 | +.0044 | 1.61 | .055 | .083 |
| All films | .7047 | .7006 | +.0041 | 2.59 | .005 | — |

Real dyads are higher than stimulus-matched pseudo-dyads in all three films. The difference survives FDR in Comedy and is significant pooled over films.

## Q3. Friends vs strangers, absolute WTC (`q3_friends_vs_strangers.csv`)

Two-tailed Mann–Whitney U test, one value per dyad (23 vs 23).

| Film | Friends mean (median) | Strangers mean (median) | Difference in means | Rank-biserial r | p | p_FDR |
|---|---|---|---|---|---|---|
| Audio | .7035 (.6990) | .6980 (.7003) | +.0056 | .09 | .598 | .742 |
| Comedy | .7129 (.7165) | .7035 (.7075) | +.0094 | .41 | .019 | .056 |
| Horror | .7034 (.7081) | .7068 (.7094) | −.0034 | −.06 | .742 | .742 |
| All films | .7066 (.7080) | .7028 (.7013) | +.0038 | .23 | .180 | — |

- **Channel level** (`q3_channels.csv`): 4 of 72 film × channel tests have p < .05 (largest: Comedy S2–D3, r = .54, p = .002). None survives FDR (lowest p_FDR = .121).
- **Over time** (`q3_time.csv`): 31 of 1,650 windows have p < .05, fewer than the 83 expected by chance. None survives FDR (lowest p_FDR = .92).
- **Reading:** no friends–strangers difference survives correction. Comedy is the closest (p_FDR = .056).

## Q4. Phase (`q4_phase.csv`, `q4_phase_histograms.csv`)

**Real dyads, mean phase angle across dyads:**

| Film | Mean angle | R-bar | Rayleigh p_FDR | V-test (toward 0°) p_FDR |
|---|---|---|---|---|
| Audio | −18.6° | .18 | .241 | .055 |
| Comedy | +2.6° | .40 | **.0006** | **.0001** |
| Horror | −1.0° | .46 | **.0001** | **< .0001** |

**Real vs pseudo:** in Comedy, real dyads have more in-phase coherence (.300 vs .278, p = .003, p_FDR = .051) and less anti-phase coherence (p = .044, p_FDR = .27). No other measure differs.

**Friends vs strangers (Mann–Whitney U):** in Audio, friends have nominally more in-phase coherence (r = .43, p = .014) and less anti-phase coherence (r = −.43, p = .012). Neither survives FDR (p_FDR = .10). Nothing else differs (all other p ≥ .22).

**Summary:** where real dyads show consistent phase (Comedy, Horror), it is in phase, with no consistent lead or lag. This is the pattern expected from a shared stimulus-locked response.

## Motion control (`m0`–`m6` files in `results/`)

Uses the headset accelerometer and gyroscope on the 44 dyads with working IMUs (friend_23 and stranger_15 each have one dead IMU). Run at 5 Hz, which reproduces the 25 Hz WTC (r = 1.00 across pairs, largest difference .0002).

**Is there motion in the signal?**

| Film | HbO variance explained by own motion | By others' motion (chance) | t(87) |
|---|---|---|---|
| Audio | 13.6 % | 8.8 % | 11.4 |
| Comedy | 15.2 % | 8.9 % | 12.0 |
| Horror | 14.7 % | 9.6 % | 11.7 |

**Motion is coupled.** Partners' motion is far more coherent in real dyads than in pseudo pairs (z = 6.7, 4.5, 6.9 for Audio, Comedy, Horror; all p < .001). Movement energy is also strongly stimulus-locked: its pairwise ISC is higher than the HbO ISC (accelerometer energy .046, .065, .125, all p < 10⁻¹⁴). Across all pairs, more coherent motion goes with more coherent HbO in Comedy and Horror (r = .08, .11; p_FDR = .005). The amount of movement does not differ between friends and strangers (Mann–Whitney, all p ≥ .23) and is unrelated to dyad WTC (all p ≥ .40).

**Q1 ISC survives motion removal** (`m6_isc.csv`; 88 participants).

| Film | Mean ISC, HbO | Mean ISC, own motion removed | Wilcoxon p, removed | Sign-flip p, removed |
|---|---|---|---|---|
| Audio | .0123 | .0125 | 3.4 × 10⁻¹¹ | .0001 |
| Comedy | .0308 | .0291 | 2.4 × 10⁻¹⁵ | .0001 |
| Horror | .0430 | .0346 | 9.7 × 10⁻¹⁵ | .0001 |

**Q2 real vs pseudo is partly explained by motion.**

| Film | HbO unchanged | Own motion removed | Motion coupling regressed out |
|---|---|---|---|
| Audio | +.0012, p = .321 | −.0026, p = .861 | +.0006, p = .412 |
| Comedy | +.0066, p = .009, **p_FDR = .027** | +.0053, p = .025, p_FDR = .075 | +.0057, p = .021, p_FDR = .064 |
| Horror | +.0042, p = .065 | +.0012, p = .323 | +.0022, p = .209 |
| All films | +.0040, z = 2.50, p = .007 | +.0013, z = 0.84, p = .201 | +.0028, z = 1.77, p = .037 |

**Q3 friends vs strangers depends on the control** (`m5_control.csv`; Mann–Whitney U, rank-biserial r in brackets; all 44 dyads).

| Film | HbO unchanged | Own motion removed | Motion coupling regressed out |
|---|---|---|---|
| Audio | +.0055 (.08), p = .647 | +.0066 (.23), p = .201 | +.0056 (.07), p = .716 |
| Comedy | +.0083 (.36), p = .042, p_FDR = .127 | **+.0105 (.38), p = .032, p_FDR = .048** | +.0081 (.36), p = .045, p_FDR = .134 |
| Horror | −.0043 (−.10), p = .565 | **−.0114 (−.40), p = .024, p_FDR = .048** | −.0047 (−.07), p = .716 |
| All films | +.0032 (.19), p = .286 | +.0019 (.11), p = .550 | +.0030 (.20), p = .255 |

With own motion removed, Comedy (friends higher) and Horror (strangers higher) just pass FDR, in opposite directions. Neither does in the other two versions, so this is not a stable group difference.

**Summary.** ISC is not a motion effect. The real-vs-pseudo effect cannot be said to be independent of motion: the pooled effect loses significance when each person's motion is removed, and only Comedy keeps a nominal effect, which no longer survives FDR. The analysis cannot tell whether the coupled motion is an artefact in the fNIRS signal or a genuine part of partners responding together, so removing it may also remove real synchrony. Q4 (phase) was not repeated with motion control.

## HbR (`hbr_q1.csv` – `hbr_q4.csv`)

| Test | Audio | Comedy | Horror |
|---|---|---|---|
| Q1 mean ISC (all p_FDR < 10⁻⁹; sign-flip p = .0001) | .0054 | .0105 | .0214 |
| Q2 real − pseudo, 46 dyads (p_FDR; six-test p_FDR) | +.0009 (.310; .310) | +.0088 (.0003; .0006) | +.0026 (.167; .167) |
| Q2, 44 dyads, own motion regressed out | −.0011 (.728) | +.0053 (.011; .023) | +.0035 (.068) |
| Q2, 44 dyads, motion coupling regressed out | −.0001 (.526) | +.0078 (.0009; .0018) | +.0005 (.526) |
| Q3 friends − strangers, 46 dyads (p_FDR) | +.0035 (.282) | +.0067 (.272) | −.0075 (.282) |
| Q4 V-test toward 0° (p_FDR); Rayleigh (p_FDR) | .404; .960 | .009; .056 | .065; .272 |

No HbR friends-vs-strangers test (any motion version) and no real-vs-pseudo or familiarity phase contrast survives FDR.

## Stimulus boundaries and filter edges (`boundary_*.csv`)

| Trim each end | ISC, z (Audio / Comedy / Horror) | Comedy real − pseudo WTC, p_FDR | Comedy friends − strangers, p |
|---|---|---|---|
| 0 s | .0127 / .0311 / .0442 | +.00615, .036 | +.00937, .019 |
| 30 s | .0129 / .0324 / .0470 | +.00653, .033 | +.00797, .087 |
| 60 s | .0122 / .0352 / .0470 | +.00679, .037 | +.00688, .292 |
| 120 s | .0123 / .0394 / .0488 | +.00693, .074 | +.00798, .345 |

All trimmed ISC tests: Wilcoxon p ≤ 4.3 × 10⁻⁹, sign-flip p = .0001. Order-matched re-matching (Comedy, 60 s): p .0071, p_FDR .021. Comedy at 60 s with filter extensions: continuous +.00679 (p .0122), reflected +.00682 (p .0122), constant +.00683 (p .0121).

## Confidence intervals (`bootstrap_ci.csv`)

95 % percentile intervals from 20,000 stratified dyad resamples, e.g. HbO Q2 Comedy +.0062 [.0019, .0103]; HbR Q2 Comedy +.0088 [.0054, .0123]; HbO Q3 Comedy +.0094 [.0004, .0185] (98.33 %: [−.0013, .0206]).
