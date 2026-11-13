# Research-question analyses (films only, channel average, per film)

These scripts answer four research questions about the three films (Audio, Comedy, Horror):

| # | Question | Script | Main output |
|---|---|---|---|
| 1 | Do the films elicit reliable stimulus-locked responses across participants? (ISC) | `run_isc.py` → `stats.py` | `q1_isc.csv` |
| 2 | Does real-dyad synchrony exceed the shared-stimulus baseline? (WTC, real vs pseudo) | `run_wtc.py` → `stats.py` | `q2_real_vs_pseudo.csv` |
| 3 | Do friends and strangers differ in absolute WTC? | `run_wtc.py` → `stats.py` | `q3_friends_vs_strangers.csv` (+ `q3_channels.csv`, `q3_time.csv`) |
| 4 | What phase relationship underlies the coherence? | `run_wtc.py` → `stats.py` | `q4_phase.csv`, `q4_phase_histograms.csv` |

The same questions are repeated for HbR, with head-motion controls, boundary-trimming and filter-edge checks, and bootstrap confidence intervals (sections below). Results are in `RESULTS.md`.

---

## Running

From the repository root, `uv sync` creates the environment (Python 3.12, pinned in `uv.lock`). Then, in `analysis/` (times on a 10-core laptop):

```bash
uv run python run_isc.py            # Q1: pairwise HbO ISC for every pair, channel and film  (under 1 min)
uv run python run_wtc.py            # Q2–Q4: HbO WTC and phase, all 46 x 46 A–B pairs, 25 Hz (about 4 h)
uv run python stats.py              # Q1–Q4 tests                                            (a few minutes)
uv run python run_motion.py         # motion control: 5 Hz WTC with and without head motion   (about 50 min)
uv run python run_motion_isc.py     # motion control: 5 Hz ISC with and without head motion   (under 1 min)
uv run python stats_motion.py       # motion-control tests                                   (a few minutes)
uv run python run_isc.py hbr        # Q1 for HbR                                             (under 1 min)
uv run python run_hbr.py            # HbR WTC at 5 Hz, unchanged and with motion controls    (about 40 min)
uv run python run_hbr_channels.py   # HbR WTC of each channel, real dyads only               (under 1 min)
uv run python run_motion_isc.py hbr # HbR ISC with own motion removed; HbR variance explained  (under 1 min)
uv run python run_windows.py hbo    # 5 Hz windowed WTC and phase, all pairs (boundary checks) (about 35 min)
uv run python run_windows.py hbr    # the same for HbR (HbR Q4)                              (about 35 min)
uv run python run_fir_edges.py      # Comedy with reflected / constant filter extensions     (about 50 min)
uv run python stats_hbr.py          # HbR Q1–Q4 tests                                        (seconds)
uv run python stats_boundary.py     # trimming, order-matched and filter-edge tests          (about 1 min)
uv run python stats_bootstrap.py    # bootstrap confidence intervals                         (about 1 min)
uv run python check_orders.py       # film blocks and Latin codes from the event markers     (under 1 min)
```

`stats_hbr.py` and `stats_bootstrap.py` read the outputs of all the steps above, so run them last.

- **Data:** the `.snirf` files are expected in `../Data/friend/` and `../Data/stranger/`; change `DATA` in `config.py` if they are elsewhere.
- **Settings:** every setting (filter band, SCI threshold, window, number of permutations, seed, phase categories) is in `config.py`.
- **Threads:** each worker is limited to one thread (`OMP_NUM_THREADS=1` etc.), so the cores are not oversubscribed.

| File | What it does |
|---|---|
| `config.py` | all settings and paths |
| `data.py` | preprocessing; cuts the three films with a common stimulus alignment |
| `coherence.py` | wavelet coherence and phase (Grinsted et al., 2004); window Fisher z; phase summary |
| `run_isc.py` | Q1: pairwise ISC for every pair of participants |
| `run_wtc.py` | Q2–Q4: coherence magnitude and phase for every A × B pair and film |
| `stats.py` | all tests and multiple-comparison corrections |
| `motion.py` | head-motion signals from the IMU; regressing a person's motion out of their HbO |
| `run_motion.py` | motion control: WTC of HbO, motion-cleaned HbO and motion, at 5 Hz |
| `run_motion_isc.py` | motion control: ISC of HbO, motion-cleaned HbO and motion, at 5 Hz; with `hbr`, the same for HbR plus the HbR variance explained by own vs others' motion |
| `stats_motion.py` | motion-control tests (`m0_check.csv` to `m6_isc.csv`) |
| `run_hbr_channels.py` | HbR: per-channel WTC of the 46 real dyads at 5 Hz (secondary Q3 channel tests) |
| `check_orders.py` | the three film blocks per recording from the H/L markers, and both partners' Latin codes and onsets |
| `run_hbr.py` | HbR: WTC for all A × B pairs at 5 Hz, unchanged and with each person's own motion removed |
| `stats_hbr.py` | HbR: Q1–Q4 tests (channel average and per channel; Q3 also per window), motion versions, six-test HbO/HbR correction, HbR motion explained and cleaned ISC |
| `run_windows.py` | 5 Hz channel-averaged WTC for every window of every A × B pair, plus phase summaries (`hbo` or `hbr`) |
| `boundary.py` | unfiltered HbO with film onsets; re-filtering a film with the signal outside it replaced (reflected or constant) |
| `run_fir_edges.py` | Comedy windowed WTC with the reflected and constant filter extensions |
| `stats_boundary.py` | ISC and WTC after trimming 30/60/120 s from each end; order-matched re-matching; filter-edge comparison at every trim; 5 Hz vs 25 Hz agreement |
| `stats_bootstrap.py` | 95 % percentile CIs from 20,000 stratified dyad resamples (motion-coupling slope refitted in every resample) |

---

## Common methods

**Preprocessing (per participant):**
1. optical density;
2. scalp coupling index (SCI; channels below 0.7 are dropped);
3. TDDR motion correction;
4. HbO and HbR by the modified Beer–Lambert law (ppf 6);
5. 0.02–0.10 Hz FIR band-pass.

**Stimulus alignment.**
- Each film starts at that participant's own film-onset (H) marker.
- Every participant's recording of a film is then cut to the same length, the shortest recording of that film (about 599 s).
- So sample *t* is the same moment of the film for every participant and every pair, real or pseudo.
- Films are analysed **separately**; every test is done per film.
- Rows labelled "all films" average each pair or participant over the three films first. They are reported as a summary, outside the correction families.

**No ROIs.**
- Every measure is computed per channel, using only channels that passed SCI in both members of a pair.
- It is then **averaged over channels** with equal weight.
- Channel-level results are given only where a question asks for them (Q1: which channels; Q3: secondary).

**No rest baseline.**
- Every comparison uses absolute values from the film blocks.
- Pearson ISC has a natural zero, so Q1 is a test against 0. For WTC, which is always positive, chance level comes from stimulus-matched pseudo-dyads (Q2). The eyes-closed rest block is never used.

**WTC.**
- Morlet wavelet (ω₀ = 6) with Grinsted smoothing; values inside the cone of influence are removed.
- Coherence is converted with Fisher z as arctanh(√ mean R²) in each 50-s window (1-s steps).
- A block value is the mean of its window values.

**Multiple comparisons.**
- Benjamini–Hochberg FDR is applied within each family listed below.
- Permutation p-values are (1 + number of null values at least as extreme) / (N + 1).

---

## Q1. Inter-subject correlation (`run_isc.py`, `stats.py`)

**Measure.**
- **Pairwise ISC:** per film and channel, Pearson r between two participants' band-passed HbO, Fisher z.
- **Pairs of real partners are left out.** They watched together, so their similarity is not purely stimulus-driven.

**Test: is the mean ISC different from 0?**
- **Unit: the participant.** Each participant's value is their mean ISC with everyone except their partner (92 values per film). The roughly 4,100 pair values cannot be tested directly, because each person appears in about 90 pairs and the pairs are not independent.
- **Primary test:** two-tailed Wilcoxon signed-rank test of the 92 participant means against 0.
- **Effect size:** matched rank-biserial r, plus the mean, median and number of participants above 0.
- **Sign-flip check (`p_signflip`):**
  - Participant means still share pairs (the pair of i and j feeds both means), which can make the Wilcoxon p too small.
  - So each participant is given a random +1 or −1, and pair (i, j) is multiplied by both signs, for 10,000 draws.
  - This keeps the pairs' dependence on shared participants, and asks how often the mean of all pairs is as far from 0 as observed.
  - It is the same idea as subject-level resampling for ISC (Chen et al., 2016). The smallest possible p is .0001.

| Level | Values tested | FDR family |
|---|---|---|
| Channel average, per film | participant means over channels | 3 films |
| Each channel, per film | participant means for that channel | 3 films × channels |

---

## Q2. Real vs stimulus-matched pseudo-dyads, absolute WTC (`run_wtc.py`, `stats.py`)

**Pairs.**
- Every participant who wore headset A is paired with every participant who wore headset B, for each film: 46 × 46 = 2,116 pairs.
- The 46 pairs on the diagonal are the real dyads. The other 2,070 are **stimulus-matched pseudo-dyads**: two people who never interacted, watching the same film, aligned to film onset.
- Coherence is computed identically for all of them.

**Test: re-matching permutation, which accounts for participant dependence.**
- **Statistic:** mean WTC of the 46 real dyads.
- **Null:** the same mean for 10,000 random **re-matchings** of the A and B participants in which nobody is paired with their real partner.
  - Each re-matching, like the real set, uses every participant exactly once.
  - So it keeps the same participant structure, including each person's general level of coherence. It differs only in who is paired with whom.
  - The null is drawn from the full 46 × 46 matrix of stimulus-matched pairs.
- **Reported:**
  - `real`: the real-dyad mean;
  - `pseudo`: the mean of all pseudo pairs;
  - `difference`: real − pseudo;
  - `z` against the re-matching null;
  - `p_greater` (one-tailed, real > pseudo; FDR over 3 films) and `p_two`.

---

## Q3. Friends vs strangers, absolute WTC (`stats.py`)

Real dyads only (23 friend and 23 stranger dyads); absolute WTC with no baseline subtracted. Each dyad contributes one value per film, so the dyad is the unit.

**Test:** two-tailed Mann–Whitney U test, reporting group means and medians and the rank-biserial r.

| Analysis | Values tested | FDR family |
|---|---|---|
| **Primary:** channel average, per film | one value per dyad | 3 films |
| Secondary: each channel, per film | one value per dyad | 3 films × channels |
| Secondary: over time, per film | channel-average WTC in each 50-s window (1-s steps) | all windows of the 3 films |

---

## Q4. Phase relationship (`coherence.phase_summary`, `stats.py`)

**Phase.**
- The phase is the angle of the smoothed cross-wavelet spectrum W_A · conj(W_B) at every time-frequency point in the 0.02–0.10 Hz band, outside the cone of influence.
- **Positive phase means partner A leads partner B.** A is the headset-215 participant.
- Every point is **weighted by its coherence R²**, so the phase description is dominated by points where the two signals are coherent.

**Per pair and film** (computed per channel, then averaged over channels):
| Measure | Definition |
|---|---|
| `in_phase` | weighted share of points with \|phase\| < 45° |
| `anti_phase` | weighted share with \|phase\| > 135° |
| `lagged` (`a_leads` + `b_leads`) | weighted share with 45° ≤ \|phase\| ≤ 135° (split by sign) |
| `phase_consistency` | length of the weighted mean phase vector (0 = no consistent phase, 1 = a single phase) |
| `mean_phase` | angle of the weighted mean phase vector |
| `lag_s` | weighted mean time lag = phase / (2π f) in seconds (positive = A leads) |
| `hist_00…hist_11` | weighted share in each 30° bin from −180° to 180° (`q4_phase_histograms.csv` averages these for friends, strangers and pseudo pairs) |

**Tests** (`q4_phase.csv`):
- **Real dyads, direction:**
  - Rayleigh test: are dyads' mean phase angles concentrated in any direction?
  - V-test: are they concentrated around 0°, i.e. in phase?
  - FDR over 3 films for each test (Zar, 2010).
- **Real vs pseudo:** each measure, re-matching permutation test as in Q2 (two-tailed); FDR over 3 films × 5 measures.
- **Friends vs strangers:** each measure, Mann–Whitney U test as in Q3 (two-tailed); FDR over 3 films × 5 measures.

---

## Motion control (`motion.py`, `run_motion.py`, `stats_motion.py`)

- **Motion signals:** three accelerometer axes band-passed to 0.02–0.10 Hz (slow head tilt), plus log movement energy from the accelerometer (above 0.5 Hz) and from the gyroscope, band-passed the same way. IMU samples share the fNIRS clock, so films are cut at the same samples as the HbO.
- **Sample:** dyads with a flat IMU are left out (44 dyads). Everything runs at 5 Hz; `m0_check.csv` compares it with the 25 Hz WTC.
- **Cleaning:** each channel of each person's HbO is regressed on that person's own 7 motion regressors (the five signals, plus both energy signals 6 s earlier for a delayed systemic response).
- **Tests:** variance explained by own vs others' motion (`m1`); amount of movement, friends vs strangers by Mann–Whitney (`m2`); motion coupling, real vs pseudo (re-matching) and friends vs strangers (Mann–Whitney) (`m3`); motion coupling vs brain coherence across pairs (`m4`); Q2 and Q3 with HbO unchanged, with HbO cleaned, and with motion coupling regressed out across pairs (`m5`, using the Q2 and Mann–Whitney tests); ISC of unchanged HbO, cleaned HbO and motion (`m6`, using the Q1 tests).
- **Limits:** the cleaning is linear. It also removes any brain signal that genuinely follows movement, so it can over-correct, and it cannot separate motion artefact from shared behaviour. Q4 is not repeated here.

---

## HbR (`run_isc.py hbr`, `run_hbr.py`, `stats_hbr.py`)

- Q1–Q4 are repeated on HbR with the same measures and tests. HbR WTC is computed at 5 Hz (`motion.DECIMATE`); HbR phase (Q4) comes from `run_windows.py hbr`.
- Q2 and Q3 are also run on the 44 motion dyads unchanged, with each person's own motion regressed out, and with motion coupling regressed out across pairs, as in `stats_motion.py`.
- Each HbR family is FDR-corrected over the three films; Q2 is also corrected over the six HbO and HbR tests (`p_fdr6`).

## Stimulus boundaries and filter edges (`run_windows.py`, `boundary.py`, `run_fir_edges.py`, `stats_boundary.py`)

- **Trimming:** ISC and WTC are recomputed after removing 30, 60 or 120 s from each end of every film. For WTC only 50-s windows lying fully inside the kept interval are used. ISC is tested as in Q1 (Wilcoxon and sign-flip), WTC as in Q2 and Q3. FDR is applied over the three films within each trim, not across trims.
- **Order-matched re-matching:** as Q2, but each dyad is re-matched only with dyads that saw the films in the same order (Latin-square code), so the null keeps presentation order.
- **Filter edges (Comedy):** the 0.02–0.10 Hz filter reaches 165 s beyond each end of a film. The film is re-filtered with the signal outside it replaced by its mirror image (`reflect`) or by its edge value (`constant`), and the 60-s-trimmed real − pseudo test is repeated.

## Confidence intervals (`stats_bootstrap.py`)

- 20,000 resamples of whole dyads, drawn separately from friend and stranger dyads so every resample keeps the 23 / 23 (or 22 / 22) balance; percentile 95 % CIs, not adjusted for multiplicity.
- Q1: mean participant ISC (also after own-motion regression). Q2: real − pseudo, re-estimated from the resampled rows and columns of the A × B matrix; for the motion-coupling version the slope of brain WTC on motion WTC is refitted in each resample. Q3: friends − strangers, with Bonferroni 98.33 % intervals, in the 44 motion dyads for each motion version, and after trimming. Q4: circular mean angle and R̄ of real dyads, and real − pseudo of each phase measure. Motion: own − others' explained variance.
- Random seed: `config.PERM_SEED` (2026), with a separate stream per script.

---

## Outputs (`results/`)

| File | Contents |
|---|---|
| `isc_matrices.npz` | per film: channels × participants × participants of Fisher z |
| `wtc_pairs.csv` | film × A × B: channel-averaged WTC and phase measures (`kind` = real / pseudo) |
| `wtc_channels.csv` | film × A × B × channel: WTC |
| `wtc_windows_real.csv` | film × real dyad × 50-s window: channel-averaged WTC |
| `isc_motion_matrices.npz` | as `isc_matrices.npz`, at 5 Hz, for HbO, cleaned HbO and each motion signal |
| `isc_matrices_hbr.npz` | as `isc_matrices.npz`, for HbR |
| `hbr_pairs.csv` | film × A × B: HbR WTC, unchanged and motion-cleaned |
| `windows_hbo.npz`, `windows_hbr.npz` | per film: A × B × 50-s window, channel-averaged WTC at 5 Hz |
| `phase_pairs_hbo.csv`, `phase_pairs_hbr.csv` | film × A × B: 5 Hz WTC and phase measures |
| `windows_fir_reflect.npz`, `windows_fir_constant.npz` | Comedy windows with the two filter extensions |
| `hbr_channels.csv` | film × real dyad × channel: HbR WTC at 5 Hz |
| `isc_motion_matrices_hbr.npz`, `motion_explained_hbr.csv` | HbR ISC unchanged and motion-cleaned; HbR variance explained by own / others' motion |
| `hbr_q1.csv` … `hbr_q4.csv`, `hbr_q1_channels.csv`, `hbr_q3_channels.csv`, `hbr_q3_time.csv`, `hbr_motion_explained.csv`, `hbr_motion_isc.csv` | HbR test results |
| `boundary_isc.csv`, `boundary_wtc.csv`, `boundary_fir.csv`, `boundary_check.csv`, `bootstrap_ci.csv`, `order_check.csv` | boundary, filter-edge, agreement, CI and order-check results |
| `q1_isc.csv` … `q4_phase_histograms.csv` | test results, one file per question (see above) |

