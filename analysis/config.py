"""All analysis settings in one place. Every other script imports this file as C."""
from pathlib import Path

# --- folders --------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data"            # contains friend/ and stranger/ folders of .snirf files
RESULTS = HERE / "results"             # every output is written here
EXCLUDE_DYADS: list = []               # e.g. ["friend_07"]; empty = analyse all dyads

# --- preprocessing --------------------------------------------------------------
SCI_MIN = 0.7            # channels with scalp coupling index below this are dropped
PPF = 6.0                # partial pathlength factor for the modified Beer-Lambert law
BAND = (0.02, 0.10)      # Hz; band-pass filter AND the frequency band used for coherence

# --- design ---------------------------------------------------------------------
FILMS = ["Audio", "Comedy", "Horror"]
# film order for each Latin-square code in the file name (<group>_<NN>_<device>_<latin>.snirf)
LATIN = {1: ["Comedy", "Audio", "Horror"], 2: ["Comedy", "Horror", "Audio"], 3: ["Horror", "Comedy", "Audio"],
         4: ["Audio", "Comedy", "Horror"], 5: ["Horror", "Audio", "Comedy"], 6: ["Audio", "Horror", "Comedy"]}
DEVICE_ROLE = {"215": "A", "330": "B"}             # headset serial number -> role in the dyad

# --- Q2-Q4: wavelet transform coherence (WTC) -----------------------------------
N_SCALES = 64            # Morlet wavelet scales spanning BAND
WINDOW_SECONDS = 50.0    # sliding window length (time-resolved analyses)
STEP_SECONDS = 1.0       # sliding window step
N_PERMUTATIONS = 10000   # permutations for the sign-flip, real-vs-pseudo and phase tests
PERM_SEED = 2026

# --- Q4: phase categories (radians, phase of partner A relative to partner B) ----
IN_PHASE_MAX = 0.25 * 3.141592653589793      # |phase| <  45 deg  -> in phase
ANTI_PHASE_MIN = 0.75 * 3.141592653589793    # |phase| > 135 deg  -> anti-phase; in between -> lagged
N_PHASE_BINS = 12                            # 30-degree bins for the phase histogram
