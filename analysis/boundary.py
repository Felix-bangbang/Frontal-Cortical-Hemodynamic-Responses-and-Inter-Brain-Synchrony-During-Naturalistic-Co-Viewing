"""HbO before band-pass filtering, for the FIR-edge sensitivity analysis (run_fir_edges.py).

unfiltered(person) repeats data.preprocess up to the haemoglobin conversion (no band-pass) and
returns the continuous HbO, the good-channel mask, the sampling rate and the film onsets.
refilter(x, first, n, fs, edge) band-passes one film block with the same FIR filter as the main
analysis, with the samples outside the block replaced:
  "continuous"  the real recording on either side (as in the main analysis)
  "reflect"     the block mirrored at each edge
  "constant"    the first / last sample of the block repeated
over HALF_SECONDS on each side, which covers the filter's one-sided length.
"""
import re

import mne
import numpy as np
from mne.preprocessing.nirs import (beer_lambert_law, optical_density, scalp_coupling_index,
                                    temporal_derivative_distribution_repair)

import config as C
from data import film_onsets

HALF_SECONDS = 165.0     # one-sided length of the 0.02-0.10 Hz FIR filter (8251 taps at 25 Hz)


def unfiltered(person, chromophore="hbo"):
    raw = mne.io.read_raw_snirf(str(person["path"]), preload=True, verbose="error")
    od = optical_density(raw, verbose="error")
    sci = scalp_coupling_index(od, verbose="error")
    od = temporal_derivative_distribution_repair(od, verbose="error")
    hb = beer_lambert_law(od, ppf=C.PPF)
    picks = [i for i, name in enumerate(hb.ch_names) if name.endswith(chromophore)]
    good = np.array([min(sci[2 * k], sci[2 * k + 1]) >= C.SCI_MIN for k in range(len(picks))])
    fs = hb.info["sfreq"]
    return dict(person, data=hb.get_data()[picks], good=good, sfreq=fs, onsets=film_onsets(hb, person["latin"]))


def refilter(x, first, n, fs, edge):
    """Band-pass one block (channels x n samples starting at `first`) with the given edge treatment."""
    half = int(round(HALF_SECONDS * fs))
    if edge == "continuous":
        ext = x[:, first - half: first + n + half]
    else:
        ext = np.pad(x[:, first: first + n], ((0, 0), (half, half)), mode="reflect" if edge == "reflect" else "edge")
    y = mne.filter.filter_data(ext, fs, *C.BAND, method="fir", fir_window="hamming", phase="zero",
                               l_trans_bandwidth=0.01, h_trans_bandwidth=0.02, verbose="error")
    return y[:, half: half + n]
