"""Load every recording, preprocess it, and cut the three films with a common stimulus alignment.

Preprocessing (per participant): optical density -> scalp coupling index (SCI) -> TDDR motion
correction -> HbO (modified Beer-Lambert law) -> 0.02-0.10 Hz band-pass.

Stimulus alignment: each film starts at that participant's own film-onset (H) marker. Each film
is then cut to the SAME number of samples for every participant (the shortest recording of that
film), so sample t means the same moment of the film for everyone.

load_films() returns one record per film:
    data     participants x channels x samples (HbO); NaN for channels that failed SCI
    good     participants x channels (bool)
    people   list of participant records (id, dyad, group, role), in the same order as `data`
    channels list of (source, detector), in the same order as the channel axis
    sfreq    sampling rate (Hz)
"""
import re

import mne
import numpy as np
from joblib import Parallel, delayed
from mne.preprocessing.nirs import (beer_lambert_law, optical_density, scalp_coupling_index,
                                    temporal_derivative_distribution_repair)

import config as C

# file names look like friend_03_215_4.snirf: group, dyad number, headset serial, Latin-square code
NAME = re.compile(r"(?P<group>friend|stranger)_(?P<n>\d{2})_(?P<device>\d{3})_(?P<latin>\d)\.snirf$")


def list_participants():
    """One record per .snirf file, parsed from the file name."""
    people = []
    for path in sorted(C.DATA.glob("*/*.snirf")):
        m = NAME.match(path.name)
        dyad = f"{m['group']}_{m['n']}"
        if dyad in C.EXCLUDE_DYADS:
            continue
        role = C.DEVICE_ROLE[m["device"]]
        people.append(dict(path=path, group=m["group"], dyad=dyad, role=role,
                           latin=int(m["latin"]), id=f"{dyad}_{role}"))
    return people


def film_onsets(raw, latin):
    """Onset (s) of each film from the H (start) / L (end) markers; checks there are 3 alternating pairs."""
    starts = sorted(a["onset"] for a in raw.annotations if a["description"].startswith("H"))
    ends = sorted(a["onset"] for a in raw.annotations if a["description"].startswith("L"))
    times = [t for pair in zip(starts, ends) for t in pair]
    if len(starts) != 3 or len(ends) != 3 or times != sorted(times):
        raise ValueError(f"expected 3 alternating H/L markers, got H={starts} L={ends}")
    return {film: (start, end - start) for film, start, end in zip(C.LATIN[latin], starts, ends)}


def preprocess(person, chromophore="hbo"):
    """Preprocess one recording; return HbO (or HbR) for each film (channels x samples, uncut length)."""
    raw = mne.io.read_raw_snirf(str(person["path"]), preload=True, verbose="error")
    od = optical_density(raw, verbose="error")
    sci = scalp_coupling_index(od, verbose="error")                 # computed before motion correction
    od = temporal_derivative_distribution_repair(od, verbose="error")
    hb = beer_lambert_law(od, ppf=C.PPF)
    hb.filter(*C.BAND, method="fir", fir_window="hamming", phase="zero",
              l_trans_bandwidth=0.01, h_trans_bandwidth=0.02, verbose="error")

    hbo = [i for i, name in enumerate(hb.ch_names) if name.endswith(chromophore)]   # "hbo" or "hbr"
    channels = [tuple(int(x) for x in re.findall(r"\d+", hb.ch_names[i].split()[0])) for i in hbo]
    # a channel's SCI is the worse of its two wavelengths (od channels come in wavelength pairs)
    good = np.array([min(sci[2 * k], sci[2 * k + 1]) >= C.SCI_MIN for k in range(len(hbo))])

    fs = hb.info["sfreq"]
    data = hb.get_data()[hbo]
    films = {}
    for film, (onset, length) in film_onsets(hb, person["latin"]).items():
        first = int(round(onset * fs))
        films[film] = data[:, first: first + int(round(length * fs))]
    return dict(person, sfreq=fs, channels=channels, good=good, films=films)


def load_films(chromophore="hbo"):
    """Preprocess everyone (in parallel) and build one aligned array per film."""
    people = Parallel(n_jobs=-1)(delayed(preprocess)(p, chromophore) for p in list_participants())
    channels = people[0]["channels"]
    fs = people[0]["sfreq"]
    for p in people:
        # sampling rates differ only in the 6th decimal place between devices (< 1 sample per hour)
        if p["channels"] != channels or abs(p["sfreq"] - fs) > 1e-3:
            raise ValueError(f"{p['id']}: channel list or sampling rate differs from {people[0]['id']}")

    info = [dict(id=p["id"], dyad=p["dyad"], group=p["group"], role=p["role"]) for p in people]
    good = np.array([p["good"] for p in people])
    out = {}
    for film in C.FILMS:
        n = min(p["films"][film].shape[1] for p in people)       # common length for this film
        data = np.array([p["films"][film][:, :n] for p in people])
        data[~good] = np.nan                                      # failed channels are missing
        out[film] = dict(data=data, good=good, people=info, channels=channels, sfreq=fs)
    return out


def partner_index(people):
    """For each participant, the index of their real partner (same dyad, other role)."""
    where = {(p["dyad"], p["role"]): i for i, p in enumerate(people)}
    return np.array([where[(p["dyad"], "B" if p["role"] == "A" else "A")] for p in people])
