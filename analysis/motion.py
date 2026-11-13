"""Head-motion signals from the headset IMU, cut with the same stimulus alignment as the HbO data.

Five motion signals are derived per participant over the whole recording and then cut per film:
  tilt_x, tilt_y, tilt_z  accelerometer axes band-passed to 0.02-0.10 Hz (slow head orientation)
  accel_energy            log magnitude of the accelerometer above 0.5 Hz, band-passed to 0.02-0.10 Hz
                          (how much the head is moving)
  gyro_energy             log magnitude of the gyroscope, band-passed to 0.02-0.10 Hz (rotation)

Each signal uses the same 0.02-0.10 Hz filter as the HbO data. The two energy signals are also kept
LAG_SECONDS earlier (energy at t - LAG_SECONDS), for a delayed systemic response to movement.
IMU samples share the fNIRS sample clock, so films are cut at the same sample as the HbO data.

Accelerometers that recorded a constant signal are flagged and have no motion signals.
"""
import h5py
import mne
import numpy as np
from joblib import Parallel, delayed

import config as C
from data import film_onsets, list_participants

SIGNALS = ["tilt_x", "tilt_y", "tilt_z", "accel_energy", "gyro_energy"]
REGRESSORS = SIGNALS + ["accel_energy_lagged", "gyro_energy_lagged"]
ENERGY_CUTOFF = 0.5      # Hz; high-pass for movement energy
LAG_SECONDS = 6.0        # delay of the lagged energy regressors
DECIMATE = 5             # analyses here use every 5th sample (5 Hz; the band stops at 0.10 Hz)


def band(x, fs):
    """Same FIR band-pass as the HbO data."""
    return mne.filter.filter_data(x, fs, *C.BAND, method="fir", fir_window="hamming", phase="zero",
                                  l_trans_bandwidth=0.01, h_trans_bandwidth=0.02, verbose="error")


def load_imu(path):
    with h5py.File(path, "r") as f:
        aux = {f[f"nirs/{k}/name"][()].decode(): k for k in f["nirs"] if k.startswith("aux")}
        read = lambda names: np.array([f[f"nirs/{aux[n]}/dataTimeSeries"][()].ravel() for n in names], float)
        return read(["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]), read(["GYRO_X", "GYRO_Y", "GYRO_Z"])


def motion_signals(person, lengths):
    """Motion regressors per film: {film: regressors x samples (decimated)}, or None for a flat sensor.

    lengths: {film: number of samples at the full rate}, the common film lengths used for the HbO data.
    """
    accel, gyro = load_imu(person["path"])
    if np.std(accel, axis=1).max() < 1e-9:
        return None
    raw = mne.io.read_raw_snirf(str(person["path"]), verbose="error")     # for the film markers only
    fs = raw.info["sfreq"]
    moving = mne.filter.filter_data(accel, fs, ENERGY_CUTOFF, None, method="fir", phase="zero", verbose="error")
    accel_energy = np.log(np.linalg.norm(moving, axis=0) + 1e-4)
    gyro_energy = np.log(np.linalg.norm(gyro - np.median(gyro, axis=1, keepdims=True), axis=0) + 1e-2)
    filtered = np.vstack([band(accel, fs), band(accel_energy[None], fs), band(gyro_energy[None], fs)])
    lag = int(round(LAG_SECONDS * fs))
    out = {}
    for film, (onset, _) in film_onsets(raw, person["latin"]).items():
        first, n = int(round(onset * fs)), lengths[film]
        block = filtered[:, first: first + n]
        lagged = filtered[3:5, first - lag: first - lag + n]
        out[film] = np.vstack([block, lagged])[:, ::DECIMATE]
    return out


def movement_amount(person):
    """Mean log accelerometer magnitude above 0.5 Hz in each film (how much the head moved)."""
    accel, _ = load_imu(person["path"])
    if np.std(accel, axis=1).max() < 1e-9:
        return None
    raw = mne.io.read_raw_snirf(str(person["path"]), verbose="error")
    fs = raw.info["sfreq"]
    moving = mne.filter.filter_data(accel, fs, ENERGY_CUTOFF, None, method="fir", phase="zero", verbose="error")
    energy = np.log(np.linalg.norm(moving, axis=0) + 1e-4)
    return {film: energy[int(round(onset * fs)): int(round((onset + length) * fs))].mean()
            for film, (onset, length) in film_onsets(raw, person["latin"]).items()}


def prepare_motion(films):
    """Motion regressors for everyone, leaving out dyads containing a flat IMU.

    Returns a dict with: info (participant records kept), keep (their indices), good
    (participants x channels), partner (index of each one's partner), fs (Hz after decimation),
    flat (ids of flat IMUs), reg ({film: participants x regressors x samples}).
    """
    first = films[C.FILMS[0]]
    people = {p["id"]: p for p in list_participants()}
    lengths = {f: films[f]["data"].shape[2] for f in C.FILMS}
    ids = [p["id"] for p in first["people"]]
    moved = dict(zip(ids, Parallel(n_jobs=-1)(delayed(motion_signals)(people[i], lengths) for i in ids)))
    flat = sorted(i for i, m in moved.items() if m is None)
    drop = {i.rsplit("_", 1)[0] for i in flat}
    keep = [k for k, i in enumerate(ids) if i.rsplit("_", 1)[0] not in drop]
    info = [first["people"][k] for k in keep]
    partner = np.array([next(j for j, q in enumerate(info) if q["dyad"] == p["dyad"] and q["role"] != p["role"])
                        for p in info])
    return dict(info=info, keep=keep, good=first["good"][keep], partner=partner, fs=first["sfreq"] / DECIMATE, flat=flat,
                reg={f: np.array([moved[p["id"]][f] for p in info]) for f in C.FILMS})


def remove_motion(hbo, regressors):
    """Least-squares residual of HbO (channels x samples) on motion regressors plus an intercept.
    Missing channels stay NaN. Also returns the variance explained per channel."""
    X = np.column_stack([np.ones(hbo.shape[1]), regressors.T])
    good = ~np.isnan(hbo).any(axis=1)
    beta = np.linalg.lstsq(X, hbo[good].T, rcond=None)[0]
    cleaned = np.full_like(hbo, np.nan)
    cleaned[good] = hbo[good] - (X @ beta).T
    r2 = np.full(hbo.shape[0], np.nan)
    r2[good] = 1 - cleaned[good].var(axis=1) / hbo[good].var(axis=1)
    return cleaned, r2
