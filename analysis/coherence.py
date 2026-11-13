"""Wavelet transform coherence (WTC) and phase between two signals, following Grinsted et al. (2004).

wavelet_coherence(x, y, fs) -> (R2, phase, period)
    R2      squared coherence (scales x time), NaN inside the cone of influence
    phase   phase of the smoothed cross-spectrum Wx * conj(Wy), in radians (-pi, pi];
            positive = x leads y. NaN inside the cone of influence
    period  Fourier period (s) of each scale
window_z(R2, fs)  -> Fisher z of the coherence magnitude in each 50-s window (1-s steps);
                     the block value used everywhere is the mean of these window values
phase_summary(R2, phase, period) -> coherence-weighted summary of the phase relationship
"""
import numpy as np
import pywt
from scipy import fft
from scipy.ndimage import convolve1d

import config as C

OMEGA0 = 6.0                                                      # standard Morlet wavelet
WAVELET = f"cmor2.0-{OMEGA0 / (2 * np.pi):.4f}"                   # PyWavelets name for it
FOURIER_FACTOR = 4 * np.pi / (OMEGA0 + np.sqrt(2 + OMEGA0 ** 2))  # scale -> period (Torrence & Compo, 1998)


def scales(fs):
    """N_SCALES log-spaced wavelet scales (in samples) whose periods span the analysis band."""
    centre = OMEGA0 / (2 * np.pi)
    low, high = C.BAND
    return np.logspace(np.log10(centre * fs / high), np.log10(centre * fs / low), C.N_SCALES)


def smooth(W, s):
    """Grinsted smoothing: Gaussian in time (width = scale), then a 0.6-octave boxcar across scales."""
    n = W.shape[1]
    sigma = s / (OMEGA0 / (2 * np.pi)) / FOURIER_FACTOR      # Gaussian width in samples, per scale
    npad = fft.next_fast_len(n + int(4 * sigma.max()) + 1)  # zero-pad so the smoothing does not wrap
    k = 2 * np.pi * fft.fftfreq(npad)
    W = fft.ifft(fft.fft(W, npad) * np.exp(-0.5 * np.outer(sigma ** 2, k ** 2)))[:, :n]
    steps = 0.6 / (2 * np.log2(s[-1] / s[0]) / (len(s) - 1))  # half-width of 0.6 octave, in scale steps
    kernel = np.r_[steps % 1, np.ones(2 * round(steps) - 1), steps % 1]
    return convolve1d(W, kernel / kernel.sum(), axis=0, mode="constant")


def wavelet_coherence(x, y, fs):
    """Squared wavelet coherence and phase of x and y (equal length)."""
    n = len(x)
    s = scales(fs)
    Wx = pywt.cwt(x, s, WAVELET, method="fft")[0]
    Wy = pywt.cwt(y, s, WAVELET, method="fft")[0]
    inv_s = 1 / s[:, None]                                    # scale normalisation before smoothing
    Sxy = smooth(inv_s * Wx * np.conj(Wy), s)
    Sxx = smooth(inv_s * np.abs(Wx) ** 2, s).real
    Syy = smooth(inv_s * np.abs(Wy) ** 2, s).real
    with np.errstate(invalid="ignore", divide="ignore"):
        R2 = np.clip(np.abs(Sxy) ** 2 / (Sxx * Syy), 0, 1)
    phase = np.angle(Sxy)
    # cone of influence: drop values closer to either edge than sqrt(2) * scale
    edge = np.minimum(np.arange(n), n - 1 - np.arange(n))
    coi = edge[None, :] < np.sqrt(2) * s[:, None]
    R2[coi] = np.nan
    phase[coi] = np.nan
    period = s / (OMEGA0 / (2 * np.pi) * fs)                   # scale -> period (s), inverse of scales()
    return R2, phase, period


def window_z(R2, fs):
    """Synchrony in each sliding window: Fisher z of sqrt(mean R^2) over the window's
    time-frequency cells (cone of influence excluded)."""
    width, step = int(round(C.WINDOW_SECONDS * fs)), int(round(C.STEP_SECONDS * fs))
    valid = ~np.isnan(R2)
    # running sums over time make every window an O(1) lookup
    total = np.r_[0, np.cumsum(np.where(valid, R2, 0).sum(0))]
    count = np.r_[0, np.cumsum(valid.sum(0))]
    starts = np.arange(0, R2.shape[1] - width + 1, step)
    mean_R2 = (total[starts + width] - total[starts]) / (count[starts + width] - count[starts])
    return np.arctanh(np.clip(np.sqrt(mean_R2), 0, 1 - 1e-9))


def phase_summary(R2, phase, period):
    """Coherence-weighted description of the phase relationship over the whole block.

    Each time-frequency cell is weighted by its R^2, so cells where the two signals are coherent
    count most. Returns:
      resultant_x, resultant_y  weighted mean of cos(phase), sin(phase): the mean phase vector
      in_phase, anti_phase      weighted share of cells with |phase| < 45 deg / > 135 deg
      a_leads, b_leads          weighted share of lagged cells (45-135 deg) with phase > 0 / < 0
      lag_s                     weighted mean time lag (s) = phase / (2 pi f), positive = A leads
      hist                      weighted share of cells in each of N_PHASE_BINS bins from -pi to pi
    """
    ok = ~np.isnan(phase)
    w, ph = R2[ok], phase[ok]
    w = w / w.sum()
    lag = (ph / (2 * np.pi) * np.broadcast_to(period[:, None], phase.shape)[ok])
    mag = np.abs(ph)
    lagged = (mag >= C.IN_PHASE_MAX) & (mag <= C.ANTI_PHASE_MIN)
    edges = np.linspace(-np.pi, np.pi, C.N_PHASE_BINS + 1)
    return dict(resultant_x=np.sum(w * np.cos(ph)), resultant_y=np.sum(w * np.sin(ph)),
                in_phase=np.sum(w[mag < C.IN_PHASE_MAX]), anti_phase=np.sum(w[mag > C.ANTI_PHASE_MIN]),
                a_leads=np.sum(w[lagged & (ph > 0)]), b_leads=np.sum(w[lagged & (ph < 0)]),
                lag_s=np.sum(w * lag), hist=np.histogram(ph, bins=edges, weights=w)[0])
