"""Checks the film order (Latin-square code in the file name) against the event markers.

For every recording: the three film blocks from the H (start) / L (end) markers (data.film_onsets,
which requires exactly three alternating H/L pairs), labelled by the Latin-square order, with their
onsets and durations. The markers do not say which film was shown, so the check can confirm the three
blocks, their durations and that both partners of a dyad carry the same Latin code and the same film
onsets, but not the film identity itself.

Writes results/order_check.csv (one row per recording and film) and prints a summary.
"""
import mne
import numpy as np
import pandas as pd

import config as C
from data import film_onsets, list_participants

if __name__ == "__main__":
    rows = []
    for p in list_participants():
        raw = mne.io.read_raw_snirf(str(p["path"]), preload=False, verbose="error")
        for film, (onset, length) in film_onsets(raw, p["latin"]).items():
            rows.append(dict(id=p["id"], dyad=p["dyad"], latin=p["latin"], film=film, onset_s=onset, duration_s=length))
    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / "order_check.csv", index=False)
    print("recordings:", d.id.nunique(), "| Latin codes:", sorted(d.latin.unique()))
    print("\nfilm durations (s) by labelled film:")
    print(d.groupby("film").duration_s.describe()[["count", "min", "50%", "max"]].round(2).to_string())
    same_code = d.groupby("dyad").latin.nunique().eq(1).all()
    gap = d.pivot_table(index=["dyad", "film"], columns=d.id.str[-1], values="onset_s")
    print("\nboth partners share a Latin code:", same_code)
    print(f"partners' film onsets differ by at most {np.abs(gap['A'] - gap['B']).max():.2f} s")
