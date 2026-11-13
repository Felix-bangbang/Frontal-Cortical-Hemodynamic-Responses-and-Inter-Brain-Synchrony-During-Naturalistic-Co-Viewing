# Trinity hyperscanning fNIRS: friends and strangers watching films

Analysis code for the paper on shared responses and partner coherence in 46 dyads (23 friend and 23 stranger pairs, 92 participants) who watched an audio story, a comedy and a horror film together while frontal fNIRS was recorded from both partners.

The code and its documentation are in [`analysis/`](analysis/):

- [`analysis/README.md`](analysis/README.md): methods, scripts, run order and outputs;
- [`analysis/RESULTS.md`](analysis/RESULTS.md): the results;
- `analysis/results/`: every output table.

## Setup

```bash
uv sync    # Python 3.12 environment with the pinned versions in uv.lock
```

The participant recordings are not included. Put them in `Data/friend/` and `Data/stranger/` at the repository root, or change `DATA` in `analysis/config.py`.
