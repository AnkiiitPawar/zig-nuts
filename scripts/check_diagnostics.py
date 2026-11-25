"""Compare znuts's convergence diagnostics with ArviZ.

With no arguments, prints ArviZ's values for the fixed test sequence used in
src/diagnostics.zig. With draw files, recomputes ESS and R-hat for every
column with ArviZ and reports the largest relative difference from the values
znuts wrote in its summary file.

    python3 scripts/check_diagnostics.py
    python3 scripts/check_diagnostics.py /tmp/es
"""
import json
import sys
from pathlib import Path

import numpy as np
from arviz_stats.base import array_stats as A


def fixture():
    mask = (1 << 64) - 1
    state = 12345
    out = np.empty((4, 100))
    for c in range(4):
        prev = 0.1 * c
        for i in range(100):
            state = (state * 6364136223846793005 + 1442695040888963407) & mask
            u = (state >> 11) / 9007199254740992.0
            prev = 0.6 * prev + (u - 0.5)
            out[c, i] = prev
    return out


def arviz_diagnostics(x):
    return {
        "ess_bulk": float(A._ess_bulk(x)),
        "ess_tail": float(A._ess_tail(x, prob=(0.05, 0.95))),
        "ess_mean": float(A._ess(A._split_chains(x))),
        "rhat": float(A._rhat_rank(x)),
    }


def read_chains(prefix):
    files = sorted(Path(prefix).parent.glob(Path(prefix).name + "-*.csv"))
    chains = []
    for f in files:
        rows = [l for l in f.read_text().splitlines() if l and not l.startswith("#")]
        header = rows[0].split(",")
        chains.append(np.array([[float(v) for v in r.split(",")] for r in rows[1:]]))
    return header, np.stack(chains)


def main():
    if len(sys.argv) == 1:
        for k, v in arviz_diagnostics(fixture()).items():
            print(f"{k:9s} {v!r}")
        return
    worst = 0.0
    for prefix in sys.argv[1:]:
        header, draws = read_chains(prefix)
        summary = json.loads(Path(prefix + "-summary.json").read_text())
        for j, name in enumerate(header):
            if name.endswith("__"):
                continue
            ours = summary["params"][name]
            ref = arviz_diagnostics(draws[:, :, j])
            for key in ("ess_bulk", "ess_tail", "rhat"):
                rel = abs(ours[key] - ref[key]) / abs(ref[key])
                worst = max(worst, rel)
        print(f"{prefix}: largest relative difference so far {worst:.2e}")
    print(f"largest relative difference from ArviZ: {worst:.2e}")


if __name__ == "__main__":
    main()
