"""The sampling checks that run in CI.

Builds znuts with safety checks on (ReleaseSafe), samples every posteriordb
model and the 100-dimensional normal with 4 chains, then checks that

  1. znuts's R-hat, bulk ESS and tail ESS match ArviZ's on the same draws, to
     a relative difference below 1e-9;
  2. every posterior mean and standard deviation is within 5 combined Monte
     Carlo standard errors of posteriordb's reference (or of the exact value
     for the normal).

A run whose largest R-hat is above 1.05 has not converged and is repeated with
the next seed, as a user would (the Gaussian mixture has a local mode that
about one chain in twenty finds from random initial values); a model that
fails three seeds fails the check.

    ZIG="python3 -m ziglang" python3 scripts/ci.py
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from arviz_stats.base import array_stats as A

from common import POSTERIORS, ROOT, read_stan_csv, reference_draws

ZIG = shlex.split(os.environ.get("ZIG", "zig"))


def sample(model, seed, prefix):
    subprocess.run([str(ROOT / "zig-out" / "bin" / "znuts"), "sample", model, "--seed", str(seed), "--out", str(prefix)],
                   check=True, capture_output=True)
    chains = [read_stan_csv(f"{prefix}-{c}.csv")[0] for c in range(1, 5)]
    summary = json.loads(Path(f"{prefix}-summary.json").read_text())
    return chains, summary


def check(model, reference, tmp):
    for seed in (1, 2, 3):
        chains, summary = sample(model, seed, tmp / f"{model}-{seed}")
        if max(p["rhat"] for p in summary["params"].values()) <= 1.05:
            break
        print(f"  {model}: seed {seed} did not converge (R-hat > 1.05), trying the next seed")
    else:
        print(f"FAIL {model}: no converged run in three seeds")
        return False
    ok = True
    worst_diag = worst_z = 0.0
    for name, ref in reference.items():
        x = np.stack([c[name] for c in chains])
        ours = summary["params"][name]
        for key, value in (("ess_bulk", A._ess_bulk(x)), ("ess_tail", A._ess_tail(x, prob=(0.05, 0.95))), ("rhat", A._rhat_rank(x))):
            worst_diag = max(worst_diag, abs(ours[key] - value) / abs(value))
        if isinstance(ref, tuple):
            mu, sd, mcse_mu, mcse_sd = ref[0], ref[1], 0.0, 0.0
        else:
            mu, sd, mcse_mu, mcse_sd = ref.mean(), ref.std(ddof=1), A._mcse_mean(ref), A._mcse_sd(ref)
        z_mean = (x.mean() - mu) / np.hypot(A._mcse_mean(x), mcse_mu)
        z_sd = (x.std(ddof=1) - sd) / np.hypot(A._mcse_sd(x), mcse_sd)
        worst_z = max(worst_z, abs(z_mean), abs(z_sd))
    if worst_diag > 1e-9:
        print(f"FAIL {model}: diagnostics differ from ArviZ by {worst_diag:.1e}")
        ok = False
    if worst_z > 5:
        print(f"FAIL {model}: a moment is {worst_z:.1f} standard errors from the reference")
        ok = False
    print(f"{'ok  ' if ok else 'FAIL'} {model:26s} seed {seed}  largest |z| {worst_z:4.2f}  diagnostics vs ArviZ {worst_diag:.1e}")
    return ok


def main():
    subprocess.run(ZIG + ["build", "-Doptimize=ReleaseSafe"], cwd=ROOT, check=True)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for posterior, model, _ in POSTERIORS:
            ok &= check(model, reference_draws(posterior), tmp)
        ok &= check("iid_normal_100", {f"x[{i + 1}]": (0.0, 1.0) for i in range(100)}, tmp)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
