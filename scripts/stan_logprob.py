"""Check znuts's log densities and gradients against Stan's at random points.

For every posteriordb model, draws 20 points uniformly in (-2, 2) on the
unconstrained scale, evaluates the log density and gradient with CmdStan's
log_prob method (Jacobian included) and with `znuts logp`, and reports the
largest relative gradient difference and the spread of the log density
difference (the two drop different constants, so only the spread matters).

    CMDSTAN_MODELS=/path/to/compiled/models python3 scripts/stan_logprob.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from common import CMDSTAN_MODELS, DATA, POSTERIORS, RESULTS, znuts_binary


def main():
    exe = znuts_binary()
    rng = np.random.default_rng(2026)
    report = {}
    for _, model, data in POSTERIORS:
        dim = int(subprocess.run([exe, "list"], capture_output=True, text=True, check=True).stdout.split(model)[1].split()[0])
        pts = rng.uniform(-2, 2, (20, dim))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "u.json").write_text(json.dumps({"params_r": pts.tolist()}))
            subprocess.run([str(CMDSTAN_MODELS / model), "log_prob", f"unconstrained_params={tmp / 'u.json'}", "jacobian=1",
                            "data", f"file={DATA / data}", "output", f"file={tmp / 'lp.csv'}", "sig_figs=18"],
                           check=True, capture_output=True)
            stan = np.array([[float(v) for v in l.split(",")] for l in (tmp / "lp.csv").read_text().splitlines()
                             if l and not l.startswith("#") and not l.startswith("lp__")])
        csv = "\n".join(",".join(repr(float(v)) for v in p) for p in pts) + "\n"
        out = subprocess.run([exe, "logp", model], input=csv, capture_output=True, text=True, check=True).stdout
        ours = np.array([[float(v) for v in l.split(",")] for l in out.splitlines()])
        dlp = ours[:, 0] - stan[:, 0]
        g_rel = np.abs(ours[:, 1:] - stan[:, 1:]) / (1 + np.abs(stan[:, 1:]))
        report[model] = {"points": len(pts), "lp_offset": float(np.median(dlp)), "lp_offset_spread": float(dlp.max() - dlp.min()),
                         "max_grad_rel_diff": float(g_rel.max())}
        print(f"{model:28s} lp - lp_stan = {np.median(dlp):+.6g} (spread {dlp.max() - dlp.min():.1e})   "
              f"largest gradient difference {g_rel.max():.1e}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stan_logprob.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
