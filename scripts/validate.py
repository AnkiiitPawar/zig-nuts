"""Compare posterior means and standard deviations with posteriordb.

Pools the replicate runs written by bench.py (5 runs x 4 chains = 20 chains of
1000 draws per sampler) and compares each parameter's mean and standard
deviation with posteriordb's reference draws (10 chains x 1000 draws from
long Stan runs with adapt_delta 0.95). A z-score divides the difference by the
combined Monte Carlo standard error of both estimates (ArviZ's mcse_mean and
mcse_sd), so for a correct sampler the z-scores should look like draws from a
standard normal. The same check is run on CmdStan's draws as a control.

    python3 scripts/validate.py [--label native]
"""
import argparse
import json

import numpy as np
from arviz_stats.base import array_stats as A

from common import POSTERIORS, RESULTS, ROOT, SYNTHETIC, read_stan_csv, reference_draws

RUNS = ROOT / ".cache" / "runs"


def pooled(sampler_dir, model):
    files = sorted((RUNS / sampler_dir).glob(f"{model}-rep*-[0-9]*.csv"))
    chains = [read_stan_csv(f)[0] for f in files]
    return chains


def stats(x):
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "mcse_mean": float(A._mcse_mean(x)), "mcse_sd": float(A._mcse_sd(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="native")
    args = ap.parse_args()
    out = {}
    for posterior, model, _ in POSTERIORS:
        ref = reference_draws(posterior)
        entry = {}
        for sampler, d in (("znuts", "znuts"), ("cmdstan", f"cmdstan-{args.label}")):
            chains = pooled(d, model)
            rows = {}
            for name, rdraws in ref.items():
                x = np.stack([c[name] for c in chains])
                s, r = stats(x), stats(rdraws)
                rows[name] = {
                    "ours": s, "reference": r,
                    "z_mean": (s["mean"] - r["mean"]) / np.hypot(s["mcse_mean"], r["mcse_mean"]),
                    "z_sd": (s["sd"] - r["sd"]) / np.hypot(s["mcse_sd"], r["mcse_sd"]),
                }
            entry[sampler] = {"chains": len(chains), "params": rows}
        out[model] = entry
        zz = [v for p in entry["znuts"]["params"].values() for v in (p["z_mean"], p["z_sd"])]
        zs = [v for p in entry["cmdstan"]["params"].values() for v in (p["z_mean"], p["z_sd"])]
        print(f"{model:26s} znuts max|z| {max(map(abs, zz)):5.2f}   cmdstan max|z| {max(map(abs, zs)):5.2f}   ({len(zz)} z-scores each)")

    # Synthetic targets with exact answers.
    truth = {"iid_normal_100": {f"x[{i + 1}]": (0.0, 1.0) for i in range(100)}, "funnel_10": {"v": (0.0, 3.0)}}
    for model, _ in SYNTHETIC:
        entry = {}
        for sampler, d in (("znuts", "znuts"), ("cmdstan", f"cmdstan-{args.label}")):
            chains = pooled(d, model)
            rows = {}
            for name, (mu, sd) in truth[model].items():
                x = np.stack([c[name] for c in chains])
                s = stats(x)
                rows[name] = {"ours": s, "truth": {"mean": mu, "sd": sd},
                              "z_mean": (s["mean"] - mu) / s["mcse_mean"], "z_sd": (s["sd"] - sd) / s["mcse_sd"]}
            entry[sampler] = {"chains": len(chains), "params": rows}
        out[model] = entry
        for sampler in ("znuts", "cmdstan"):
            z = [v for p in entry[sampler]["params"].values() for v in (p["z_mean"], p["z_sd"])]
            print(f"{model:26s} {sampler:8s} max|z| {max(map(abs, z)):6.2f}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
