"""Run znuts and CmdStan on the same posteriors with the same settings.

Each run is 4 chains, one after another on one core, with 1000 warmup and 1000
sampling iterations, adapt_delta 0.8 and maximum tree depth 10. Every run is
repeated with several seeds. Effective sample sizes are computed with ArviZ
for both samplers, from the draws each one wrote.

    CMDSTAN_MODELS=/path/to/models python3 scripts/bench.py [--reps 5] [--label native]

Draws go to .cache/runs/<sampler>/; per-run numbers go to docs/data/bench-<label>.json.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from arviz_stats.base import array_stats as A

from common import CMDSTAN_MODELS, DATA, POSTERIORS, RESULTS, ROOT, SYNTHETIC, read_stan_csv, znuts_binary

RUNS = ROOT / ".cache" / "runs"
CHAINS, WARMUP, SAMPLES = 4, 1000, 1000


def run_znuts(model, seed, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(znuts_binary()), "sample", model, "--chains", str(CHAINS), "--warmup", str(WARMUP),
                    "--samples", str(SAMPLES), "--seed", str(seed), "--out", str(out)], check=True, capture_output=True)
    return [read_stan_csv(f"{out}-{c + 1}.csv") for c in range(CHAINS)]


def run_cmdstan(model, stan_file, data, seed, out, models_dir):
    out.parent.mkdir(parents=True, exist_ok=True)
    chains = []
    for c in range(CHAINS):
        args = [str(models_dir / stan_file), "sample", f"num_warmup={WARMUP}", f"num_samples={SAMPLES}", "adapt", "delta=0.8",
                "algorithm=hmc", "engine=nuts", "max_depth=10", "init=2", "random", f"seed={seed}", f"id={c + 1}",
                "output", f"file={out}-{c + 1}.csv", "refresh=0"]
        if data:
            args += ["data", f"file={DATA / data}"]
        subprocess.run(args, check=True, capture_output=True)
        chains.append(read_stan_csv(f"{out}-{c + 1}.csv"))
    return chains


def metrics(chains, names):
    cols = {n: np.stack([c[0][n] for c in chains]) for n in names}
    ess = {n: float(A._ess_bulk(v)) for n, v in cols.items()}
    rhat = {n: float(A._rhat_rank(v)) for n, v in cols.items()}
    n_leapfrog = float(sum(c[0]["n_leapfrog__"].sum() for c in chains))
    return {
        "min_ess_bulk": min(ess.values()),
        "min_ess_param": min(ess, key=ess.get),
        "max_rhat": max(rhat.values()),
        "sampling_seconds": sum(c[1]["sampling"] for c in chains),
        "warmup_seconds": sum(c[1]["warmup"] for c in chains),
        "n_leapfrog": n_leapfrog,
        "divergent": int(sum(c[0]["divergent__"].sum() for c in chains)),
        "stepsize": [float(c[0]["stepsize__"][0]) for c in chains],
        "mean_treedepth": float(np.mean([c[0]["treedepth__"].mean() for c in chains])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="native")
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--stan-file", default=None, help="run this Stan program instead (with --only and --cmdstan-only)")
    ap.add_argument("--cmdstan-only", action="store_true")
    args = ap.parse_args()

    targets = [(m, m, d) for _, m, d in POSTERIORS] + [(z, s, None) for z, s in SYNTHETIC]
    if args.only:
        targets = [t for t in targets if t[0] in args.only.split(",")]
    if args.stan_file:
        targets = [(m, args.stan_file, d) for m, _, d in targets]
    results = []
    for model, stan_file, data in targets:
        for rep in range(1, args.reps + 1):
            seed = 1000 + rep
            t0 = time.time()
            names = None
            pairs = []
            if not args.cmdstan_only:
                z = run_znuts(model, seed, RUNS / "znuts" / f"{model}-rep{rep}")
                names = [n for n in z[0][0] if not n.endswith("__")]
                pairs.append(("znuts", z))
            s = run_cmdstan(model, stan_file, data, seed, RUNS / f"cmdstan-{args.label}" / f"{model}-rep{rep}", CMDSTAN_MODELS)
            if names is None:
                ref = run_znuts(model, seed, RUNS / "tmp" / model)  # only for the column names
                names = [n for n in ref[0][0] if not n.endswith("__")]
            pairs.append(("cmdstan", s))
            for sampler, chains in pairs:
                m = metrics(chains, names)
                m.update(model=model, sampler=sampler, rep=rep, seed=seed, stan_file=stan_file)
                results.append(m)
                print(f"{model:26s} rep {rep} {sampler:8s} ESS/s {m['min_ess_bulk'] / m['sampling_seconds']:9.0f}  "
                      f"us/leapfrog {1e6 * m['sampling_seconds'] / m['n_leapfrog']:8.2f}  ({time.time() - t0:.0f} s)", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"bench-{args.label}.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
