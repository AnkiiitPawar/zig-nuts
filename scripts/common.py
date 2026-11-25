"""Shared paths and helpers for the scripts."""
import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "posteriordb"
RESULTS = ROOT / "docs" / "data"
FIGURES = ROOT / "docs"

# posteriordb posterior name, model name used by znuts and by the Stan file, data file
POSTERIORS = [
    ("eight_schools-eight_schools_noncentered", "eight_schools_noncentered", "eight_schools.json"),
    ("arK-arK", "arK", "arK.json"),
    ("gp_pois_regr-gp_pois_regr", "gp_pois_regr", "gp_pois_regr.json"),
    ("low_dim_gauss_mix-low_dim_gauss_mix", "low_dim_gauss_mix", "low_dim_gauss_mix.json"),
    ("wells_data-wells_dist", "wells_dist", "wells_data.json"),
    ("garch-garch11", "garch11", "garch.json"),
    ("dogs-dogs", "dogs", "dogs.json"),
]
# synthetic models: znuts name, Stan file name
SYNTHETIC = [("iid_normal_100", "iid_normal100"), ("funnel_10", "funnel10")]

# posteriordb at a fixed commit, for the reference draws and the Stan programs
POSTERIORDB_URL = "https://github.com/stan-dev/posteriordb.git"
POSTERIORDB_COMMIT = "91027f9cdd13b7e76283fec3f2898f59d4b83ec4"
POSTERIORDB = Path(os.environ.get("POSTERIORDB") or ROOT / ".cache" / "posteriordb")

# CmdStan, built separately; see README
CMDSTAN_MODELS = Path(os.environ.get("CMDSTAN_MODELS") or ROOT / ".cache" / "stan")


def znuts_binary():
    """The znuts executable, built with ReleaseFast if it is not there yet.
    Set ZIG to the compiler command if it is not `zig` (e.g. "python3 -m ziglang")."""
    exe = ROOT / "zig-out" / "bin" / "znuts"
    if not exe.exists():
        subprocess.run(shlex.split(os.environ.get("ZIG", "zig")) + ["build", "-Doptimize=ReleaseFast"], cwd=ROOT, check=True)
    return exe


def posteriordb():
    """A sparse checkout of posteriordb at the pinned commit."""
    if not (POSTERIORDB / "posterior_database").exists():
        POSTERIORDB.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", POSTERIORDB_URL, str(POSTERIORDB)], check=True)
        subprocess.run(["git", "-C", str(POSTERIORDB), "sparse-checkout", "set", "posterior_database/reference_posteriors/draws", "posterior_database/models/stan"], check=True)
        subprocess.run(["git", "-C", str(POSTERIORDB), "checkout", POSTERIORDB_COMMIT], check=True)
    return POSTERIORDB / "posterior_database"


def reference_draws(posterior):
    """posteriordb reference draws as {name: array(chains, draws)}."""
    import zipfile
    path = posteriordb() / "reference_posteriors" / "draws" / "draws" / f"{posterior}.json.zip"
    with zipfile.ZipFile(path) as z:
        chains = json.loads(z.read(z.namelist()[0]))
    return {k: np.array([c[k] for c in chains]) for k in chains[0]}


def read_stan_csv(path):
    """Columns of a Stan-format CSV file (ours or CmdStan's) and its timing."""
    header, rows, timing = None, [], {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if "Elapsed Time" in line or "seconds (Sampling)" in line:
                parts = line.replace("#", "").replace("Elapsed Time:", "").split()
                timing["warmup" if "Warm-up" in line else "sampling"] = float(parts[0])
            continue
        if not line.strip():
            continue
        if header is None:
            header = line.split(",")
        else:
            rows.append([float(v) for v in line.split(",")])
    arr = np.array(rows)
    return {h.replace(".", "[", 1) + "]" if "." in h else h: arr[:, j] for j, h in enumerate(header)}, timing


def stan_name(column):
    """CmdStan writes theta.1, znuts and posteriordb write theta[1]."""
    if "." in column:
        base, idx = column.split(".", 1)
        return f"{base}[{idx.replace('.', ',')}]"
    return column
