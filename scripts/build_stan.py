"""Compile the Stan programs used by the comparison scripts with CmdStan.

Copies the seven posteriordb programs (from the pinned posteriordb checkout)
and the programs in scripts/stan/ into CMDSTAN_MODELS and builds each one.

    CMDSTAN_MODELS=/path/to/out python3 scripts/build_stan.py /path/to/cmdstan-2.40.0

For the timings in the README, CmdStan's make/local contained
`CXXFLAGS += -march=native`.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from common import CMDSTAN_MODELS, POSTERIORS, ROOT, posteriordb


def main():
    cmdstan = Path(sys.argv[1]).resolve()
    CMDSTAN_MODELS.mkdir(parents=True, exist_ok=True)
    sources = [posteriordb() / "models" / "stan" / f"{m}.stan" for _, m, _ in POSTERIORS]
    sources += sorted((ROOT / "scripts" / "stan").glob("*.stan"))
    for src in sources:
        shutil.copy(src, CMDSTAN_MODELS / src.name)
        target = CMDSTAN_MODELS / src.stem
        print("building", target, flush=True)
        subprocess.run(["make", "-j2", str(target)], cwd=cmdstan, check=True, capture_output=True)


if __name__ == "__main__":
    main()
