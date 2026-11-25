"""Figures for the README.

    python3 scripts/bench.py && python3 scripts/validate.py   # docs/data/*.json
    python3 scripts/figures.py
"""
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import FIGURES, POSTERIORS, RESULTS, ROOT, reference_draws, read_stan_csv, znuts_binary  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})
RUNS = ROOT / ".cache" / "runs"
SHORT = {"eight_schools_noncentered": "eight schools", "arK": "arK", "gp_pois_regr": "gp_pois_regr",
         "low_dim_gauss_mix": "gauss mixture", "wells_dist": "wells", "garch11": "garch11", "dogs": "dogs",
         "iid_normal_100": "normal, 100-D", "funnel_10": "funnel, 10-D"}


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def trajectory():
    """One transition on a correlated 2-D normal: the doublings and the U-turn."""
    out = subprocess.run([str(znuts_binary()), "trace", "corr_normal_2", "--seed", "4", "--stepsize", "0.15"],
                         capture_output=True, text=True, check=True).stdout
    tr = json.loads(out)
    pts = tr["points"]
    start = pts[0]
    states = pts[1:]
    depth = tr["treedepth"]
    # Position of each state in time: backward states are negative.
    pos = {}
    n_fwd = n_bwd = 0
    for p in states:
        if p["dir"] > 0:
            n_fwd += 1
            pos[id(p)] = n_fwd
        else:
            n_bwd += 1
            pos[id(p)] = -n_bwd
    pos[id(start)] = 0
    accepted = [p for p in states if p["level"] < depth]
    rejected = [p for p in states if p["level"] >= depth]
    ordered = sorted(accepted + [start], key=lambda p: pos[id(p)])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.45})
    # Panel A: the path in parameter space over the target's contours.
    g = np.linspace(-3.2, 3.2, 300)
    X, Y = np.meshgrid(g, g)
    rho = 0.8
    Z = (X**2 - 2 * rho * X * Y + Y**2) / (1 - rho**2)
    ax.contour(X, Y, Z, levels=[1, 4, 9], colors=[GRID], linewidths=1.2)
    qs = np.array([p["q"] for p in ordered])
    ax.plot(qs[:, 0], qs[:, 1], color=AXIS, lw=1, zorder=1)
    for p in accepted:
        ax.scatter(*p["q"], s=34, color=RAMP[min(p["level"], len(RAMP) - 1)], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    if rejected:
        rq = np.array([start["q"]] + [p["q"] for p in sorted(rejected, key=lambda p: abs(pos[id(p)]))])
        ax.plot(rq[:, 0], rq[:, 1], color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.scatter(rq[1:, 0], rq[1:, 1], s=26, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2, zorder=2)
        far = rq[-1]
        ax.annotate("turned back:\nlast doubling rejected", far, xytext=(-6, -26), textcoords="offset points",
                    color=INK2, fontsize=8.5, ha="left")
    ax.scatter(*start["q"], s=80, facecolor=SURFACE, edgecolor=INK, linewidth=2, zorder=4)
    ax.annotate("start", start["q"], xytext=(10, -14), textcoords="offset points", color=INK, fontsize=9)
    ax.scatter(*tr["selected"], s=180, marker="*", color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=5)
    ax.annotate("chosen state", tr["selected"], xytext=(10, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("x[1]")
    ax.set_ylabel("x[2]")
    ax.set_title("A. One transition on a correlated normal")

    # Panel B: which steps each doubling added, one row per doubling.
    labels = ["final trajectory"]
    for lvl in range(depth + (1 if rejected else 0)):
        grp = [p for p in states if p["level"] == lvl]
        d = "forward" if grp[0]["dir"] > 0 else "backward"
        n_new = 2**lvl
        if lvl < depth:
            labels.append(f"doubling {lvl + 1}: {n_new} step{'s' if n_new > 1 else ''} {d}")
        else:
            labels.append(f"doubling {lvl + 1}: {d}, rejected")
        for p in grp:
            if lvl < depth:
                bx.scatter(pos[id(p)], lvl + 1, s=30, color=RAMP[min(lvl, len(RAMP) - 1)], edgecolor=SURFACE,
                           linewidth=1, zorder=3)
            else:
                bx.scatter(pos[id(p)], lvl + 1, s=24, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2, zorder=3)
        if lvl >= depth:
            bx.annotate(f"turned back after {len(grp)} of {n_new} steps", (min(pos[id(p)] for p in grp), lvl + 1),
                        xytext=(0, -14), textcoords="offset points", fontsize=8.5, color=INK2)
    lo = min(pos[id(p)] for p in ordered)
    hi = max(pos[id(p)] for p in ordered)
    bx.plot([lo, hi], [0, 0], color=INK2, lw=2, zorder=2)
    bx.scatter(0, 0, s=70, facecolor=SURFACE, edgecolor=INK, linewidth=2, zorder=4)
    sel = min(accepted + [start], key=lambda p: np.hypot(p["q"][0] - tr["selected"][0], p["q"][1] - tr["selected"][1]))
    bx.scatter(pos[id(sel)], 0, s=160, marker="*", color=ORANGE, edgecolor=SURFACE, zorder=5)
    bx.set_yticks(range(len(labels)), labels)
    bx.set_ylim(len(labels) - 0.3, -0.6)
    bx.set_xlabel("leapfrog step relative to the start (negative: backward in time)")
    span = max(abs(v) for v in pos.values())
    bx.set_xlim(-span - 2, span + 2)
    bx.grid(axis="y", visible=False)
    bx.set_title(f"B. Doublings in random directions ({tr['n_leapfrog']} leapfrog steps)")
    save(fig, "trajectory.png")


def zscores():
    v = json.loads((RESULTS / "validation.json").read_text())
    models = [m for _, m, _ in POSTERIORS] + ["iid_normal_100"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(-2, 2, color="#f0efec", zorder=0)
    for x in (-3, 3):
        ax.axvline(x, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1)
    for i, m in enumerate(models):
        for k, (sampler, color) in enumerate((("znuts", BLUE), ("cmdstan", ORANGE))):
            params = v[m][sampler]["params"].values()
            zs = [p["z_mean"] for p in params] + [p["z_sd"] for p in params]
            y = i + (-0.17 if k == 0 else 0.17) + np.random.default_rng(i * 2 + k).uniform(-0.07, 0.07, len(zs))
            ax.scatter(np.clip(zs, -5.8, 5.8), y, s=14, color=color, alpha=0.85, edgecolor="none", zorder=3)
    ax.set_yticks(range(len(models)), [SHORT[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(-6, 6)
    ax.set_xlabel("z-score of a posterior mean or standard deviation against the reference")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="znuts"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="CmdStan 2.40")],
              loc="upper right")
    ax.set_title("Both samplers agree with posteriordb's reference draws within Monte Carlo error")
    save(fig, "zscores.png")


def speed():
    runs = json.loads((RESULTS / "bench-native.json").read_text())
    models = [m for _, m, _ in POSTERIORS] + ["iid_normal_100"]

    def per(sampler, key):
        out = []
        for m in models:
            vals = []
            for r in runs:
                if r["model"] == m and r["sampler"] == sampler:
                    vals.append(key(r))
            out.append((np.median(vals), min(vals), max(vals)))
        return np.array(out)

    panels = [
        ("ESS per 1,000 gradients", lambda r: 1000 * r["min_ess_bulk"] / r["n_leapfrog"], False),
        ("microseconds per gradient (leapfrog step)", lambda r: 1e6 * r["sampling_seconds"] / r["n_leapfrog"], True),
        ("ESS per second", lambda r: r["min_ess_bulk"] / r["sampling_seconds"], True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    y = np.arange(len(models))
    for ax, (title, key, logx) in zip(axes, panels):
        for k, (sampler, color, label) in enumerate((("znuts", BLUE, "znuts"), ("cmdstan", ORANGE, "CmdStan 2.40"))):
            d = per(sampler, key)
            yy = y + (-0.18 if k == 0 else 0.18)
            ax.barh(yy, d[:, 0], height=0.34, color=color, label=label, zorder=2)
            ax.errorbar(d[:, 0], yy, xerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], fmt="none", ecolor=INK2, elinewidth=1,
                        capsize=2, zorder=3)
        if logx:
            ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [SHORT[m] for m in models])
    axes[0].invert_yaxis()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
               labels=["znuts", "CmdStan 2.40"], loc="upper right", bbox_to_anchor=(0.98, 1.0), ncol=2)
    fig.suptitle("Same algorithm, same efficiency per gradient; the difference is the cost of each gradient",
                 x=0.06, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.06, -0.02, "Bars: median of 5 runs of 4 chains (1000 warmup, 1000 draws each); whiskers: smallest and largest "
             "run. ESS is the smallest bulk ESS over the parameters. One core of a 2.1 GHz Xeon.", fontsize=8.5, color=INK2)
    save(fig, "speed.png")


def marginals():
    picks = [("eight_schools-eight_schools_noncentered", "eight_schools_noncentered", "tau"),
             ("gp_pois_regr-gp_pois_regr", "gp_pois_regr", "rho"),
             ("garch-garch11", "garch11", "alpha1"),
             ("low_dim_gauss_mix-low_dim_gauss_mix", "low_dim_gauss_mix", "theta")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, (posterior, model, name) in zip(axes, picks):
        ref = reference_draws(posterior)[name].ravel()
        files = sorted((RUNS / "znuts").glob(f"{model}-rep*-[0-9]*.csv"))
        ours = np.concatenate([read_stan_csv(f)[0][name] for f in files])
        lo, hi = np.quantile(np.concatenate([ref, ours]), [0.002, 0.998])
        bins = np.linspace(lo, hi, 60)
        ax.hist(ref, bins=bins, density=True, color="#dcdbd5", label="posteriordb reference", zorder=1)
        ax.hist(ours, bins=bins, density=True, histtype="step", color=BLUE, lw=1.8, label="znuts", zorder=2)
        ax.set_title(f"{SHORT[model]}: {name}", fontsize=10)
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="upper right", fontsize=8.5)
    save(fig, "marginals.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    trajectory()
    marginals()
    zscores()
    speed()


if __name__ == "__main__":
    main()
