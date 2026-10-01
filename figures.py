"""Figures for the paper, drawn from results/*.json and results/*.csv."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"two": "#2a78d6", "fix1": "#eb6834", "fix10": "#1baf7a"}   # validated slots 1-3
STYLE = {"two": "-", "fix1": "--", "fix10": ":"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fig_trajectory(figsize=(3.4, 4.6), out="trajectory.pdf", ylim=(0.70, 1.0), legend_loc="lower left"):
    tr = json.load(open("results/digits_trajectory.json"))
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 0.8, 1.1], "hspace": 0.32})
    t0 = tr["shift_at"]
    # (a) accuracy
    a = ax[0]
    a.axvspan(t0, 20000, color=GRID, alpha=0.5, lw=0)
    a.plot(tr["rolling_acc_t"], tr["rolling_acc"], color=INK2, lw=1.0, label="true accuracy (500-item window)")
    a.axhline(tr["declared"], color=INK, lw=0.9, ls="-")
    a.axhline(tr["violation"], color=INK, lw=0.9, ls="--")
    a.text(19800, tr["declared"] + 0.006, f"declared {tr['declared']*100:.1f}%", color=INK, fontsize=7, ha="right")
    a.text(19800, tr["violation"] - 0.026, f"violation level {tr['violation']*100:.1f}%", color=INK, fontsize=7, ha="right")
    est = [(t, e) for t, e in tr["two-tier, relative trigger"]["atc"] if e is not None]
    a.plot([t for t, _ in est], [e for _, e in est], color=C["two"], lw=1.0,
           label="Tier 1 label-free estimate (ATC)")
    a.set_ylim(*ylim); a.set_ylabel("accuracy")
    a.legend(loc=legend_loc, fontsize=6.5, handlelength=1.6)
    a.set_title("(a) Deployed accuracy drops; the declaration stays", loc="left")
    # (b) audit rate
    b = ax[1]
    for key, name in (("two", "two-tier, relative trigger"), ("fix1", "fixed 1%")):
        r = tr[name]["rates"]
        xs = [t for t, _ in r] + [20000]; ys = [v * 100 for _, v in r] + [r[-1][1] * 100]
        b.step(xs, ys, where="post", color=C[key], ls=STYLE[key], lw=1.2,
               label=name if key == "fix1" else "two-tier")
    b.set_ylim(0, 11.5); b.set_yticks([0, 5, 10]); b.set_ylabel("audit rate (%)")
    b.legend(loc="center right", fontsize=6.5, ncol=2, handlelength=1.8)
    b.set_title("(b) Tier 1 raises the audit rate after the shift", loc="left")
    # (c) wealth
    c = ax[2]
    for key, name in (("two", "two-tier, relative trigger"), ("fix1", "fixed 1%")):
        w = tr[name]["wealth_trace"]
        c.plot([t for t, _ in w], [max(v, 1e-3) for _, v in w], color=C[key], ls=STYLE[key], lw=1.2)
        ra = tr[name]["rejected_at"]
        if ra is not None:
            c.plot([ra], [1 / tr["delta"]], "o", ms=4, color=C[key])
            if key == "two":
                c.text(ra + 350, 0.35, f"two-tier:\ninvalid at item {ra:,}", fontsize=6.5,
                       color=INK, ha="left", va="center")
            else:
                c.text(ra + 350, 60, f"fixed 1%:\ninvalid at item {ra:,}", fontsize=6.5,
                       color=INK, ha="left", va="center")
    c.axhline(1 / tr["delta"], color=INK, lw=0.9, ls="--")
    c.text(150, 1 / tr["delta"] * 1.35, r"reject at $1/\delta = 20$", fontsize=7, color=INK)
    c.set_yscale("log"); c.set_ylim(0.05, 200); c.set_ylabel("wealth $M_n$ (log)")
    c.set_xlabel("deployment item"); c.set_xlim(0, 20000)
    c.set_title("(c) The labeled test decides", loc="left")
    fig.savefig(f"{OUT}/{out}")
    plt.close(fig)


def fig_effect_size(figsize=(3.4, 1.75), out="effect_size.pdf"):
    df = pd.read_csv("results/effect_size.csv")
    names = {"two-tier 1% -> 10%": "two", "fixed 1%": "fix1", "fixed 10%": "fix10"}
    labels = {"two": "two-tier 1% to 10%", "fix1": "fixed 1%", "fix10": "fixed 10%"}
    fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.45})
    for pname, key in names.items():
        d = df[df.policy == pname].sort_values("post_accuracy")
        x = d.post_accuracy * 100
        ax[0].plot(x, d.median_delay, color=C[key], ls=STYLE[key], lw=1.2, marker="o", ms=2.5)
        ax[1].plot(x, d.mean_labels, color=C[key], ls=STYLE[key], lw=1.2, marker="o", ms=2.5)
    ax[0].set_yscale("log"); ax[0].set_ylabel("items to invalidation")
    ax[1].set_ylabel("labels used")
    for a in ax:
        a.set_xlabel("accuracy after shift (%)"); a.invert_xaxis()
    handles = [plt.Line2D([], [], color=C[k], ls=STYLE[k], lw=1.2, marker="o", ms=2.5) for k in names.values()]
    fig.legend(handles, [labels[k] for k in names.values()], loc="upper center", ncol=3,
               fontsize=6.5, bbox_to_anchor=(0.5, 1.1), handlelength=2.2)
    ax[0].set_title("(a) Detection speed", loc="left"); ax[1].set_title("(b) Labeling cost", loc="left")
    fig.savefig(f"{OUT}/{out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_trajectory(); fig_effect_size()
    # single-column versions for the ACM Small (CS&Law) layout
    fig_trajectory(figsize=(4.4, 5.0), out="trajectory_wide.pdf", ylim=(0.62, 1.0), legend_loc="lower right")
    fig_effect_size(figsize=(5.0, 2.0), out="effect_size_wide.pdf")
    print(os.listdir(OUT))
