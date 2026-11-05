"""Figures, drawn only from results JSON (summary.json and ground_truth_*.json)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wildfire.io import ROOT, load_result, results_dir  # noqa: E402

COL = {"sim": "#1f1b18", "mpnn1": "#2a78d6", "structured": "#eb6834", "mpnn1_windblind": "#1baf7a",
       "logreg": "#eda100", "mpnn2": "#6250d6"}
LABEL = {"mpnn1": "generic MPNN, 1-hop", "mpnn2": "generic MPNN, 2-hop", "structured": "structured (true form)",
         "logreg": "logistic regression", "mpnn1_band": "1-hop + band data", "structured_band": "structured + band data",
         "mpnn1_windblind": "1-hop, wind-blind"}
PANELS = ["logreg", "mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band", "mpnn1_windblind"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#6b645c", "axes.labelcolor": "#2b2622",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fbf8f2",
                     "axes.facecolor": "#fbf8f2", "savefig.facecolor": "#fbf8f2"})


def logistic(x, xc, w):
    return 1 / (1 + np.exp(-(x - xc) / w))


def out_dir():
    d = results_dir().parent / "figures" if results_dir() == ROOT / "results" else results_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def crossing(S):
    fams = list(S["families"])
    fig, axes = plt.subplots(len(fams), len(PANELS), figsize=(15, 2.3 * len(fams)), sharey=True)
    for i, fam in enumerate(fams):
        F = S["families"][fam]
        gt = load_result(f"ground_truth_{fam}")["result"]["L128"]
        x = np.asarray(gt["x"])
        tc = F["truth"]["L128"]["xc"]
        xs = np.linspace(tc - 0.25, tc + 0.25, 300)
        for j, kind in enumerate(PANELS):
            ax = axes[i, j]
            if kind not in F["models"]:
                ax.axis("off")
                continue
            ax.plot(xs, logistic(xs, gt["xc"], gt["w"]), color=COL["sim"], lw=2, label="simulator")
            ax.plot(x, np.asarray(gt["k"]) / np.asarray(gt["n"]), "o", ms=2.5, color=COL["sim"])
            c = COL.get(kind.replace("_band", ""), "#888")
            ls = "--" if kind.endswith("_band") else "-"
            for d in F["models"][kind]["seeds"]:
                if d["xc"] is None:
                    continue
                ax.plot(xs, logistic(xs, d["xc"], d["width_10_90"] / (2 * np.log(9))), color=c, lw=1.2,
                        ls=ls, alpha=0.75)
            ax.axvline(tc, color="#9a948c", lw=0.6, ls=":")
            if i == 0:
                ax.set_title(LABEL[kind], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"family {fam}\nP(cross), L=128")
            if i == len(fams) - 1:
                ax.set_xlabel("base spread s")
            ax.set_xlim(xs[0], xs[-1])
    fig.suptitle("Crossing curves: simulator (black, points = measured) vs each model's 5 seeds (coloured)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir() / "crossing_curves.png", dpi=130)
    plt.close(fig)


def excess_vs_error(S):
    fams = list(S["families"])
    fig, axes = plt.subplots(1, len(fams), figsize=(12, 3.6))
    for ax, fam in zip(axes, fams, strict=True):
        F = S["families"][fam]
        thr = F["h2_threshold"]
        ax.axhspan(-thr, thr, color="#e9e3d8", lw=0)
        ax.axvline(S["margins"]["excellent"], color="#6b645c", lw=0.8, ls="--")
        for kind in PANELS:
            if kind not in F["models"]:
                continue
            pts = [(d["test"]["offband"]["excess"], d["err"]) for d in F["models"][kind]["seeds"]
                   if d["err"] is not None]
            if not pts:
                continue
            e, r = zip(*pts, strict=True)
            mk = "s" if kind.endswith("_band") else ("^" if kind == "mpnn1_windblind" else "o")
            ax.scatter(e, r, s=22, marker=mk, color=COL.get(kind.replace("_band", ""), "#888"),
                       edgecolor="#fbf8f2", lw=0.8, label=LABEL[kind], zorder=3)
        ax.axhline(0, color="#1f1b18", lw=0.6)
        ax.set_xscale("log")
        ax.set_title(f"family {fam}  (shaded: |error| below H2 threshold {thr:.3f})", fontsize=8.5)
        ax.set_xlabel("excess active-cell Brier over Bayes floor (off-band test)")
        ax.set_ylabel("critical s error (model - simulator)")
    axes[-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir() / "excess_vs_error.png", dpi=130)
    plt.close(fig)


def accuracy_trap(S):
    fams = list(S["families"])
    fig, axes = plt.subplots(1, len(fams), figsize=(12, 3.0), sharey=True)
    for ax, fam in zip(axes, fams, strict=True):
        F = S["families"][fam]
        rows = F["models"]["never"]["accuracy_by_s"]
        mid = [(r["s_lo"] + r["s_hi"]) / 2 for r in rows]
        ax.plot(mid, [100 * r["acc"] for r in rows], "-o", ms=3, color=COL["sim"], label="never-ignite")
        if "mpnn1" in F["models"]:
            acc = np.mean([d["test"]["all"]["all_cell_accuracy"] for d in F["models"]["mpnn1"]["seeds"]])
            ax.axhline(100 * acc, color=COL["mpnn1"], lw=1, ls="--", label="1-hop MPNN (all s)")
        ax.axvline(F["truth"]["L128"]["xc"], color="#9a948c", lw=0.6, ls=":")
        ax.set_title(f"family {fam}: never-ignite never crosses", fontsize=8.5)
        ax.set_xlabel("base spread s (L=64 test frames)")
        ax.set_ylabel("all-cell accuracy, %")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir() / "accuracy_trap.png", dpi=130)
    plt.close(fig)


def main():
    S = load_result("summary")["result"]
    crossing(S)
    excess_vs_error(S)
    accuracy_trap(S)


if __name__ == "__main__":
    main()
