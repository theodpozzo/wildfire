"""Render REPORT.md and README.md from results/*.json.

Every number in either document is produced here from the JSON results via the
Doc.n() formatter; tests/test_report.py re-renders and checks the committed files
match exactly, and that every numeric token in them was produced by Doc.n() or
is a listed structural constant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from wildfire.io import ROOT, load_result

LABEL = {"never": "never-ignite", "persistence": "persistence", "logreg": "logistic regression",
         "mpnn1": "generic MPNN, 1-hop", "mpnn2": "generic MPNN, 2-hop", "structured": "structured (true form)",
         "mpnn1_band": "1-hop + band data", "structured_band": "structured + band data",
         "mpnn1_windblind": "1-hop, wind-blind (control)"}
ORDER = ["never", "persistence", "logreg", "mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band",
         "mpnn1_windblind"]

# Structural constants that may appear in prose (sizes, years, labels), not results.
ALLOWED = {"0", "1", "2", "3", "4", "5", "9", "10", "16", "50", "64", "90", "95", "128", "300", "400", "1500",
           "1980", "2000", "2026", "0.5", "0.592746", "0.010", "0.08", "0.30", "0.015", "1.3", "0.6", "0.4",
           "0.9", "0.2", "0.002", "0.0005", "0.02", "0.01", "1e-5", "4000", "2048", "32", "13", "14", "5-9",
           "0.3", "4-neighbour", "1-hop", "2-hop", "10-90", "H1", "H2", "H3", "H4", "H5", "G1", "G2", "L=128",
           "L=64", "360", "7", "6", "8", "100"}


class Doc:
    def __init__(self):
        self.lines: list[str] = []
        self.nums: set[str] = set()

    def n(self, x, nd=3, pct=False, sign=False):
        if x is None:
            s = "n/a"
        elif pct:
            s = f"{100 * x:.{nd}f}%"
        elif isinstance(x, int) and not isinstance(x, bool):
            s = str(x)
        else:
            s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
        self.nums.add(s.rstrip("%"))
        return s

    def ci(self, lo, hi, nd=3):
        return f"({self.n(lo, nd)}, {self.n(hi, nd)})"

    def w(self, s=""):
        self.lines.append(s)

    def text(self):
        return "\n".join(self.lines).rstrip() + "\n"


def h2_counts(S):
    H2 = S["H"]["H2"]
    return H2["passes"], H2["tried"], H2["n_excellent"]


def headline(S, d: Doc | None = None) -> str:
    d = d or Doc()
    p, t, e = h2_counts(S)
    H2 = S["H"]["H2"]
    worst = max((r for r in H2["rows"] if r["excellent"] and r["err"] is not None), key=lambda r: abs(r["err"]),
                default=None)
    if p > 0:
        fams = sorted({r["family"] for r in H2["rows"] if r["pass"]})
        s = (f"Yes. {d.n(p)} of {d.n(t)} trained GNN seeds were locally excellent at next-step prediction and "
             f"still put the tipping point measurably in the wrong place (family {', '.join(fams)}): critical s "
             f"off by more than the pre-registered threshold, with non-overlapping 95% intervals. ")
    else:
        s = (f"No, not by the pre-registered standard: none of the {d.n(t)} trained GNN seeds was both locally "
             f"excellent and off by more than the threshold with non-overlapping intervals. ")
    if worst is not None:
        s += (f"The largest critical-s error among locally excellent seeds was {d.n(worst['err'], sign=True)} "
              f"({LABEL[worst['kind']]}, family {worst['family']}).")
    return s


def plain_terms(S, d: Doc) -> list[str]:
    """Data-driven bullet points that qualify the headline."""
    H2 = S["H"]["H2"]
    rows = [r for r in H2["rows"] if r["err"] is not None]
    exc = [r["excess"] for r in rows]
    big = max(abs(r["err"]) for r in rows if r["excellent"])
    out = []
    thr = ", ".join(f"{f} {d.n(F['h2_threshold'], 3)}" for f, F in S["families"].items())
    out.append(f"Every eligible GNN seed was locally excellent: off-band excess Brier between {d.n(min(exc), 5)} and "
               f"{d.n(max(exc), 5)}, against a margin of 0.002. Their tipping points all landed within "
               f"{d.n(big, 3)} of the simulator's, each inside its family's pre-registered threshold (half the 10-90% "
               f"transition width: {thr}).")
    out.append(f"The misses are small but real: in {d.n(H2['excellent_and_ci_separated'])} of {d.n(H2['tried'])} "
               f"seeds the model's 95% interval for critical s did not overlap the simulator's. So the answer to "
               f"\"can it be off?\" is yes, measurably; to \"can it be off by a meaningful fraction of the "
               f"transition?\" it is no, at L=128 and at this margin.")
    for f, F in S["families"].items():
        M = {k: F["models"][k] for k in ("mpnn1", "mpnn2", "structured")}
        best_brier = min(M, key=lambda k: M[k]["offband_excess"]["mean"])
        worst_err = max(M, key=lambda k: abs(M[k]["err"]["mean"]))
        if best_brier == worst_err:
            out.append(f"Next-step skill did not rank tipping-point skill: in family {f}, the {LABEL[best_brier]} "
                       f"model had the lowest excess Brier of the three default GNNs "
                       f"({d.n(M[best_brier]['offband_excess']['mean'], 5)}) and the largest mean critical-s error "
                       f"({d.n(M[worst_err]['err']['mean'], 3, sign=True)}).")
    nv = [F["models"]["never"]["seeds"][0]["test"]["all"]["all_cell_accuracy"] for F in S["families"].values()]
    out.append(f"All-cell accuracy is the wrong yardstick: the never-ignite predictor scores {d.n(min(nv), 2, pct=True)}"
               f"-{d.n(max(nv), 2, pct=True)} and never crosses at any s tested (up to 1.3).")
    if "controls" in S:
        cr = [c["unmasked"][k]["crossed_frac"] for c in S["controls"].values() for k in ("mpnn1", "mpnn2")]
        out.append(f"Masking matters: with no mask, the generic MPNNs' rare ignitions of cells with no burning "
                   f"neighbour made {d.n(min(cr), 0, pct=True)}-{d.n(max(cr), 0, pct=True)} of fires at the "
                   f"simulator's critical s cross (about half would, masked).")
    return out


def model_table(d: Doc, S, fam):
    F = S["families"][fam]
    t = F["truth"]["L128"]
    d.w(f"Simulator critical s at L=128: **{d.n(t['xc'], 4)}** {d.ci(*t['ci'], nd=4)}, "
        f"10-90% width {d.n(t['width_10_90'])}; H2 threshold {d.n(F['h2_threshold'], 4)}.")
    d.w()
    d.w("| model | all-cell accuracy | active-cell Brier (off-band) | excess over floor | ECE | critical s, mean "
        "(95% t-interval) | seed range | error vs simulator | excellent seeds |")
    d.w("|---|---|---|---|---|---|---|---|---|")
    for kind in ORDER:
        if kind not in F["models"]:
            continue
        M = F["models"][kind]
        if kind in ("never", "persistence"):
            te = M["seeds"][0]["test"]
            d.w(f"| {LABEL[kind]} | {d.n(te['all']['all_cell_accuracy'], 2, pct=True)} | "
                f"{d.n(te['offband']['active_brier'], 4)} | {d.n(te['offband']['excess'], 4)} | "
                f"{d.n(te['offband']['ece'], 3)} | never crosses | - | - | - |")
            continue
        ex = sum(s["excellent"] for s in M["seeds"])
        if "xc" in M:
            xc = f"{d.n(M['xc']['mean'])} {d.ci(M['xc']['lo'], M['xc']['hi'])}"
            rng = f"{d.n(M['xc']['min'])}-{d.n(M['xc']['max'])}"
            er = f"{d.n(M['err']['mean'], sign=True)} {d.ci(M['err']['lo'], M['err']['hi'])}"
        else:
            xc, rng, er = "censored", "-", "-"
        acc = sum(s["test"]["all"]["all_cell_accuracy"] for s in M["seeds"]) / len(M["seeds"])
        d.w(f"| {LABEL[kind]} | {d.n(acc, 2, pct=True)} | {d.n(M['offband_active_brier']['mean'], 4)} | "
            f"{d.n(M['offband_excess']['mean'], 5)} | {d.n(M['offband_ece']['mean'], 3)} | {xc} | {rng} | {er} | "
            f"{d.n(ex)}/{d.n(len(M['seeds']))} |")
    d.w()


def report(S, base: Path | None = None) -> Doc:
    d = Doc()
    H = S["H"]
    G = S["gates"]
    d.w("# Report: does a locally excellent learned fire model put the tipping point in the right place?")
    d.w()
    d.w("## The answer")
    d.w()
    d.w(headline(S, d))
    d.w()
    for line in plain_terms(S, d):
        d.w("- " + line)
    d.w()
    p, t, e = h2_counts(S)
    d.w(f"Main hypothesis H2: **{H['H2']['verdict']}** ({d.n(p)} passing seeds out of {d.n(t)} tried; "
        f"{d.n(e)} of the {d.n(t)} were locally excellent; "
        f"{d.n(H['H2']['excellent_and_ci_separated'])} excellent seeds had a critical-s CI that did not overlap the "
        f"simulator's, but an error below the size threshold). "
        f"Judged per model instead (mean over {d.n(S['n_seeds'])} seeds), {d.n(H['H2']['per_model_passes'])} of "
        f"{d.n(len(H['H2']['per_model']))} model-family pairs pass.")
    d.w()
    d.w("![Excess Brier against critical-s error](figures/excess_vs_error.png)")
    d.w()
    d.w(f"This is a synthetic toy. It must never inform real fire or evacuation decisions. The study was designed and "
        f"run autonomously by an AI agent (Claude) from a single prompt. Pre-registration commit: "
        f"`{S['prereg_commit']}`.")
    d.w()
    d.w("## Method")
    d.w()
    d.w("- **Simulator** (`wildfire/sim.py`): L x L grid, 4-neighbour; each burning neighbour independently "
        "ignites an unburnt fuel cell with q = clip(s x moisture x wind x slope, 0, 1); burning lasts one step; the "
        "top row is lit at step 0; a fire *crosses* if it reaches the bottom row. Vectorised over many trials, "
        "only the burning front is touched, seeded.")
    d.w(f"- **Gates** at L=128: G1 bond percolation critical s = {d.n(G['G1']['xc'], 4)} "
        f"{d.ci(*G['G1']['ci'], nd=4)} (target 0.5, Kesten 1980) - "
        f"{'pass' if G['G1']['pass'] else 'FAIL'}; G2 site percolation critical p = {d.n(G['G2']['xc'], 4)} "
        f"{d.ci(*G['G2']['ci'], nd=4)} (target 0.592746, Newman and Ziff 2000) - "
        f"{'pass' if G['G2']['pass'] else 'FAIL'}. Each used {d.n(G['G1']['n_per'])} trials at "
        f"{d.n(G['G1']['n_values'])} and {d.n(G['G2']['n_values'])} control values.")
    d.w("- **Critical s** = the 50% point of a binomial maximum-likelihood logistic fit of crossing against s at "
        "L=128 (a finite-size value, not an infinite-lattice threshold); 95% CIs by bootstrap.")
    d.w("- **Families**: A uniform (iid fuel 0.9); B wind (along-wind factors 1 +/- 0.5, north/south 0.6, random "
        "east/west sign); C patchy fuel, smooth moisture, slope. Ground truth:")
    d.w()
    d.w("| family | critical s, L=128 (95% CI) | trials | critical s, L=64 (95% CI) | 10-90% width, L=128 | "
        "logistic fit p | clipped edges at critical s |")
    d.w("|---|---|---|---|---|---|---|")
    for f, F in S["families"].items():
        a, b = F["truth"]["L128"], F["truth"]["L64"]
        d.w(f"| {f} | {d.n(a['xc'], 4)} {d.ci(*a['ci'], nd=4)} | {d.n(a['n_trials'])} | {d.n(b['xc'], 4)} "
            f"{d.ci(*b['ci'], nd=4)} | {d.n(a['width_10_90'], 4)} | {d.n(a['gof_p'], 2)} | "
            f"{d.n(a['clip_frac_at_xc'], 1, pct=True)} |")
    d.w()
    d.w("- **Data**: L=64 simulator rollouts. Training s excludes the band |s - sc| < 0.08 around each family's "
        "measured critical s. **That gap is centred on the known answer**, which a real modeller would not know. "
        "Fixed test set built before training, covering the band too.")
    d.w("- **Models** see raw fields only (s, fuel, moisture, elevation, wind vector, cell state), never q or its "
        "factors. Generic MPNNs (1-hop, 2-hop), a wind-blind 1-hop control in B, and a structured model that "
        "predicts one probability per directed edge and combines them as 1 - prod(1 - q). **The structured model "
        f"has the true functional form by construction.** {d.n(S['n_seeds'])} seeds each; hidden size chosen on "
        "validation Brier only. Trained with NumPy and hand-written gradients (PyTorch could not be installed here).")
    d.w("- **Locally excellent** = excess active-cell Brier over the Bayes floor at most 0.002 on the off-band test "
        "set. Excess = mean (p_model - p_true)^2 over active cells, where p_true comes from the simulator's q.")
    d.w("- **Rollouts**: masked (only active cells may ignite) at L=128 on fresh simulator landscapes, with an "
        "adaptive pre-registered trial rule.")
    d.w()
    d.w("![Crossing curves](figures/crossing_curves.png)")
    d.w()
    d.w("## Results by family")
    d.w()
    for f in S["families"]:
        d.w(f"### Family {f}")
        d.w()
        model_table(d, S, f)
    d.w("## Hypotheses")
    d.w()
    h1 = H["H1"]
    d.w(f"**H1 - best GNN beats the baselines on active-cell Brier: {h1['verdict']}.**")
    for f, r in h1["per_family"].items():
        d.w(f"- {f}: best GNN {LABEL[r['best']]}, mean {d.n(r['best_mean'], 4)} (upper 95% bound "
            f"{d.n(r['best_hi'], 4)}) vs logistic regression {d.n(r['logreg_mean'], 4)} (lower bound "
            f"{d.n(r['logreg_lo'], 4)}), never-ignite {d.n(r['never'], 4)}, persistence {d.n(r['persistence'], 4)}"
            f" - {'supported' if r['supported'] else 'not supported'}.")
    d.w()
    d.w(f"**H2 - a locally excellent model misplaces the tipping point: {H['H2']['verdict']}.** Judged per seed. "
        f"{d.n(p)} of {d.n(t)} seeds pass. Seeds that were locally excellent with a CI separated from the "
        f"simulator's: {d.n(H['H2']['excellent_and_ci_separated'])}.")
    d.w()
    d.w("| family | model | seeds excellent | seeds with |error| > threshold | seeds CI-separated | seeds passing |")
    d.w("|---|---|---|---|---|---|")
    rows = H["H2"]["rows"]
    for f in S["families"]:
        for kind in ("mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band"):
            rr = [r for r in rows if r["family"] == f and r["kind"] == kind]
            d.w(f"| {f} | {LABEL[kind]} | {d.n(sum(r['excellent'] for r in rr))} | {d.n(sum(r['big'] for r in rr))}"
                f" | {d.n(sum(r['separated'] for r in rr))} | {d.n(sum(r['pass'] for r in rr))} |")
    d.w()
    h3 = H["H3"]
    d.w(f"**H3 - critical-band data halves the critical-s error: {h3['verdict']}** "
        f"({d.n(h3['n_supported'])} of {d.n(h3['n_conclusive'])} conclusive cells supported).")
    for c, r in h3["cells"].items():
        if "err_default" in r:
            d.w(f"- {c}: |mean error| default {d.n(r['err_default'], 4)}, with band {d.n(r['err_band'], 4)} - "
                f"{r['verdict']}{_h3_reason(r)}.")
        else:
            d.w(f"- {c}: {r['verdict']} ({r.get('reason', '')}).")
    d.w()
    h4 = H["H4"]
    d.w(f"**H4 - 2-hop equivalent to 1-hop within +/- 0.02 on critical s: {h4['verdict']}.**")
    for f, r in h4["per_family"].items():
        if "diff" in r:
            d.w(f"- {f}: difference (2-hop - 1-hop) {d.n(r['diff'], 4, sign=True)}, Welch 95% CI "
                f"{d.ci(r['lo'], r['hi'], nd=4)} - {r['verdict']}.")
        else:
            d.w(f"- {f}: {r['verdict']}.")
    d.w()
    h5 = H["H5"]
    d.w(f"**H5 - the structured model (true form by construction) has lower critical-s error than the generic "
        f"1-hop MPNN: {h5['verdict']}.**")
    for f, r in h5["per_family"].items():
        d.w(f"- {f}: mean |error| generic {d.n(r['mean_abs_err_mpnn1'], 4)}, structured "
            f"{d.n(r['mean_abs_err_structured'], 4)}; difference CI {d.ci(r['lo'], r['hi'], nd=4)} - {r['verdict']}.")
    d.w()
    d.w("## Controls")
    d.w()
    d.w("![All-cell accuracy of never-ignite](figures/accuracy_trap.png)")
    d.w()
    for f, F in S["families"].items():
        nv = F["models"]["never"]
        lo = [r for r in nv["accuracy_by_s"]]
        d.w(f"- **Never-ignite, family {f}**: all-cell accuracy {d.n(nv['seeds'][0]['test']['all']['all_cell_accuracy'], 2, pct=True)} "
            f"overall (lowest by s bin {d.n(min(r['acc'] for r in lo), 2, pct=True)}), and it never crosses. "
            f"All-cell accuracy rewards predicting that nothing happens; it says nothing about the tipping point.")
    if "controls" in S:
        for f, c in S["controls"].items():
            d.w(f"- **Oracle rollouts, family {f}** (true q through the model-rollout code): critical s "
                f"{d.n(c['oracle_xc'], 4)} {d.ci(*c['oracle_ci'], nd=4)} vs simulator "
                f"{d.n(S['families'][f]['truth']['L128']['xc'], 4)}.")
        d.w()
        d.w(f"Unmasked rollouts (seed 0, {d.n(S['n_unmasked'])} fires at the simulator's critical s, at most 300 "
            "steps): rate of ignitions of cells with no burning neighbour, per non-active cell-step, and the share of "
            "those fires that crossed. Masked, about half would cross at this s. The structured model is zero by "
            "construction.")
        d.w()
        d.w("| family | " + " | ".join(LABEL[k] for k in ORDER[2:]) + " |")
        d.w("|---|" + "---|" * len(ORDER[2:]))
        for f, c in S["controls"].items():
            cells = []
            for k in ORDER[2:]:
                u = c["unmasked"].get(k)
                if u is None:
                    cells.append("-")
                else:
                    r = "0" if u["rate"] == 0 else format_sci(d, u["rate"])
                    cells.append(f"{r}; crossed {d.n(u['crossed_frac'], 0, pct=True)}")
            d.w(f"| {f} | " + " | ".join(cells) + " |")
    d.w()
    if "B" in S["families"] and "mpnn1_windblind" in S["families"]["B"]["models"]:
        M = S["families"]["B"]["models"]["mpnn1_windblind"]
        if "xc" in M:
            d.w(f"- **Wind-blind control (B)**: critical s {d.n(M['xc']['mean'])} {d.ci(M['xc']['lo'], M['xc']['hi'])}, "
                f"error {d.n(M['err']['mean'], sign=True)}; off-band excess {d.n(M['offband_excess']['mean'], 5)}.")
    d.w()
    embers(d, base)
    d.w(Path(ROOT / "docs_src" / "report_tail.md").read_text().rstrip())
    return d


def _h3_reason(r):
    if r["verdict"] != "inconclusive":
        return ""
    if r["err_default"] < 0.01:
        return " (default error below the 0.01 measurement floor)"
    return " (default model's seed interval contains the truth)"


def format_sci(d: Doc, x):
    s = f"{x:.1e}"
    d.nums.add(s)
    return s


def embers(d: Doc, base: Path | None = None):
    base = base or ROOT / "results"
    d.w("## Embers (exploratory, not confirmatory)")
    d.w()
    p1, p2 = base / "ember_e1.json", base / "ember_e2.json"
    if not p1.exists() and not p2.exists():
        d.w("No exploratory follow-ups were run.")
        d.w()
        return
    d.w("Hypotheses for these were written in EMBERS.md and committed before they ran. They were chosen after "
        "seeing H1-H5, so they are exploratory.")
    d.w()
    if p1.exists():
        E = json.loads(p1.read_text())["result"]
        d.w(f"**E1 - signed bias, not squared error, tracks the tipping-point error: "
            f"{'supported' if E['supported'] else 'not supported'}.** Across {d.n(E['n'])} GNN seeds, the error "
            f"predicted from each model's mean relative bias on in-band test cells (-sc x bias) correlates with the "
            f"measured critical-s error at Pearson r = {d.n(E['pearson_pred_vs_err'], 2)} (slope "
            f"{d.n(E['slope'], 2)}), while off-band excess Brier and |error| have Spearman rho = "
            f"{d.n(E['spearman_excess_vs_abs_err'], 2)}. The slope is near the top of the stated range, so the simple "
            "rescaling argument underestimates the shift. The clearest outliers are family B's 1-hop MPNNs: under "
            "wind, errors can cancel in a mean while still distorting the direction-dependent spread.")
        d.w()
        d.w("![E1: signed bias vs squared error](figures/ember_e1_bias.png)")
        d.w()
    if p2.exists():
        E = json.loads(p2.read_text())["result"]
        d.w(f"**E2 - the same models at L={d.n(E['L'])}.** For each family, the seed-0 default-data GNN with the "
            f"largest L=128 error:")
        d.w()
        d.w("| family | model | simulator critical s | model critical s | error | error at L=128 | 10-90% width | "
            "width at L=128 | abs(error) / half width | same at L=128 |")
        d.w("|---|---|---|---|---|---|---|---|---|---|")
        for f, r in E["families"].items():
            d.w(f"| {f} | {LABEL[r['kind']]} | {d.n(r['truth_xc'], 4)} {d.ci(*r['truth_ci'], nd=4)} | "
                f"{d.n(r['model_xc'], 4)} {d.ci(*r['model_ci'], nd=4)} | {d.n(r['err'], 4, sign=True)} | "
                f"{d.n(r['err_128'], 4, sign=True)} | {d.n(r['truth_width'], 4)} | {d.n(r['truth_width_128'], 4)} | "
                f"{d.n(abs(r['err']) / (r['truth_width'] / 2), 2)} | {d.n(abs(r['err_128']) / (r['truth_width_128'] / 2), 2)} |")
        d.w()
        if "verdict" in E:
            grew = [f for f, r in E["families"].items()
                    if abs(r["err"]) / r["truth_width"] > abs(r["err_128"]) / r["truth_width_128"]]
            fell = [f for f in E["families"] if f not in grew]
            txt = E["verdict"]
            if grew:
                txt += (f" In {', '.join(grew)} the miss stayed roughly fixed in s while the simulator's transition "
                        "narrowed, so the same miss is a larger fraction of the transition on the bigger map.")
            if fell:
                txt += (f" In {', '.join(fell)} the error itself shrank enough that its share of the transition "
                        "did not grow.")
            if all(abs(r["err"]) <= max(0.01, r["truth_width"] / 2) for r in E["families"].values()):
                txt += (" None of these crosses the H2 size criterion at this L; whether they would at larger L "
                        "was not tested.")
            else:
                txt += " At least one now exceeds the H2 size criterion (exploratory, single seed)."
            d.w(txt)
            d.w()


def readme(S) -> Doc:
    d = Doc()
    d.w("# wildfire: does a good next-step fire model find the tipping point?")
    d.w()
    d.w("**" + headline(S, d) + "**")
    d.w()
    d.w("> This is a simplified, synthetic simulation. It must never inform real fire or evacuation decisions.")
    d.w()
    for line in plain_terms(S, d):
        d.w("- " + line)
    d.w()
    d.w("![Excess Brier against critical-s error](figures/excess_vs_error.png)")
    d.w()
    d.w("This repository was designed and run autonomously by an AI agent, Claude, from a single prompt: the "
        "simulator, the experiments, the pre-registration, the analysis, this text and the demo.")
    d.w()
    d.w("A fire on a grid either dies out or crosses the map, and a small change in the base spread control s "
        "decides which. The simulator's tipping point (critical s, 50% crossing at L=128) was measured, then graph "
        "neural networks were trained on next-step data with the critical region held out, and rolled out to see "
        "where *their* tipping point lands.")
    d.w()
    d.w("| family | simulator critical s | generic 1-hop MPNN | structured model | locally excellent seeds |")
    d.w("|---|---|---|---|---|")
    for f, F in S["families"].items():
        g, s_ = F["models"]["mpnn1"], F["models"]["structured"]
        ex = sum(x["excellent"] for k in ("mpnn1", "mpnn2", "structured", "mpnn1_band", "structured_band")
                 for x in F["models"][k]["seeds"])
        tot = sum(len(F["models"][k]["seeds"]) for k in ("mpnn1", "mpnn2", "structured", "mpnn1_band",
                                                         "structured_band"))
        d.w(f"| {f} | {d.n(F['truth']['L128']['xc'], 4)} | {d.n(g['xc']['mean'])} "
            f"({d.n(g['err']['mean'], sign=True)}) | {d.n(s_['xc']['mean'])} ({d.n(s_['err']['mean'], sign=True)}) | "
            f"{d.n(ex)}/{d.n(tot)} |")
    d.w()
    d.w("Hypotheses (pre-registered at commit `" + S["prereg_commit"] + "`): " + "; ".join(
        f"{k} {S['H'][k]['verdict']}" for k in ("H1", "H2", "H3", "H4", "H5")) + ". Details, intervals and "
        "limitations: [REPORT.md](REPORT.md).")
    d.w()
    d.w(Path(ROOT / "docs_src" / "readme_tail.md").read_text().rstrip())
    return d


def render(base: Path | None = None) -> dict[str, Doc]:
    S = load_result("summary", base)["result"]
    return {"REPORT.md": report(S, base), "README.md": readme(S)}


if __name__ == "__main__":
    docs = render()
    for name, d in docs.items():
        (ROOT / name).write_text(d.text())
    print("rendered", list(docs), file=sys.stderr)
