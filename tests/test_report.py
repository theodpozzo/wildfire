"""REPORT.md and README.md must be exactly what scripts/render.py produces from the
committed JSON results, and every number in them must come from that rendering."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "summary.json").exists(), reason="no summary yet")

NUM = re.compile(r"(?<![\w.`/-])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\w`])")


def _rendered():
    from scripts.render import render

    return render(ROOT / "results")


def test_docs_match_render():
    for name, doc in _rendered().items():
        assert (ROOT / name).read_text() == doc.text(), f"{name} is stale: run `make render`"


def test_every_number_comes_from_results():
    from scripts.render import ALLOWED

    for name, doc in _rendered().items():
        text = (ROOT / name).read_text()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`]*`", "", text)  # code spans (hashes, paths)
        text = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
        for tok in NUM.findall(text):
            t = tok.rstrip("%")
            t2 = t.lstrip("+-")
            assert t in doc.nums or t2 in ALLOWED or t in ALLOWED or ("+" + t2) in doc.nums, (name, tok)


def test_numbers_match_json():
    """Spot-check: key numbers in REPORT.md equal the JSON values."""
    import json

    S = json.loads((ROOT / "results" / "summary.json").read_text())["result"]
    rep = (ROOT / "REPORT.md").read_text()
    for F in S["families"].values():
        assert f"{F['truth']['L128']['xc']:.4f}" in rep
        assert f"{F['models']['mpnn1']['xc']['mean']:.3f}" in rep
    assert f"{S['H']['H2']['passes']} of {S['H']['H2']['tried']}" in rep
