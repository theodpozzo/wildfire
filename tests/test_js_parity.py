"""The demo's hand-written JS inference must match the Python models within 1e-5."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wildfire.models import predict_grid

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "js_parity.json"


def cases():
    return json.loads(FX.read_text())["cases"]


def test_fixture_is_python_truth():
    for c in cases():
        L = c["L"]
        fields = (np.array(c["fuel"], bool).reshape(1, L, L), np.array(c["moisture"], np.float32).reshape(1, L, L),
                  np.array(c["elevation"], np.float32).reshape(1, L, L), np.array([c["wind"]], np.float32),
                  np.array([c["s"]]))
        w = {k: np.array(v, np.float32) for k, v in c["weights"].items()}
        p = predict_grid(c["kind"], w, fields, np.array(c["state"], np.int8).reshape(1, L, L))
        # float32 matmul rounding differs across CPUs/BLAS builds at the 1e-7 level
        assert np.allclose(p.ravel(), c["expected"], atol=2e-6, rtol=0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    out = json.loads(subprocess.check_output(["node", str(ROOT / "tests" / "js_parity.js")]))
    worst = 0.0
    for c, js in zip(cases(), out, strict=True):
        d = np.abs(np.array(js) - np.array(c["expected"]))
        worst = max(worst, float(d.max()))
        # sanity: the fixture exercises non-trivial predictions
        assert np.array(c["expected"]).max() > 0.05
    assert worst < 1e-5, worst
