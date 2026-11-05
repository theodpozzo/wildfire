"""Parameter shapes for each learned model (shared by training and export)."""

from wildfire.models import F


def shapes(kind: str, H: int) -> dict:
    E = 2
    if kind in ("mpnn1", "mpnn1_windblind"):
        return {"l1.msg0": (2 * F + E, H), "l1.msg1": (H, H), "l1.upd0": (F + H, H), "l1.upd1": (H, 1)}
    if kind == "mpnn2":
        return {"l1.msg0": (2 * F + E, H), "l1.msg1": (H, H), "l1.upd0": (F + H, H), "l1.upd1": (H, H),
                "l2.msg0": (2 * H + E, H), "l2.msg1": (H, H), "l2.upd0": (2 * H, H), "l2.upd1": (H, 1)}
    if kind == "structured":
        return {"edge0": (2 * F + E, H), "edge1": (H, H), "edge2": (H, 1)}
    if kind == "logreg":
        return {}
    raise ValueError(kind)
