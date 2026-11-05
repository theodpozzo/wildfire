"""Scenario families (ground-truth distributions), fixed before pre-registration."""

from wildfire.sim import Family

FAMILIES = {
    # A. Uniform: iid fuel (density 0.9), no moisture, slope or wind.
    "A": Family("A", fuel_density=0.9),
    # B. Wind: as A, plus wind blowing east or west (random per landscape) with
    # downwind/upwind factors 1 +/- 0.5 and a perpendicular (north/south) factor 0.6.
    "B": Family("B", fuel_density=0.9, wind_w=0.5, wind_ns=0.6, wind_random_sign=True),
    # C. Patchy fuel (smoothed field, density 0.9), smooth random moisture, slope.
    "C": Family(
        "C", fuel_density=0.9, fuel_corr=2.0, moisture_amp=0.6, moisture_corr=3.0,
        slope_beta=0.4, elev_amp=1.0, elev_corr=3.0,
    ),
}
