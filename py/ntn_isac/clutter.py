"""Ephemeris-driven stationary-clutter exclusion for a LEO illuminator.

A zero-Doppler notch is not a stationary-clutter filter when the illuminator
moves.  If the satellite line of sight rotates at angular speed ``omega``, a
stationary object at receiver range ``d`` can have residual Doppler up to

    |f_clutter| <= (fc / c) * d * omega.

The receiver does not observe ``d`` directly; its delay axis is bistatic excess
path.  For a ground target seen at satellite elevation ``E``,

    excess = d * (1 - cos(E) * cos(relative_azimuth))

so ``d <= excess / (1 - cos(E))``.  Combining that geometric bound with an
explicit maximum surveillance range gives a conservative, range-dependent
clutter envelope.  The maximum range is required: excess delay alone cannot
identify receiver range, and pretending otherwise would make the filter look
more selective than the geometry permits.
"""

from __future__ import annotations

import numpy as np


C = 299792458.0


def max_stationary_doppler(range_m, los_rate_rad_s, carrier_hz):
    """Conservative residual-Doppler bound for stationary objects."""
    return (np.asarray(range_m, float) * abs(float(los_rate_rad_s))
            * float(carrier_hz) / C)


def max_range_for_excess(excess_m, elevation_deg, max_range_m):
    """Largest ground-target range consistent with each excess-path cell.

    ``max_range_m`` is the surveillance footprint.  At very low elevation the
    excess-to-range mapping is ill-conditioned; the explicit footprint keeps
    the bound finite and makes that limitation visible to the caller.
    """
    e = np.radians(float(elevation_deg))
    den = max(1.0 - np.cos(e), 1e-9)
    inferred = np.maximum(np.asarray(excess_m, float), 0.0) / den
    return np.minimum(inferred, max(float(max_range_m), 0.0))


def doppler_bound_by_delay(delay_s, carrier_hz, los_rate_rad_s,
                           elevation_deg, max_range_m):
    """Stationary-clutter half-band, Hz, for every excess-delay bin."""
    d = max_range_for_excess(
        np.asarray(delay_s, float) * C, elevation_deg, max_range_m)
    return max_stationary_doppler(d, los_rate_rad_s, carrier_hz)


def keep_mask(delay_s, doppler_hz, carrier_hz, los_rate_rad_s,
              elevation_deg, max_range_m, guard_hz=0.0):
    """Boolean [Doppler, delay] mask; true cells lie outside clutter locus."""
    bound = doppler_bound_by_delay(
        delay_s, carrier_hz, los_rate_rad_s, elevation_deg, max_range_m)
    return np.abs(np.asarray(doppler_hz, float))[:, None] > (
        bound[None, :] + max(float(guard_hz), 0.0))


def model_from_meta(meta, max_range_m, guard_hz=0.0):
    """Build a conservative model from trajectory metadata.

    The metadata currently carries extrema over the pass, not per-dwell
    elevation and line-of-sight rate.  Using maximum angular rate and minimum
    elevation is conservative.  A future trajectory format may supply the
    per-dwell values without changing the mask API.
    """
    if not isinstance(meta, dict):
        raise ValueError("trajectory metadata is required for clutter locus")
    if "los_rotation_deg_s" not in meta:
        raise ValueError("trajectory metadata has no los_rotation_deg_s")
    elev = meta.get("elevation_deg", {})
    if not isinstance(elev, dict) or "min" not in elev:
        raise ValueError("trajectory metadata has no minimum elevation")
    return dict(
        carrier_hz=float(meta["carrier_hz"]),
        los_rate_rad_s=np.radians(float(meta["los_rotation_deg_s"])),
        elevation_deg=float(elev["min"]),
        max_range_m=float(max_range_m),
        guard_hz=float(guard_hz),
    )
