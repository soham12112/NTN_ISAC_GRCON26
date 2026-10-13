"""Bistatic far-illuminator echo-to-direct link-budget checks."""

from __future__ import annotations

import numpy as np


def echo_to_direct_db(rcs_m2, receiver_range_m):
    """Far-illuminator ratio sigma/(4*pi*Rr^2), in dB."""
    sigma = float(rcs_m2)
    rr = float(receiver_range_m)
    if sigma <= 0 or rr <= 0:
        raise ValueError("rcs_m2 and receiver_range_m must be positive")
    return float(10.0 * np.log10(sigma / (4.0 * np.pi * rr * rr)))


def range_for_ratio_m(rcs_m2, echo_to_direct_db_value):
    """Receiver range giving a requested echo-to-direct ratio."""
    sigma = float(rcs_m2)
    if sigma <= 0:
        raise ValueError("rcs_m2 must be positive")
    ratio = 10.0 ** (float(echo_to_direct_db_value) / 10.0)
    return float(np.sqrt(sigma / (4.0 * np.pi * ratio)))


def coherent_gain_db(bandwidth_hz, dwell_s):
    """Ideal time-bandwidth processing gain."""
    product = float(bandwidth_hz) * float(dwell_s)
    if product <= 0:
        raise ValueError("bandwidth_hz*dwell_s must be positive")
    return float(10.0 * np.log10(product))


def assess(rcs_m2, receiver_range_m, direct_cn_db=15.0,
           bandwidth_hz=4.5e6, dwell_s=0.1, threshold_db=13.0,
           implementation_loss_db=0.0):
    """Ideal one-dwell SNR and margin, with losses stated explicitly."""
    ratio = echo_to_direct_db(rcs_m2, receiver_range_m)
    gain = coherent_gain_db(bandwidth_hz, dwell_s)
    snr = ratio + float(direct_cn_db) + gain - float(implementation_loss_db)
    return dict(
        echo_to_direct_db=ratio,
        coherent_gain_db=gain,
        post_dwell_snr_db=snr,
        threshold_db=float(threshold_db),
        margin_db=snr - float(threshold_db),
        detected=bool(snr >= float(threshold_db)),
    )
