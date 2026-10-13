"""NTN ISAC prototype: shared Python library.

Requires the GNU Radio interpreter (/usr/bin/python3, 3.10). See env.sh.

Signal chain, in the order the data flows:

    fileio      generator bundle readers, CPI tiling
    ofdm        CP-OFDM matched to nrOFDMModulate, symbol layout, slot rate
    trajectory  traj.csv reader, cubic-spline range interpolation
    leo_channel time-varying multipath channel (the Python reference; the
                live path is the C++ block in gr-ntnisac)
    receiver    ephemeris tracker, de-ramp, de-walk, tracked demodulation
    sensing     LS channel estimate, MMSE, slicer, delay-Doppler map, and the
                three sensing variants of handoff section 7

Two-stage coarse-to-fine sensing, on top of those. See docs/two-stage.md:

    coarse      stage 1: SSB matched filter, MTI, pulse-pair Doppler
    aperture    Renyi entropy of the aperture and what it predicts
    detect      CA-CFAR, DBSCAN, frame-rate comb rejection
    clutter     ephemeris-driven stationary-clutter locus
    track       Kalman filter on (excess range, range rate), and forecasting
    twostage    the whole flow: one dwell of samples in, one dict out
"""
__all__ = ["aperture", "clutter", "coarse", "detect", "fileio", "leo_channel",
           "link_budget", "ofdm", "receiver", "sensing", "track", "trajectory",
           "twostage"]
