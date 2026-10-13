#!/usr/bin/env python3
"""Add a rotating physical target to a trajectory, so the bench flowgraph can
produce one with no radio and no new blocks.

    $NTN_PY tools/add_bench_target.py --in data/traj_ota --out data/traj_bench

The over-the-air demo has two targets. One is emulated and already in
traj.csv - the drone at 640 m of excess path, with exact commanded truth. The
other is physically present between the radios, and on the bench there is no
"between the radios". So it is injected here instead, as one more path.

That works because a rotating scatterer is a range history like any other, and
because the channel block already accepts an arbitrary number of paths. Nothing
new is built and nothing is special-cased: the bench flowgraph uses the same
`ntnisac.leo_channel` the transmit flowgraph does, reading a trajectory with one
extra column pair.

## What the path is

A point at radius r on a blade turning at f_rot swings the bistatic excess path
sinusoidally:

    R(t) = R_0(t) + excess + r*cos(2*pi*f_rot*t)

Delay and Doppler then agree by construction, which is handoff section 6's
requirement and the reason there is no Doppler parameter anywhere in this
codebase. The interesting quantity is the modulation index
beta = 2*pi*r/lambda, which at 0.25 m and 12.1 cm is 13 radians - so the echo
is a Bessel comb of lines spaced at f_rot, with its envelope edge at the
tip-speed Doppler 2*pi*r*f_rot/lambda.

## Two numbers that are not free choices

**The table rate.** traj.csv is sampled at 100 Hz and the channel interpolates
it with a not-a-knot cubic spline. A 7 Hz oscillation is 14 samples per cycle,
where the spline error is about 2e-5 m, which is 1e-3 radians of carrier phase.
That is fine, and it stops being fine somewhere above 20 rev/s - the check
below refuses rather than quietly degrading, because a smeared micro-Doppler
comb looks like a real measurement.

**The rotation rate.** Slot 0 of every frame carries the SSB and no PDSCH, so
the sensing weight changes once per 20 ms and every tap acquires a Doppler comb
at multiples of 25 Hz (findings.md sections 11 and 15). A rate that divides 25
puts its entire signature on those bins. 7 rev/s does not, and the check below
warns when a rate does.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

C = 299792458.0
MIN_SAMPLES_PER_CYCLE = 12


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--in", dest="src", default=os.path.join(root, "data", "traj_ota"))
    p.add_argument("--out", dest="dst", default=os.path.join(root, "data", "traj_bench"))
    p.add_argument("--radius-m", type=float, default=0.25)
    p.add_argument("--rot-hz", type=float, default=7.0)
    p.add_argument("--excess-m", type=float, default=0.4,
                   help="bistatic excess path at the centre of rotation. A "
                        "metre off a five-metre baseline is 0.39 m")
    p.add_argument("--gain-db", type=float, default=-30.0,
                   help="echo-to-direct ratio")
    p.add_argument("--name", default="blade")
    p.add_argument("--solo", action="store_true", default=True,
                   help="drive the emulated clutter and drone to the gain "
                        "floor, leaving the direct path and the rotating "
                        "target. On by default")
    p.add_argument("--keep-emulated", action="store_false", dest="solo",
                   help="leave them on, for a capture with both kinds of "
                        "target in it")
    p.add_argument("--rate-hz", type=float, default=1000.0, dest="rate_hz",
                   help="resample the table to this rate before adding the "
                        "target. gen_trajectory writes 100 Hz, which is only "
                        "14 samples per rotation at 7 rev/s")
    a = p.parse_args()

    with open(os.path.join(a.src, "traj.csv")) as f:
        hdr = f.readline().strip().split(",")
    M = np.loadtxt(os.path.join(a.src, "traj.csv"), delimiter=",", skiprows=1)
    with open(os.path.join(a.src, "traj_meta.json")) as f:
        meta = json.load(f)

    t = M[:, 0]
    R = M[:, 1::2]
    G = M[:, 2::2]
    n_base = R.shape[1]
    rate = 1.0 / np.median(np.diff(t))
    fc = float(meta["carrier_hz"])
    lam = C / fc

    # Resample the base paths onto a finer grid before adding the target.
    #
    # gen_trajectory writes 100 Hz, which the channel's not-a-knot spline
    # interpolates perfectly well for a satellite pass - the ranges are smooth
    # over seconds. A 7 rev/s rotation is a different kind of signal: 14
    # samples per cycle, and while the spline's amplitude error is small its
    # harmonic structure is not preserved. Measured on the bench flowgraph, a
    # 100 Hz table produced a confident 4.00 rev/s against a true 7.00, because
    # what the analyser sees is the spline's comb and not the blade's. At
    # 1000 Hz it reads 7.00.
    #
    # The satellite paths are resampled with the same cubic spline the channel
    # uses, so nothing is introduced that the channel would not have computed
    # anyway; the extra rows only matter for the target.
    if a.rate_hz > rate * 1.001:
        from scipy.interpolate import CubicSpline
        t_new = np.arange(t[0], t[-1] + 0.5 / a.rate_hz, 1.0 / a.rate_hz)
        R = np.column_stack([CubicSpline(t, R[:, k])(t_new)
                             for k in range(R.shape[1])])
        # Gains are held piecewise constant on purpose: G is a control knob,
        # not a trajectory, and splining it would smear the deliberate on/off
        # edges that produce the target-absent epochs.
        idx = np.clip(np.searchsorted(t, t_new, side="right") - 1, 0, t.size - 1)
        G = G[idx]
        print(f"  resampled {t.size} rows at {rate:.0f} Hz -> "
              f"{t_new.size} rows at {a.rate_hz:.0f} Hz")
        t, rate = t_new, a.rate_hz

    spc = rate / a.rot_hz
    if spc < MIN_SAMPLES_PER_CYCLE:
        sys.exit(
            f"ERROR: {a.rot_hz} rev/s against a {rate:.0f} Hz trajectory table "
            f"is {spc:.1f} samples per cycle. The channel's cubic spline needs "
            f"at least {MIN_SAMPLES_PER_CYCLE}, and below that the "
            f"micro-Doppler comb smears in a way that still looks like a "
            f"measurement. Lower --rot-hz, or regenerate the trajectory at a "
            f"higher Rate.")

    comb = rate and 25.0
    if abs(comb / a.rot_hz - round(comb / a.rot_hz)) < 1e-6:
        print(f"WARNING: {a.rot_hz} rev/s divides the {comb:.0f} Hz aperture "
              f"comb, so every harmonic of the target lands on a masked bin. "
              f"Pick something like 7.")

    Rb = R[:, 0] + a.excess_m + a.radius_m * np.cos(2 * np.pi * a.rot_hz * t)
    Gb = np.full(t.size, a.gain_db)

    if a.solo and G.shape[1] > 1:
        # The emulated clutter and drone occupy the same Doppler band a
        # rotating target does - at t = 300 s the drone is a couple of delay
        # cells out with about 20 Hz of residual Doppler - so with them on, the
        # micro-Doppler comb is a mixture and the rotation rate comes out a
        # harmonic. Measured on the bench flowgraph: 3-4 rev/s reported against
        # a true 7, with the Doppler edge still correct, which is exactly what
        # a contaminated comb looks like. Both kinds of target in one capture
        # is the demo; a measurement of one has to exclude the other.
        G = G.copy()
        G[:, 1:] = -200.0

    Rall = np.column_stack([R, Rb])
    Gall = np.column_stack([G, Gb])
    n = Rall.shape[1]

    os.makedirs(a.dst, exist_ok=True)
    cols = ["t_s"]
    for k in range(n):
        cols += [f"R{k}_m", f"G{k}_db"]
    out = np.empty((t.size, 1 + 2 * n))
    out[:, 0] = t
    out[:, 1::2] = Rall
    out[:, 2::2] = Gall
    with open(os.path.join(a.dst, "traj.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        np.savetxt(f, out, delimiter=",",
                   fmt=["%.6f"] + ["%.6f", "%.4f"] * n)

    edge = 2 * np.pi * a.radius_m * a.rot_hz / lam
    beta = 2 * np.pi * a.radius_m / lam
    # r_nominal must be the floor-quantised minimum over present paths on the
    # same 1e-6 grid the CSV is written at, or a consumer that takes the
    # minimum from the file disagrees with this field by enough that fc/c turns
    # it into visible carrier phase.
    present = Gall > -100
    Rp = np.where(present, Rall, np.inf)
    meta["r_nominal_m"] = float(np.floor(Rp.min() * 1e6) / 1e6)
    meta["bench_target"] = dict(
        path=n_base, name=a.name, radius_m=a.radius_m, rot_hz=a.rot_hz,
        rpm=a.rot_hz * 60, excess_m=a.excess_m, gain_db=a.gain_db,
        tip_speed_mps=2 * np.pi * a.radius_m * a.rot_hz,
        doppler_edge_hz=edge, modulation_index_rad=beta,
        line_spacing_hz=a.rot_hz, samples_per_cycle=spc,
        note="Physical rotating target, injected so the bench flowgraph can "
             "produce one with no radio. Statistically it is truth; it is a "
             "commanded path like any other.")
    paths = meta.get("paths", [])
    paths.append(dict(
        id=n_base, name=a.name, gain_db=a.gain_db,
        excess_m_max=a.excess_m + a.radius_m, always_present=True,
        description=(f"Rotating scatterer, r={a.radius_m} m at {a.rot_hz} "
                     f"rev/s, {a.excess_m} m of excess path. Doppler comb "
                     f"every {a.rot_hz} Hz out to +-{edge:.1f} Hz.")))
    meta["paths"] = paths
    with open(os.path.join(a.dst, "traj_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cell = C / 4.5e6
    print(f"add_bench_target -> {a.dst}")
    print(f"  {n_base} existing paths + 1 rotating target = {n} paths")
    print(f"  r={a.radius_m} m at {a.rot_hz} rev/s ({a.rot_hz*60:.0f} RPM), "
          f"tip speed {2*np.pi*a.radius_m*a.rot_hz:.2f} m/s")
    print(f"  excess path {a.excess_m} m = {a.excess_m/cell:.4f} delay cells: "
          f"in the direct path's own cell, as a real object would be")
    print(f"  modulation index {beta:.1f} rad -> Doppler comb every "
          f"{a.rot_hz:.1f} Hz out to +-{edge:.1f} Hz")
    print(f"  trajectory table {rate:.0f} Hz = {spc:.1f} samples per rotation")
    print(f"  echo {a.gain_db:.0f} dB below the direct path")


if __name__ == "__main__":
    main()
