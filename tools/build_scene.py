#!/usr/bin/env python3
"""Compile a channel scene into the trajectory files the flowgraphs read.

    $NTN_PY tools/build_scene.py scenes/bench.yml
    $NTN_PY tools/build_scene.py scenes/bench.yml --set paths.3.rot_hz=11
    $NTN_PY tools/build_scene.py scenes/bench.yml --out data/traj_try --plot

Replaces running gen_trajectory.m, gen_ntn_channel.m and add_bench_target.py in
order, each rewriting the file the next one reads. The output is the same
`traj.csv` and `traj_meta.json`, so nothing downstream changes - what changes is
that the channel is now one file you can read, diff and version.

`--set` overrides any field by dotted path, so a sweep is a shell loop rather
than a directory of near-identical YAML:

    for g in -20 -30 -40 -50; do
      $NTN_PY tools/build_scene.py scenes/bench.yml \
              --set paths.3.gain_db=$g --out data/traj_g$g
    done
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from ntn_isac import scene as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def apply_override(d, dotted, value):
    """`paths.3.rot_hz=11` -> d["paths"][3]["rot_hz"] = 11.0"""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    last = keys[-1]
    try:
        v = json.loads(value)
    except Exception:                                  # noqa: BLE001
        v = value
    if isinstance(cur, list):
        cur[int(last)] = v
    else:
        cur[last] = v


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene")
    p.add_argument("--out", default="", help="default is the scene's out_dir")
    p.add_argument("--set", action="append", default=[], dest="sets",
                   metavar="a.b=v", help="override a field, repeatable")
    p.add_argument("--plot", action="store_true",
                   help="write a picture of the excess path and Doppler of "
                        "every path against pass time")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    sc = S.load_yaml(a.scene)
    for s in a.sets:
        k, _, v = s.partition("=")
        apply_override(sc, k, v)

    c = S.compile_scene(sc, base_dir=ROOT)
    out = a.out or os.path.join(ROOT, sc.get("out_dir", "data/traj_scene"))
    S.write(c, out)

    if not a.quiet:
        m = c.meta
        print(f"build_scene {a.scene} -> {out}")
        print(f"  {m['n_paths']} paths, {m['duration_s']:.0f} s at "
              f"{m['rate_hz']:.0f} Hz ({m['n_samples']} rows), "
              f"carrier {m['carrier_hz']/1e6:.3f} MHz")
        if "elevation_deg" in m:
            print(f"  elevation {m['elevation_deg']['min']:.1f} to "
                  f"{m['elevation_deg']['max']:.1f} deg, slant range "
                  f"{m['slant_range_m']['min']/1e3:.0f} to "
                  f"{m['slant_range_m']['max']/1e3:.0f} km")
        # Range rate is what sets Doppler and delay walk, and both are worth
        # seeing at build time rather than discovering in the receiver.
        rr = np.gradient(c.R[:, 0], c.t)
        lam = 299792458.0 / m["carrier_hz"]
        print(f"  range rate +-{np.abs(rr).max():.0f} m/s -> Doppler "
              f"+-{np.abs(rr).max()/lam/1e3:.1f} kHz, delay walk "
              f"{np.abs(rr).max()/299792458.0*7.68e6:.0f} samples/s")
        for pth in m["paths"]:
            print(f"  p{pth['id']} {pth['name']:<14} {pth['description']}")
        print(f"  estimated channel throughput "
              f"{m['throughput_estimate_ms']:.1f} MS/s against 7.68 required")
        for w in c.warnings:
            print(f"  WARNING: {w}")

    if a.plot:
        plot(c, os.path.join(out, "scene.png"))
        print(f"  figure -> {os.path.join(out, 'scene.png')}")


def plot(c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = 299792458.0 / c.meta["carrier_hz"]
    ex = c.R - c.R[:, [0]]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, d in enumerate(c.meta["paths"]):
        live = c.G[:, i] > -100
        e = np.where(live, ex[:, i], np.nan)
        ax[0].plot(c.t, e, lw=1.0, label=f"p{i} {d['name']}")
        ax[1].plot(c.t, np.where(live, -np.gradient(e, c.t) / lam, np.nan),
                   lw=1.0)
        ax[2].plot(c.t, np.where(live, c.G[:, i], np.nan), lw=1.0)
    ax[0].set_ylabel("excess path, m")
    ax[0].axhline(c.meta["cp_budget_m"], color="k", ls="--", lw=0.8)
    ax[0].text(c.t[0], c.meta["cp_budget_m"], " cyclic-prefix budget",
               fontsize=7, va="bottom")
    ax[0].set_yscale("symlog", linthresh=1.0)
    ax[0].legend(fontsize=7, ncol=3)
    ax[1].set_ylabel("residual Doppler, Hz")
    ax[1].axhspan(-30, 30, color="0.85", zorder=0)
    ax[1].text(c.t[0], 32, " zero-Doppler notch at a 100 ms dwell", fontsize=7)
    ax[2].set_ylabel("gain, dB")
    ax[2].set_xlabel("pass time, s")
    for x in ax:
        x.grid(alpha=0.3)
    fig.suptitle(f"{c.meta['n_paths']} paths, excess path and residual Doppler "
                 f"relative to the direct ray", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
