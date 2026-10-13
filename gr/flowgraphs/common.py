"""Shared setup for the transmit and receive flowgraphs."""

from __future__ import annotations

import argparse
import json
import os

# 7.68 MS/s needs a master clock rate that divides evenly. The B2xx default of
# 32 MHz gives 4.1667, which UHD will silently approximate and which puts a
# fractional resampler in the path. 30.72 MHz gives exactly 4.
DEFAULT_MCR = 30.72e6


def load_cfg(wf_dir):
    with open(os.path.join(wf_dir, "config.json")) as f:
        return json.load(f)


def load_traj_meta(traj_dir):
    with open(os.path.join(traj_dir, "traj_meta.json")) as f:
        return json.load(f)


def base_parser(desc):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--wf", default=os.path.join(root, "data", "wf"),
                   help="waveform bundle directory")
    p.add_argument("--traj", default=os.path.join(root, "data", "traj"),
                   help="trajectory directory")
    p.add_argument("--mcr", type=float, default=DEFAULT_MCR,
                   help="USRP master clock rate")
    p.add_argument("--freq", type=float, default=None,
                   help="RF centre frequency, default carrier_hz from config")
    return p


def usrp_args(dev_args, mcr):
    a = f"master_clock_rate={mcr:.0f}"
    return f"{dev_args},{a}" if dev_args else a
