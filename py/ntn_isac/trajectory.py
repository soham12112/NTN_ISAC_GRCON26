"""traj.csv / traj_meta.json reader and range interpolation.

The trajectory table is 100 Hz; the waveform is 7.68 MS/s. Everything in
between is a cubic spline, evaluated on an absolute sample clock that never
resets - the 20 ms waveform file loops, the pass does not.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

C = 299792458.0


@dataclass
class Path:
    """One propagation path: a range history and a power ratio history."""

    pid: int
    name: str
    t: np.ndarray          # seconds, monotone
    R: np.ndarray          # metres
    G_db: np.ndarray       # echo-to-direct power ratio, dB
    _R: CubicSpline = field(default=None, repr=False)

    def __post_init__(self):
        self._R = CubicSpline(self.t, self.R, extrapolate=True)

    def range_at(self, t: np.ndarray) -> np.ndarray:
        return self._R(t)

    def gain_at(self, t: np.ndarray) -> np.ndarray:
        """Amplitude (not power). Held piecewise-constant: G is a control knob,
        not a physical trajectory, and splining it would smear the deliberate
        on/off edges that produce the target-absent runs."""
        i = np.searchsorted(self.t, t, side="right") - 1
        np.clip(i, 0, self.t.size - 1, out=i)
        return 10.0 ** (self.G_db[i] / 20.0)

    def present(self, t: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
        i = np.searchsorted(self.t, t, side="right") - 1
        np.clip(i, 0, self.t.size - 1, out=i)
        return self.G_db[i] > floor_db


def linear_path(pid: int, name: str, R0_m: float, doppler_hz: float,
                gain_db: float, fc_hz: float, duration_s: float = 1e6) -> Path:
    """A manual tap specified by Doppler, expressed as a range rate.

    Section 6 is emphatic that delay is the only thing injected and Doppler
    must fall out of the carrier phase. So a "100 Hz tap" is a tap whose range
    drifts at -f_d * c / f_c; there is no separate Doppler term anywhere in
    this codebase.
    """
    rdot = -doppler_hz * C / fc_hz
    t = np.array([0.0, duration_s])
    return Path(pid=pid, name=name, t=t, R=R0_m + rdot * t,
                G_db=np.full(2, float(gain_db)))


@dataclass
class Trajectory:
    paths: list
    meta: dict

    @property
    def r_nominal(self) -> float:
        return float(self.meta["r_nominal_m"])

    @property
    def fc(self) -> float:
        return float(self.meta["carrier_hz"])

    @property
    def duration(self) -> float:
        return float(self.paths[0].t[-1])


def load(path: str) -> Trajectory:
    """Read a directory containing traj.csv and traj_meta.json."""
    with open(os.path.join(path, "traj_meta.json")) as f:
        meta = json.load(f)
    csv_path = os.path.join(path, "traj.csv")
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)

    if header[0] != "t_s":
        raise ValueError(f"{csv_path}: first column is {header[0]!r}, expected 't_s'")
    n_paths = (len(header) - 1) // 2
    if len(header) != 1 + 2 * n_paths:
        raise ValueError(f"{csv_path}: header has {len(header)} columns")

    t = arr[:, 0]
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{csv_path}: t_s is not strictly increasing")

    names = {p["id"]: p["name"] for p in meta.get("paths", [])}
    paths = []
    for p in range(n_paths):
        if header[1 + 2 * p] != f"R{p}_m" or header[2 + 2 * p] != f"G{p}_db":
            raise ValueError(f"{csv_path}: unexpected header for path {p}")
        paths.append(Path(pid=p, name=names.get(p, f"path{p}"), t=t,
                          R=arr[:, 1 + 2 * p], G_db=arr[:, 2 + 2 * p]))
    return Trajectory(paths=paths, meta=meta)
