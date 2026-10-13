"""Declarative channel scenes: one file describes the whole propagation channel.

Before this, composing a channel meant running three tools in order, each
rewriting the file the next one read - `gen_trajectory.m` for the orbit and its
geometric scatterers, `gen_ntn_channel.m` for 3GPP multipath, then
`add_bench_target.py` for a physical object. Order mattered, nothing recorded
what had been applied, and two of the three needed MATLAB. That is fine for a
demo and wrong for an instrument.

A scene is a YAML file listing the orbit and every path, compiled in one step:

    $NTN_PY tools/build_scene.py scenes/bench.yml

It emits exactly the same `traj.csv` and `traj_meta.json` the old tools did, so
nothing downstream changes and the C++ channel block is untouched. What changes
is that the channel is now a thing you can read, diff, version and hand to
someone else.

## Why this is only a layer, and not a rewrite

`traj.csv` was already general: an arbitrary list of paths, each with a range
history and a gain history. That is why a rotating blade and a set of 3GPP
clusters both dropped in without touching the block. Every path type below is a
recipe for filling in one more `R_p(t)` column.

The one rule they all obey is handoff section 6's: **delay is the only thing
injected, and Doppler falls out of the carrier phase.** So a path that should
show a Doppler is expressed as a range that changes at the rate implying it.
There is no Doppler parameter anywhere here, and adding one would let delay and
Doppler disagree - which describes an object that cannot exist, and makes any
receiver that jointly tracks them produce confident nonsense.

## No MATLAB

The orbit is propagated analytically for a circular orbit, and the 3GPP tap
tables are transcribed from TR 38.811. Both are cross-checked against MATLAB by
`tests/verify_scene.py`, which skips itself when MATLAB is absent. The receive
laptop needs no licence.

`orbit.source: file` reuses an existing MATLAB-generated trajectory instead,
which is what to do when the SGP4-propagated pass and its 3-D scatterer
geometry are what is wanted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

C = 299792458.0
RE = 6371e3            # mean Earth radius, m
MU = 3.986004418e14    # Earth gravitational parameter, m^3/s^2


# --------------------------------------------------------------- 3GPP tables

# 3GPP TR 38.811 Table 6.9.2-1 and following: non-terrestrial tapped delay line
# profiles. Delays are NORMALISED and are scaled by the chosen delay spread;
# gains are dB relative to the profile total. NTN-TDL-A and -B are NLOS and have
# no line-of-sight component; -C and -D are LOS, with the K factor of the first
# tap given.
#
# Transcribed rather than computed, and cross-checked against MATLAB's
# nrTDLChannel by tests/verify_scene.py: at a 100 ns delay spread the delays
# agree to the nanosecond and the K factors to 0.1 dB.
NTN_TDL = {
    "NTN-TDL-A": dict(delays=[0.0, 1.0811, 2.8416],
                      gains_db=[0.0, -4.675, -6.482], k_db=None),
    "NTN-TDL-B": dict(delays=[0.0, 0.7249, 0.7410, 5.7392],
                      gains_db=[0.0, -1.973, -4.332, -11.914], k_db=None),
    "NTN-TDL-C": dict(delays=[0.0, 14.8124],
                      gains_db=[0.0, -23.373], k_db=10.224),
    "NTN-TDL-D": dict(delays=[0.0, 0.5596, 7.3340],
                      gains_db=[0.0, -9.887, -16.771], k_db=11.707),
}
# The two fixed-delay-spread variants of the same standard.
NTN_TDL["NTN-TDLA100"] = dict(NTN_TDL["NTN-TDL-A"], fixed_ds_ns=100.0)
NTN_TDL["NTN-TDLC5"] = dict(NTN_TDL["NTN-TDL-C"], fixed_ds_ns=5.0)

GAIN_FLOOR_DB = -200.0     # "path absent", matching gen_trajectory


# ------------------------------------------------------------------- orbit

def circular_pass(alt_km=550.0, max_elev_deg=67.5, duration_s=600.0,
                  rate_hz=1000.0, return_los=False):
    """Slant range against time for a circular-orbit pass, centred on closest
    approach.

    Spherical geometry, not SGP4. The satellite moves on a great circle whose
    closest approach to the ground station is a central angle `theta0`, so the
    central angle at time t is

        cos(theta(t)) = cos(theta0) * cos(omega_g * t)

    and the slant range follows from the law of cosines. `theta0` is found from
    the requested maximum elevation via

        tan(E) = (cos(theta) - Re/(Re+h)) / sin(theta)

    This reproduces the quantities that matter - slant range, range rate,
    Doppler, Doppler rate and delay walk - to well inside a percent of the
    SGP4-propagated pass, and it needs nothing installed. Use
    `orbit.source: file` when the real ephemeris matters.
    """
    h = alt_km * 1e3
    a = RE + h
    n = np.sqrt(MU / a ** 3)                # orbital angular rate, rad/s

    # Central angle at the requested maximum elevation.
    E = np.radians(max_elev_deg)
    th = np.linspace(1e-6, np.pi / 2, 200000)
    elev = np.arctan2(np.cos(th) - RE / a, np.sin(th))
    theta0 = float(np.interp(-E, -elev, th))   # elev is decreasing in th

    t = np.arange(0.0, duration_s + 0.5 / rate_hz, 1.0 / rate_hz)
    tc = t - duration_s / 2.0
    phase = n * tc
    cos_th = np.cos(theta0) * np.cos(phase)
    R = np.sqrt(RE ** 2 + a ** 2 - 2 * RE * a * cos_th)
    el = np.degrees(np.arctan2(cos_th - RE / a,
                               np.sqrt(np.maximum(1 - cos_th ** 2, 0.0))))
    if return_los:
        # Ground station at [RE, 0, 0].  The orbit is a great circle whose
        # closest central angle is theta0.  Only the norm of du/dt is used
        # downstream, so the arbitrary local east/north orientation cancels.
        sat = a * np.column_stack((
            np.cos(theta0) * np.cos(phase),
            np.sin(phase),
            np.sin(theta0) * np.cos(phase),
        ))
        los = sat - np.array([RE, 0.0, 0.0])
        los /= np.linalg.norm(los, axis=1, keepdims=True)
        return t, R, el, los
    return t, R, el


def orbit_from_file(path, rate_hz=None):
    """Path 0 of an existing trajectory, optionally resampled."""
    from scipy.interpolate import CubicSpline
    M = np.loadtxt(os.path.join(path, "traj.csv"), delimiter=",", skiprows=1)
    t, R = M[:, 0], M[:, 1]
    if rate_hz:
        tn = np.arange(t[0], t[-1] + 0.5 / rate_hz, 1.0 / rate_hz)
        R = CubicSpline(t, R)(tn)
        t = tn
    return t, R, np.full(t.size, np.nan)


# -------------------------------------------------------------- path types

def _window(t, spec):
    """Gain-floor mask outside an optional [start, stop] visibility window."""
    if not spec:
        return np.ones(t.size, bool)
    lo, hi = spec
    return (t >= float(lo)) & (t <= float(hi))


def build_path(kind, t, R0, fc, **kw):
    """One path's (excess_m_over_time, gain_db_over_time, description).

    Every type returns an EXCESS path relative to the direct ray, in metres.
    The caller adds R0. Doppler is never returned because it is never injected:
    a path that should show f_d Hz is given an excess that changes at
    -f_d * c / fc metres per second, and the channel's carrier phase term does
    the rest.
    """
    lam = C / fc
    g = float(kw.get("gain_db", -25.0))
    win = _window(t, kw.get("window_s"))

    if kind == "direct":
        return np.zeros(t.size), np.zeros(t.size), "Direct satellite path."

    if kind == "static":
        e = float(kw["excess_m"])
        return (np.full(t.size, e), np.where(win, g, GAIN_FLOOR_DB),
                f"Static scatterer at {e:.1f} m of excess path.")

    if kind == "linear":
        # A target crossing at constant excess-range rate. The Doppler it
        # produces is rate/lambda, which is what to size against the
        # zero-Doppler notch.
        e = float(kw["excess_m"])
        rate = float(kw.get("excess_rate_mps", 0.0))
        t0 = float(kw.get("t_ref_s", t[0] + (t[-1] - t[0]) / 2))
        ex = e + rate * (t - t0)
        return (ex, np.where(win, g, GAIN_FLOOR_DB),
                f"Crossing target, {e:.1f} m of excess at t={t0:.0f} s, "
                f"{rate:+.2f} m/s ({-rate/lam:+.1f} Hz).")

    if kind == "rotating":
        # A point on a blade of radius r turning at f_rot. The phase modulation
        # index is 2*pi*r/lambda, so the echo is a Bessel comb of lines spaced
        # at f_rot with its envelope edge at 2*pi*r*f_rot/lambda. Both numbers
        # are recoverable, which is what gives this target its own ground truth.
        e = float(kw.get("excess_m", 0.4))
        r = float(kw["radius_m"])
        f = float(kw["rot_hz"])
        ph = float(kw.get("phase_rad", 0.0))
        ex = e + r * np.cos(2 * np.pi * f * t + ph)
        edge = 2 * np.pi * r * f / lam
        return (ex, np.where(win, g, GAIN_FLOOR_DB),
                f"Rotating scatterer r={r} m at {f} rev/s ({f*60:.0f} RPM), "
                f"{e:.2f} m of excess, Doppler comb every {f:.1f} Hz out to "
                f"+-{edge:.1f} Hz.")

    if kind == "pendulum":
        e = float(kw.get("excess_m", 0.5))
        amp = float(kw["amplitude_m"])
        per = float(kw["period_s"])
        ex = e + amp * np.sin(2 * np.pi * t / per)
        return (ex, np.where(win, g, GAIN_FLOOR_DB),
                f"Pendulum, {amp} m amplitude at {per} s period, peak Doppler "
                f"+-{2*np.pi*amp/per/lam:.1f} Hz.")

    if kind == "custom":
        # An arbitrary excess history, as a Python expression in t. The escape
        # hatch: anything this file cannot describe can still be written here
        # rather than forcing a new path type.
        ex = eval(str(kw["excess_m_expr"]),                    # noqa: S307
                  {"np": np, "t": t, "pi": np.pi, "lam": lam})
        ex = np.broadcast_to(np.asarray(ex, float), t.shape).copy()
        return (ex, np.where(win, g, GAIN_FLOOR_DB),
                f"Custom excess history: {kw['excess_m_expr']}")

    raise ValueError(f"unknown path type {kind!r}")


def ntn_tdl_paths(t, fc, profile="NTN-TDL-C", delay_spread_ns=100.0,
                  mobile_speed_mps=0.83, sub_rays=1, t_ref_s=None,
                  max_excess_m=1260.0, seed=2026):
    """3GPP TR 38.811 clusters, as ramped-range paths.

    Two things this does that are easy to get wrong.

    **Tap 0's line of sight is dropped.** On a LOS profile tap 0 is Rician, so
    it is a direct ray plus a diffuse part - and the scene's `direct` path
    already is the direct ray. Emitting tap 0 whole would count it twice. Only
    its diffuse part is emitted, at -K dB, and every other cluster is
    referenced to the same line-of-sight power. On an NLOS profile (-A, -B)
    there is no line of sight at all, so the clusters are referenced to the
    scene's direct path and the profile describes what sits on top of it.

    **Each cluster becomes a ramped range, not a Doppler.** A sub-ray with a
    constant Doppler is a constant range rate, which is why the ramp is
    referenced to `t_ref_s`: a constant Doppler accumulates without bound, so
    by 300 s a 0.83 m/s ray has drifted 249 m from where it was placed.
    """
    if profile not in NTN_TDL:
        raise ValueError(f"unknown profile {profile!r}; have "
                         f"{sorted(NTN_TDL)}")
    p = NTN_TDL[profile]
    ds = p.get("fixed_ds_ns", delay_spread_ns) * 1e-9
    lam = C / fc
    fD = mobile_speed_mps / lam
    if t_ref_s is None:
        t_ref_s = t[0] + (t[-1] - t[0]) / 2

    tau = np.asarray(p["delays"], float) * ds
    gdb = np.asarray(p["gains_db"], float)
    excess = tau * C

    if p["k_db"] is None:
        rel_db = gdb.copy()                     # NLOS: no LOS to remove
        k_note = "NLOS profile, no line-of-sight tap to remove"
    else:
        klin = 10 ** (p["k_db"] / 10.0)
        los = klin / (klin + 1.0)
        los_db = gdb[0] + 10 * np.log10(los)
        rel_db = gdb - los_db
        rel_db[0] = gdb[0] + 10 * np.log10(1 - los) - los_db
        k_note = (f"K = {p['k_db']:.2f} dB on tap 0, its line of sight dropped "
                  f"in favour of the scene's direct path")

    rng = np.random.default_rng(seed)
    n_sub = max(1, int(sub_rays))
    out = []
    for k in range(tau.size):
        if excess[k] > max_excess_m:
            continue
        if n_sub == 1:
            cos_t = np.array([2 * rng.random() - 1.0])
        else:
            # Jakes: rays uniform in arrival angle, so Doppler is fD*cos(theta)
            # and their sum fades at the difference frequencies.
            cos_t = np.cos(2 * np.pi * (np.arange(n_sub) + rng.random()) / n_sub)
        for s, ct in enumerate(cos_t):
            fd = fD * ct
            ex = excess[k] - fd * lam * (t - t_ref_s)
            out.append((ex, np.full(t.size, rel_db[k] - 10 * np.log10(n_sub)),
                        f"{profile} cluster {k} ray {s}: {excess[k]:.1f} m of "
                        f"excess at t={t_ref_s:.0f} s, {fd:+.2f} Hz, "
                        f"{rel_db[k] - 10*np.log10(n_sub):+.1f} dB vs the LOS"))
    return out, dict(profile=profile, delay_spread_s=ds,
                     mobile_speed_mps=mobile_speed_mps, max_doppler_hz=fD,
                     k_factor_db=p["k_db"], sub_rays=n_sub, note=k_note,
                     cluster_excess_m=excess.tolist(),
                     cluster_rel_db=rel_db.tolist())


# ----------------------------------------------------------------- compile

@dataclass
class Compiled:
    t: np.ndarray
    R: np.ndarray               # [N, n_path], absolute ranges
    G: np.ndarray               # [N, n_path], dB
    meta: dict
    warnings: list = field(default_factory=list)

    @property
    def n_path(self):
        return self.R.shape[1]


def compile_scene(scene: dict, base_dir=".") -> Compiled:
    fc = float(scene.get("carrier_hz", 2.478e9))
    orb = dict(scene.get("orbit", {}) or {})
    src = orb.pop("source", "circular")
    rate = float(scene.get("rate_hz", 1000.0))

    los = None
    if src == "file":
        t, R0, el = orbit_from_file(
            os.path.join(base_dir, orb["path"]), rate_hz=rate)
    elif src == "circular":
        t, R0, el, los = circular_pass(
            alt_km=float(orb.get("alt_km", 550.0)),
            max_elev_deg=float(orb.get("max_elev_deg", 67.5)),
            duration_s=float(scene.get("duration_s", 600.0)), rate_hz=rate,
            return_los=True)
    else:
        raise ValueError(f"unknown orbit source {src!r}")

    warn = []
    cols, descs = [], []
    ntn_meta = None
    for spec in scene.get("paths", []):
        s = dict(spec)
        kind = s.pop("type")
        name = s.pop("name", kind)
        if kind == "ntn_tdl":
            got, ntn_meta = ntn_tdl_paths(t, fc, **s)
            for ex, g, d in got:
                cols.append((R0 + ex, g))
                descs.append(dict(name="ntn_cluster", description=d))
        else:
            ex, g, d = build_path(kind, t, R0, fc, **s)
            cols.append((R0 + ex, g))
            descs.append(dict(name=name, description=d))

    if not cols:
        raise ValueError("scene has no paths; at least a 'direct' is needed")

    R = np.column_stack([c[0] for c in cols])
    G = np.column_stack([c[1] for c in cols])

    # Checks that are cheap here and expensive to discover later.
    cp_m = float(scene.get("cp_limit_m", 1406.0)) * 0.9
    present = G > -100.0
    ex = R - R[:, [0]]
    worst = np.where(present, ex, 0.0).max()
    if worst > cp_m:
        warn.append(
            f"a path reaches {worst:.0f} m of excess, past the {cp_m:.0f} m "
            f"cyclic-prefix budget; outside it the receiver cannot see it")

    # The rotation-rate and comb interaction, checked per rotating path rather
    # than left for the operator to rediscover on the dashboard.
    for spec in scene.get("paths", []):
        if spec.get("type") == "rotating":
            f = float(spec["rot_hz"])
            if abs(25.0 / f - round(25.0 / f)) < 1e-6:
                warn.append(
                    f"rotation rate {f} rev/s divides the 25 Hz aperture comb, "
                    f"so every harmonic lands on a masked bin; pick 7 or 11")
            spc = rate / f
            if spc < 12:
                warn.append(
                    f"{f} rev/s at a {rate:.0f} Hz table is {spc:.1f} samples "
                    f"per rotation; below 12 the spline's own comb is what "
                    f"gets measured. Raise rate_hz")

    n = R.shape[1]
    thr = 20.4 * 3 / n
    if thr < 7.68:
        warn.append(
            f"{n} paths is about {thr:.1f} MS/s against 7.68 required; reduce "
            f"sub_rays or paths, or the channel will not keep up live")

    Rp = np.where(present, R, np.inf)
    meta = dict(
        generator="scene", scene=scene, carrier_hz=fc,
        r_nominal_m=float(np.floor(Rp.min() * 1e6) / 1e6),
        duration_s=float(t[-1] - t[0]), rate_hz=rate, n_samples=int(t.size),
        n_paths=int(n), orbit=dict(source=src, **orb),
        slant_range_m=dict(min=float(R0.min()), max=float(R0.max())),
        cp_budget_m=cp_m, throughput_estimate_ms=float(thr),
        paths=[dict(id=i, gain_db=float(G[0, i]),
                    excess_m_max=float(ex[:, i].max()), **d)
               for i, d in enumerate(descs)],
        warnings=warn)
    if np.isfinite(el).any():
        meta["elevation_deg"] = dict(min=float(np.nanmin(el)),
                                     max=float(np.nanmax(el)))
    if los is not None:
        los_rate = np.linalg.norm(np.gradient(los, t, axis=0), axis=1)
        meta["los_rotation_deg_s"] = float(np.degrees(np.max(los_rate)))
    if ntn_meta:
        meta["ntn"] = ntn_meta
    if "noise" in scene:
        meta["noise"] = scene["noise"]
    return Compiled(t=t, R=R, G=G, meta=meta, warnings=warn)


def write(c: Compiled, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n = c.n_path
    cols = ["t_s"]
    for k in range(n):
        cols += [f"R{k}_m", f"G{k}_db"]
    body = np.empty((c.t.size, 1 + 2 * n))
    body[:, 0] = c.t
    body[:, 1::2] = c.R
    body[:, 2::2] = c.G
    with open(os.path.join(out_dir, "traj.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        np.savetxt(f, body, delimiter=",", fmt=["%.6f"] + ["%.6f", "%.4f"] * n)
    with open(os.path.join(out_dir, "traj_meta.json"), "w") as f:
        json.dump(c.meta, f, indent=2)
    return out_dir


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
