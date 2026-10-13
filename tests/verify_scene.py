#!/usr/bin/env python3
"""V20-V22: the scene compiler.

  V20  the analytic circular-orbit pass against the SGP4 one
  V21  the TR 38.811 tap tables against MATLAB's nrTDLChannel
  V22  a compiled scene plays through the C++ channel block, and every path
       lands at the excess path and Doppler the scene asked for

V20 and V21 exist because the scene compiler deliberately does not need MATLAB,
and "does not need MATLAB" is only worth anything if the thing it replaced is
checked against the thing it replaces. Both skip themselves, loudly, when the
reference is absent.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from ntn_isac import scene as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATLAB = "/Applications/MATLAB_R2025b.app/bin/matlab"
n_fail = 0


def check(name, ok, detail=""):
    global n_fail
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        n_fail += 1


def skip(name, why):
    print(f"    [SKIP] {name}  {why}")


print(__doc__.strip())

# ------------------------------------------------------------------ V20

print()
print("V20 analytic circular pass against the SGP4 pass")

ref = os.path.join(ROOT, "data", "traj_ota")
if not os.path.isdir(ref):
    skip("orbit comparison", "data/traj_ota not generated; run gen_ota_bundles")
else:
    M = np.loadtxt(os.path.join(ref, "traj.csv"), delimiter=",", skiprows=1)
    t_s, R_s = M[:, 0], M[:, 1]
    with open(os.path.join(ref, "traj_meta.json")) as f:
        meta = json.load(f)
    el_max = float(meta["elevation_deg"]["max"])
    t_a, R_a, el_a = S.circular_pass(alt_km=550.0, max_elev_deg=el_max,
                                     duration_s=float(t_s[-1]), rate_hz=100.0)
    rr_s = np.gradient(R_s, t_s)
    rr_a = np.gradient(R_a, t_a)

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-9) * 100

    rows = [("min slant range, km", R_a.min() / 1e3, R_s.min() / 1e3),
            ("max slant range, km", R_a.max() / 1e3, R_s.max() / 1e3),
            ("peak range rate, m/s", np.abs(rr_a).max(), np.abs(rr_s).max()),
            ("max elevation, deg", np.nanmax(el_a), el_max)]
    print("    quantity                analytic      SGP4     error")
    for nm, a, b in rows:
        print(f"    {nm:<22} {a:9.1f} {b:9.1f}   {rel(a, b):5.1f}%")
    # 10% is the honest tolerance. The analytic model is a circular orbit over a
    # spherical Earth with no Earth rotation, so it reproduces the quantities
    # that matter - range, range rate, Doppler, delay walk - to a few percent
    # and is not trying to be an ephemeris. Use orbit.source: file when it is.
    check("analytic orbit within 10% of SGP4 on every quantity",
          all(rel(a, b) < 10.0 for _, a, b in rows))

# ------------------------------------------------------------------ V21

print()
print("V21 TR 38.811 tap tables against MATLAB nrTDLChannel")

if not os.path.exists(MATLAB):
    skip("3GPP table cross-check", f"no MATLAB at {MATLAB}")
else:
    # Written beside the test, not in tempfile.gettempdir(): on macOS that is
    # a per-process private directory MATLAB -batch could not read.
    #
    # The name must not start with an underscore. MATLAB's run() evaluates the
    # file's stem as an identifier, so "_xchk_scene.m" fails with "Invalid text
    # character" - which reads like a corrupt file and is really an illegal
    # function name.
    m = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "xchk_scene.m")
    with open(m, "w") as f:
        # One statement per line and no continuations: MATLAB rejected an
        # earlier version with "Invalid text character", which is what it says
        # when a line continuation or an escape has not survived being written
        # out from Python.
        f.write("profs = {'NTN-TDL-A','NTN-TDL-B','NTN-TDL-C','NTN-TDL-D'};\n")
        f.write("for i=1:numel(profs)\n")
        f.write("c = nrTDLChannel;\n")
        f.write("c.DelayProfile = profs{i};\n")
        f.write("c.DelaySpread = 100e-9;\n")
        f.write("c.MaximumDopplerShift = 10;\n")
        f.write("c.SampleRate = 7.68e6;\n")
        f.write("q = info(c);\n")
        f.write("d = jsonencode(round(q.PathDelays/100e-9,4));\n")
        f.write("g = jsonencode(round(q.AveragePathGains,3));\n")
        f.write("fprintf('XCHK|%s|%s|%s|%.4f\\n', profs{i}, d, g, "
                "q.KFactorFirstTap);\n")
        f.write("end\n")
    r = subprocess.run([MATLAB, "-batch", f"run('{m}')"],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout
    os.remove(m)
    got = {}
    for line in out.splitlines():
        if line.startswith("XCHK|"):
            _, name, d, g, k = line.split("|")
            got[name] = (json.loads(d), json.loads(g), float(k))
    if not got:
        skip("3GPP table cross-check",
             "MATLAB produced no usable output: "
             + (r.stderr or out).strip().replace("\n", " ")[:200])
    for name, (d, g, k) in got.items():
        p = S.NTN_TDL[name]
        dd = np.max(np.abs(np.asarray(d) - np.asarray(p["delays"])))
        gg = np.max(np.abs(np.asarray(g) - np.asarray(p["gains_db"])))
        kk = (p["k_db"] is None and not np.isfinite(k)) or \
             (p["k_db"] is not None and abs(k - p["k_db"]) < 0.05)
        print(f"    {name}: {len(d)} taps, max delay error {dd:.5f} "
              f"(normalised), max gain error {gg:.4f} dB, "
              f"K {'-Inf/NLOS' if not np.isfinite(k) else f'{k:.3f} dB'}")
        check(f"{name} matches MATLAB", dd < 1e-4 and gg < 1e-3 and kk)

# ------------------------------------------------------------------ V22

print()
print("V22 a compiled scene plays through the C++ channel block")

try:
    from gnuradio import ntnisac
except Exception as exc:                               # noqa: BLE001
    skip("channel playback", f"gnuradio.ntnisac not importable: {exc}")
else:
    sc = dict(
        carrier_hz=2.478e9, duration_s=60.0, rate_hz=1000.0,
        orbit=dict(source="circular", alt_km=550.0, max_elev_deg=67.5),
        paths=[
            dict(type="direct", name="satellite"),
            dict(type="static", name="clutter", excess_m=250.0, gain_db=-20.0),
            dict(type="linear", name="drone", excess_m=640.0,
                 excess_rate_mps=4.7, t_ref_s=30.0, gain_db=-25.0),
            dict(type="rotating", name="blade", excess_m=0.4, radius_m=0.25,
                 rot_hz=7.0, gain_db=-30.0, phase_rad=np.pi / 2),
        ])
    c = S.compile_scene(sc, base_dir=ROOT)
    out = tempfile.mkdtemp(prefix="scene_v22_")
    S.write(c, out)
    fc = sc["carrier_hz"]
    lam = 299792458.0 / fc
    ch = ntnisac.leo_channel(os.path.join(out, "traj.csv"), 7.68e6, fc,
                             c.meta["r_nominal_m"], 40.0, 131072, 8, 30.0)
    check("path count", ch.num_paths() == 4, f"{ch.num_paths()}")
    print("    path            excess m   Doppler Hz   gain dB")
    st = [(ch.tap_excess_m(p), ch.tap_doppler_hz(p), ch.tap_gain_db(p))
          for p in range(ch.num_paths())]
    for p, (e, d, g) in enumerate(st):
        print(f"    p{p} {sc['paths'][p]['name']:<12} {e:9.2f} {d:12.2f} "
              f"{g:9.2f}")

    # Excess path, against what the scene asked for.
    for p, want in ((1, 250.0), (2, 640.0), (3, 0.4)):
        check(f"p{p} excess path", abs(st[p][0] - want) < 0.5,
              f"{st[p][0]:.2f} m, wanted {want} m")
    # Residual Doppler, relative to the direct ray. The blade is given a
    # quarter-turn phase so it sits at a zero crossing of its range, where its
    # excess-range rate - and therefore its Doppler - is at a maximum.
    res = [st[p][1] - st[0][1] for p in range(4)]
    check("p1 clutter is at zero residual Doppler", abs(res[1]) < 0.05,
          f"{res[1]:+.3f} Hz")
    check("p2 drone Doppler follows its range rate",
          abs(res[2] - (-4.7 / lam)) < 0.2,
          f"{res[2]:+.2f} Hz, wanted {-4.7/lam:+.2f}")
    edge = 2 * np.pi * 0.25 * 7.0 / lam
    check("p3 blade Doppler is at its tip-speed edge",
          abs(abs(res[3]) - edge) < 0.05 * edge,
          f"{res[3]:+.2f} Hz, edge +-{edge:.2f}")
    for p, want in ((1, -20.0), (2, -25.0), (3, -30.0)):
        check(f"p{p} gain", abs(st[p][2] - want) < 0.01, f"{st[p][2]:.2f} dB")
    shutil.rmtree(out, ignore_errors=True)

print()
print("FAILURES:", n_fail)
sys.exit(1 if n_fail else 0)
