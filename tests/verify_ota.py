#!/usr/bin/env python3
"""V16-V19: the over-the-air additions, against a simulated two-laptop link.

The conducted setup commands the pass from one process, so the receiver knows
tau0(t) exactly. Two laptops do not, and a real object between the radios has
its echo land in the same delay cell as the direct path. These checkpoints
cover the code that closes both gaps, before any radio is involved.

  V16  acquire() recovers a known frame offset and carrier offset
  V17  OtaTracker converges on the injected clock parameters
  V18  CLEAN cancels the direct path, measured against a fixed reference
  V19  the micro-Doppler of a co-delay rotating target, and where its
       detection floor actually is

The link model is in py/ntn_isac/otasim.py, including why a receive-clock
offset needs no new channel code and why the target is a rotating blade.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from ntn_isac import clean, ota, otasim, receiver, sensing, udoppler
from ntn_isac.trajectory import C

ROOT = os.path.join(os.path.dirname(__file__), "..")
WF = os.path.join(ROOT, "data", "wf_ota")
TRAJ = os.path.join(ROOT, "data", "traj_ota")
N_FRAMES = 5

n_fail = 0


def check(name, ok, detail=""):
    global n_fail
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        n_fail += 1


def close(name, got, want, tol, unit=""):
    check(f"{name}: {got:+.4f} {unit}".rstrip(), abs(got - want) <= tol,
          f"(expected {want:+.4f}, tol {tol:g} {unit})".replace("  ", " "))


def sim(**kw):
    return otasim.simulate(WF, TRAJ, **kw)


print(__doc__.strip())

# ------------------------------------------------------------------ V16

print()
print("V16 coarse acquisition of frame offset and carrier offset")

LAG = 51234
EPS_PPM = 3.2                                   # two free-running B200 TCXOs
y, b, tr, cfg, truth = sim(seconds=0.35, eps_ppm=EPS_PPM, esn0_db=30.0)
n_frame = int(cfg["frame_len_samples"])
n_dwell = N_FRAMES * n_frame
margin = int(0.006 * b.fs)

# Rotate the capture so the frame boundary sits somewhere arbitrary, which is
# what a receiver that started at an unrelated moment actually sees.
t_acq = time.time()
acq = ota.acquire(
    np.roll(y, LAG), b.iq(), cfg, cfo_max_hz=12e3, cfo_step_hz=2.5e3,
    n_sym_probe=24)
t_acq = time.time() - t_acq

# The frame origin the correlator should find is where the roll put it plus the
# channel's own bulk delay. At t = 300 s the pass is at closest approach, where
# R0 equals r_nominal, so the LEO term is zero and what is left is the
# per-stage tau_bias.
want_lag = int(round(LAG + truth["bias_samples"])) % n_frame
print(f"    injected  lag {want_lag} (roll {LAG % n_frame} + "
      f"{truth['bias_samples']:.0f} samples of bulk delay)  "
      f"carrier offset {truth['cfo_hz']:+.1f} Hz ({EPS_PPM} ppm)")
print(f"    measured  lag {acq.lag_samples}  carrier offset {acq.cfo_hz:+.1f} Hz  "
      f"peak {acq.peak_db:.1f} dB over the floor  [{t_acq*1e3:.0f} ms]")
check("acquisition declares lock", acq.locked, f"{acq.peak_db:.1f} dB")
# The lag only has to land inside the cyclic prefix; the per-dwell tracker
# takes it from there. 36 samples is the short CP.
lag_err = (acq.lag_samples - want_lag + n_frame // 2) % n_frame - n_frame // 2
close("lag error", float(lag_err), 0.0, 36.0, "samples")
close("carrier offset", acq.cfo_hz, truth["cfo_hz"], 1.5e3, "Hz")

# ------------------------------------------------------------------ V17

print()
print("V17 OtaTracker converges on the injected clock parameters")



D0 = 12.0
y, b, tr, cfg, truth = sim(seconds=0.9, eps_ppm=EPS_PPM, d0_samples=D0,
                           esn0_db=30.0)
trk = ota.OtaTracker(tr, b.fs, tau_bias_samples=truth["bias_samples"] - D0,
                     gps_locked=False)
acq = ota.acquire(y[:n_frame], b.iq(), cfg, n_sym_probe=24)
trk.set_acquisition(acq, t0=300.0)
trk.d0 = 0.0                     # the lag is a frame alignment, not a bias
print(f"    injected  d0 {D0:+.1f} samples   eps {EPS_PPM:+.3f} ppm "
      f"({truth['cfo_hz']:+.1f} Hz at {b.fc/1e9:.3f} GHz)")
print(f"    seeded    eps {trk.eps*1e6:+.3f} ppm from acquisition")

hist = []
for i in range(6):
    t0 = 300.0 + i * n_dwell / b.fs
    seg = y[i * n_dwell:i * n_dwell + n_dwell + margin]
    if seg.size < n_dwell + margin:
        break
    Y, H, tb, st = otasim.demod_dwell(seg, b, trk, t0, N_FRAMES)
    d, f, _, _ = ota.residual_from_H(H, tb.occ_mask, cfg)
    trk.update(d, f, t0)
    hist.append((d, f))
    print(f"    dwell {i}  residual {d:+8.3f} samples {f:+8.2f} Hz "
          f"-> d0 {trk.d0*b.fs:+8.3f}  eps {trk.eps*1e6:+.4f} ppm")

close("eps after convergence", trk.eps * 1e6, EPS_PPM, 0.25, "ppm")
check("residual delay is driven toward zero", abs(hist[-1][0]) < abs(hist[0][0]),
      f"{hist[0][0]:+.3f} -> {hist[-1][0]:+.3f} samples")
check("residual Doppler is driven toward zero",
      abs(hist[-1][1]) < abs(hist[0][1]),
      f"{hist[0][1]:+.2f} -> {hist[-1][1]:+.2f} Hz")

# ------------------------------------------------------------------ setup

# The emulated scatterers are switched off from here on. At t = 300 s the
# emulated drone sits about two delay cells from the direct path with ~20 Hz of
# residual Doppler, which is the same cell and the same band a rotating blade
# occupies: with it on, a target-absent run still showed an 85 dB line at
# +-20 Hz and the blade's echo could be swept over 50 dB without the measured
# number moving. A demo wants both targets in one map; a measurement of one
# has to exclude the other.
DISABLE = (1, 2)

# The direct path has been de-ramped to zero Doppler and static room clutter
# sits there with it, so CLEAN is confined to a few cells around the origin. A
# rotating target does pass through zero Doppler twice a revolution and that
# part of it goes too - unavoidable, and the same blindness the zero-Doppler
# notch already imposes. Detection uses the Doppler extremes, where a rotating
# scatterer's energy concentrates anyway.
DIRECT = dict(search=(-200e-9, 200e-9), search_doppler=(-15.0, 15.0))


def prepare(n_frames=N_FRAMES, **kw):
    """Simulate, demodulate one dwell and form Z."""
    y, b, tr, cfg, truth = sim(**kw)
    trk = receiver.EphemerisTracker(tr, b.fs,
                                    tau_bias_samples=truth["bias_samples"])
    nd = n_frames * int(cfg["frame_len_samples"])
    Y, H, tb, st = otasim.demod_dwell(y[:nd + margin], b, trk, 300.0, n_frames)
    X, W = sensing.build_variant(Y, tb, sensing.PAYLOAD)
    return sensing.form_z(Y, X, W), W, cfg, b, truth


# ------------------------------------------------------------------ V18

print()
print("V18 CLEAN cancels the direct path")

Z, W, cfg, b, truth = prepare(seconds=0.25, esn0_db=40.0, disable=DISABLE)

for win in ("hann", "blackmanharris"):
    m0 = sensing.map_from_z(Z, float(W.sum()), cfg, 2, win)
    _, comps, maps = clean.clean(Z, W, cfg, n_components=4, pad=2, window=win,
                                 **DIRECT)
    can = clean.cancellation_db(maps["before"], maps["after"],
                                m0.peak_delay_s, m0.peak_doppler_hz)
    print(f"    {win:15s} peak/noise {m0.pk_nf_db:5.1f} dB  "
          f"direct path down {can['cancelled_db']:5.1f} dB  "
          f"{len(comps)} component(s)")
    if win == "blackmanharris":
        check("CLEAN removes at least 30 dB of the direct path",
              can["cancelled_db"] >= 30.0, f"{can['cancelled_db']:.1f} dB")
        check("the noise floor is not materially raised",
              can["noise_after_db"] - can["noise_before_db"] < 3.0,
              f"{can['noise_after_db'] - can['noise_before_db']:+.2f} dB")

# ------------------------------------------------------------------ V19

print()
print("V19 a co-delay micro-Doppler target, the over-the-air geometry")

BLADE = dict(excess_m=0.4, radius_m=0.25, rot_hz=7.0)
lam = C / b.fc
fd_peak = otasim.blade_doppler_hz(b.fc, BLADE["radius_m"], BLADE["rot_hz"])
cell_m = C / (float(cfg["n_sc"]) * float(cfg["scs_khz"]) * 1e3)
FRAME_HZ = float(cfg["fs_hz"]) / float(cfg["frame_len_samples"])

print(f"    blade r={BLADE['radius_m']} m at {BLADE['rot_hz']} rev/s, "
      f"excess {BLADE['excess_m']} m, lambda {lam*100:.2f} cm")
print(f"    delay cell {cell_m:.1f} m, so the target is "
      f"{BLADE['excess_m']/cell_m:.4f} cells from the direct path: not "
      f"resolvable in delay, and no bandwidth here can make it so")
print(f"    predicted Doppler extent +-{fd_peak:.1f} Hz "
      f"(= 2 pi r f_rot / lambda)")


def blade_band(res, ref_delay_s):
    """Per-Doppler level at the direct path's delay, with the comb masked out.

    Slot 0 of every frame carries the SSB and no PDSCH, so W changes once per
    20 ms and every tap acquires a 50 Hz-spaced Doppler comb about 9 dB down -
    findings.md section 11 - plus the notch-edge residue section 15 measured
    at half the frame rate. That lands squarely in a rotating target's band.
    An earlier
    version of this check measured the 100 Hz comb order and called it the
    blade: it read exactly +50.0 Hz and did not move when the echo was swept
    over 50 dB.

    Which is the design rule for the physical target, not just for the test:
    pick a rotation rate whose harmonics miss the comb. At 7 rev/s the blade's
    lines fall on multiples of 7 Hz and the comb on multiples of 25, and they
    do not coincide below 175 Hz.
    """
    j = int(np.argmin(np.abs(res.delay_s - ref_delay_s)))
    fd = res.doppler_hz
    cellf = fd[1] - fd[0]
    keep = np.abs(np.abs(fd) - np.round(np.abs(fd) / (FRAME_HZ / 2))
                  * (FRAME_HZ / 2)) > 2 * cellf
    lvl = res.M[:, max(j - 2, 0):j + 3].max(axis=1)
    return fd, lvl, keep


def blade_snr(res, ref_delay_s, noise_ref, lo, hi):
    """Strongest non-comb line in a Doppler band, against a fixed noise scale.

    `noise_ref` is passed in rather than read from `res` so a cancelled map and
    an uncancelled one are measured on the same scale. Each map's own median
    flatters cancellation twice: removing the direct path lowers the median, so
    the ratio rises even when the numerator has not moved.
    """
    fd, lvl, keep = blade_band(res, ref_delay_s)
    sel = keep & (np.abs(fd) >= lo) & (np.abs(fd) <= hi)
    i = int(np.argmax(np.where(sel, lvl, 0.0)))
    return 20 * np.log10(lvl[i] / max(noise_ref, 1e-30)), float(fd[i])


# -- the physics: does the Doppler extent match the tip speed? ----------------
# Measured on the cancelled map, where the floor is the noise rather than the
# direct path's own structure. The search is capped at +-500 Hz because that is
# where the map stops meaning anything: W is periodic at the 1000 Hz slot rate,
# so beyond half of it a sideband of an in-band tap is indistinguishable from a
# tap (findings.md section 11).
def micro(n_frames, gain_db=-30.0, esn0_db=55.0):
    Z, W, cfg_, b_, _ = prepare(seconds=n_frames * 0.020 + 0.02, esn0_db=esn0_db,
                                disable=DISABLE, n_frames=n_frames,
                                blade=dict(BLADE, gain_db=gain_db))
    m0 = sensing.map_from_z(Z, float(W.sum()), cfg_, 2, "blackmanharris")
    _, _, mp = clean.clean(Z, W, cfg_, n_components=2, pad=2,
                           window="blackmanharris", **DIRECT)
    return udoppler.analyse(mp["after"], cfg_, m0.peak_delay_s,
                            noise_ref=m0.noise, fc=b_.fc,
                            rot_search_hz=(3.0, 40.0))


# The tip-speed edge is readable from the sensing stage's own 100 ms dwell. The
# rotation rate is not, and the table shows why rather than asserting it: at
# 100 ms the Doppler cell is 5 Hz against a 7 Hz line spacing, so the comb is
# simply not resolved and the best candidate is a harmonic. Only the 1 s row is
# a measurement; the others are what an unresolved estimate looks like, which
# is worth seeing because it is confident and wrong.
print("    dwell   extent Hz   best rate rev/s   implied radius cm   (true "
      f"{BLADE['rot_hz']:.0f} rev/s, {BLADE['radius_m']*100:.0f} cm)")
for nf in (5, 25, 50):
    md = micro(nf)
    rot = "unresolved" if not np.isfinite(md.rotation_hz) else f"{md.rotation_hz:.2f}"
    rad = "     -" if not np.isfinite(md.radius_m) else f"{md.radius_m*100:6.1f}"
    print(f"    {nf*20:4d} ms  {md.extent_hz:8.1f}   {rot:>15}   {rad}")

md = micro(5)
close("Doppler extent matches the tip speed at a 100 ms dwell", md.extent_hz,
      fd_peak, 0.25 * fd_peak, "Hz")
md = micro(50)
close("Doppler extent at a 1 s dwell", md.extent_hz, fd_peak, 0.15 * fd_peak,
      "Hz")
close("recovered rotation rate at a 1 s dwell", md.rotation_hz,
      BLADE["rot_hz"], 0.7, "rev/s")
close("recovered tip radius at a 1 s dwell", md.radius_m, BLADE["radius_m"],
      0.06, "m")

# -- where the detection floor actually is -----------------------------------
# Two things can hide this target and they dominate in different places: the
# slow-time window's Doppler skirt off the direct path, and the noise. Sweeping
# the echo ratio separates them, which is what says whether CLEAN is
# load-bearing or merely tidy.
lo, hi = 0.45 * fd_peak, 1.2 * fd_peak
print()
print(f"    strongest non-comb line in {lo:.0f}-{hi:.0f} Hz at the direct "
      f"path's delay, dB over a fixed noise reference")
print("     echo    hann  hann+CLEAN     b-h  b-h+CLEAN")
rows = []
for g_db in (None, -20.0, -40.0, -55.0):
    kw = dict(seconds=0.25, esn0_db=55.0, disable=DISABLE)
    if g_db is not None:
        kw["blade"] = dict(BLADE, gain_db=g_db)
    Z, W, cfg, b, truth = prepare(**kw)
    row = {}
    for win in ("hann", "blackmanharris"):
        m0 = sensing.map_from_z(Z, float(W.sum()), cfg, 2, win)
        _, comps, maps = clean.clean(Z, W, cfg, n_components=2, pad=2,
                                     window=win, **DIRECT)
        row[win] = blade_snr(m0, m0.peak_delay_s, m0.noise, lo, hi)[0]
        row[win + "+clean"] = blade_snr(maps["after"], m0.peak_delay_s,
                                        m0.noise, lo, hi)[0]
    rows.append((g_db, row))
    tag = "absent" if g_db is None else f"{g_db:.0f}"
    print(f"    {tag:>6}  {row['hann']:6.1f}  {row['hann+clean']:10.1f}  "
          f"{row['blackmanharris']:6.1f}  {row['blackmanharris+clean']:9.1f}")

absent, strong, mid, weak = (r[1] for r in rows)
best = "blackmanharris+clean"
print()
print("    each route against its own target-absent floor:")
for win in ("hann", "hann+clean", "blackmanharris", "blackmanharris+clean"):
    print(f"      {win:22s} -20 dB echo {strong[win]-absent[win]:+6.1f} dB   "
          f"-40 dB echo {mid[win]-absent[win]:+6.1f} dB   "
          f"-55 dB echo {weak[win]-absent[win]:+6.1f} dB")

# This is the claim the over-the-air demo rests on. Uncancelled, every row of
# the sweep reads the same number whatever the echo, because what is being
# measured is the direct path's own structure and not the target at all.
check("uncancelled, a -40 dB co-delay target is indistinguishable from absent",
      abs(mid["blackmanharris"] - absent["blackmanharris"]) < 3.0,
      f"{mid['blackmanharris'] - absent['blackmanharris']:+.1f} dB")
check("cancelled, the same target stands well clear",
      mid[best] - absent[best] > 30.0,
      f"{mid[best] - absent[best]:+.1f} dB")
# 15 dB of echo should cost about 15 dB of detection SNR once the floor is the
# noise. If it costs much less, something other than the target is being read.
slope = mid[best] - weak[best]
close("15 dB of echo costs 15 dB of SNR after cancellation", slope, 15.0, 4.0,
      "dB")
check("the weakest echo still clears its floor by 10 dB",
      weak[best] > absent[best] + 10.0,
      f"{weak[best] - absent[best]:+.1f} dB")

print()
print("FAILURES:", n_fail)
sys.exit(1 if n_fail else 0)
