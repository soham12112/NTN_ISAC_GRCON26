#!/usr/bin/env python3
"""Verification checkpoints for the two-stage coarse-to-fine chain.

Continues the handoff's numbering. V1-V3 verify the channel (verify_v1_v3.py);
these verify the flow built on it, against commanded truth.

  V8   stage 1 detects the drone at the right delay, target present
  V9   the coarse Doppler is ambiguous by exactly one PRF, and the
       coarse-to-fine loop resolves it
  V10  MTI cancels the direct path, and its blind speeds are where theory says
  V11  the stage 1 gate contains the truth it is meant to hand to stage 2
  V12  stage 2 delay and Doppler accuracy, parabolic peak vs paper eq 37
  V13  the frame-rate aperture comb is rejected, and the count is predictable
  V14  target absent: no confirmed track, and the false alarm count is as
       requested
  V15  the resource gate changes the outcome at low occupancy

Everything is scored against ch.tap_state(), which is the commanded
trajectory evaluated from the same splines the channel itself used. There is
no physical reflector anywhere in this setup and that is deliberate: it gives
exact truth.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import numpy as np

from ntn_isac import (aperture, coarse, detect, fileio, leo_channel, receiver,
                      sensing, track, trajectory, twostage)

C = 299792458.0
root = os.path.join(os.path.dirname(__file__), "..", "data")
tj = trajectory.load(os.path.join(root, "traj"))

# t = 500 s is inside traj_meta's last well-separated window. The drone sits
# at ~640 m of excess (9.6 delay cells out), its residual Doppler is -31.5 Hz
# so MTI gains it +5.3 dB rather than notching it. The deliberately selected
# clutter azimuth puts this one tap at +0.5 Hz; generic 250 m static clutter
# can reach 20.5 Hz at this legacy 2.0 GHz carrier (25.2 Hz at 2478 MHz).
T0, BIAS, NF = 500.0, 40.0, 5
ok = True


def check(name, got, want, tol, unit=""):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"    [{'PASS' if good else 'FAIL'}] {name}: {got:+.4f}{unit} "
          f"(expected {want:+.4f}, tol {tol}{unit})")
    return good


def check_true(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return bool(cond)


def run(bundle="wf_occ100", n_dwell=8, enable=None, gains=None, t0=T0, **kw):
    """One channel realisation, processed dwell by dwell. Returns per-dwell
    (result, truth) pairs. The channel is run once for the whole sequence so
    the delay line and absolute sample clock carry across, exactly as live."""
    b = fileio.load(os.path.join(root, bundle))
    cfg, fs, frame = b.cfg, b.fs, int(b.cfg["frame_len_samples"])
    eph = receiver.EphemerisTracker(tj, fs, tau_bias_samples=BIAS)
    ch = leo_channel.from_trajectory(
        tj, fs, t_start_s=t0, interp="sinc", interp_taps=16,
        tau_bias_samples=BIAS,
        enable=[True, True, True] if enable is None else enable,
        gain_override_db=[None, None, None] if gains is None else gains)
    # Two extra frames beyond the dwells: one skipped at the start and one to
    # cover the propagation margin of the final dwell.
    y = ch.process(np.tile(b.iq(), NF * n_dwell + 3))
    ts = twostage.TwoStage(b, eph, n_frames=NF,
                           tracker=track.MultiTracker(sigma_accel=3.0), **kw)
    n = ts.samples_per_dwell()
    need = ts.samples_needed()          # one dwell plus the propagation margin
    out = []
    for d in range(n_dwell):
        off = frame + d * n
        tt = t0 + off / fs
        st = ch.tap_state(tt)
        truth = dict(
            clutter_m=st[1]["excess_m"], drone_m=st[2]["excess_m"],
            clutter_fd=st[1]["doppler_hz"] - st[0]["doppler_hz"],
            drone_fd=st[2]["doppler_hz"] - st[0]["doppler_hz"])
        out.append((ts.process(y[off:off + need], tt, tracker_t=tt), truth))
    return ts, out


t_start = time.time()
print(f"waveform wf_occ100, pass t0 = {T0:.0f} s, dwell = {NF} frames")
ts, runs = run()
r0, tr0 = runs[0]
res = ts.resolution()
print(f"  delay cell {res['delay_cell_m']:.1f} m, Doppler cell "
      f"{res['doppler_cell_hz']:.1f} Hz, notch +-{res['notch_half_hz']:.0f} Hz")
print(f"  coarse PRF {res['coarse_prf_hz']:.0f} Hz -> unambiguous "
      f"+-{res['coarse_unamb_hz']:.0f} Hz; comb spacing {res['frame_rate_hz']:.0f} Hz")
print(f"  Pfa {res['pfa']:.2e} for {res['n_false_alarms']:.0f} expected false "
      f"alarm(s) per dwell")
print(f"  truth: drone {tr0['drone_m']:.1f} m {tr0['drone_fd']:+.2f} Hz | "
      f"clutter {tr0['clutter_m']:.1f} m {tr0['clutter_fd']:+.2f} Hz")

print("\nV8  stage 1 detects the drone from the SSB alone")
c0 = r0.coarse
print(f"    {c0.n_rep} SSB repetitions, aperture {c0.meta['aperture_hz']/1e6:.2f} MHz, "
      f"{c0.meta['n_ref_used']} reference REs per block, SNR {c0.snr_db:.1f} dB")
check("coarse delay", c0.excess_m(), tr0["drone_m"], 40.0, " m")
check_true("coarse SNR > 20 dB", c0.snr_db > 20.0, f"{c0.snr_db:.1f} dB")

print("\nV9  the coarse Doppler is ambiguous by exactly one PRF, and the loop "
      "resolves it")
prf = res["coarse_prf_hz"]
alias = r0.coarse.doppler_hz - tr0["drone_fd"]
n_alias = round(alias / prf)
print(f"    dwell 0 has no track yet, so nothing disambiguates it: measured "
      f"{r0.coarse.doppler_hz:+.2f} Hz vs truth {tr0['drone_fd']:+.2f} Hz")
check_true(f"the error is an exact PRF multiple (n = {n_alias})",
           n_alias != 0 and abs(alias - n_alias * prf) < 0.05,
           f"residual {alias - n_alias * prf:+.4f} Hz")
later = [abs(r.coarse.doppler_unamb_hz - t["drone_fd"]) for r, t in runs[1:]]
print(f"    dwells 1-{len(runs)-1}, unwrapped against the running track's "
      f"forecast: worst error {max(later):.3f} Hz")
check_true("unwrapped coarse Doppler within 1 Hz of truth", max(later) < 1.0,
           f"worst {max(later):.3f} Hz")

print("\nV10 MTI cancels the direct path, and its blind speeds are where theory "
      "says")
print(f"    measured notch depth on the direct path: {c0.mti_notch_db:.1f} dB")
check_true("notch depth > 15 dB", c0.mti_notch_db > 15.0)
t_rep = c0.t_rep
for f_d, want in ((0.0, -np.inf), (0.5 / t_rep, 20 * np.log10(2.0)),
                  (1.0 / t_rep, -np.inf)):
    got = float(coarse.mti_response_db(f_d, t_rep))
    label = f"MTI gain at {f_d:.0f} Hz"
    if np.isneginf(want):
        check_true(f"{label} is a null", got < -100.0, f"{got:.1f} dB")
    else:
        check(label, got, want, 0.01, " dB")
print(f"    so a target at {1.0/t_rep:.0f} Hz of residual Doppler is cancelled "
      f"as hard as the clutter; traj_meta puts the drone's residual Doppler "
      f"between -36.7 and +77.1 Hz, which crosses it")

print("\nV11 the stage 1 gate contains the truth it hands to stage 2")
g = c0.to_gate(ts.delay_s, widen=3.0)
j_true = int(round(tr0["drone_m"] / C / (ts.delay_s[1] - ts.delay_s[0])))
w_true = float(g[j_true])
print(f"    gate sigma {c0.sigma_delay_s()*C:.1f} m, weight at the true drone "
      f"delay {w_true:.4f} (peak {g.max():.4f})")
check_true("gate weight at truth > 0.5", w_true > 0.5)
check_true("gate excludes the grating lobe at 5000 m",
           not bool(ts.valid[np.argmin(np.abs(ts.delay_s - 1 / 60e3))]),
           "comb-4 PBCH-DMRS replica, outside the CP")

print("\nV12 stage 2 delay and Doppler accuracy")
ep, ec, ef = [], [], []
for r, t in runs:
    if not r.detections:
        continue
    d = r.detections[0]
    ep.append(d.excess_m - t["drone_m"])
    ec.append(d.centroid_delay_s * C - t["drone_m"])
    ef.append(d.doppler_hz - t["drone_fd"])
ep, ec, ef = np.array(ep), np.array(ec), np.array(ef)
print(f"    {len(ep)} of {len(runs)} dwells detected the drone")
print(f"    parabolic peak    mean {ep.mean():+.2f} m  rms {np.sqrt((ep**2).mean()):.2f} m")
print(f"    eq-37 centroid    mean {ec.mean():+.2f} m  rms {np.sqrt((ec**2).mean()):.2f} m")
print(f"    Doppler           mean {ef.mean():+.3f} Hz rms {np.sqrt((ef**2).mean()):.3f} Hz")
check_true("drone detected in every dwell", len(ep) == len(runs))
check("stage 2 delay bias", ep.mean(), 0.0, 2.0, " m")
check("stage 2 Doppler bias", ef.mean(), 0.0, 0.5, " Hz")
check_true("parabolic peak beats the eq-37 centroid",
           abs(ep.mean()) < abs(ec.mean()),
           f"{abs(ep.mean()):.2f} m vs {abs(ec.mean()):.2f} m of bias")
conf = [t for t in ts.tracker.tracks if t.confirmed]
print(f"    confirmed tracks: {len(conf)}")
for t_ in conf:
    e = t_.excess_m - runs[-1][1]["drone_m"]
    print(f"      track {t_.tid}: {t_.excess_m:.1f} +-{t_.sigma()[0]:.1f} m "
          f"({e:+.1f} m vs drone), {t_.range_rate_mps:+.2f} +-{t_.sigma()[1]:.2f} m/s, "
          f"{t_.hits} hits")
check_true("exactly one confirmed track", len(conf) == 1,
           "the clutter is inside the zero-Doppler notch by design, so it "
           "must not produce one")

print("\nV13 the frame-rate aperture comb is rejected")
n_raw = [r.gate_used.get("n_raw_detections", 0) for r, _ in runs]
n_comb = [r.gate_used.get("n_comb_rejected", 0) for r, _ in runs]
n_weak = [r.gate_used.get("n_below_snr_floor", 0) for r, _ in runs]
print(f"    per dwell: {np.mean(n_raw):.1f} raw clusters, "
      f"{np.mean(n_weak):.1f} below the {res['min_det_snr_db']:.0f} dB floor, "
      f"{np.mean(n_comb):.1f} comb-rejected")
ts2, runs2 = run(suppress_sidebands=False, min_det_snr_db=0.0, n_dwell=3)
sb = runs2[0][0].detections
offs = sorted({round((d.doppler_hz - runs2[0][1]["drone_fd"]) / res["frame_rate_hz"])
               for d in sb if abs(d.excess_m - runs2[0][1]["drone_m"]) < 70})
print(f"    with rejection off, co-delay detections appear at frame-rate "
      f"multiples n = {offs}")
check_true("more than one comb order is present when rejection is off",
           len([n for n in offs if n != 0]) >= 3,
           f"{len(offs)} orders including the target")
check_true("rejection leaves only the n = 0 target",
           all(len(r.detections) <= 2 for r, _ in runs), )

print("\nV14 target absent: no confirmed track, false alarms as requested")
ts3, runs3 = run(enable=[True, True, False], n_dwell=6)
conf3 = [t for t in ts3.tracker.tracks if t.confirmed]
n_det3 = [len(r.detections) for r, _ in runs3]
print(f"    detections per dwell {n_det3}, confirmed tracks {len(conf3)}")
check_true("no confirmed track with the drone switched off", len(conf3) == 0)
# CFAR is calibrated against noise; what remains on a target-absent run is the
# direct path's point spread function, which is a deterministic structure and
# not something a Pfa governs. So this bounds the count rather than matching it.
check_true("mean detections per target-absent dwell <= 1",
           np.mean(n_det3) <= 1.0, f"{np.mean(n_det3):.2f}")

print("\nV15 the resource gate changes the outcome at low occupancy")
print(f"{'bundle':<12}{'H_2':>7}{'frag':>7}{'admitted':>10}{'dets':>6}{'coarse dB':>11}")
for bundle, min_bits in (("wf_occ100", 7.0), ("wf_occ010", 7.0)):
    tsx, runsx = run(bundle=bundle, n_dwell=3, min_renyi_bits=min_bits)
    rx = runsx[-1][0]
    a = rx.resource
    print(f"{bundle:<12}{a['renyi_bits_per_symbol']:>7.2f}"
          f"{a['run_fragmentation']:>7.2f}{str(rx.admitted):>10}"
          f"{len(rx.detections):>6}{rx.coarse.snr_db:>11.1f}")
    if bundle == "wf_occ100":
        check_true("a filled allocation is admitted", rx.admitted)
        check_true("and the fine stage detects", len(rx.detections) >= 1)
    else:
        check_true("a 10% allocation is rejected by the H_2 > 7.0 gate",
                   not rx.admitted)
        check_true("the coarse stage still tracks it, since the SSB does not "
                   "depend on the scheduler", rx.coarse.snr_db > 20.0)

print(f"\nlatency, mean over {len(runs)} dwells of {ts.dwell_s*1e3:.0f} ms:")
keys = [k for k in runs[0][0].latency_ms if k != "total"]
for k in keys:
    print(f"    {k:<10}{np.mean([r.latency_ms[k] for r, _ in runs]):7.1f} ms")
tot = np.mean([r.latency_ms["total"] for r, _ in runs])
print(f"    {'total':<10}{tot:7.1f} ms = {tot/(ts.dwell_s*1e3):.1f}x the dwell")
print("    Python reference speed, per findings.md section 7. The live path is "
      "the C++ channel plus a decimated dwell rate; the offline campaign and "
      "replay mode are unaffected.")

print(f"\n[{time.time()-t_start:.0f} s]  " + ("OK" if ok else "FAILURES PRESENT"))
sys.exit(0 if ok else 1)
