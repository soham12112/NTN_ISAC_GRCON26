#!/usr/bin/env python3
"""Verification checkpoints V1-V3 (handoff section 9): channel block only,
digital loopback, no radios.

  V1  single static tap at exactly 3.5 samples, zero Doppler
  V2  same, with a fixed 100 Hz
  V3  two taps with a known amplitude ratio

The channel injects delay only; the Doppler in V2 is realised as a range rate,
so it comes out of the carrier phase exactly as section 6 requires.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import numpy as np

from ntn_isac import fileio, leo_channel, ofdm, sensing
from ntn_isac.leo_channel import DEFAULT_TAU_BIAS

# Full PRB occupancy on purpose. These checkpoints verify the CHANNEL, and a
# partially-filled grid is a thinned aperture whose sidelobe pedestal sits near
# -20 dBc - the same order as V3's -13 dB second tap. Randomising the PRB set
# per slot also decorrelates that pedestal slot to slot, which smears it across
# the whole Doppler axis instead of leaving it at the strong tap's Doppler. On a
# filled aperture the pedestal is -63 dBc and the measurement is clean.
WF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "wf_occ100")

b = fileio.load(WF)
cfg = b.cfg
fs, fc = b.fs, b.fc
tx = b.iq()
frame = tx.size
genie = np.asarray(b.grid())

TOL_DELAY = 0.05      # samples
TOL_DOPP = 2.0        # Hz
TOL_RATIO = 0.5       # dB
ok = True


def run(taps, variant=sensing.GENIE, n_frames=3, use=1):
    """Push `n_frames` looped copies through the channel, demodulate frame `use`."""
    ch = leo_channel.manual(fs, fc, taps)
    y = ch.process(np.tile(tx, n_frames))
    # The emulator's causal bias is a known integer bulk delay; retime it out
    # exactly as the tracker does for a real pass, leaving the taps at the
    # delays they were asked for.
    Y = ofdm.demodulate(y[use * frame:], cfg, sym_offset=int(DEFAULT_TAU_BIAS))
    X, W = sensing.build_variant(Y, b, variant, genie_grid=genie)
    return sensing.delay_doppler_map(Y, X, W, cfg, pad=2), Y


def check(name, got, want, tol, unit=""):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"    [{'PASS' if good else 'FAIL'}] {name}: {got:+.4f} {unit} "
          f"(expected {want:+.4f}, tol {tol}{unit})")


def local_peak(res, delay_samples, half=6):
    """Interpolated peak magnitude and location near a predicted delay.

    Parabolic interpolation on both axes: a tap that lands between bins reads
    up to ~1.4 dB low if you take the peak bin alone, which would fail the
    amplitude-ratio check on a channel that is in fact exact.
    """
    amp, d_s, fd = sensing.peak_near(res, delay_samples / fs, half_bins=half)
    return amp, d_s * fs, fd


print("V1  single static tap, 3.5 samples, 0 Hz")
r, _ = run([(3.5, 0.0, 0.0)])
print(f"    map {r.M.shape}, sum(W)={r.w_sum:.0f}, peak/noise {r.pk_nf_db:.1f} dB")
check("delay", r.peak_delay_s * fs, 3.5, TOL_DELAY, " samples")
check("doppler", r.peak_doppler_hz, 0.0, TOL_DOPP, " Hz")

print("V2  same tap, +100 Hz")
r, _ = run([(3.5, 100.0, 0.0)])
print(f"    peak/noise {r.pk_nf_db:.1f} dB")
check("delay", r.peak_delay_s * fs, 3.5, TOL_DELAY, " samples")
check("doppler", r.peak_doppler_hz, 100.0, TOL_DOPP, " Hz")

print("V3  two taps, 3.5 and 23.5 samples, -13 dB ratio, +0 / +250 Hz")
r, _ = run([(3.5, 0.0, 0.0), (23.5, 250.0, -13.0)])
a0, d0, f0 = local_peak(r, 3.5)
a1, d1, f1 = local_peak(r, 23.5)
print(f"    tap0 {a0:.4g} @ {d0:.3f} samples, {f0:+.1f} Hz")
print(f"    tap1 {a1:.4g} @ {d1:.3f} samples, {f1:+.1f} Hz")
check("tap0 delay", d0, 3.5, 0.10, " samples")
check("tap1 delay", d1, 23.5, 0.10, " samples")
check("tap0 doppler", f0, 0.0, 3.0, " Hz")
check("tap1 doppler", f1, 250.0, 3.0, " Hz")
check("amplitude ratio", 20 * np.log10(a1 / a0), -13.0, TOL_RATIO, " dB")

print("V3b same channel on the 50% random-PRB grid, where the thinned-aperture")
print("    pedestal is comparable to the weaker tap (expected to be degraded):")
_b, _g = b, genie
b = fileio.load(os.path.join(os.path.dirname(__file__), "..", "data", "wf_occ050"))
genie = np.asarray(b.grid()); tx = b.iq()
r2, _ = run([(3.5, 0.0, 0.0), (23.5, 250.0, -13.0)])
c0, _, _ = local_peak(r2, 3.5)
c1, e1, _ = local_peak(r2, 23.5)
print(f"    tap1/tap0 {20*np.log10(c1/c0):+.2f} dB at {e1:.3f} samples "
      f"(vs {-13.0:+.2f} dB at 23.500 on the filled grid)")
b, genie = _b, _g
tx = b.iq()

print("\nall three variants on the V3 channel:")
for v in sensing.VARIANTS:
    rv, _ = run([(3.5, 0.0, 0.0), (23.5, 250.0, -13.0)], variant=v)
    p0, _, _ = local_peak(rv, 3.5)
    p1, _, _ = local_peak(rv, 23.5)
    print(f"    {v:<15} sum(W)={rv.w_sum:6.0f}  peak/noise {rv.pk_nf_db:5.1f} dB  "
          f"tap1/tap0 {20*np.log10(p1/p0):+.2f} dB")

print("\nOK" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
