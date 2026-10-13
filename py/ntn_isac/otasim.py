"""A simulated two-laptop over-the-air link, for testing the OTA chain dry.

Everything here models something the conducted setup does not have: an
independent receive clock, and a physical object between the radios whose echo
arrives in the same delay cell as the direct path.

## The clock offset needs no new channel code

A B210 derives its RF local oscillator and its sample clock from one TCXO, so a
single fractional error eps sets both: the carrier lands at +eps*fc and the
waveform's apparent delay drifts at -eps per second. That is exactly the
"delay is the only thing injected, Doppler falls out of the carrier phase"
invariant of handoff section 6, so a receive-clock offset is physically
indistinguishable from a single Doppler-shifted path and `leo_channel.manual`
already implements it. Cascading one stage after the emulated pass is the whole
simulation - and the identity is also why `ota.OtaTracker` estimates one eps
rather than a carrier offset and a timing drift separately.

## Why the target is a rotating blade

A real object a metre off a five-metre baseline has a bistatic excess path of
0.39 m against a 66.6 m delay resolution cell. It is 0.006 of a cell from the
direct path and no bandwidth this setup can reach will separate them, so the
only observable is Doppler. At 2.478 GHz a person walking reaches about 21 Hz
of excess-range rate, which the zero-Doppler notch already blinds; a 0.25 m
blade at 7 rev/s reaches 91 Hz, which it does not. See docs/ota.md.
"""

from __future__ import annotations

import numpy as np

from . import fileio, leo_channel, receiver, sensing, trajectory
from .trajectory import C


def blade_path(pid, t_end, excess_m=0.4, radius_m=0.25, rot_hz=7.0,
               gain_db=-25.0, rate_hz=8000.0, name="blade"):
    """A rotating scatterer, as a range history.

    A point at `radius_m` on a blade turning at `rot_hz` swings the bistatic
    excess path by +-radius about `excess_m`. Expressed as a range history for
    the same reason every other tap in this codebase is: injecting a Doppler
    directly would let delay and Doppler disagree.

    Peak excess-range rate is 2 pi radius rot_hz, so the Doppler reaches
    2 pi radius rot_hz / lambda. The *spectrum* is not a line there: the phase
    is sinusoidal with modulation index beta = 2 pi radius / lambda, so the
    echo is a Bessel comb at multiples of rot_hz whose envelope is largest just
    inside n = beta. At 0.25 m and 12.1 cm, beta is 13.0, so the strongest
    lines sit near 11-12 times rot_hz rather than at the 13th.
    """
    t = np.arange(0.0, t_end + 2.0 / rate_hz, 1.0 / rate_hz)
    R = excess_m + radius_m * np.cos(2 * np.pi * rot_hz * t)
    return trajectory.Path(pid=pid, name=name, t=t, R=R,
                           G_db=np.full(t.size, float(gain_db)))


def blade_doppler_hz(fc, radius_m=0.25, rot_hz=7.0):
    """Peak micro-Doppler of a blade tip, Hz."""
    return 2 * np.pi * radius_m * rot_hz / (C / fc)


def simulate(wf_dir, traj_dir, seconds=0.25, t_start=300.0, eps_ppm=0.0,
             d0_samples=0.0, blade=None, esn0_db=40.0, seed=7, tau_bias=40.0,
             interp_taps=16, disable=()):
    """One over-the-air capture. Returns (y, bundle, traj, cfg, truth).

    Three cascaded stages, in the order the real link applies them:

      1. the emulated LEO pass, applied digitally before the DAC exactly as
         gr/flowgraphs/tx_usrp.py does it;
      2. the physical link - a direct ray plus, optionally, the rotating
         target, both at essentially the same delay;
      3. the receive clock.

    `disable` switches off emulated paths by index, as the GRC block's
    set_enable does. Worth doing deliberately when measuring the *physical*
    target: at t = 300 s the emulated drone is only a couple of delay cells
    from the direct path with about 20 Hz of residual Doppler, so it lands in
    the same band as a rotating blade and will be mistaken for it. For a demo
    the opposite is wanted - a synthetic target at 640 m with exact truth and a
    real one at 0 m, in the same map - but a measurement of one has to exclude
    the other.

    Each stage carries its own constant tau_bias to keep its interpolator
    causal. That is not physics, and a real receiver sees the same thing as a
    fixed delay through cable, filters and pipeline. Handoff section 9's V4
    says to record it as a calibration constant rather than pretend it is zero,
    so the total is returned as `truth["bias_samples"]` and callers hand it to
    the tracker as exactly that.
    """
    b = fileio.load(wf_dir)
    cfg = b.cfg
    fs, fc = b.fs, b.fc
    tr = trajectory.load(traj_dir)
    n = int(seconds * fs)

    tx = b.iq()
    x = np.tile(tx, int(np.ceil(n / tx.size)) + 1)[:n]
    ch = leo_channel.from_trajectory(tr, fs, t_start_s=t_start,
                                     tau_bias_samples=tau_bias,
                                     interp="sinc", interp_taps=interp_taps)
    for p in disable:
        ch.set_enable(int(p), False)
    y = ch.process(x)

    if blade is not None:
        paths = [trajectory.linear_path(0, "direct", 0.0, 0.0, 0.0, fc),
                 blade_path(1, seconds + 0.1, **blade)]
        y = leo_channel.LeoChannel(fs=fs, fc=fc, paths=paths, r_nominal=0.0,
                                   tau_bias_samples=tau_bias, interp="sinc",
                                   interp_taps=interp_taps).process(y)

    eps = eps_ppm * 1e-6
    clk_bias = 0.0
    if eps != 0.0 or d0_samples != 0.0:
        # The clock stage's delay *decreases* at eps per second, so its bias
        # has to cover the whole capture or the interpolator's read pointer
        # overtakes the write pointer partway through. At 2.5 ppm over 4.5 s
        # that is 86 samples, against a default bias of 40, and the channel
        # correctly refuses. This is an artefact of simulating the clock as a
        # delay line; a real receiver has no such limit.
        clk_bias = float(tau_bias + max(0.0, eps * seconds * fs) + 8.0)
        y = leo_channel.manual(fs, fc, [(d0_samples, eps * fc, 0.0)],
                               tau_bias_samples=clk_bias, interp="sinc",
                               interp_taps=interp_taps).process(y)

    if esn0_db is not None:
        rng = np.random.default_rng(seed)
        per_sym = b.occ_mask.sum(axis=1)
        n_occ = int(np.median(per_sym[per_sym > 0]))
        sig = float(np.mean(np.abs(y) ** 2))
        n0 = sig * int(cfg["n_fft"]) / n_occ / (10 ** (esn0_db / 10.0))
        y = (y + np.sqrt(n0 / 2) * (rng.standard_normal(y.size)
                                    + 1j * rng.standard_normal(y.size))
             ).astype(np.complex64)

    # The constant part of the total delay, at t = t_start. The eps term is not
    # in here: it is time-varying and it is what OtaTracker is for. d0_samples
    # is also left out, so a tracker seeded with this converges its own d0 to
    # d0_samples, which is what V17 checks.
    n_geo = 1 + (blade is not None)
    truth = dict(eps=eps, d0_samples=d0_samples, cfo_hz=eps * fc,
                 t_start=t_start, tau_bias=tau_bias, clk_bias=clk_bias,
                 n_stage=n_geo + (1 if clk_bias else 0),
                 disabled=list(disable),
                 bias_samples=n_geo * tau_bias + clk_bias)
    if blade is not None:
        truth["blade"] = dict(blade)
        truth["blade_doppler_hz"] = blade_doppler_hz(
            fc, blade.get("radius_m", 0.25), blade.get("rot_hz", 7.0))
    return y, b, tr, cfg, truth


def demod_dwell(y, b, tracker, t0, n_frames=5):
    """Demodulate one dwell. Returns (Y, H, tiled_bundle, state)."""
    cfg = b.cfg
    n_sym = n_frames * int(cfg["n_sym_total"])
    tb = fileio.TiledBundle(b, n_frames)
    Y, st = receiver.demod_tracked(y, cfg, tracker, t0, n_sym=n_sym)
    H = sensing.ls_channel_estimate(Y, tb.ref_mask, tb.ref_grid(), tb.occ_mask)
    return Y, H, tb, st
