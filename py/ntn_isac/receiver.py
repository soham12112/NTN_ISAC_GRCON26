"""Tracking, de-ramp, de-walk and OFDM demodulation (section 7, steps 3-4).

Step 3 must happen before step 4, for two reasons the handoff gives and one it
does not.

Doppler: at 2 GHz a 550 km pass reaches ~550 Hz/s of Doppler rate, which keeps
smear inside one Doppler bin only for coherent integration under roughly 30 ms.

Cyclic prefix: symbols 0 and 7 carry four extra CP samples, so symbol starts are
not perfectly uniform. After de-ramp the residual Doppler makes that negligible;
before de-ramp it does not.

Delay walk - not in section 7, and it bites hardest: the radial rate reaches
~6.6 km/s, which is 170 samples/s at 7.68 MS/s. The delay resolution is
1/4.5 MHz = 1.71 samples, so a 20 ms CPI walks 3.4 samples = 2 resolution cells
and a 100 ms CPI walks 17. Left alone this smears the delay axis and the 100 ms
CPI of section 7 cannot work. Correcting it is nearly free: a delay is a linear
phase across subcarriers, so the residual after integer retiming comes off in
the frequency domain.
"""

from __future__ import annotations

import numpy as np

from . import ofdm
from .leo_channel import DEFAULT_TAU_BIAS
from .trajectory import C


class EphemerisTracker:
    """Feedforward from the commanded trajectory, plus a residual estimate.

    We command the pass, so the bulk delay and Doppler are known rather than
    acquired. That unblocks the sensing branches without waiting on blind
    PSS/SSS acquisition, which slots in later behind the same interface: all a
    replacement has to provide is tau0(t).
    """

    def __init__(self, traj, fs: float, tau_bias_samples: float = DEFAULT_TAU_BIAS,
                 path: int = 0):
        self.traj = traj
        self.fs = float(fs)
        self.fc = traj.fc
        self.bias = float(tau_bias_samples) / self.fs
        self.p0 = traj.paths[path]
        self.r_nominal = traj.r_nominal

    def tau0(self, t):
        """Direct-path delay in seconds, on the same clock the channel uses."""
        return (self.p0.range_at(np.asarray(t, float)) - self.r_nominal) / C + self.bias

    def doppler(self, t, eps: float = 1e-3):
        t = np.asarray(t, float)
        return -self.fc * (self.tau0(t + eps) - self.tau0(t - eps)) / (2 * eps)


def deramp(y: np.ndarray, t: np.ndarray, tau0: np.ndarray, fc: float) -> np.ndarray:
    """Remove the direct path's carrier phase exp(-j 2 pi fc tau0(t)).

    The fractional part of fc*tau0 is taken in float64 before scaling by 2 pi,
    for the same reason the channel does it: fc*tau0 is ~1e7 cycles here and
    float32 truncation raises the floor across the whole map.
    """
    ph = np.mod(fc * np.asarray(tau0, float), 1.0)
    return (y * np.exp(np.complex64(2j * np.pi) * ph.astype(np.float32))).astype(np.complex64)


def demod_tracked(y: np.ndarray, cfg: dict, tracker: EphemerisTracker,
                  t0: float, n_sym: int = None, allow_short: bool = False):
    """De-ramp, retime and demodulate one CPI.

    `t0` is the transmit-side time of the first sample of `y`. Returns the grid
    plus the tracking state that produced it.

    `y` must be longer than the CPI. Symbol s of the CPI is read at receive
    index s + tau0*fs, and tau0 is the *bulk* delay - 23664 samples at
    t = 500 s of this pass, 40386 at its start - so a slice exactly one CPI
    long cannot supply the last symbols. Use TwoStage.margin_samples() to size
    it. `allow_short` restores the old truncating behaviour for callers that
    genuinely want a partial grid.
    """
    fs = float(cfg["fs_hz"])
    fc = float(cfg["carrier_hz"])
    n_fft = int(cfg["n_fft"])
    n_sc = int(cfg["n_sc"])
    if n_sym is None:
        n_sym = int(cfg["n_sym_total"])

    t = t0 + np.arange(y.size) / fs
    y = deramp(y, t, tracker.tau0(t), fc)

    # Per-symbol delay, split into an integer retime and a residual.
    #
    # The channel evaluates tau at the ARRIVAL time: y(t) = x(t - tau(t)). So
    # the sample carrying transmit index s is at receive index n solving
    # n = s + tau(t0 + n/fs)*fs, not n = s + tau(t0 + s/fs)*fs. The difference
    # is tau itself - up to 5 ms here - and at 170 samples/s of walk that is
    # most of a sample of timing error, which shows up as a delay-axis bias.
    # Two fixed-point passes converge: the map |n -> tau| is a contraction with
    # slope |dtau/dt| ~ 2e-5.
    cp, start = ofdm.symbol_layout(cfg, n_sym)
    s_sym = (start[:n_sym] + cp[:n_sym]).astype(float)
    d_sym = tracker.tau0(t0 + s_sym / fs) * fs
    for _ in range(2):
        d_sym = tracker.tau0(t0 + (s_sym + d_sym) / fs) * fs
    off = np.round(d_sym).astype(np.int64)
    resid = d_sym - off                              # samples, |.| <= 0.5

    Y = ofdm.demodulate(y, cfg, sym_offset=off, n_sym=n_sym,
                        allow_short=allow_short)

    # De-walk: undo the residual delay as a linear phase across subcarriers.
    k = (np.arange(n_sc) - n_sc // 2).astype(np.float32)
    Y *= np.exp(np.complex64(2j * np.pi) *
                (k[None, :] * resid[:, None].astype(np.float32) / n_fft))

    return Y.astype(np.complex64), dict(
        n_sym=int(n_sym), n_sym_filled=int(getattr(ofdm.demodulate, "n_filled",
                                                   n_sym)),
        t0=t0, tau0_s=float(tracker.tau0(np.array([t0]))[0]),
        doppler_hz=float(tracker.doppler(np.array([t0]))[0]),
        delay_walk_samples=float(d_sym[-1] - d_sym[0]),
        resid_max_samples=float(np.abs(resid).max()),
        sym_offset_span=int(off[-1] - off[0]))
