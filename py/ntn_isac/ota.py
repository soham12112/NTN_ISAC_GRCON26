"""Over-the-air synchronisation: what the conducted setup never needed.

`receiver.EphemerisTracker` supplies tau0(t) by *commanding* the pass. That is
legitimate over a cable driven by one process, and its docstring promises that
"all a replacement has to provide is tau0(t)". This is that replacement.

Across two laptops the receiver does not know when the transmitter started, and
the two radios run off independent references. What it does know is the whole
transmitted waveform and the commanded trajectory, because both are files that
ship with the bundle. So this is not blind acquisition of an unknown signal; it
is estimation of two scalars on top of a known one.

## The clock model, and why it is two parameters rather than three

Let eps be the fractional frequency error of the transmitter's reference
against the receiver's. On a B200 the RF local oscillator and the sample clock
are both derived from the same TCXO, so eps sets both at once:

- the transmit LO sits at fc*(1 + eps), so the signal lands at +eps*fc of
  receive baseband;
- transmit samples are emitted at fs*(1 + eps), so the waveform measured on the
  receiver's clock is time-compressed and its apparent delay *decreases* at
  eps per second.

Write that as a delay and the carrier offset follows from it:

    tau_clk(t) = d0 - eps*t        =>   -fc * d(tau_clk)/dt = +eps*fc

which is the same "delay is the only thing injected, Doppler falls out of the
carrier phase" invariant that handoff section 6 imposes on the channel block.
Injecting a carrier offset independently of a timing offset would describe a
transmitter that cannot exist, and any receiver that jointly tracks the two
would see data that contradicts itself.

So the total direct-path delay is

    tau0(t) = tau_LEO(t) + d0 - eps*t

with tau_LEO known from traj.csv and (d0, eps) estimated here. Everything
downstream - deramp, per-symbol retiming, de-walk, demodulation - is unchanged,
which is the point.

## Magnitudes

At 2.478 GHz, with both GPSDOs locked, eps is around 1e-11: 0.025 Hz of carrier
offset and 8e-5 samples per second of drift, i.e. nothing. Free-running, two
B200 TCXOs at +-2 ppm give eps up to 4e-6, which is 9.9 kHz of carrier offset
and 31 samples per second of delay drift - the same order as the LEO pass's own
170 samples/s, and far too much to ignore.

That 9.9 kHz is also what forces a coarse acquisition step. The per-dwell
tracker below reads residual Doppler from the phase advance of the channel
estimate between OFDM symbols, which wraps beyond 1/(2*t_sym) = 7.0 kHz. Under
GPS lock the tracker can start cold; free-running it cannot, and `acquire`
exists for that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ofdm
from .leo_channel import DEFAULT_TAU_BIAS
from .trajectory import C


# --------------------------------------------------------------- acquisition


def _symbol_slices(cfg, n_sym_probe, n_sym_total):
    """Evenly spaced OFDM symbol windows to correlate against, as (start, len).

    Spread across the frame rather than taken consecutively so that the
    non-coherent sum averages over the whole aperture, including the SSB slot,
    instead of over one slot's PRB draw.
    """
    cp, start = ofdm.symbol_layout(cfg, n_sym_total)
    n_fft = int(cfg["n_fft"])
    idx = np.unique(np.linspace(0, n_sym_total - 1, n_sym_probe).astype(int))
    return [(int(start[l]), int(cp[l] + n_fft)) for l in idx]


@dataclass
class Acquisition:
    lag_samples: int
    cfo_hz: float
    peak_db: float            # correlation peak over the non-coherent floor
    locked: bool
    cfo_grid_hz: np.ndarray = field(default=None, repr=False)
    score: np.ndarray = field(default=None, repr=False)


def acquire(y, ref_iq, cfg, cfo_max_hz=12e3, cfo_step_hz=2.5e3,
            n_sym_probe=24, min_peak_db=12.0):
    """Coarse (lag, carrier offset) of a known reference frame inside `y`.

    Partial-coherent, non-coherently combined. Each probe symbol is correlated
    coherently over its own 71 us, which tolerates a carrier offset out to
    about 1/(2*t_sym) = 7 kHz, and the magnitudes are summed across probe
    symbols. That buys most of the processing gain of the whole frame while
    only needing a carrier grid as coarse as one symbol's tolerance. A
    full-frame coherent correlation would resolve 25 Hz instead and need a
    thousand-point grid to cover the same span.

    The reference is periodic - `loopable` is true for the MATLAB generator,
    and the transmit flowgraph plays it on repeat - so one frame of `y`
    contains one whole period at some rotation, the correlation is circular,
    and every lag in 0..frame_len-1 is searched at once. That periodicity is
    only approximate under the LEO channel, whose delay walks 3.4 samples per
    20 ms frame; the resulting smear is well inside the cyclic prefix that the
    lag only has to resolve to.

    Frame length is 153600 at 7.68 MS/s, so one FFT bin is exactly 50 Hz and a
    carrier shift by a multiple of 50 Hz is exactly a roll of the spectrum.
    The grid is snapped to that, which turns the whole search into one forward
    FFT of `y`, one per probe symbol, and an inverse per (symbol, hypothesis).

    Returns an `Acquisition`. `lag_samples` is the index into `y` at which the
    reference frame begins; `cfo_hz` is what must be *removed* from `y`.
    """
    fs = float(cfg["fs_hz"])
    n = int(cfg["frame_len_samples"])
    y = np.asarray(y, np.complex64)
    if y.size < n:
        raise ValueError(f"acquire needs a whole frame ({n} samples), got {y.size}")
    ref = np.asarray(ref_iq, np.complex64)
    if ref.size != n:
        raise ValueError("reference must be exactly one frame")

    df = fs / n                                     # 50.0 Hz
    k_max = int(round(cfo_max_hz / df))
    k_step = max(int(round(cfo_step_hz / df)), 1)
    bins = np.arange(-k_max, k_max + 1, k_step)
    grid = bins * df

    Yf = np.fft.fft(y[:n])
    Rf, offs = [], []
    for (s0, ln) in _symbol_slices(cfg, n_sym_probe, int(cfg["n_sym_total"])):
        r = np.zeros(n, np.complex64)
        r[:ln] = ref[s0:s0 + ln]
        Rf.append(np.conj(np.fft.fft(r)))
        offs.append(s0)

    best, score = None, np.zeros(bins.size)
    for gi, k0 in enumerate(bins):
        # Removing f = k0*df from y is Y[k] -> Y[k + k0].
        Ysh = np.roll(Yf, -int(k0))
        acc = np.zeros(n)
        for R, s0 in zip(Rf, offs):
            # Correlate against one symbol's window, then undo that symbol's
            # own offset so every symbol votes for the same lag: the frame
            # origin, not the symbol's position within it.
            acc += np.abs(np.roll(np.fft.ifft(Ysh * R), -s0)) ** 2
        score[gi] = acc.max()
        if best is None or acc.max() > best[0]:
            best = (acc.max(), int(np.argmax(acc)), acc)

    _, lag, acc = best
    floor = float(np.median(acc))
    peak_db = 10 * np.log10(max(acc.max(), 1e-30) / max(floor, 1e-30))

    # Interpolate the carrier offset off the grid. Worth doing: a 2.5 kHz grid
    # otherwise leaves up to 1.25 kHz of error, and the per-dwell tracker
    # below wraps beyond 7 kHz, so the margin is thinner than it looks once
    # the LEO Doppler rate has moved during the dwell.
    gi = int(np.argmax(score))
    cfo = float(grid[gi])
    if 0 < gi < score.size - 1:
        a, b, cc = score[gi - 1], score[gi], score[gi + 1]
        den = a - 2 * b + cc
        if den != 0:
            cfo += 0.5 * (a - cc) / den * (k_step * df)

    return Acquisition(lag_samples=int(lag), cfo_hz=float(cfo),
                       peak_db=float(peak_db),
                       locked=bool(peak_db >= min_peak_db),
                       cfo_grid_hz=grid, score=score)


# ------------------------------------------------------------- residual read


def residual_from_H(H, occ_mask, cfg):
    """Residual (delay, Doppler) of the dominant tap, read off a channel estimate.

    Both come out as a phase slope, and both are computed from an inner product
    of adjacent bins rather than by unwrapping, which needs no branch and is
    unbiased at low SNR:

      delay   the slope of arg H across subcarriers. A delay of `d` samples is
              a phase ramp exp(-j 2 pi k d / n_fft) across the grid, so
              arg <H[:, k], H[:, k+1]> = -2 pi d / n_fft.
      Doppler the slope of arg H across symbols, arg <H[l], H[l+1]> = 2 pi f t_sym.

    The direct path dominates H by more than 20 dB in every geometry this
    setup can produce, so these are its slopes and not a multipath average.
    Returned in samples and Hz, as corrections to *add* to the current model.
    """
    H = np.asarray(H)
    n_sym, n_sc = H.shape
    n_fft = int(cfg["n_fft"])
    m = np.asarray(occ_mask, bool)

    # Delay: adjacent-subcarrier product, over REs occupied in both bins.
    mk = m[:, :-1] & m[:, 1:]
    a = np.sum(np.conj(H[:, :-1][mk]) * H[:, 1:][mk])
    d_samples = -np.angle(a) * n_fft / (2 * np.pi) if a != 0 else 0.0

    # Doppler: adjacent-symbol product, over REs occupied in both symbols.
    cp = np.asarray(cfg["cp_lengths"], float)
    t_sym = (n_fft + cp.mean()) / float(cfg["fs_hz"])
    ml = m[:-1] & m[1:]
    b = np.sum(np.conj(H[:-1][ml]) * H[1:][ml])
    f_hz = np.angle(b) / (2 * np.pi * t_sym) if b != 0 else 0.0

    return float(d_samples), float(f_hz), float(np.abs(a)), float(np.abs(b))


# ------------------------------------------------------------------- tracker


class OtaTracker:
    """Drop-in for `EphemerisTracker`, with the clock difference estimated.

    Exposes exactly `tau0(t)` and `doppler(t)`, so `TwoStage`, `demod_tracked`
    and the sensing blocks take it without modification.

    The commanded LEO delay is feedforward; (d0, eps) are corrected once per
    dwell from `update()`. The loop is a two-state alpha-beta rather than a
    Kalman because the measurement noise here is dominated by how well the
    direct path stands above the multipath, which is not stationary and which
    a covariance would therefore misreport.
    """

    def __init__(self, traj, fs, tau_bias_samples=DEFAULT_TAU_BIAS, path=0,
                 d0_samples=0.0, eps=0.0, t_ref=0.0,
                 alpha=0.35, beta=0.05, gps_locked=False, n_pullin=4):
        self.traj = traj
        self.fs = float(fs)
        self.fc = traj.fc
        self.bias = float(tau_bias_samples) / self.fs
        self.p0 = traj.paths[path]
        self.r_nominal = traj.r_nominal

        self.d0 = float(d0_samples) / self.fs      # seconds
        self.eps = float(eps)                      # dimensionless
        self.t_ref = float(t_ref)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Under GPS lock eps is ~1e-11 and any drift the loop "measures" is
        # multipath, not clock. Holding it at zero is both more accurate and
        # what makes the delay axis comparable between captures.
        self.gps_locked = bool(gps_locked)
        # Acquisition leaves a carrier error of order the grid step divided by
        # the interpolator's accuracy - measured, about 70 Hz off a 2.5 kHz
        # grid. A residual carrier offset *is* a residual eps under the
        # single-reference model, so it can be removed outright rather than
        # integrated: the first `n_pullin` updates apply the full correction
        # and only then does the loop drop to `beta`. Left on beta = 0.05
        # throughout, a 70 Hz seed error decayed at 2.7 Hz per dwell and the
        # direct path sat tens of hertz off zero for the first second - long
        # enough that a canceller looking for it near zero Doppler found
        # nothing and reported 0.0 dB of cancellation.
        self.n_pullin = int(n_pullin)
        self.n_update = 0
        self.last = {}

    # -- the EphemerisTracker interface --------------------------------------

    def tau_leo(self, t):
        return (self.p0.range_at(np.asarray(t, float)) - self.r_nominal) / C + self.bias

    def tau_clk(self, t):
        return self.d0 - self.eps * (np.asarray(t, float) - self.t_ref)

    def tau0(self, t):
        return self.tau_leo(t) + self.tau_clk(t)

    def doppler(self, t, eps_t=1e-3):
        t = np.asarray(t, float)
        return -self.fc * (self.tau0(t + eps_t) - self.tau0(t - eps_t)) / (2 * eps_t)

    # -- estimation -----------------------------------------------------------

    def set_acquisition(self, acq: Acquisition, t0: float):
        """Seed the model from a coarse acquisition at dwell time `t0`.

        `acq.cfo_hz` is a carrier offset, and under the single-reference model
        above a carrier offset *is* a clock ratio: eps = cfo / fc. Seeding eps
        from it rather than leaving it at zero is what lets the per-dwell
        tracker start inside its 7 kHz wrap limit.
        """
        self.t_ref = float(t0)
        self.d0 = float(acq.lag_samples) / self.fs
        self.eps = 0.0 if self.gps_locked else float(acq.cfo_hz) / self.fc
        self.last = dict(acq_lag=acq.lag_samples, acq_cfo_hz=acq.cfo_hz,
                         acq_peak_db=acq.peak_db)

    def update(self, d_samples, f_hz, t, weight=1.0):
        """Fold one dwell's residual (delay, Doppler) into (d0, eps).

        `d_samples` and `f_hz` are what `residual_from_H` measured *after*
        demodulating with the current model, so they are errors to be driven
        to zero, not absolute quantities.
        """
        w = float(np.clip(weight, 0.0, 1.0))
        pull = self.n_update < self.n_pullin
        self.d0 += (1.0 if pull else self.alpha) * w * float(d_samples) / self.fs
        if not self.gps_locked:
            # A residual carrier offset is a residual clock ratio. The sign:
            # tau_clk = d0 - eps*t, so -fc*dtau/dt = +eps*fc, and a measured
            # +f_hz of leftover offset means eps is short by f_hz/fc.
            self.eps += (1.0 if pull else self.beta) * w * float(f_hz) / self.fc
        self.n_update += 1
        self.last.update(resid_delay_samples=float(d_samples),
                         resid_doppler_hz=float(f_hz),
                         d0_samples=float(self.d0 * self.fs),
                         eps=float(self.eps),
                         eps_ppm=float(self.eps * 1e6),
                         clk_doppler_hz=float(self.eps * self.fc),
                         clk_drift_samples_s=float(self.eps * self.fs),
                         t=float(t))
        return self.last

    def state(self):
        # `last` carries the most recent measurement and overlaps this on
        # purpose, so it is merged rather than splatted: dict(a=1, **{"a": 2})
        # is a TypeError, not a later-wins update, and it fired on every dwell.
        d = dict(self.last)
        d.update(d0_samples=float(self.d0 * self.fs), eps=float(self.eps),
                 eps_ppm=float(self.eps * 1e6),
                 clk_doppler_hz=float(self.eps * self.fc),
                 clk_drift_samples_s=float(self.eps * self.fs),
                 t_ref=self.t_ref, n_update=self.n_update,
                 gps_locked=self.gps_locked)
        return d
