"""Time-varying multipath channel for an emulated LEO pass (handoff section 6).

Per path, per sample:

    y_p(t) = a_p(t) * x(t - tau_p(t)) * exp(-j*2*pi*f_c*tau_p(t))

summed over paths. Delay is the only thing injected. Doppler is not a
parameter anywhere in this file: it falls out of the derivative of the carrier
phase term, which is exactly the Doppler implied by the delay rate. Injecting
the two independently makes them disagree, and any receiver that jointly tracks
delay and Doppler then sees physically impossible data.

Three details that are easy to get wrong and expensive to debug:

* The sample counter is absolute and never resets. The waveform file loops
  every 20 ms; the pass does not.
* tau stays in float64 and the phase is reduced modulo one cycle *before*
  multiplying by 2*pi. f_c*tau is around 1e7 cycles here; doing this in float32
  raises the noise floor across the whole delay-Doppler map and looks exactly
  like a receiver bug.
* The fractional delay uses a cubic Farrow (Lagrange) interpolator, and a
  constant bias keeps the read pointer at least two samples behind the write
  pointer so the four-tap window is always causal.
"""

from __future__ import annotations

import numpy as np

from .trajectory import C, Path, Trajectory

# Shared by the channel and by the receiver's tracker: a constant delay bias
# that keeps the interpolator's read pointer causally behind the write pointer.
# Large enough for the default 24-tap sinc window. Both ends must agree, or the
# receiver's timing is offset from the channel's by the difference.
DEFAULT_TAU_BIAS = 16.0


class FracDelay:
    """Polyphase fractional-delay interpolator.

    Two kernels. 'lagrange' is the cubic Farrow of section 6: four taps, cheap,
    and exact at mu = 0. 'sinc' is a Kaiser-windowed sinc bank.

    The distinction matters more than it looks. Cubic Lagrange at mu = 0.5 has
    |H(f)| = |1.125 cos(pi f) - 0.125 cos(3 pi f)|, which is -1.96 dB at the
    0.293 cycles/sample band edge of this waveform. A static tap does not care -
    the droop is a fixed filter the channel estimate absorbs. But during a pass
    the delay walks ~170 samples/s, so mu sweeps a sawtooth at that rate and the
    droop turns into amplitude modulation, painting spurious Doppler sidebands
    on the direct path at multiples of the wrap rate. Those land near zero delay
    where a low-excess target sits.
    """

    def __init__(self, kind="sinc", n_taps=24, n_phase=1024, cutoff=0.45,
                 beta=8.0):
        self.kind = kind
        if kind == "lagrange":
            self.n_taps, self.k0 = 4, 1
            self.bank = None
            return
        if kind != "sinc":
            raise ValueError(f"unknown interpolator {kind!r}")
        if n_taps % 2:
            raise ValueError("n_taps must be even")
        self.n_taps = int(n_taps)
        self.n_phase = int(n_phase)
        self.k0 = self.n_taps // 2 - 1
        mu = np.arange(n_phase)[:, None] / n_phase
        t = np.arange(self.n_taps)[None, :]
        arg = mu + self.k0 - t
        h = 2 * cutoff * np.sinc(2 * cutoff * arg) * np.kaiser(self.n_taps, beta)[None, :]
        h /= h.sum(axis=1, keepdims=True)          # exact DC gain per phase
        self.bank = h.astype(np.float32)

    def response(self, f, mu):
        """|H| and the taps' frequency response at normalised frequency f."""
        h, k0 = self.taps(np.atleast_1d(mu)), self.k0
        t = np.arange(self.n_taps)[None, :] - k0
        f = np.atleast_1d(f)[:, None, None]
        return np.abs(np.sum(h[None, :, :] * np.exp(-2j * np.pi * f * t), axis=2))

    def taps(self, mu):
        mu = np.atleast_1d(mu)
        if self.kind == "lagrange":
            m1, m2, p1 = mu - 1.0, mu - 2.0, mu + 1.0
            return np.stack([(-mu * m1 * m2) / 6.0, (p1 * m1 * m2) * 0.5,
                             (-p1 * mu * m2) * 0.5, (p1 * mu * m1) / 6.0], axis=1)
        return self.bank[np.minimum((mu * self.n_phase).astype(np.int64),
                                    self.n_phase - 1)]

    def interp(self, ring, mask, i0, mu):
        """Gather-and-filter from a power-of-two ring buffer."""
        h = self.taps(mu)
        out = np.zeros(i0.size, np.complex64)
        for t in range(self.n_taps):
            out += h[:, t].astype(np.float32) * ring[(i0 + t - self.k0) & mask]
        return out


class LeoChannel:
    """Streaming channel. Feed it arbitrary chunk sizes; state persists."""

    def __init__(self, fs: float, fc: float, paths, r_nominal: float,
                 delay_line: int = 131072, tau_bias_samples: float = DEFAULT_TAU_BIAS,
                 t_start_s: float = 0.0, rate_scale: float = 1.0,
                 enable=None, gain_override_db=None, interp="sinc",
                 interp_taps=24):
        if delay_line & (delay_line - 1):
            raise ValueError("delay_line must be a power of two")
        if rate_scale != 1.0:
            # Section 6: compressing the trajectory makes the Doppler rate
            # unphysical and invalidates the NTN claim. Allowed for bench
            # debugging, never for a measurement.
            import warnings
            warnings.warn(
                f"rate_scale={rate_scale} makes the Doppler rate unphysical; "
                "results from this run are not valid NTN measurements",
                stacklevel=2)

        self.fs = float(fs)
        self.fc = float(fc)
        self.paths = list(paths)
        self.r_nominal = float(r_nominal)
        self.tau_bias = float(tau_bias_samples)
        self.t_start = float(t_start_s)
        self.rate_scale = float(rate_scale)
        self.enable = (np.ones(len(self.paths), bool) if enable is None
                       else np.asarray(enable, bool).copy())
        self.gain_override_db = ([None] * len(self.paths) if gain_override_db is None
                                 else list(gain_override_db))

        self.frac = FracDelay(interp, n_taps=interp_taps)
        if tau_bias_samples < self.frac.k0 + 2:
            raise ValueError(
                f"tau_bias_samples must be >= {self.frac.k0 + 2} for a causal "
                f"{self.frac.n_taps}-tap window")
        self.L = int(delay_line)
        self._ring = np.zeros(self.L, np.complex64)
        self._n = 0                      # absolute samples written, never resets
        self.max_delay_samples = 0.0

    # -- introspection -------------------------------------------------------
    @property
    def t_now(self) -> float:
        return self.t_start + self.rate_scale * self._n / self.fs

    def reset(self):
        self._ring[:] = 0
        self._n = 0
        self.max_delay_samples = 0.0

    def set_enable(self, pid: int, on: bool):
        self.enable[pid] = bool(on)

    def set_gain_db(self, pid: int, gain_db):
        """Override the trajectory's G column. None restores it."""
        self.gain_override_db[pid] = gain_db

    def tap_state(self, t: float = None):
        """Per-path (delay_samples, doppler_hz, amplitude) at time t. Truth for
        the dashboard and for scoring the receiver, computed from the same
        splines the channel itself uses."""
        t = self.t_now if t is None else t
        eps = 1e-3
        out = []
        for i, p in enumerate(self.paths):
            tt = np.array([t - eps, t, t + eps])
            R = p.range_at(tt)
            tau = (R - self.r_nominal) / C + self.tau_bias / self.fs
            fd = -self.fc / C * (R[2] - R[0]) / (2 * eps)
            a = p.gain_at(np.array([t]))[0]
            if self.gain_override_db[i] is not None:
                a = 10.0 ** (self.gain_override_db[i] / 20.0)
            out.append(dict(pid=p.pid, name=p.name, delay_samples=tau[1] * self.fs,
                            excess_m=R[1] - self.paths[0].range_at(np.array([t]))[0],
                            doppler_hz=fd, amplitude=float(a) * self.enable[i]))
        return out

    # -- the work ------------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Any length. Split internally so a caller can hand over a whole
        capture; the delay line and the absolute sample clock carry across."""
        x = np.ascontiguousarray(x, dtype=np.complex64)
        if x.size == 0:
            return x
        step = self.L // 2
        if x.size > step:
            return np.concatenate([self._chunk(x[i:i + step])
                                   for i in range(0, x.size, step)])
        return self._chunk(x)

    def _chunk(self, x: np.ndarray) -> np.ndarray:
        n = x.size
        L, mask = self.L, self.L - 1
        n0 = self._n
        abs_idx = n0 + np.arange(n, dtype=np.int64)
        self._ring[abs_idx & mask] = x

        t = self.t_start + self.rate_scale * (abs_idx / self.fs)
        y = np.zeros(n, np.complex64)

        for i, p in enumerate(self.paths):
            if not self.enable[i]:
                continue
            if self.gain_override_db[i] is None:
                a = p.gain_at(t)
                live = p.present(t)
            else:
                a = np.full(n, 10.0 ** (self.gain_override_db[i] / 20.0))
                live = np.ones(n, bool)
            if not live.any():
                continue

            tau = (p.range_at(t) - self.r_nominal) / C + self.tau_bias / self.fs
            d = tau * self.fs                                  # delay in samples
            if d.min() < self.frac.k0 + 2:
                raise ValueError(
                    f"path {p.pid}: delay {d.min():.2f} samples is not causal; "
                    "raise tau_bias_samples or lower r_nominal")
            if d.max() > L - self.frac.n_taps:
                raise ValueError(
                    f"path {p.pid}: delay {d.max():.0f} samples exceeds the "
                    f"{L}-sample delay line")
            self.max_delay_samples = max(self.max_delay_samples, float(d.max()))

            pos = abs_idx - d
            i0 = np.floor(pos).astype(np.int64)
            mu = pos - i0
            s = self.frac.interp(self._ring, mask, i0, mu)

            # Reduce f_c*tau modulo one cycle in float64 before scaling by
            # 2*pi, and keep the rotation in float64 too. Letting numpy take a
            # complex64 exp here costs ~1.6e-7 of phase, which is small but is
            # gratuitous divergence from the C++ block's sincos.
            ph = np.mod(self.fc * tau, 1.0) * (-2.0 * np.pi)
            s *= (a * (np.cos(ph) + 1j * np.sin(ph))).astype(np.complex64)
            if not live.all():
                s[~live] = 0
            y += s

        self._n = n0 + n
        return y


def from_trajectory(traj: Trajectory, fs: float, **kw) -> LeoChannel:
    return LeoChannel(fs=fs, fc=traj.fc, paths=traj.paths,
                      r_nominal=traj.r_nominal, **kw)


def manual(fs: float, fc: float, taps, **kw) -> LeoChannel:
    """Channel from a list of (delay_samples, doppler_hz, gain_db) tuples.

    Used by V1-V3. The delay is realised as a range offset and the Doppler as a
    range rate, so these taps go through exactly the same physics as a real
    pass - there is no separate 'test mode' code path to diverge.

    `d_samp` is the delay relative to the emulator's timing origin. On top of
    it sits the constant tau_bias that keeps the interpolator causal, so the
    tap lands at d_samp + tau_bias in absolute terms. A receiver removes the
    bias the same way it removes any bulk delay - integer retiming - and then
    sees the tap at exactly d_samp.
    """
    from .trajectory import linear_path
    bias = kw.get("tau_bias_samples", DEFAULT_TAU_BIAS)
    paths = []
    for k, (d_samp, fd_hz, g_db) in enumerate(taps):
        R0 = d_samp / fs * C
        paths.append(linear_path(k, f"tap{k}", R0, fd_hz, g_db, fc))
    return LeoChannel(fs=fs, fc=fc, paths=paths, r_nominal=0.0, **kw)
