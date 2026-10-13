"""CLEAN deconvolution of the direct path's point spread function.

findings.md asks for this twice, in sections 4 and 11, and both times for the
same reason: CFAR fixes the false alarm rate against *noise* and says nothing
about deterministic structure. The direct path is deterministic structure. On a
filled aperture its two-dimensional point spread function still stands about
31 dB above the noise floor at the synthetic drone's delay, and it produced
4-5 dB detections at the edge of the zero-Doppler notch.

Over a cable that was the biggest remaining improvement. Over the air it is the
whole game. A real object a metre off a five-metre baseline has a bistatic
excess path of 0.39 m against a 66.6 m delay resolution cell, so it does not sit
hundreds of metres away from the direct path like the emulated drone does - it
sits *inside the same delay cell*, separated only in Doppler. What is left of
the direct path after cancellation is the noise floor of the entire OTA
measurement.

## Why this is exact rather than adaptive

The usual passive-radar canceller - ECA and its block variants - projects the
surveillance channel onto the orthogonal complement of a delay-Doppler span of
the reference, because the reference is measured and noisy. Here the aperture
W is *known exactly*: it is `occ_mask` or `ref_mask`, written by the generator
and shipped in the bundle. So a tap's contribution to Z is known in closed form
up to one complex scalar, and cancellation is a least-squares fit of that
scalar rather than a subspace projection. No degrees of freedom are spent, and
nothing is removed that a target could have occupied.

## The signal model

`sensing.form_z` builds Z = Y conj(X_est)/|X_est| W. A tap of complex amplitude
A at delay tau and Doppler f_d contributes

    Z[l, k] = A * exp(-2j pi k df tau) * exp(+2j pi f_d t_l) * W[l, k]

with df the subcarrier spacing and t_l the symbol's FFT window start. The signs
are those of `sensing.map_from_z`: its delay transform is an IFFT over k onto
an axis of m/(nk*df), and its Doppler transform is an FFT over l.

t_l is taken from `ofdm.symbol_layout`, not as l times a mean symbol period.
Symbols 0 and 7 of every slot carry four extra cyclic-prefix samples, so the
true sampling instants are not uniform, and the map's Doppler axis - which does
use the mean - is therefore a slight approximation. Fitting against the true
instants is what lets the subtraction reach the level the residual needs;
fitting against uniform ones leaves a floor at the jitter, about -43 dBc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ofdm, sensing


def symbol_times(cfg, n_sym):
    """FFT-window start time of every symbol, seconds, origin at the dwell."""
    cp, start = ofdm.symbol_layout(cfg, n_sym)
    return (start + cp).astype(np.float64) / float(cfg["fs_hz"])


def steering(cfg, n_sym, n_sc, tau_s, fd_hz, t_l=None):
    """The [n_sym, n_sc] structure a unit tap at (tau, f_d) writes into Z."""
    df = float(cfg["scs_khz"]) * 1e3
    k = np.arange(n_sc, dtype=np.float64)
    if t_l is None:
        t_l = symbol_times(cfg, n_sym)
    return (np.exp(-2j * np.pi * df * float(tau_s) * k)[None, :]
            * np.exp(2j * np.pi * float(fd_hz) * t_l)[:, None]).astype(np.complex64)


@dataclass
class Component:
    tau_s: float
    doppler_hz: float
    amplitude: complex
    power_db: float          # relative to the first (strongest) component

    def as_dict(self):
        return dict(tau_s=self.tau_s, delay_m=self.tau_s * 299792458.0,
                    doppler_hz=self.doppler_hz,
                    amp=abs(self.amplitude), power_db=self.power_db)


def _fit(Z, S, W):
    """Least-squares complex amplitude of structure S within Z, over W."""
    SW = S * W
    den = float(np.vdot(SW, SW).real)
    if den <= 0:
        return 0j, SW
    return complex(np.vdot(SW, Z) / den), SW


def _refine(Z, W, cfg, n_sym, n_sc, tau0, fd0, t_l, n_iter=3):
    """Local coordinate refinement of (tau, f_d) by maximising |<Z, S W>|^2/|SW|^2.

    The map peak is quantised to a pad-oversampled bin and parabolically
    interpolated on magnitude, which is good to a few percent of a cell. A few
    percent of a cell is 30 dB of residual, so the peak location is refined
    directly against Z before anything is subtracted.
    """
    df = float(cfg["scs_khz"]) * 1e3
    d_tau = 1.0 / (df * n_sc) * 0.25          # quarter of a delay cell
    d_fd = 1.0 / (t_l[-1] - t_l[0] + 1e-12) * 0.25

    def cost(tau, fd):
        A, SW = _fit(Z, steering(cfg, n_sym, n_sc, tau, fd, t_l), W)
        return abs(A) ** 2 * float(np.vdot(SW, SW).real)

    tau, fd, best = tau0, fd0, cost(tau0, fd0)
    for _ in range(n_iter):
        improved = False
        for dt in (-d_tau, d_tau):
            c = cost(tau + dt, fd)
            if c > best:
                tau, best, improved = tau + dt, c, True
                break
        for dfd in (-d_fd, d_fd):
            c = cost(tau, fd + dfd)
            if c > best:
                fd, best, improved = fd + dfd, c, True
                break
        if not improved:
            d_tau *= 0.4
            d_fd *= 0.4
    return tau, fd


def clean(Z, W, cfg, n_components=6, loop_gain=1.0, pad=2, window="hann",
          stop_db=40.0, search=None, search_doppler=None, refine=True):
    """Subtract the strongest taps from Z and return the residual.

    Iterates: map the residual, take its peak, refine that peak's (tau, f_d)
    against the residual directly, fit one complex amplitude by least squares,
    subtract `loop_gain` times it, repeat.

    `loop_gain` is 1.0 by default, which is a full subtraction rather than the
    0.1-0.2 of image-deconvolution CLEAN. The small gain there guards against a
    dirty beam that is only approximately known; here the aperture is exact, so
    the fitted amplitude is the right one and taking a fraction of it merely
    spends iterations.

    `stop_db` halts once the strongest remaining component is that far below
    the first, which stops the loop from spending its last iterations
    subtracting noise peaks - each of which removes a little real target energy.

    `search` and `search_doppler` restrict where components may be taken from,
    as (min, max) in seconds and in Hz. Both matter, for different reasons.

    The delay bound keeps CLEAN on the direct path and its near-in clutter.
    Leaving it open lets CLEAN eat a genuine target, which is the classic way
    to make a deconvolution look spectacular and mean nothing.

    The Doppler bound is what stops CLEAN locking onto the aperture's own comb.
    The map's Doppler axis runs to +-1/(2 t_sym) = +-7 kHz, but findings.md
    section 11 shows the weight W is periodic at the frame and slot rates, so
    every tap carries sidebands at multiples of 50 Hz and 1000 Hz. Unbounded,
    CLEAN's first three components came out at -4000 Hz, -1001 Hz and -0.55 Hz:
    it spent two of them subtracting slot-rate sidebands of the direct path
    before reaching the direct path itself. For cancelling the direct path a
    window of a few tens of hertz around zero is right, because the receiver
    has already de-ramped it to zero Doppler by construction.

    Returns (residual_Z, components, maps) with `maps` the MapResult before and
    after, on identical arithmetic, for a before/after display.
    """
    Z = np.asarray(Z, np.complex64).copy()
    W = np.asarray(W, np.float32)
    n_sym, n_sc = Z.shape
    w_sum = float(np.sum(W))
    t_l = symbol_times(cfg, n_sym)

    m_before = sensing.map_from_z(Z, w_sum, cfg, pad, window)
    comps, first_p = [], None

    for _ in range(int(n_components)):
        res = sensing.map_from_z(Z, w_sum, cfg, pad, window)
        Mm = res.M
        if search is not None:
            lo, hi = search
            keep = (res.delay_s >= lo) & (res.delay_s <= hi)
            if not keep.any():
                break
            Mm = np.where(keep[None, :], Mm, 0.0)
        if search_doppler is not None:
            lo, hi = search_doppler
            keepd = (res.doppler_hz >= lo) & (res.doppler_hz <= hi)
            if not keepd.any():
                break
            Mm = np.where(keepd[:, None], Mm, 0.0)
        i, j = np.unravel_index(np.argmax(Mm), Mm.shape)
        tau = float(sensing._parabolic(Mm[i], j, res.delay_s))
        fd = float(sensing._parabolic(Mm[:, j], i, res.doppler_hz))

        if refine:
            tau, fd = _refine(Z, W, cfg, n_sym, n_sc, tau, fd, t_l)

        S = steering(cfg, n_sym, n_sc, tau, fd, t_l)
        A, SW = _fit(Z, S, W)
        p = abs(A) ** 2 * float(np.vdot(SW, SW).real)
        if p <= 0:
            break
        if first_p is None:
            first_p = p
        rel_db = 10 * np.log10(max(p, 1e-30) / max(first_p, 1e-30))
        if comps and rel_db < -abs(stop_db):
            break

        Z -= (loop_gain * A) * SW
        comps.append(Component(tau_s=tau, doppler_hz=fd, amplitude=A,
                               power_db=rel_db))

    m_after = sensing.map_from_z(Z, w_sum, cfg, pad, window)
    return Z, comps, dict(before=m_before, after=m_after)


def cancellation_db(before, after, tau_s, doppler_hz, half_delay_cells=3,
                    half_doppler_cells=3):
    """How far the direct path came down, in a box around where it was.

    Reported rather than assumed. A canceller that looks like it worked because
    the whole map dropped has not cancelled anything, it has rescaled, so this
    is a local ratio and the noise floor either side is reported with it.
    """
    def box(res):
        dj = res.delay_s[1] - res.delay_s[0]
        di = res.doppler_hz[1] - res.doppler_hz[0]
        j = int(round(tau_s / dj))
        i = int(np.argmin(np.abs(res.doppler_hz - doppler_hz)))
        j0, j1 = max(j - half_delay_cells, 0), min(j + half_delay_cells + 1,
                                                   res.M.shape[1])
        i0, i1 = max(i - half_doppler_cells, 0), min(i + half_doppler_cells + 1,
                                                     res.M.shape[0])
        return float(res.M[i0:i1, j0:j1].max())

    b, a = box(before), box(after)
    return dict(before_db=20 * np.log10(max(b, 1e-30)),
                after_db=20 * np.log10(max(a, 1e-30)),
                cancelled_db=20 * np.log10(max(b, 1e-30) / max(a, 1e-30)),
                noise_before_db=20 * np.log10(max(before.noise, 1e-30)),
                noise_after_db=20 * np.log10(max(after.noise, 1e-30)))
