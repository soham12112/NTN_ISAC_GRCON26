"""Micro-Doppler of a target that shares the direct path's delay cell.

The whole sensing chain up to here reports a target as a cell in a delay-Doppler
map, which is right when the target is the emulated drone 640 m out. It is
useless for a physical object between two radios on a bench: at 4.5 MHz of
occupied bandwidth the delay resolution is 66.6 m, and an object a metre off a
five-metre baseline has 0.39 m of bistatic excess path. It is six thousandths
of a cell from the direct path. There is no bandwidth this setup can reach that
would separate them - even 100 MHz leaves 3 m cells - so delay is not an
observable here and Doppler is the only one.

That is not a degraded mode, it is a different and well-posed measurement. A
rotating scatterer writes a signature that identifies it outright:

* its phase is sinusoidal with modulation index beta = 2 pi r / lambda, so the
  echo is a Bessel comb of lines at multiples of the rotation rate;
* the comb's envelope is largest just inside n = beta and falls off
  super-exponentially past it, putting a sharp edge at the tip-speed Doppler
  2 pi r f_rot / lambda;
* the line spacing *is* the rotation rate.

So two numbers come out - the spacing and the edge - and they give the rotation
rate and the tip radius separately. Both are checkable against a tachometer,
which is why a rotating target is worth more as a demonstration than a person
walking past: it carries its own ground truth, at a Doppler that clears the
zero-Doppler notch, and it repeats.

Two things have to be masked before any of that is readable.

The aperture comb. Slot 0 of every frame carries the SSB and no PDSCH, so the
sensing weight W changes once per 20 ms and every tap acquires a 50 Hz-spaced
Doppler comb about 9 dB down. The apparent half-rate line was resolved in
findings.md section 15 as residue piled up at a hard notch edge. Both land in
the band a rotating target occupies and are masked here.

The direct path itself. Its own structure sits tens of dB above a real target
at the same delay, and no windowing removes it - see clean.py. This module
expects to be handed a map that has already been through `clean.clean`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299792458.0


def frame_rate_hz(cfg) -> float:
    return float(cfg["fs_hz"]) / float(cfg["frame_len_samples"])


def comb_mask(doppler_hz, frame_hz, n_cells=2.0):
    """False on bins within n_cells of a multiple of half the frame rate.

    Half, not whole: a hard notch can pile clipped residue into the first
    surviving bin, measured at +25.00 Hz in findings.md section 15. Masking
    only whole frame-rate multiples leaves that deterministic residue to be
    read as a target.
    """
    fd = np.asarray(doppler_hz, float)
    cell = abs(fd[1] - fd[0]) if fd.size > 1 else 1.0
    step = frame_hz / 2.0
    return np.abs(fd - np.round(fd / step) * step) > n_cells * cell


def doppler_profile(res, ref_delay_s, half_cells=2):
    """Level against Doppler at one delay, as (doppler_hz, level).

    Takes the maximum over a few delay cells rather than one, because the
    target and the direct path share a cell and the peak interpolates between
    bins; picking a single column throws away up to a couple of dB for no gain.
    """
    j = int(np.argmin(np.abs(res.delay_s - ref_delay_s)))
    lo, hi = max(j - half_cells, 0), min(j + half_cells + 1, res.M.shape[1])
    return res.doppler_hz, res.M[:, lo:hi].max(axis=1)


@dataclass
class MicroDoppler:
    rotation_hz: float           # line spacing -> revolutions per second
    rotation_conf: float         # 0..1, the autocorrelation peak's prominence
    extent_hz: float             # tip-speed edge
    tip_speed_mps: float
    radius_m: float              # from extent and rotation_hz
    peak_hz: float               # strongest non-comb line
    peak_snr_db: float
    n_lines: int
    doppler_hz: np.ndarray = field(default=None, repr=False)
    level_db: np.ndarray = field(default=None, repr=False)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if k not in ("doppler_hz", "level_db")}


def analyse(res, cfg, ref_delay_s, noise_ref=None, fc=None,
            fd_limit_hz=500.0, notch_hz=15.0, snr_db=10.0,
            rot_search_hz=(2.0, 40.0), edge_db=12.0, min_occupancy=0.5,
            comb_cells=2.0) -> MicroDoppler:
    """Extract a rotating target's signature from a cancelled map.

    `fd_limit_hz` caps the search at half the slot rate. Beyond that W's own
    1000 Hz periodicity makes a sideband of an in-band tap indistinguishable
    from a tap (findings.md section 11), so a reading there is not evidence.

    `notch_hz` excludes the band CLEAN has emptied around zero Doppler. A
    rotating scatterer does pass through zero twice a revolution and that part
    of it is gone; the lines either side are what carry the information.

    `edge_db` sets how far below the band's peak a bin may be and still count
    as part of the signature, which is what fixes the tip-speed edge.

    Resolving the rotation rate needs the Doppler cell to be well inside the
    line spacing, so it needs a long dwell - much longer than the 100 ms the
    delay-Doppler stage uses. Measured on a 0.25 m blade: at 100 ms the cell is
    5 Hz against a 7 Hz spacing and the rate is unresolvable; at 1 s it comes
    out exactly, 7.00 for 7 and 11.00 for 11. The tip-speed extent needs no
    such thing and is good to about 15% from 100 ms. `rotation_hz` is NaN when
    no candidate clears `min_occupancy`, which is the honest report at a dwell
    too short to resolve it.
    """
    fd, lvl = doppler_profile(res, ref_delay_s)
    noise = float(noise_ref if noise_ref is not None else res.noise)
    db = 20 * np.log10(np.maximum(lvl, 1e-30) / max(noise, 1e-30))

    band = (comb_mask(fd, frame_rate_hz(cfg), comb_cells)
            & (np.abs(fd) > notch_hz) & (np.abs(fd) <= fd_limit_hz))
    if not band.any():
        return MicroDoppler(np.nan, 0.0, np.nan, np.nan, np.nan, np.nan,
                            -np.inf, 0, fd, db)

    live = band & (db > snr_db)
    n_lines = int(live.sum())
    ip = int(np.argmax(np.where(band, db, -np.inf)))
    peak_hz, peak_snr = float(fd[ip]), float(db[ip])

    # -- tip-speed edge ------------------------------------------------------
    # The outermost bin still within `edge_db` of the band's peak. A Bessel
    # comb's largest lines sit just inside n = beta and the envelope then falls
    # super-exponentially, so the edge is sharp and this finds it.
    #
    # Two other definitions were tried and both failed on a strong target. The
    # outermost bin over a fixed absolute threshold ran to the search cap,
    # because at -20 dB of echo the target perturbs the channel estimate enough
    # that the slicer's residual is broadband. Containing 99% of the band
    # energy chased the same tail. Referencing the peak makes the measurement
    # about the envelope's shape rather than about where the noise sits.
    edge = band & (db > peak_snr - edge_db) & (db > snr_db)
    extent = float(np.abs(fd[edge]).max()) if edge.any() else np.nan

    # -- line spacing --------------------------------------------------------
    # Harmonic template matching, not autocorrelation. The comb bins are masked
    # out, and an autocorrelation of a spectrum with a 25 Hz-periodic *mask*
    # reports 25 Hz whatever the target is doing - which is exactly what the
    # first version of this did, confidently, for a blade turning at 7.
    #
    # Instead, score each candidate rate by the mean level on the surviving
    # bins nearest its harmonics. Masked bins are skipped rather than counted
    # as zero, so the mask contributes nothing to the score.
    # Scoring a candidate by the mean level on its harmonics is degenerate, and
    # the degeneracy is the octave error familiar from pitch detection: twice
    # the true rate hits every second line, which are on average stronger than
    # the full set, so it wins. For a blade at 7 rev/s that version reported
    # 14.4, 27.8 and 27.9 at three dwell lengths - all harmonics, all confident.
    #
    # Two factors together break it. `cap` is the share of the band's energy
    # lying on the candidate's harmonics: the true rate takes nearly all of it,
    # twice the rate about half. `occ` is the share of the candidate's own
    # harmonics that carry anything: the true rate fills all of them, half the
    # rate fills every second one. Their product peaks at the fundamental.
    cell = float(abs(fd[1] - fd[0]))
    power = np.where(band, lvl ** 2, 0.0)
    total = float(power.sum())
    hi = extent if np.isfinite(extent) else fd_limit_hz
    # Lines are 1/dwell wide, so a candidate's harmonic is credited with the
    # best bin within one cell of it rather than the bin it rounds to.
    lo_f = max(rot_search_hz[0], 2.5 * cell)
    rot_hz, conf, best = np.nan, 0.0, -1.0
    if total > 0 and hi > 3 * lo_f:
        for f in np.arange(lo_f, rot_search_hz[1], cell / 4.0):
            h = np.arange(1, int(hi / f) + 1) * f
            h = h[h > notch_hz]
            if h.size < 4:
                continue
            idx = np.clip(np.round((h - fd[0]) / cell).astype(int), 1,
                          fd.size - 2)
            win = np.stack([power[idx - 1], power[idx], power[idx + 1]])
            ok = np.stack([band[idx - 1], band[idx], band[idx + 1]])
            vals = np.where(ok, win, 0.0).max(axis=0)
            cap = float(vals.sum()) / total
            occ = float(np.count_nonzero(
                vals > (10 ** (snr_db / 10.0)) * noise ** 2) / h.size)
            # A sub-harmonic still captures the energy - its harmonics are a
            # superset - but most of its own lines are empty. Requiring the
            # candidate's comb to be mostly filled rejects it. Without this
            # floor a 500 ms dwell on a 7 rev/s blade reported 3.00, which is
            # the bottom of the search range, with no indication anything had
            # gone wrong.
            if occ < min_occupancy:
                continue
            s = cap * occ
            if s > best:
                best, rot_hz, conf = s, float(f), float(min(s, 1.0))

    lam = C / float(fc if fc is not None else cfg["carrier_hz"])
    tip = extent * lam if np.isfinite(extent) else np.nan
    radius = (tip / (2 * np.pi * rot_hz)
              if np.isfinite(tip) and np.isfinite(rot_hz) and rot_hz > 0
              else np.nan)
    return MicroDoppler(rotation_hz=rot_hz, rotation_conf=conf,
                        extent_hz=extent, tip_speed_mps=tip, radius_m=radius,
                        peak_hz=peak_hz, peak_snr_db=peak_snr,
                        n_lines=n_lines, doppler_hz=fd, level_db=db)


def predicted_extent_hz(fc, radius_m, rot_hz):
    """2 pi r f_rot / lambda, the tip-speed Doppler."""
    return 2 * np.pi * radius_m * rot_hz * fc / C


def choose_rotation_hz(cfg, radius_m, fc, prefer=(4.0, 12.0), n=400):
    """Rotation rates whose harmonics avoid the aperture comb.

    The blade's lines fall on multiples of f_rot and the comb on multiples of
    half the frame rate, 25 Hz here. A rate that divides 25 puts every one of
    its harmonics on a masked bin and the target disappears into the mask. The
    score is the smallest distance from any blade harmonic inside the tip-speed
    edge to any comb line, so a high score is a rate whose whole signature
    stays clear.
    """
    step = frame_rate_hz(cfg) / 2.0
    out = []
    for f in np.linspace(prefer[0], prefer[1], n):
        edge = predicted_extent_hz(fc, radius_m, f)
        h = np.arange(1, max(int(edge / f), 1) + 1) * f
        d = np.abs(h - np.round(h / step) * step)
        out.append((float(d.min()), float(f), float(edge)))
    out.sort(reverse=True)
    return out
