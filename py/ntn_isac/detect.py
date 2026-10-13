"""Stage 2 detection: CA-CFAR then density clustering (paper section IV-B).

Paper eq 34-35 threshold a range-azimuth map with cell-averaging CFAR and eq
36-37 group the surviving cells with DBSCAN. Two adaptations, both forced by
having one receive antenna:

The map is delay-Doppler, not range-azimuth. With a single B200 channel there
is no spatial aperture, so the second axis is Doppler. Everything else in the
paper's detection stage is axis-agnostic: cell averaging, guard cells and
density clustering do not care what the axes mean, only that neighbouring
cells are correlated, which is true of a delay-Doppler map for the same
windowing reason it is true of a range-azimuth one.

The clustering distance is in *cells*, not metres and degrees. Eq 36 mixes
range in metres with angle in degrees under one radius, which only works
because the two happen to be numerically comparable in that scenario. A delay
cell here is 66.6 m and a Doppler cell is a few hertz, so an unnormalised
radius would be meaningless. Distances are therefore taken in units of the
map's own resolution cells, which is scale-free and makes eps interpretable.

No sklearn in this environment, so DBSCAN is implemented here. It is the real
algorithm - core points, density reachability, noise labels - not a
connected-components stand-in, because the two differ exactly where it
matters: a sidelobe ridge is connected but not dense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cfar_alpha(n_train: int, pfa: float) -> float:
    """Scale factor for square-law CA-CFAR: alpha = N (Pfa^(-1/N) - 1).

    Exact for exponentially distributed cells, which is what |.|^2 of a
    complex Gaussian map is. Note this is the *finite-N* form: the N -> inf
    limit -ln(Pfa) under-thresholds badly at the training-cell counts a small
    map affords, and would leak false alarms at several times the requested
    rate.
    """
    n = max(int(n_train), 1)
    return float(n * (float(pfa) ** (-1.0 / n) - 1.0))


def pfa_for_count(n_cells: int, n_expected: float = 1.0, pad: int = 1) -> float:
    """Per-cell Pfa giving `n_expected` false alarms over a map of `n_cells`.

    The paper quotes Pfa = 1e-3, which is the right *per-cell* number for the
    range-azimuth map it thresholds. This chain's delay-Doppler map is 2800 by
    600 display bins, so 1e-3 budgets 1680 false alarms per dwell - measured,
    605 clusters survived, the tracker then spawned a track for each of them
    and the dwell took four seconds. Per-cell Pfa is not portable between maps
    of different size; the expected count is.

    `pad` is the map's oversampling factor per axis. Neighbouring bins of an
    oversampled map are correlated and do not each get an independent throw,
    so the effective cell count is divided by pad^2.
    """
    n = max(int(n_cells) // max(int(pad) ** 2, 1), 1)
    return float(min(max(float(n_expected) / n, 1e-15), 0.5))


def _vertex(v: np.ndarray, i: int) -> float:
    """Sub-bin offset of a peak from a three-point parabolic fit, in bins."""
    if i <= 0 or i >= v.size - 1:
        return 0.0
    a, b, c = float(v[i - 1]), float(v[i]), float(v[i + 1])
    den = a - 2 * b + c
    return 0.5 * (a - c) / den if den != 0 else 0.0


def _integral(P: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row and column, for O(1) box sums."""
    S = np.zeros((P.shape[0] + 1, P.shape[1] + 1), np.float64)
    np.cumsum(np.cumsum(P.astype(np.float64), axis=0), axis=1, out=S[1:, 1:])
    return S


def _box_sum(S: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    """Inclusive box sums from a summed-area table, on clipped index arrays."""
    return (S[r1 + 1, c1 + 1] - S[r0, c1 + 1] - S[r1 + 1, c0] + S[r0, c0])


def doppler_notch(M: np.ndarray, doppler_hz: np.ndarray, half_hz: float):
    """Null the zero-Doppler component. Returns (M_notched, kept_rows).

    Paper section IV-B: "Static clutter is suppressed by nulling the
    zero-Doppler component, followed by matched filtering". Applied here in
    the Doppler domain after the slow-time FFT, and *not* as an
    adjacent-symbol difference along the symbol axis, which is what the phrase
    "MTI filtering" would suggest by analogy with the coarse stage.

    The difference is not stylistic. Adjacent-symbol MTI is a high-pass whose
    first null sits at the symbol rate: T_sym = 71.35 us here, so a PRF of
    14 kHz and a response of 2|sin(pi f_d T_sym)|. The residual Dopplers this
    scenario contains are tens of hertz - the drone reaches 77 Hz - which puts
    every target deep in the notch: -37 dB at 30 Hz. It would cancel the
    targets along with the clutter.

    A Doppler-domain notch is sized against the dwell's own Doppler
    resolution instead. Over a 100 ms dwell that is 10 Hz per cell, so a
    +-1 cell notch discards excess range rates within about +-1.5 m/s of the
    direct path's - which is what "static clutter" means here.
    """
    d = np.asarray(doppler_hz, float)
    kept = np.abs(d) > float(half_hz)
    out = np.array(M, copy=True)
    out[~kept, :] = 0.0
    return out, kept


def map_noise_power(M: np.ndarray, kept: np.ndarray = None,
                    valid: np.ndarray = None) -> float:
    """Noise power from the median of a magnitude map, excluding nulled cells.

    Same median-of-an-exponential convention as sensing.delay_doppler_map, so
    the two stages' SNR figures are directly comparable. Notched rows must be
    excluded or the median is dragged toward zero and every remaining cell
    looks like a detection.
    """
    sub = M if kept is None else M[kept, :]
    if valid is not None:
        sub = sub[:, valid]
    if sub.size == 0:
        return 0.0
    return float(np.median(sub.astype(np.float64) ** 2) / np.log(2.0))


def ca_cfar_2d(P: np.ndarray, guard=(2, 2), train=(4, 8), pfa: float = 1e-3):
    """Cell-averaging CFAR over a 2-D power map. Returns (mask, threshold).

    `guard` and `train` are half-widths in (row, column) = (Doppler, delay).
    Guard cells exist because a real target spreads over several cells: without
    them a strong target's own skirt inflates its local threshold and censors
    it. The training region is the ring between guard and train.

    The training count is computed per cell from the *clipped* window rather
    than assumed constant, and alpha is derived from that actual count. At the
    map edges a fixed alpha would be applied to a third of the intended
    training cells, which raises the local false alarm rate by more than an
    order of magnitude - precisely at the zero-Doppler and zero-delay edges
    where the direct path's residue lives.
    """
    P = np.asarray(P, np.float64)
    n_r, n_c = P.shape
    gr, gc = int(guard[0]), int(guard[1])
    tr, tc = int(train[0]), int(train[1])
    if tr < gr or tc < gc:
        raise ValueError("train half-width must be >= guard half-width")

    S = _integral(P)
    rows = np.arange(n_r)[:, None]
    cols = np.arange(n_c)[None, :]

    def window(hr, hc):
        r0 = np.clip(rows - hr, 0, n_r - 1)
        r1 = np.clip(rows + hr, 0, n_r - 1)
        c0 = np.clip(cols - hc, 0, n_c - 1)
        c1 = np.clip(cols + hc, 0, n_c - 1)
        n = (r1 - r0 + 1) * (c1 - c0 + 1)
        return _box_sum(S, r0, r1, c0, c1), n

    s_out, n_out = window(tr, tc)
    s_in, n_in = window(gr, gc)
    n_train = n_out - n_in
    total = np.maximum(s_out - s_in, 0.0)
    # Cells with no training region at all (a degenerate tiny map) get an
    # infinite threshold rather than a division by zero, so they never detect.
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n_train > 0, total / np.maximum(n_train, 1), np.inf)
    alpha = np.where(n_train > 0,
                     n_train * (float(pfa) ** (-1.0 / np.maximum(n_train, 1)) - 1.0),
                     np.inf)
    thr = alpha * mean
    return P > thr, thr


def dbscan(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN labels; -1 is noise. Paper eq 36's neighbour rule, O(n^2).

    n is the number of CFAR detections - tens, occasionally hundreds - so the
    brute-force neighbour query is the right implementation. A spatial index
    would cost more to build than it saves.
    """
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    n = pts.shape[0]
    labels = np.full(n, -1, np.int64)
    if n == 0:
        return labels

    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    neigh = d2 <= float(eps) ** 2
    core = neigh.sum(axis=1) >= int(min_samples)

    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        # Breadth-first density expansion. Border points join the first
        # cluster that reaches them, which is DBSCAN's documented
        # non-determinism and is harmless for a centroid.
        stack = [i]
        labels[i] = cid
        while stack:
            j = stack.pop()
            if not core[j]:
                continue                      # border point: reachable, not expanding
            for k in np.flatnonzero(neigh[j]):
                if labels[k] == -1:
                    labels[k] = cid
                    stack.append(k)
        cid += 1
    return labels


@dataclass
class Detection:
    """One clustered candidate. Paper eq 37's centroid, plus what it took."""

    delay_s: float                # parabolic peak - the primary estimate
    doppler_hz: float
    power: float                  # peak cell power in the cluster
    snr_db: float                 # peak over the CFAR threshold at that cell
    n_cells: int
    delay_cells: float            # power-weighted centroid, in map cells
    doppler_cells: float
    excess_m: float
    centroid_delay_s: float = float("nan")     # paper eq 37, kept for comparison
    centroid_doppler_hz: float = float("nan")

    def range_rate_mps(self, fc: float, c: float = 299792458.0) -> float:
        """One-way bistatic: Rdot = -lambda * f_d. See coarse.py on the 4pi."""
        return -self.doppler_hz * c / fc


def cluster_detections(mask: np.ndarray, P: np.ndarray, thr: np.ndarray,
                       delay_s: np.ndarray, doppler_hz: np.ndarray,
                       eps_cells: float = 2.5, min_samples: int = 2,
                       cell_r: float = 1.0, cell_c: float = 1.0,
                       c: float = 299792458.0):
    """CFAR mask -> list of Detection, strongest first. Paper eq 36-37.

    `cell_r` and `cell_c` are the map's oversampling factors: with pad = 2 a
    resolution cell is 2 bins wide, so distances are divided by them to make
    eps_cells mean resolution cells rather than display bins.

    Two estimators are computed per cluster and both are returned.

    `centroid_*` is eq 37, power-weighted rather than eq 37's plain mean - a
    plain mean drags toward whichever side of the target happened to clear the
    threshold. `delay_s`/`doppler_hz` are instead a three-point parabolic fit
    about the cluster's strongest cell, and that is the primary estimate.

    Measured reason for the swap, on a -25 dB drone at 638.3 m of true excess
    over eight consecutive dwells: the power-weighted centroid read +3.3 m
    high every time, while the coarse stage's parabolic fit on the same target
    was within 0.2 m. A centroid over the CFAR-surviving cells is biased
    whenever that cell set is asymmetric, which next to the direct path's
    skirt it always is. 3.3 m is only 0.05 of a delay cell, but it is a bias,
    not noise, so it does not average away over a track.
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return []
    pts = np.column_stack((idx[:, 0] / float(cell_r), idx[:, 1] / float(cell_c)))
    labels = dbscan(pts, eps_cells, min_samples)

    out = []
    for cid in range(labels.max() + 1 if labels.size else 0):
        sel = idx[labels == cid]
        if sel.size == 0:
            continue
        w = P[sel[:, 0], sel[:, 1]]
        tw = w.sum()
        if tw <= 0:
            continue
        ri = float(np.dot(sel[:, 0], w) / tw)
        ci = float(np.dot(sel[:, 1], w) / tw)
        k = int(np.argmax(w))
        r0, c0 = int(sel[k, 0]), int(sel[k, 1])
        d_step = delay_s[1] - delay_s[0]
        f_step = doppler_hz[1] - doppler_hz[0]

        # Parabolic fit about the peak cell, on the full map rather than on
        # the masked cells, so a peak whose neighbour failed CFAR is still
        # refined against the real shoulder.
        dc = _vertex(P[r0, :], c0)
        dr = _vertex(P[:, c0], r0)
        tau = float(delay_s[c0] + dc * d_step)
        fd = float(doppler_hz[r0] + dr * f_step)
        out.append(Detection(
            delay_s=tau, doppler_hz=fd, power=float(w[k]),
            snr_db=float(10 * np.log10(max(w[k], 1e-300)
                                       / max(thr[r0, c0], 1e-300))),
            n_cells=int(sel.shape[0]), delay_cells=ci / float(cell_c),
            doppler_cells=ri / float(cell_r), excess_m=tau * c,
            centroid_delay_s=float(delay_s[0] + ci * d_step),
            centroid_doppler_hz=float(doppler_hz[0] + ri * f_step)))
    out.sort(key=lambda d: d.power, reverse=True)
    return out


def suppress_comb(dets, M_raw: np.ndarray, delay_s: np.ndarray,
                  doppler_hz: np.ndarray, comb_hz: float,
                  doppler_tol_hz: float = None, delay_tol_bins: int = 1,
                  max_order: int = 16):
    """Drop detections that are known aperture sidebands. Returns (keep, drop).

    The sensing weight W is not uniform in slow time. Slot 0 of every frame
    carries the SSB and no PDSCH (handoff section 4), so the aperture changes
    structurally once per 20 ms frame. That is a periodic amplitude modulation
    at the 50 Hz frame rate, and it convolves every tap in the map with a
    50 Hz-spaced Doppler comb. Measured on a filled grid with a -25 dB drone:
    sidebands at +-50, +-100, +-150 and +-200 Hz from the tap, the strongest
    about 9 dB below it, each surviving CFAR and each spawning its own track.

    findings.md section 4 already noted that per-slot PRB randomisation
    "spreads the pedestal across the whole Doppler axis". This is the precise
    form of that spreading: not a smooth pedestal, a comb at the frame rate.

    Two reasons this rejection is legitimate rather than a fudge. The receiver
    knows W exactly, so it knows the comb's positions exactly - this is a
    known point spread function, the same information CLEAN would use. And the
    reference is taken from the *un-notched* map, which matters: the
    zero-Doppler notch removes a static scatterer's main peak but not its
    sidebands. In the selected verification geometry the clutter tap is at
    +0.5 Hz, so its main peak vanishes while its +-50 Hz harmonics remain and
    would otherwise read as two moving targets. Generic 250 m clutter can
    reach 25.2 Hz at 2478 MHz and is handled by the optional locus filter.

    The cost, stated: a genuine target lying within a delay bin of a strong
    scatterer *and* within tolerance of an exact multiple of 50 Hz from it is
    discarded. The delay coincidence makes that unlikely, and it is the
    trade any known-PSF rejection makes.
    """
    if not dets:
        return [], []
    d_step = float(delay_s[1] - delay_s[0])
    f_step = float(doppler_hz[1] - doppler_hz[0])
    # One display bin of tolerance. It has to absorb the detection centroid's
    # own error, which is a few tenths of a hertz once the reference below is
    # interpolated. Before interpolating the reference the systematic error
    # was half a bin - 1.5 Hz here - and an n = -1 sideband at 2.6 Hz residual
    # survived a 2.5 Hz tolerance and became a permanent false track.
    tol = 1.0 * abs(f_step) if doppler_tol_hz is None else float(doppler_tol_hz)
    n_c = M_raw.shape[1]
    keep, drop = [], []
    for d in dets:
        j = int(round((d.delay_s - delay_s[0]) / d_step))
        c0, c1 = max(j - delay_tol_bins, 0), min(j + delay_tol_bins + 1, n_c)
        if c1 <= c0:
            keep.append(d)
            continue
        col = np.abs(M_raw[:, c0:c1]).max(axis=1)
        i = int(np.argmax(col))
        # Interpolate the reference to the same sub-bin precision the
        # detection has, or the comb offsets are compared against a quantised
        # origin and half a bin of that error goes straight into the residual.
        f_pk = float(doppler_hz[i] + _vertex(col, i) * f_step)
        off = d.doppler_hz - f_pk
        n = int(round(off / comb_hz))
        if n != 0 and abs(n) <= max_order and abs(off - n * comb_hz) <= tol:
            drop.append((d, n, f_pk))
        else:
            keep.append(d)
    return keep, drop


def measured_pfa(mask: np.ndarray, valid: np.ndarray = None) -> float:
    """Fraction of cells that fired. On a target-absent run this is the
    realised false alarm rate, which is the only honest way to confirm the
    requested Pfa - eq 34's alpha is exact only for the noise model assumed."""
    m = mask if valid is None else mask[..., valid]
    return float(np.count_nonzero(m) / max(m.size, 1))
