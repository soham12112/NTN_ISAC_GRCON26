"""Channel estimation, equalisation and delay-Doppler mapping (section 7, 5-9).

One code path, one flag. The three variants differ only in the weight mask W
and in where X_est comes from:

    reference-only   W = ref_mask   X_est = known reference values
    payload-aided    W = occ_mask   X_est = references + sliced decisions
    genie            W = occ_mask   X_est = tx_grid

There is no LDPC decoding here and none is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REF_ONLY = "reference-only"
PAYLOAD = "payload-aided"
GENIE = "genie"
VARIANTS = (REF_ONLY, PAYLOAD, GENIE)


def ls_channel_estimate(Y, ref_mask, ref_grid, occ_mask=None):
    """Least squares on the flagged reference REs, then separable interpolation.

    Frequency first, across the subcarriers that carry references in a given
    symbol; then time, between the symbols that have any. Symbols outside the
    reference span hold the nearest estimate.
    """
    n_sym, n_sc = Y.shape
    H = np.zeros((n_sym, n_sc), np.complex64)
    have = np.zeros(n_sym, bool)
    k = np.arange(n_sc)

    for l in range(n_sym):
        m = ref_mask[l]
        if m.sum() < 2:
            continue
        h = Y[l, m] / ref_grid[l, m]
        H[l] = (np.interp(k, k[m], h.real).astype(np.float32)
                + 1j * np.interp(k, k[m], h.imag).astype(np.float32))
        have[l] = True

    if not have.any():
        raise ValueError("no symbol carries enough reference REs to estimate H")

    idx = np.flatnonzero(have)
    for part in ("real", "imag"):
        src = getattr(H[idx], part)
        interp = np.empty((n_sym, n_sc), np.float32)
        for c in range(n_sc):
            interp[:, c] = np.interp(np.arange(n_sym), idx, src[:, c])
        if part == "real":
            out = interp.astype(np.complex64)
        else:
            out = out + 1j * interp
    H = out
    if occ_mask is not None:
        H = np.where(occ_mask, H, 0).astype(np.complex64)
    return H


def estimate_noise_var(Y, H, ref_mask, ref_grid):
    """Residual power on the reference REs after removing the estimate."""
    r = Y[ref_mask] - H[ref_mask] * ref_grid[ref_mask]
    return float(np.mean(np.abs(r) ** 2)) if r.size else 0.0


def mmse_equalize(Y, H, noise_var):
    return (np.conj(H) * Y / (np.abs(H) ** 2 + noise_var + 1e-30)).astype(np.complex64)


def slice_qpsk(Xeq, amplitude=1.0):
    """Hard decision to the nearest QPSK point, scaled to `amplitude`.

    Constant modulus is why section 4 pins the modulation to QPSK: it makes
    this slicer reliable and makes step 8 equivalent to reciprocal filtering
    without the noise enhancement that hits the inner points of 16QAM.
    """
    s = np.sign(Xeq.real) + 1j * np.sign(Xeq.imag)
    s[s.real == 0] += 1
    s[s.imag == 0] += 1j
    return (s * (amplitude / np.sqrt(2.0))).astype(np.complex64)


@dataclass
class MapResult:
    M: np.ndarray            # |.|, [n_doppler, n_delay], Doppler fftshifted
    delay_s: np.ndarray
    doppler_hz: np.ndarray
    w_sum: float
    peak: float
    noise: float
    pk_nf_db: float
    peak_delay_s: float
    peak_doppler_hz: float

    @property
    def peak_delay_samples_at(self):
        return lambda fs: self.peak_delay_s * fs


def delay_axis(cfg, pad, n_sc):
    """Delay bin centres, seconds, for a pad-times-oversampled IFFT over k.

    Public because the coarse SSB stage must land on exactly this axis: its
    Gaussian range gate is handed to the fine stage as a delay mask, and any
    rescaling between the two would bias the gate by a fraction of a cell.
    """
    nk = int(pad) * int(n_sc)
    return np.arange(nk) / (nk * float(cfg["scs_khz"]) * 1e3), nk


def doppler_axis(cfg, pad, n_sym):
    """Doppler bin centres, Hz, fftshifted, for a pad-times FFT over l."""
    nl = int(pad) * int(n_sym)
    cp = np.asarray(cfg["cp_lengths"], float)
    t_sym = (int(cfg["n_fft"]) + cp.mean()) / float(cfg["fs_hz"])
    return (np.arange(nl) - nl // 2) / (nl * t_sym), nl


def _axes(cfg, pad, n_sym, n_sc):
    delay, nk = delay_axis(cfg, pad, n_sc)
    dopp, nl = doppler_axis(cfg, pad, n_sym)
    return delay, dopp, nk, nl


def taper(n: int, kind: str = "hann") -> np.ndarray:
    """Symmetric analysis window of length n, endpoints excluded.

    "hann" reproduces `np.hanning(n + 2)[1:-1]` exactly and is the default
    everywhere, because every verified checkpoint from V1 to V15 was measured
    with it and the numbers in findings.md are its numbers.

    "blackmanharris" exists for the over-the-air path. Hann's first sidelobe is
    -31 dB and findings.md section 11 measured the direct path's own point
    spread function standing ~31 dB above the noise floor at the target's
    delay - the same number, and not a coincidence. Over a cable the target is
    a synthetic tap hundreds of metres away in delay and that skirt is
    survivable. Over the air a real object sits at delay bin zero, under the
    direct path, separated from it only in Doppler, so the slow-time sidelobe
    level *is* the detection floor. A 4-term Blackman-Harris trades 1.9x
    mainlobe width for -92 dB, which costs Doppler resolution that the OTA
    path has to spare and buys 60 dB exactly where it is needed.
    """
    if kind == "hann":
        return np.hanning(n + 2)[1:-1]
    if kind == "blackmanharris":
        a = (0.35875, 0.48829, 0.14128, 0.01168)
        m = np.arange(1, n + 1) / (n + 1)
        return (a[0] - a[1] * np.cos(2 * np.pi * m) + a[2] * np.cos(4 * np.pi * m)
                - a[3] * np.cos(6 * np.pi * m))
    if kind in ("none", "rect"):
        return np.ones(n)
    raise ValueError(f"unknown window {kind!r}")


def map_from_z(Z, w_sum, cfg, pad=2, window="hann") -> MapResult:
    """The transform half of `delay_doppler_map`, on an already-formed Z.

    Split out so that a canceller can modify Z - subtracting a tap's exact
    contribution, which is what `clean` does - and then map the residual
    through identical arithmetic. Duplicating these six lines instead would
    make a cancelled map and an uncancelled one incomparable for reasons no
    one would find.

    `Z` must be unwindowed. The taper belongs to the transform, not to the
    signal model, so a canceller fits and subtracts in the untapered domain.
    """
    n_sym, n_sc = Z.shape
    Zw = Z * taper(n_sc, window)[None, :].astype(np.float32)
    Zw = Zw * taper(n_sym, window)[:, None].astype(np.float32)

    delay, dopp, nk, nl = _axes(cfg, pad, n_sym, n_sc)
    z = np.fft.ifft(Zw, n=nk, axis=1) * nk           # undo numpy's 1/N
    M = np.fft.fftshift(np.fft.fft(z, n=nl, axis=0), axes=0)

    M = np.abs(M) / np.sqrt(max(w_sum, 1.0))

    noise = float(np.median(M ** 2) / np.log(2.0))   # median of an exponential
    i, j = np.unravel_index(np.argmax(M), M.shape)
    peak = float(M[i, j])
    return MapResult(
        M=M, delay_s=delay, doppler_hz=dopp, w_sum=w_sum, peak=peak,
        noise=float(np.sqrt(noise)),
        pk_nf_db=10 * np.log10(peak ** 2 / max(noise, 1e-30)),
        peak_delay_s=float(_parabolic(M[i], j, delay)),
        peak_doppler_hz=float(_parabolic(M[:, j], i, dopp)),
    )


def form_z(Y, X_est, W, mag_floor=1e-12) -> np.ndarray:
    """Z = Y * conj(X_est)/|X_est| * W, untapered.

    Dividing by |X_est| is what makes the three variants comparable. It is a
    no-op for a constant-modulus grid, but a real scheduler boosts PSS/SSS and
    applies per-layer power offsets, and without it whichever REs are boosted
    would dominate the map.
    """
    Xn = X_est / np.maximum(np.abs(X_est), mag_floor)
    return (Y * np.conj(Xn) * W).astype(np.complex64)


def delay_doppler_map(Y, X_est, W, cfg, pad=2, mag_floor=1e-12,
                      window="hann") -> MapResult:
    """Z = Y * conj(X_est)/|X_est| * W, windowed, then IFFT over k, FFT over l."""
    Z = form_z(Y, X_est, W, mag_floor)
    return map_from_z(Z, float(np.sum(W)), cfg, pad, window)


def _parabolic(v, i, axis):
    """Sub-bin peak location by three-point parabolic fit on magnitude."""
    return axis[i] + _vertex(v, i)[0] * (axis[1] - axis[0])


def _vertex(v, i):
    """(offset in bins, interpolated peak value) from a three-point fit.

    Reading the peak bin alone under-reports amplitude by up to ~1.4 dB for a
    Hann-windowed tap that lands between bins, which is enough to fail an
    amplitude-ratio check that the channel actually satisfies exactly.
    """
    if i <= 0 or i >= v.size - 1:
        return 0.0, float(v[i])
    a, b, c = float(v[i - 1]), float(v[i]), float(v[i + 1])
    den = a - 2 * b + c
    if den == 0:
        return 0.0, b
    d = 0.5 * (a - c) / den
    return d, b - 0.25 * (a - c) * d


def peak_near(res, delay_s, half_bins=6):
    """Interpolated (amplitude, delay_s, doppler_hz) near a predicted delay."""
    step = res.delay_s[1] - res.delay_s[0]
    c = int(round(delay_s / step))
    lo, hi = max(c - half_bins, 0), min(c + half_bins + 1, res.M.shape[1])
    sub = res.M[:, lo:hi]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    dj, amp = _vertex(res.M[i], lo + j)
    di, amp_f = _vertex(res.M[:, lo + j], i)
    return (max(amp, amp_f), (lo + j + dj) * step,
            res.doppler_hz[i] + di * (res.doppler_hz[1] - res.doppler_hz[0]))


def link_metrics(Y, bundle, H=None, noise_var=None, genie_grid=None,
                 dwell_s=None) -> dict:
    """Communication-side metrics for the dashboard: EVM, SER, throughput.

    Shares the channel estimate with build_variant - pass the same H and
    noise_var to both - because ls_channel_estimate is the most expensive
    single step in the chain and computing it twice per dwell would cost more
    than the second sensing map does.

    On the three quantities handoff section 8 asks for:

    EVM is measured, on the occupied data resource elements only. Reference
    elements are excluded: their error is what the channel estimate was fitted
    to, so including them reports the fit residual rather than the data EVM
    and flatters the number.

    BLER is **not available and is not reported**. It needs LDPC decoding,
    which handoff section 12 forbids in the live receive path and which this
    chain therefore does not contain. What is reported instead is the uncoded
    symbol error rate against `tx_grid`, which is a genuine measurement of the
    same thing one layer down. A dashboard must label it as such; presenting
    an uncoded SER as a BLER would overstate the link by orders of magnitude
    at any useful operating point.

    Throughput is likewise the **offered** rate computed from the allocation,
    modulation and code rate - not a delivered rate, which without a decoder
    cannot be known.
    """
    cfg = bundle.cfg
    ref_mask, occ_mask = bundle.ref_mask, bundle.occ_mask
    ref_grid = bundle.ref_grid()
    data = occ_mask & ~ref_mask
    out = dict(n_data_re=int(data.sum()), n_ref_re=int(ref_mask.sum()),
               n_occ_re=int(occ_mask.sum()))
    if not data.any():
        return out

    if H is None:
        H = ls_channel_estimate(Y, ref_mask, ref_grid, occ_mask)
    if noise_var is None:
        noise_var = estimate_noise_var(Y, H, ref_mask, ref_grid)
    Xeq = mmse_equalize(Y, H, noise_var)
    d = Xeq[data]

    # EVM against the best-fit reference constellation: normalise the measured
    # symbols to unit rms and compare with unit-rms QPSK. Deliberately
    # insensitive to a common gain, because a receiver's AGC and the scale of
    # the channel estimate both apply one and neither is an impairment. An
    # absolute-amplitude definition instead reports whatever scale error the
    # slicer happens to carry - which is how the sqrt(2) bug in build_variant
    # above was found, as a flat 29% EVM on a noiseless channel.
    r = float(np.sqrt(np.mean(np.abs(d) ** 2)))
    if r <= 0:
        return out
    dn = d / r
    ideal = (np.sign(dn.real) + 1j * np.sign(dn.imag)) / np.sqrt(2.0)
    ideal[ideal == 0] = (1 + 1j) / np.sqrt(2.0)
    err = dn - ideal
    ep = float(np.mean(np.abs(err) ** 2))
    evm = float(np.sqrt(ep / max(float(np.mean(np.abs(ideal) ** 2)), 1e-30)))
    out.update(evm=evm, evm_db=float(20 * np.log10(max(evm, 1e-12))),
               noise_var=float(noise_var),
               snr_db=float(-20 * np.log10(max(evm, 1e-12))))

    if genie_grid is not None:
        # Quadrant comparison, which for QPSK is exactly a symbol error. The
        # slicer is only valid for constant modulus (handoff section 4), so
        # this number is meaningless for the 16QAM bundle and the caller is
        # told the modulation so it can say so.
        g = np.asarray(genie_grid)[data]
        se = ((np.sign(dn.real) != np.sign(g.real))
              | (np.sign(dn.imag) != np.sign(g.imag)))
        out["ser_uncoded"] = float(np.count_nonzero(se) / se.size)

    pd = cfg.get("pdsch", {})
    mod = str(pd.get("modulation", "QPSK")).upper()
    bits = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}.get(mod, 2)
    rate = float(pd.get("target_code_rate", 308 / 1024))
    out.update(modulation=mod, bits_per_symbol=bits, code_rate=rate,
               slicer_valid=bool(mod == "QPSK"))
    if dwell_s:
        out["offered_mbps"] = float(
            out["n_data_re"] * bits * rate / float(dwell_s) / 1e6)
    return out


def build_variant(Y, bundle, variant, H=None, noise_var=None, genie_grid=None):
    """Return (X_est, W) for one variant. Steps 5-8 share everything but this."""
    ref_grid = bundle.ref_grid()
    ref_mask, occ_mask = bundle.ref_mask, bundle.occ_mask

    if variant == REF_ONLY:
        return ref_grid, ref_mask.astype(np.float32)

    if variant == GENIE:
        if genie_grid is None:
            raise ValueError("genie variant needs tx_grid")
        return np.asarray(genie_grid, np.complex64), occ_mask.astype(np.float32)

    if variant != PAYLOAD:
        raise ValueError(f"unknown variant {variant!r}")

    if H is None:
        H = ls_channel_estimate(Y, ref_mask, ref_grid, occ_mask)
    if noise_var is None:
        noise_var = estimate_noise_var(Y, H, ref_mask, ref_grid)
    # slice_qpsk returns points of magnitude `amplitude`, so this is the
    # reference RE magnitude, not that times sqrt(2). The extra factor was
    # here and was invisible: delay_doppler_map normalises X_est by |X_est|,
    # so a common magnitude error on the data REs is divided straight back
    # out and every sensing checkpoint passes either way. It is still wrong -
    # X_est is documented as an estimate of the transmitted grid, and with the
    # factor present the grid it returns is internally inconsistent, true
    # amplitude on reference REs and sqrt(2) too large on data ones. Anything
    # that reads X_est without normalising, such as an EVM, inherits a 29%
    # error. Verified against tx_grid: the data-RE rms and the mean reference
    # magnitude are equal to 1 part in 1e7.
    amp = float(np.mean(np.abs(ref_grid[ref_mask]))) if ref_mask.any() else 1.0
    X = slice_qpsk(mmse_equalize(Y, H, noise_var), amplitude=amp)
    X = np.where(ref_mask, ref_grid, X).astype(np.complex64)
    X = np.where(occ_mask, X, 0).astype(np.complex64)
    return X, occ_mask.astype(np.float32)
