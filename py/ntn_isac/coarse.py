"""Stage 1: coarse detection from the SSB (paper section II).

The paper's first stage uses the beam-swept SSB as a pulsed illuminator:
frequency-domain matched filtering against the known SSB (eq 6), moving target
indication by adjacent-repetition subtraction (eq 7), a non-coherent range
profile (eq 8), a Gaussian gate around the predicted delay (eq 9), parabolic
sub-cell refinement (eq 10), and radial velocity from the inter-repetition
phase difference (eq 11-12) rather than a Doppler FFT - because, as the paper
puts it, "standard Doppler FFT processing is unsuitable for the irregular and
limited SSB repetitions in 5G NR".

That reasoning transfers to this testbed unchanged, and the arithmetic does
not. Four deviations, all forced by the hardware and all of them load-bearing:

No angle of arrival. The paper beamforms across an 8-element ULA (eq 4, 13) to
get AoA and hence a range-AoA map. This implementation consumes one receive
channel, so it has no spatial aperture. Stage 1 here produces delay and
residual Doppler only, and stage 2's map is delay-Doppler rather than
range-azimuth. Additional B210 channels make a future multi-channel branch
possible; they do not create angle data in this branch.

One beam, not a sweep. The illuminator is a LEO satellite with a fixed beam
over the pass, so SSB index carries no angular sector information. The "search
radar scanning a field of view" framing does not apply; what remains is the
pulsed-illuminator framing, which is the part stage 1 actually uses.

Bistatic one-way, so no factor of 4pi. Eq 11 reads v = lambda*dphi/(4*pi*T),
which is the monostatic two-way round trip. This channel injects
exp(-j*2*pi*f_c*tau) on a one-way excess path (handoff section 6), so the tap
phase is -2*pi*f_c*tau and

    f_d = dphi / (2*pi*T_rep),      excess range rate = -lambda * f_d

Using the paper's constant would halve every velocity estimate.

One SSB per 20 ms, which sets a hard ambiguity. The waveform carries a single
SSB in slot 0 with a period equal to the file length, so the repetition
interval is the 20 ms frame and the pulse-pair PRF is 50 Hz: unambiguous only
to +-25 Hz. traj_meta.json puts the drone's residual Doppler between -36.7 and
+77.1 Hz, so the coarse Doppler is genuinely ambiguous over much of the pass.
That is not a defect to hide - it is the reason a fine stage exists. Stage 1
returns the ambiguous measurement and the ambiguity interval;
`unwrap_doppler` resolves it against a prediction, exactly as the paper's own
prediction gating does.

MTI blind speeds come with the same 50 Hz PRF. Adjacent-repetition
subtraction has response 2|sin(pi f_d T_rep)|, which nulls at every multiple
of the PRF. The direct path sits at f_d ~ 0 after de-ramp and is therefore
cancelled, which is the point; but a target at 50 Hz residual Doppler is
cancelled just as hard, and the drone passes through exactly that. Callers get
`mti_gain_db` so a blind-speed epoch is reported rather than mistaken for a
lost target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import sensing


def ssb_span(cfg: dict):
    """(first subcarrier, count) of the SSB inside the carrier grid.

    Read from config.json rather than recomputed from NR mapping rules; the
    whole point of the file interface is that this module never has to know
    what Case A means.
    """
    ssb = cfg["ssb"]
    k0 = int(ssb["ncrb_ssb"]) * 12 + int(ssb.get("kssb", 0))
    return k0, int(ssb["n_sc"])


def syms_per_frame(cfg: dict) -> int:
    """Symbols in one waveform file, which is the SSB repetition stride.

    Deliberately *not* `n_sym_total`: fileio.TiledBundle rewrites that key to
    the length of the whole dwell, so reading it here strides by the dwell
    instead of by the frame and walks straight off the end of the grid.
    n_slots and n_sym_slot are per-file and survive tiling.
    """
    return int(cfg["n_slots"]) * int(cfg["n_sym_slot"])


def n_frames_in(cfg: dict) -> int:
    """How many looped frames the grid described by `cfg` covers."""
    per = syms_per_frame(cfg)
    total = int(cfg["n_sym_total"])
    if total % per:
        raise ValueError(f"{total} symbols is not a whole number of {per}-symbol frames")
    return total // per


def ssb_symbols(cfg: dict, n_frames: int = None) -> np.ndarray:
    """Global symbol indices of the SSB, shape [n_frames, n_ssb_sym].

    The waveform file loops, so frame f puts its SSB at f*syms_per_frame plus
    the in-frame symbol offsets. `n_frames` defaults to whatever the grid
    covers, which is the form callers should prefer - passing it by hand is
    how the stride bug above got in.
    """
    ssb = cfg["ssb"]
    off = np.asarray(ssb["symbols"], dtype=np.int64)
    per = syms_per_frame(cfg)
    if n_frames is None:
        n_frames = n_frames_in(cfg)
    if n_frames > 1 and not cfg.get("loopable", False):
        raise ValueError("cannot repeat SSB frames on a non-loopable capture")
    return off[None, :] + per * np.arange(int(n_frames), dtype=np.int64)[:, None]


def rep_interval(cfg: dict) -> float:
    """Seconds between SSB repetitions: one frame, since Period == n_slots."""
    return float(cfg["frame_len_samples"]) / float(cfg["fs_hz"])


def cp_excess_limit_s(cfg: dict) -> float:
    """Largest excess delay the cyclic prefix can carry, seconds.

    handoff section 4: the shortest CP is 36 samples at 7.68 MS/s, so 4.69 us
    or about 1400 m of usable excess path. A tap beyond this is not merely
    weak, it has fallen outside the CP and the grid-based receiver cannot see
    it, so anything the profile shows out there is an artefact.
    """
    return float(min(cfg["cp_lengths"])) / float(cfg["fs_hz"])


def valid_delays(delay_s: np.ndarray, cfg: dict,
                 min_excess_s: float = 0.0) -> np.ndarray:
    """Delay bins inside the CP budget: where a peak can physically be.

    `min_excess_s` additionally blanks the direct path. MTI suppresses it by a
    finite amount - 20.7 dB measured here - so on a target-absent dwell the
    residual illuminator is still the strongest thing in the profile, and
    without blanking the coarse stage locks onto it and hands the tracker a
    confirmed "target" at zero excess. Stage 2 blanks the same cell, so this
    keeps the two stages' notions of a reachable delay identical.

    Needed because the SSB reference aperture is not a filled block. PBCH-DMRS
    is comb-4, i.e. one subcarrier in every four over the 240-subcarrier SSB,
    and a comb of spacing 4*15 kHz = 60 kHz puts grating lobes on the delay
    axis at multiples of 1/60 kHz = 16.67 us, which is 5000 m. Measured, those
    replicas sit 26 dB above the noise floor - well above a -25 dB drone echo.
    They are all far outside the 1400 m CP budget, so confining the peak
    search is enough; without it a flat gate can lock the coarse stage onto a
    grating lobe of the direct path instead of onto a target.
    """
    d = np.asarray(delay_s, float)
    return (d <= cp_excess_limit_s(cfg)) & (d >= float(min_excess_s))


def mti_response_db(f_d, t_rep: float) -> np.ndarray:
    """Gain of single-delay MTI at residual Doppler f_d, dB relative to unity.

    2|sin(pi f_d T_rep)|. Zero at every multiple of the PRF - the clutter notch
    at f_d = 0 that MTI is for, and the blind speeds that come with it.
    """
    g = 2.0 * np.abs(np.sin(np.pi * np.asarray(f_d, float) * t_rep))
    return 20 * np.log10(np.maximum(g, 1e-12))


def matched_filter(Y: np.ndarray, ref_grid: np.ndarray, ref_mask: np.ndarray,
                   cfg: dict, n_frames: int = None, pad: int = 2,
                   mag_floor: float = 1e-12):
    """Paper eq 6, done in the frequency domain: C[rep, sym, delay], complex.

    Eq 6 is written as FFT of the windowed receive block times the conjugate
    FFT of the reference, then an IFFT. On an OFDM waveform the two FFTs have
    already been done by the demodulator, so the matched filter is a
    per-subcarrier product and one IFFT - no time-domain correlation anywhere.

    Only the SSB reference elements are used. PBCH payload is occupied but not
    known without decoding the MIB, so `ref_mask` excludes it and so does
    this; including it would flatter the coarse stage with knowledge a UE does
    not have at this point.

    The Hamming window of eq 6 is evaluated across the SSB *span*, not across
    the sparse reference elements within it. The window shapes an aperture,
    and the aperture is the 240-subcarrier block; applying it to the sample
    index instead would taper by position-in-list and mean nothing.
    """
    n_sc = int(cfg["n_sc"])
    k0, n_ssb = ssb_span(cfg)
    idx = ssb_symbols(cfg, n_frames)
    delay_s, nk = sensing.delay_axis(cfg, pad, n_sc)

    # Hamming across the SSB span, zero outside it.
    win = np.zeros(n_sc, np.float32)
    win[k0:k0 + n_ssb] = np.hamming(n_ssb).astype(np.float32)

    C = np.zeros(idx.shape + (nk,), np.complex64)
    for f in range(idx.shape[0]):
        for j, l in enumerate(idx[f]):
            m = ref_mask[l].copy()
            m[:k0] = False
            m[k0 + n_ssb:] = False
            if not m.any():
                continue
            x = ref_grid[l]
            # Normalise by |X| for the same reason sensing.delay_doppler_map
            # does: PSS and SSS are boosted relative to DMRS by a real gNB,
            # and without this whichever is boosted would dominate C.
            z = np.zeros(n_sc, np.complex64)
            z[m] = Y[l, m] * np.conj(x[m]) / np.maximum(np.abs(x[m]), mag_floor)
            C[f, j] = np.fft.ifft(z * win, n=nk) * nk       # undo numpy's 1/N
    return C, delay_s


def mti(C: np.ndarray) -> np.ndarray:
    """Paper eq 7: MTI(tau, xi) = C(tau, xi+1) - C(tau, xi), along repetitions.

    Requires at least two repetitions. With one frame there is nothing to
    subtract and the direct path is not cancelled at all, so callers that ask
    for a single-frame dwell get the un-cancelled profile and a notch depth of
    0 dB rather than a silent pass.
    """
    if C.shape[0] < 2:
        return np.zeros((0,) + C.shape[1:], C.dtype)
    return (C[1:] - C[:-1]).astype(np.complex64)


def range_profile(Z: np.ndarray) -> np.ndarray:
    """Paper eq 8: P(tau) = sum over repetitions and symbols of |Z|^2."""
    if Z.size == 0:
        return np.zeros(0, np.float64)
    return np.sum(np.abs(Z.astype(np.complex128)) ** 2, axis=(0, 1))


def gaussian_gate(delay_s: np.ndarray, tau_pred: float, sigma_tau: float
                  ) -> np.ndarray:
    """Paper eq 9: W(tau) = exp(-(tau - tau_pred)^2 / (2 sigma_tau^2)).

    A non-negative prediction weight, not a hard window: a target just outside
    the gate is attenuated rather than deleted, so a bad prediction degrades
    the estimate instead of losing the track.
    """
    if not np.isfinite(tau_pred) or sigma_tau <= 0:
        return np.ones_like(delay_s)
    d = (np.asarray(delay_s, float) - float(tau_pred)) / float(sigma_tau)
    return np.exp(-0.5 * d * d)


def refine_peak(P: np.ndarray, delay_s: np.ndarray, gate: np.ndarray = None,
                valid: np.ndarray = None):
    """Paper eq 10: gated peak, then a three-point parabolic fit in power.

    Returns (delay_s, bin index, gated peak power). The fit is on P, not on
    |.|, because eq 10 is written on the power profile; doing it on magnitude
    shifts the vertex by a few hundredths of a cell.

    `valid` restricts the search to physically reachable delays - see
    `valid_delays`. The parabolic fit still reads its neighbours from the full
    profile, so a peak in the last valid bin is refined, not clipped.
    """
    Pg = P if gate is None else P * gate
    if Pg.size == 0:
        return float("nan"), -1, 0.0
    if valid is not None:
        Pg = np.where(valid, Pg, 0.0)
    j = int(np.argmax(Pg))
    step = delay_s[1] - delay_s[0]
    if 0 < j < Pg.size - 1:
        a, b, c = float(Pg[j - 1]), float(Pg[j]), float(Pg[j + 1])
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        d = 0.0
    return float(delay_s[j] + d * step), j, float(Pg[j])


def pulse_pair(C: np.ndarray, j: int, t_rep: float):
    """Paper eq 12 then eq 11, corrected to one-way bistatic geometry.

    dphi = angle( mean over repetitions of C(tau_hat, xi+1) conj(C(tau_hat, xi)) )
    f_d  = dphi / (2 pi T_rep)

    Averaged over the SSB symbols in a block as well as over repetitions,
    which the paper's eq 12 does implicitly through its per-antenna sum.
    Returns (f_d, unambiguous half-interval, coherence in [0, 1]).
    """
    prf = 1.0 / float(t_rep)
    if C.shape[0] < 2 or j < 0:
        return float("nan"), prf / 2.0, 0.0
    a = C[1:, :, j]
    b = C[:-1, :, j]
    r = np.mean(a * np.conj(b))
    # Coherence: |mean| over mean|.|. A pulse-pair phase from a low-coherence
    # lag is meaningless, and the tracker needs to know that rather than
    # receiving a confident wrong number.
    denom = np.mean(np.abs(a) * np.abs(b))
    coh = float(np.abs(r) / denom) if denom > 0 else 0.0
    return float(np.angle(r) / (2 * np.pi * t_rep)), prf / 2.0, coh


def unwrap_doppler(f_meas: float, f_pred: float, prf: float) -> float:
    """Move the ambiguous pulse-pair estimate to the alias nearest a prediction.

    With one SSB per 20 ms the PRF is 50 Hz, so the drone's +77 Hz residual
    Doppler aliases to -23 Hz. There is no signal-processing fix at this PRF;
    the ambiguity is resolved by prediction, which is what the paper's own
    forecast gating provides and what any real tracker does.
    """
    if not np.isfinite(f_meas):
        return f_meas
    if not np.isfinite(f_pred) or prf <= 0:
        return f_meas
    return float(f_meas + prf * np.round((f_pred - f_meas) / prf))


@dataclass
class CoarseResult:
    """Stage 1 output. `to_gate` is the whole coarse-to-fine handoff."""

    delay_s: np.ndarray
    P: np.ndarray                  # MTI range profile, power
    P_raw: np.ndarray              # same without MTI, for the notch depth
    gate: np.ndarray
    valid: np.ndarray              # delay bins inside the CP budget
    n_rep: int
    peak_delay_s: float
    peak_bin: int
    peak_power: float
    noise_power: float
    snr_db: float
    doppler_hz: float              # ambiguous, as measured
    doppler_unamb_hz: float        # after unwrapping to the prediction
    doppler_half_interval_hz: float
    coherence: float
    mti_notch_db: float
    t_rep: float
    meta: dict = field(default_factory=dict)

    @property
    def prf_hz(self) -> float:
        return 1.0 / self.t_rep

    def excess_m(self, c: float = 299792458.0) -> float:
        """Delay converted to excess path length, relative to the delay origin."""
        return self.peak_delay_s * c

    def range_rate_mps(self, fc: float, c: float = 299792458.0) -> float:
        """Excess range rate. One-way bistatic: Rdot = -lambda * f_d."""
        return -self.doppler_unamb_hz * c / fc

    def sigma_delay_meas_s(self, min_snr_db: float = 6.0) -> float:
        """Delay standard deviation of the *measurement*, for a tracker's R.

        One delay cell over sqrt(2*SNR), the matched-filter bound, with no
        floor. At the 34-38 dB coarse SNR this scenario gives, that is about
        1.2 m against an 83 m cell, and the measured coarse delay error is
        0.2-0.6 m - so the bound is right and using the floored `sigma_delay_s`
        here instead throws away a factor of 70.

        That factor decides whether stage 1 can bootstrap on its own. The
        coarse Doppler is ambiguous at this PRF and cannot seed a range rate,
        so the rate has to come from the *sequence* of delay measurements
        instead. With sigma = 1.2 m at a 100 ms dwell the rate converges below
        the 3.7 m/s alias spacing after about 5 dwells; with the floored 83 m
        it would take 84.
        """
        cell = 1.0 / (float(self.meta.get("aperture_hz", 3.6e6)))
        if self.snr_db < min_snr_db:
            return np.inf
        return float(cell / np.sqrt(2 * 10 ** (self.snr_db / 10.0)))

    def sigma_delay_s(self, min_snr_db: float = 6.0) -> float:
        """Delay standard deviation for the *gate* handed to stage 2.

        The measurement bound above, floored at one delay cell so a
        confident-but-wrong coarse peak cannot close the gate around the wrong
        place. Below `min_snr_db` the estimate is not trusted at all and the
        gate opens fully. Deliberately looser than the measurement sigma: a
        gate that is too tight discards the fine stage's whole advantage,
        while one that is too loose only readmits sidelobes CFAR then rejects.
        """
        cell = 1.0 / (float(self.meta.get("aperture_hz", 3.6e6)))
        s = self.sigma_delay_meas_s(min_snr_db)
        return float(np.inf if not np.isfinite(s) else max(s, cell))

    def to_gate(self, delay_s: np.ndarray, widen: float = 3.0) -> np.ndarray:
        """The Gaussian mask stage 2 applies to its own delay axis.

        `widen` multiplies the coarse delay sigma. The coarse stage is
        deliberately allowed to be loose here: a gate that is too tight throws
        away the fine stage's whole advantage, while one that is too loose only
        readmits sidelobes the CFAR then has to reject.
        """
        s = self.sigma_delay_s()
        if not np.isfinite(s) or not np.isfinite(self.peak_delay_s):
            return np.ones_like(delay_s)
        return gaussian_gate(delay_s, self.peak_delay_s, widen * s)


def process(Y: np.ndarray, bundle, cfg: dict = None, n_frames: int = None,
            pad: int = 2, tau_pred: float = None, sigma_pred_s: float = None,
            f_pred_hz: float = None, min_excess_s: float = 0.0) -> CoarseResult:
    """Run the whole of stage 1 on a de-ramped, retimed dwell.

    `Y` is the demodulated grid for a dwell of looped frames, as produced by
    receiver.demod_tracked. `bundle` supplies ref_mask and the reference
    values; nothing here knows any NR mapping rules. `cfg` and `n_frames`
    default to the bundle's own config and to the dwell length it describes.

    The prediction arguments are the paper's gating. With none supplied the
    gate is flat and the stage runs open-loop, which is what the first dwell
    of a track has to do.
    """
    if cfg is None:
        cfg = bundle.cfg
    if n_frames is None:
        n_frames = n_frames_in(cfg)
    t_rep = rep_interval(cfg)
    ref_grid = bundle.ref_grid()
    C, delay_s = matched_filter(Y, ref_grid, bundle.ref_mask, cfg, n_frames, pad)

    Z = mti(C)
    P = range_profile(Z)
    P_raw = range_profile(C)
    if P.size == 0:                       # single frame: nothing to subtract
        P = P_raw.copy()

    gate = (np.ones_like(delay_s) if tau_pred is None
            else gaussian_gate(delay_s, tau_pred,
                               sigma_pred_s if sigma_pred_s else 1e-6))
    valid = valid_delays(delay_s, cfg, min_excess_s)
    tau_hat, j, pk = refine_peak(P, delay_s, gate, valid)

    # Noise floor from the median of the profile: the map is mostly noise and
    # sidelobe, and a median is not dragged by the peak the way a mean is. The
    # log(2) divides out the median of an exponential distribution, matching
    # the convention in sensing.delay_doppler_map so the two stages' SNRs are
    # comparable numbers.
    noise = float(np.median(P) / np.log(2.0)) if P.size else 0.0
    snr_db = 10 * np.log10(max(pk, 1e-300) / max(noise, 1e-300))

    f_d, half, coh = pulse_pair(Z if Z.size else C, j, t_rep)
    f_unamb = f_d if f_pred_hz is None else unwrap_doppler(f_d, f_pred_hz, 1.0 / t_rep)

    # How much the direct path actually cancelled. This is the honest measure
    # of whether stage 1 is seeing a target or the residue of the illuminator.
    notch = (10 * np.log10(max(P_raw[valid].max(), 1e-300)
                           / max(P[valid].max(), 1e-300)) if Z.size else 0.0)

    _, n_ssb = ssb_span(cfg)
    return CoarseResult(
        delay_s=delay_s, P=P, P_raw=P_raw, gate=gate, n_rep=int(C.shape[0]),
        peak_delay_s=tau_hat, peak_bin=j, peak_power=pk, noise_power=noise,
        snr_db=float(snr_db), doppler_hz=float(f_d),
        doppler_unamb_hz=float(f_unamb), doppler_half_interval_hz=float(half),
        coherence=float(coh), mti_notch_db=float(notch), t_rep=t_rep,
        valid=valid,
        meta=dict(n_ssb_sc=n_ssb, n_ssb_sym=int(C.shape[1]),
                  cp_limit_m=cp_excess_limit_s(cfg) * 299792458.0,
                  aperture_hz=n_ssb * float(cfg["scs_khz"]) * 1e3,
                  n_frames=int(n_frames),
                  n_ref_used=int(np.count_nonzero(
                      bundle.ref_mask[ssb_symbols(cfg, n_frames)[0]]))))
