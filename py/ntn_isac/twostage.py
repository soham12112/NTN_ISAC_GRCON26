"""The two-stage coarse-to-fine flow (paper Fig. 2), wired end to end.

Fig. 2 of the paper runs gNB -> channel -> UEs (SSB processing) -> Gaussian
estimates uplinked to the gNB -> trajectory forecast -> selected PDSCH
downlink -> UE (DM-RS processing). This module is that flowgraph for one
receiver:

    demodulate (de-ramp, retime, de-walk)      receiver.demod_tracked
      |
      +-- stage 1  SSB matched filter -> MTI -> range profile -> gate ->
      |            parabolic peak -> pulse-pair Doppler          coarse.py
      |
      +-- resource gate   Renyi entropy of the fine aperture   aperture.py
      |
      +-- stage 2  W/X_est variant -> delay-Doppler map -> zero-Doppler
      |            notch -> coarse Gaussian delay gate -> CA-CFAR ->
      |            DBSCAN -> centroids                 sensing.py, detect.py
      |
      +-- tracker  associate, update, forecast                    track.py
                        |
                        +--> next dwell's gate  (this is the closed loop)

Which parts of Fig. 2 are absent, and why - all four are hardware, not effort:

  * AoA and the range-AoA map. One receive channel, so no spatial aperture.
    The fine map is delay-Doppler.
  * Multiple UEs and the WLS fusion of eq 24. One receiver.
  * The SDT/PRACH uplink reporting path. There is no second node to report to;
    the coarse estimate is handed to the fine stage in-process, which is the
    same information flow with the radio link removed.
  * Cartesian position. Not observable from one bistatic delay-Doppler pair -
    see the module docstring of track.py.

What is present and faithful: the staging itself. Stage 1 runs on signals that
are always transmitted regardless of traffic, is cheap, and is ambiguous in
Doppler at this PRF. Stage 2 runs only on a rich enough allocation, costs the
whole grid, and resolves what stage 1 could not. That is the paper's claim,
and it is testable here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import aperture, clean, clutter, coarse, detect, ofdm, receiver, sensing, track

C = 299792458.0


def map_roi(M: np.ndarray, delay_s: np.ndarray, doppler_hz: np.ndarray,
            noise: float, max_delay_s: float, max_doppler_hz: float):
    """Crop a delay-Doppler map to the physically meaningful region, in dB.

    Returns (dB array [n_delay, n_doppler], delay_m, doppler_hz).

    Cropping rather than decimating. The full map is 2800 x 600 bins, which is
    6.7 MB of float32 per dwell and saturates a socket long before it reaches
    a dashboard; but blind decimation throws away the peak, and the region
    that matters is small anyway. Delay beyond the cyclic prefix cannot carry
    a target at all, and Doppler beyond a few hundred hertz is the
    slot-periodic comb region - see ofdm.slot_rate_hz. What is left is about
    43 by 101 cells.

    Values are dB above *this map's own* noise floor. That is what makes a
    shared colour scale legitimate across variants: handoff section 7 requires
    each map be normalised by sqrt(sum(W)) and reported relative to its own
    noise floor, so absolute magnitudes are not comparable between variants
    but dB-above-own-noise is. Plotting absolute magnitude would show the
    audience an 11 dB scaling artefact instead of a result.
    """
    kd = np.flatnonzero(np.asarray(delay_s) <= max_delay_s)
    kf = np.flatnonzero(np.abs(np.asarray(doppler_hz)) <= max_doppler_hz)
    if kd.size == 0 or kf.size == 0:
        return np.zeros((0, 0), np.float32), np.zeros(0), np.zeros(0)
    sub = np.asarray(M)[np.ix_(kf, kd)].T          # -> [delay, doppler]
    ref = max(float(noise), 1e-30)
    return (20 * np.log10(np.maximum(sub, 1e-30) / ref)).astype(np.float32), \
        np.asarray(delay_s)[kd] * C, np.asarray(doppler_hz)[kf]


@dataclass
class TwoStageResult:
    """One dwell's worth of everything, ready for ZMQ and for scoring."""

    t0: float
    dwell_s: float
    n_frames: int
    variant: str
    coarse: coarse.CoarseResult
    resource: dict
    fine: sensing.MapResult = None
    fine_M: np.ndarray = None        # notched and gated, what CFAR ran on
    fine_noise: float = 0.0
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    gate_used: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)
    admitted: bool = True
    link: dict = field(default_factory=dict)
    resolution_info: dict = field(default_factory=dict)
    maps: dict = field(default_factory=dict)      # variant -> roi payload
    map_axes: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Flat, JSON-safe dict. This is the ZMQ payload the dashboard reads;
        no numpy arrays, so a display crash cannot be a deserialisation bug."""
        c = self.coarse
        d = dict(
            t0=self.t0, dwell_s=self.dwell_s, n_frames=self.n_frames,
            variant=self.variant, admitted=bool(self.admitted),
            coarse=dict(excess_m=c.excess_m(), snr_db=c.snr_db,
                        doppler_hz=c.doppler_hz,
                        doppler_unamb_hz=c.doppler_unamb_hz,
                        half_interval_hz=c.doppler_half_interval_hz,
                        coherence=c.coherence, mti_notch_db=c.mti_notch_db,
                        n_rep=c.n_rep, prf_hz=c.prf_hz,
                        sigma_delay_m=float(c.sigma_delay_s() * C)),
            resource={k: (float(v) if isinstance(v, (int, float, np.floating))
                          else v) for k, v in self.resource.items()},
            gate=self.gate_used, tracking=self.tracking,
            latency_ms=self.latency_ms,
            detections=[dict(excess_m=x.excess_m, doppler_hz=x.doppler_hz,
                             snr_db=x.snr_db, n_cells=x.n_cells)
                        for x in self.detections],
            tracks=self.tracks)
        if self.fine is not None:
            d["fine"] = dict(pk_nf_db=self.fine.pk_nf_db,
                             peak_delay_m=self.fine.peak_delay_s * C,
                             peak_doppler_hz=self.fine.peak_doppler_hz,
                             w_sum=self.fine.w_sum,
                             noise=float(self.fine_noise))
        if self.resolution_info:
            d["resolution"] = self.resolution_info
        if self.link:
            d["link"] = self.link
        if self.maps:
            d["maps"] = self.maps
            d["map_axes"] = self.map_axes
        return d


class TwoStage:
    """Stateful two-stage processor. One instance per receive chain.

    State is the tracker, and the tracker is what makes this coarse-to-fine
    rather than merely two-stage: each dwell's gate comes from the previous
    dwell's forecast, and the Doppler ambiguity of stage 1 is resolved against
    that same forecast.
    """

    def __init__(self, bundle, eph, tracker=None, cfg: dict = None, n_frames: int = 5,
                 variant: str = sensing.PAYLOAD, pad: int = 2,
                 pfa: float = None, n_false_alarms: float = 1.0,
                 guard=(2, 2), train=(4, 8),
                 notch_cells: float = 2.0, notch_guard_cells: float = 1.0,
                 exclude_delay_cells: float = 1.0,
                 doppler_search_hz: float = None, max_detections: int = 8,
                 suppress_sidebands: bool = True, min_det_snr_db: float = 6.0,
                 min_coarse_snr_db: float = 12.0,
                 eps_cells: float = 2.5,
                 min_samples: int = 2, gate_widen: float = 3.0,
                 min_renyi_bits: float = 0.0, max_fragmentation: float = None,
                 sigma_accel: float = 3.0, genie_grid=None,
                 extra_variants=(), export_maps: bool = False,
                 map_doppler_hz: float = 250.0, link_metrics: bool = False,
                 clutter_locus: bool = False,
                 clutter_max_range_m: float = 250.0,
                 clutter_guard_bins: float = 1.0,
                 clean_components: int = 0,
                 clean_delay_cells: float = 1.0,
                 clean_doppler_hz: float = 20.0):
        self.base = bundle
        self.eph = eph                 # receiver.EphemerisTracker: supplies tau0(t)
        self.cfg = cfg if cfg is not None else bundle.cfg
        self.n_frames = int(n_frames)
        self.variant = variant
        self.pad = int(pad)
        self.guard, self.train = tuple(guard), tuple(train)
        self.notch_cells = float(notch_cells)
        self.notch_guard_cells = float(notch_guard_cells)
        self.exclude_delay_cells = float(exclude_delay_cells)
        self.max_detections = int(max_detections)
        self.suppress_sidebands = bool(suppress_sidebands)
        # A secondary floor on top of CFAR. CFAR fixes the false alarm rate
        # against *noise*; it does nothing about the direct path's own point
        # spread function, whose 2-D skirt on a filled aperture still stands
        # ~31 dB above the noise floor at the drone's delay and produced
        # 4-5 dB detections at the zero-Doppler notch edge. Those are neither
        # noise nor targets, and CLEAN - deconvolving the known PSF, named but
        # not built in findings.md section 4 - is the real fix. Until then a
        # floor well below the target and well above the skirt is honest and
        # keeps them out of the tracker.
        self.min_det_snr_db = float(min_det_snr_db)
        # The coarse fallback can confirm a track on its own when the fine
        # stage is rejected or empty, so it needs a stricter floor than the
        # fine stage's: 6 dB was enough for the MTI residue of the illuminator
        # to confirm a track on a target-absent dwell.
        self.min_coarse_snr_db = float(min_coarse_snr_db)
        self.eps_cells = float(eps_cells)
        self.min_samples = int(min_samples)
        self.gate_widen = float(gate_widen)
        self.min_renyi_bits = float(min_renyi_bits)
        self.max_fragmentation = max_fragmentation
        self.genie_grid = genie_grid
        # Additional variants mapped from the *same* dwell. Handoff section 8
        # wants two maps side by side; they are only a comparison if they come
        # from identical received samples, so this is a loop inside one dwell
        # rather than two instances or two runs.
        self.extra_variants = tuple(v for v in extra_variants if v != variant)
        self.export_maps = bool(export_maps)
        self.map_doppler_hz = float(map_doppler_hz)
        self.want_link = bool(link_metrics)
        self.use_clutter_locus = bool(clutter_locus)
        self.clutter_max_range_m = float(clutter_max_range_m)
        self.clutter_guard_bins = float(clutter_guard_bins)
        self.clean_components = int(clean_components)
        self.clean_delay_cells = float(clean_delay_cells)
        self.clean_doppler_hz = float(clean_doppler_hz)
        self.tracker = tracker if tracker is not None else track.MultiTracker(
            sigma_accel=sigma_accel)

        # The dwell bundle: the same file looped n_frames times. Built once,
        # because tiling the reference grid every dwell is the single most
        # expensive avoidable operation in the chain.
        from . import fileio
        self.dwell = (bundle if self.n_frames == 1
                      else fileio.TiledBundle(bundle, self.n_frames))
        self.n_sym = int(self.dwell.cfg["n_sym_total"])
        self.delay_s, _ = sensing.delay_axis(self.cfg, self.pad,
                                             int(self.cfg["n_sc"]))
        self.doppler_hz, _ = sensing.doppler_axis(self.cfg, self.pad, self.n_sym)
        self.dwell_s = self.n_frames * coarse.rep_interval(self.cfg)
        self.fc = float(self.cfg["carrier_hz"])
        self.slot_rate_hz = ofdm.slot_rate_hz(self.cfg)
        # Frame rate: the period of the SSB-versus-PDSCH structural change in
        # slow time, hence the spacing of the Doppler comb around every tap.
        self.frame_rate_hz = float(self.cfg["fs_hz"]) / float(
            self.cfg["frame_len_samples"])
        self._genie_dwell = None

        # Delay cells a detection may occupy: inside the CP, and clear of the
        # direct path. Excluding the first cell is direct-path blanking, which
        # any bistatic receiver has to do; the cost is that a target closer
        # than one delay cell of excess path is invisible, and traj_meta puts
        # the drone's minimum excess at 80 m against a 66.6 m cell, so it is
        # only just outside.
        d_cell = 1.0 / (int(self.cfg["n_sc"]) * float(self.cfg["scs_khz"]) * 1e3)
        self.exclude_delay_s = self.exclude_delay_cells * d_cell
        self.valid = coarse.valid_delays(self.delay_s, self.cfg,
                                         self.exclude_delay_s)

        # Doppler search window. Defaults to half the slot rate, beyond which
        # the aperture's own slot-periodic sidebands live - see
        # ofdm.slot_rate_hz. Without it the strongest "detections" in a clean
        # run were the drone's own sidebands at +-1000 and +-2000 Hz.
        self.notch_half_hz = self.notch_cells * float(
            self.doppler_hz[1] - self.doppler_hz[0]) * self.pad
        # A feature whose true Doppler lies inside the notch does not vanish:
        # the notch is a hard mask, so its clipped residue piles up in the
        # first surviving bin and reads as a detection one cell outside the
        # edge. Measured, moving the notch moves the artefact with it: a
        # 10 Hz notch put a spurious detection at +15.0 Hz, a 20 Hz notch at
        # +25.3 Hz, and a 40 Hz notch - which swallows the drone's own
        # -31.5 Hz - reported the drone at -45.0 Hz with an 18 dB SNR and a
        # 13.5 m/s velocity error. Excluding a guard band just outside the
        # edge turns all of those into an honest non-detection.
        #
        # The cost is real and is a blind band, not free: with the default
        # 2-cell notch and 1-cell guard, excess range rates within +-4.5 m/s
        # of the direct path's are undetectable rather than +-3.0 m/s.
        self.notch_guard_hz = self.notch_half_hz + self.notch_guard_cells * float(
            self.doppler_hz[1] - self.doppler_hz[0]) * self.pad
        self.doppler_search_hz = (0.5 * self.slot_rate_hz
                                  if doppler_search_hz is None
                                  else float(doppler_search_hz))
        self.clutter_model = None
        if self.use_clutter_locus:
            if self.eph is None or not hasattr(self.eph, "traj"):
                raise ValueError("clutter_locus needs ephemeris trajectory metadata")
            self.clutter_model = clutter.model_from_meta(
                self.eph.traj.meta, self.clutter_max_range_m,
                guard_hz=self.clutter_guard_bins
                * abs(float(self.doppler_hz[1] - self.doppler_hz[0])))

        # Pfa is specified as an expected false-alarm count per dwell and
        # converted per cell, because a per-cell Pfa is not portable across
        # map sizes. See detect.pfa_for_count.
        self.n_false_alarms = float(n_false_alarms)
        n_cells = int(self.doppler_hz.size) * int(self.delay_s.size)
        self.pfa = (float(pfa) if pfa is not None
                    else detect.pfa_for_count(n_cells, n_false_alarms, self.pad))

    # -- geometry the caller needs to size a dwell ---------------------------
    def resolution(self) -> dict:
        """Cell sizes, search limits, and what every exclusion costs."""
        d_cell = 1.0 / (int(self.cfg["n_sc"]) * float(self.cfg["scs_khz"]) * 1e3)
        f_cell = float(self.doppler_hz[1] - self.doppler_hz[0]) * self.pad
        return dict(
            delay_cell_m=d_cell * C, doppler_cell_hz=f_cell,
            notch_half_hz=self.notch_half_hz,
            notch_half_mps=self.notch_half_hz * C / self.fc,
            notch_guard_hz=self.notch_guard_hz,
            blind_mps=self.notch_guard_hz * C / self.fc,
            slot_rate_hz=self.slot_rate_hz,
            frame_rate_hz=self.frame_rate_hz,
            doppler_search_hz=self.doppler_search_hz,
            exclude_delay_m=self.exclude_delay_cells * d_cell * C,
            pfa=self.pfa, n_false_alarms=self.n_false_alarms,
            min_det_snr_db=self.min_det_snr_db,
            min_coarse_snr_db=self.min_coarse_snr_db,
            coarse_prf_hz=1.0 / coarse.rep_interval(self.cfg),
            coarse_unamb_hz=0.5 / coarse.rep_interval(self.cfg),
            coarse_unamb_mps=0.5 / coarse.rep_interval(self.cfg) * C / self.fc,
            cp_limit_m=coarse.cp_excess_limit_s(self.cfg) * C,
            dwell_s=self.dwell_s,
            clutter_locus=bool(self.clutter_model),
            clutter_max_range_m=(self.clutter_max_range_m
                                 if self.clutter_model else None),
            clutter_bound_hz=(float(clutter.max_stationary_doppler(
                self.clutter_max_range_m,
                self.clutter_model["los_rate_rad_s"], self.fc))
                if self.clutter_model else None),
            clean_components=self.clean_components)

    def samples_per_dwell(self) -> int:
        return self.n_frames * int(self.cfg["frame_len_samples"])

    def margin_samples(self) -> int:
        """Extra samples `process` needs beyond one dwell.

        Symbol s of a dwell is read at receive index s + tau0*fs, and tau0 is
        the bulk propagation delay, not the walk: 23664 samples at t = 500 s
        of this pass and 40386 at its start. A slice exactly one dwell long
        therefore cannot supply the last symbols, and until
        ofdm.demodulate was made to raise it returned zeros for them without
        saying so.

        Taken as the maximum over the whole trajectory rather than over this
        dwell, so the figure is constant and a streaming buffer can be sized
        once. It costs 41000 samples of buffer, 5.3 ms, against a dwell of
        768000.
        """
        n_fft = int(self.cfg["n_fft"])
        if self.eph is None:
            return n_fft + 64
        t = np.asarray(self.eph.p0.t, float)
        tau_max = float(np.max(self.eph.tau0(t)))
        return int(np.ceil(tau_max * float(self.cfg["fs_hz"]))) + n_fft + 64

    def samples_needed(self) -> int:
        """Length of the slice `process` wants: one dwell plus the margin."""
        return self.samples_per_dwell() + self.margin_samples()

    # -- the flow ------------------------------------------------------------
    def process(self, y: np.ndarray, t0: float, tracker_t: float = None,
                truth=None, gate=None) -> TwoStageResult:
        """One dwell. `y` starts at `t0` and must be `samples_needed()` long -
        one dwell plus `margin_samples()`, because the last symbols of the
        dwell arrive after the bulk propagation delay. Passing exactly one
        dwell now raises rather than silently returning a truncated grid.

        `t0` is the transmit-side time of the first sample, the same clock the
        channel and the tracker use.

        `gate` is an optional externally supplied (tau_pred_s, sigma_s) pair.
        In the GNU Radio flowgraph it is the Gaussian gate that arrives from
        ntnisac.ssb_coarse - the paper's Fig. 2 handoff - and it takes
        precedence over this instance's own tracker. Without it the two-stage
        split is only notional: stage 2 would gate itself from a track it
        built alone, which is not coarse-to-fine, and the coarse message would
        be decorative.
        """
        lat = {}
        w0 = time.perf_counter()

        # Prediction from the running track: the gate, and the Doppler
        # hypothesis that disambiguates stage 1. Both come from the same
        # forecast, which is the paper's arrangement.
        best = self.tracker.best()
        tau_pred = sig_pred = f_pred = None
        if best is not None:
            tau_pred, sig_pred = best.gate_delay_s(
                self.gate_widen, floor_m=self.resolution()["delay_cell_m"])
            f_pred = best.doppler_hz(self.fc)
        if gate is not None:
            tau_pred, sig_pred = float(gate[0]), float(gate[1])

        Y, tm = receiver.demod_tracked(y, self.cfg, self.eph, t0, n_sym=self.n_sym)
        lat["demod"] = 1e3 * (time.perf_counter() - w0)

        # -- stage 1 --------------------------------------------------------
        w = time.perf_counter()
        cr = coarse.process(Y, self.dwell, self.dwell.cfg, self.n_frames,
                            pad=self.pad, tau_pred=tau_pred,
                            sigma_pred_s=sig_pred, f_pred_hz=f_pred,
                            min_excess_s=self.exclude_delay_s)
        lat["coarse"] = 1e3 * (time.perf_counter() - w)

        res = TwoStageResult(
            t0=float(t0), dwell_s=self.dwell_s, n_frames=self.n_frames,
            variant=self.variant, coarse=cr, resource={},
            gate_used=dict(from_track=(None if gate is not None
                                       else (best.tid if best else None)),
                           external=gate is not None,
                           tau_pred_m=(tau_pred * C) if tau_pred else None,
                           sigma_pred_m=(sig_pred * C) if sig_pred else None,
                           f_pred_hz=f_pred),
            tracking={k: (float(v) if isinstance(v, float) else v)
                      for k, v in tm.items()},
            resolution_info=self.resolution())

        # -- resource gate ---------------------------------------------------
        w = time.perf_counter()
        # The channel estimate is the single most expensive step, and every
        # variant plus the link metrics need the same one. Computed once here
        # and passed in, rather than letting each build_variant call recompute
        # it - which is what makes a second map cost a map instead of a map
        # plus a channel estimate.
        ref_grid = self.dwell.ref_grid()
        H = sensing.ls_channel_estimate(Y, self.dwell.ref_mask, ref_grid,
                                        self.dwell.occ_mask)
        nv = sensing.estimate_noise_var(Y, H, self.dwell.ref_mask, ref_grid)
        X, W = sensing.build_variant(Y, self.dwell, self.variant, H=H,
                                     noise_var=nv, genie_grid=self._genie())
        res.resource = aperture.describe(W)
        res.admitted = aperture.gate(W, self.min_renyi_bits,
                                     max_fragmentation=self.max_fragmentation)
        res.resource["admitted"] = bool(res.admitted)
        lat["variant"] = 1e3 * (time.perf_counter() - w)

        if not res.admitted:
            # Paper section IV-A: an allocation too sparse to sense on is
            # skipped rather than processed into a misleading map. The coarse
            # estimate still updates the track, which is the whole reason the
            # cheap stage exists.
            self._update_track(res, cr, tracker_t if tracker_t is not None else t0)
            lat["total"] = 1e3 * (time.perf_counter() - w0)
            res.latency_ms = lat
            return res

        # -- stage 2 ---------------------------------------------------------
        if self.want_link:
            w = time.perf_counter()
            res.link = sensing.link_metrics(
                Y, self.dwell, H=H, noise_var=nv,
                genie_grid=self._genie_always(), dwell_s=self.dwell_s)
            lat["link"] = 1e3 * (time.perf_counter() - w)

        w = time.perf_counter()
        fine, clean_info = self._map_variant(Y, X, W)
        lat["map"] = 1e3 * (time.perf_counter() - w)
        if clean_info:
            res.gate_used["clean"] = clean_info

        w = time.perf_counter()
        if self.clutter_model:
            locus_keep = clutter.keep_mask(
                self.delay_s, fine.doppler_hz, **self.clutter_model)
            M = np.where(locus_keep, fine.M, 0.0)
            noise_cells = locus_keep & self.valid[None, :]
            noise_power = (
                np.median(fine.M[noise_cells].astype(np.float64) ** 2)
                / np.log(2.0) if np.any(noise_cells) else 0.0)
            noise = float(max(float(noise_power), 0.0))
            kept = np.ones(fine.doppler_hz.size, dtype=bool)
        else:
            locus_keep = None
            M, kept = detect.doppler_notch(fine.M, fine.doppler_hz,
                                           self.notch_half_hz)
            noise = detect.map_noise_power(fine.M, kept, self.valid)

        # The coarse Gaussian gate of eq 9, on the shared delay axis. Applied
        # multiplicatively to the map before CFAR, so a target outside it is
        # attenuated rather than deleted.
        #
        # An externally supplied gate is used directly rather than being
        # allowed to steer the internal coarse stage and then reading that
        # stage's gate back out. The indirection would work, but it discards
        # the width the sender computed and substitutes a locally recomputed
        # one - and the width is half of what a gate is.
        if gate is not None:
            g = coarse.gaussian_gate(self.delay_s, gate[0],
                                     self.gate_widen * gate[1])
        else:
            g = cr.to_gate(self.delay_s, widen=self.gate_widen)
        M = M * g[None, :].astype(M.dtype)
        M = M * self.valid[None, :].astype(M.dtype)

        if locus_keep is None:
            searchable = ((kept
                           & (np.abs(fine.doppler_hz) > self.notch_guard_hz)
                           & (np.abs(fine.doppler_hz)
                              <= self.doppler_search_hz))[:, None])
        else:
            searchable = (locus_keep
                          & (np.abs(fine.doppler_hz)
                             <= self.doppler_search_hz)[:, None])
        mask, thr = detect.ca_cfar_2d(M ** 2, guard=self.guard,
                                      train=self.train, pfa=self.pfa)
        mask &= searchable & self.valid[None, :]
        dets = detect.cluster_detections(
            mask, M ** 2, thr, self.delay_s, fine.doppler_hz,
            eps_cells=self.eps_cells, min_samples=self.min_samples,
            cell_r=self.pad, cell_c=self.pad)
        # Reject the known frame-rate aperture comb first, against the
        # un-notched map so a notched static scatterer's sidebands are still
        # attributed to it. Before the SNR floor, deliberately: the comb is a
        # specific artefact with a known cause and its count is worth
        # reporting, while the floor is a blunt catch-all that would otherwise
        # absorb it and hide the mechanism.
        n_comb = 0
        if self.suppress_sidebands and dets:
            dets, dropped = detect.suppress_comb(
                dets, fine.M, self.delay_s, fine.doppler_hz, self.frame_rate_hz)
            n_comb = len(dropped)

        n_weak = sum(1 for d in dets if d.snr_db < self.min_det_snr_db)
        dets = [d for d in dets if d.snr_db >= self.min_det_snr_db]

        # Cap what reaches the tracker. Association is O(tracks x detections)
        # and every unassociated detection seeds a track, so an unexpectedly
        # noisy dwell must not be able to make the next one quadratically
        # slower. Strongest first, so the cap discards the least likely.
        n_raw = len(dets)
        dets = dets[:self.max_detections]
        lat["detect"] = 1e3 * (time.perf_counter() - w)

        res.fine, res.fine_M, res.fine_noise = fine, M, noise
        res.detections = dets
        res.gate_used["n_raw_detections"] = int(n_raw)

        # -- extra variants and map export -----------------------------------
        if self.export_maps or self.extra_variants:
            w = time.perf_counter()
            max_d = coarse.cp_excess_limit_s(self.cfg)
            roi, dly_m, dop = map_roi(fine.M, self.delay_s, fine.doppler_hz,
                                      noise, max_d, self.map_doppler_hz)
            res.map_axes = dict(delay_m=dly_m.tolist(), doppler_hz=dop.tolist())
            if self.export_maps:
                res.maps[self.variant] = self._map_payload(fine, roi, noise)
            for v in self.extra_variants:
                Xv, Wv = sensing.build_variant(Y, self.dwell, v, H=H,
                                               noise_var=nv,
                                               genie_grid=self._genie_always())
                fv, clean_v = self._map_variant(Y, Xv, Wv)
                _, keptv = detect.doppler_notch(fv.M, fv.doppler_hz,
                                                self.notch_half_hz)
                nvv = detect.map_noise_power(fv.M, keptv, self.valid)
                rv, _, _ = map_roi(fv.M, self.delay_s, fv.doppler_hz, nvv,
                                   max_d, self.map_doppler_hz)
                res.maps[v] = self._map_payload(fv, rv, nvv)
                res.resource.setdefault("aperture_by_variant", {})[v] = \
                    aperture.describe(Wv)["renyi_bits_per_symbol"]
                if clean_v:
                    res.resource.setdefault("clean_by_variant", {})[v] = clean_v
            lat["extra"] = 1e3 * (time.perf_counter() - w)
        res.gate_used["n_comb_rejected"] = int(n_comb)
        res.gate_used["n_below_snr_floor"] = int(n_weak)
        res.gate_used["clutter_locus"] = bool(self.clutter_model)

        self._update_track(res, cr, tracker_t if tracker_t is not None else t0,
                           dets=dets)
        lat["total"] = 1e3 * (time.perf_counter() - w0)
        res.latency_ms = lat
        if truth is not None:
            res.tracking["truth"] = truth
        return res

    # -- internals -----------------------------------------------------------
    def _map_payload(self, res_map, roi, noise):
        return dict(db=roi.ravel().tolist(), shape=list(roi.shape),
                    pk_nf_db=float(res_map.pk_nf_db),
                    peak_delay_m=float(res_map.peak_delay_s * C),
                    peak_doppler_hz=float(res_map.peak_doppler_hz),
                    w_sum=float(res_map.w_sum), noise=float(noise))

    def _map_variant(self, Y, X, W):
        """Map one variant, optionally cancelling only the direct component."""
        Z = sensing.form_z(Y, X, W)
        w_sum = float(np.sum(W))
        if self.clean_components <= 0:
            return sensing.map_from_z(Z, w_sum, self.cfg, self.pad), None

        before = sensing.map_from_z(Z, w_sum, self.cfg, self.pad)
        delay_cell = 1.0 / (
            int(self.cfg["n_sc"]) * float(self.cfg["scs_khz"]) * 1e3)
        half_delay = self.clean_delay_cells * delay_cell
        _, components, maps = clean.clean(
            Z, W, self.cfg, n_components=self.clean_components, pad=self.pad,
            search=(max(0.0, before.peak_delay_s - half_delay),
                    before.peak_delay_s + half_delay),
            search_doppler=(before.peak_doppler_hz - self.clean_doppler_hz,
                            before.peak_doppler_hz + self.clean_doppler_hz))
        cancellation = clean.cancellation_db(
            maps["before"], maps["after"], before.peak_delay_s,
            before.peak_doppler_hz)
        return maps["after"], dict(
            n_components=len(components),
            components=[c.as_dict() for c in components],
            **{k: float(v) for k, v in cancellation.items()})

    def _genie_always(self):
        """tx_grid whatever the primary variant is: the genie map and the
        uncoded SER both need it even when the primary branch does not."""
        if self._genie_dwell is None:
            self._genie_dwell = (np.asarray(self.genie_grid)
                                 if self.genie_grid is not None
                                 else np.asarray(self.dwell.grid()))
        return self._genie_dwell

    def _genie(self):
        if self.variant != sensing.GENIE:
            return None
        if self._genie_dwell is None:
            self._genie_dwell = (np.asarray(self.genie_grid)
                                 if self.genie_grid is not None
                                 else np.asarray(self.dwell.grid()))
        return self._genie_dwell

    def _outside_clutter_locus(self, delay_s, doppler_hz):
        if not self.clutter_model:
            return abs(float(doppler_hz)) > self.notch_half_hz
        return bool(clutter.keep_mask(
            np.array([float(delay_s)]), np.array([float(doppler_hz)]),
            **self.clutter_model)[0, 0])

    def _update_track(self, res, cr, t, dets=None):
        """Feed the tracker. Fine detections when there are any, coarse
        otherwise, so the track survives a rejected or empty dwell."""
        zs, Rs = [], []
        f_cell = float(self.doppler_hz[1] - self.doppler_hz[0]) * self.pad
        if dets:
            for d in dets:
                zs.append([d.excess_m, d.range_rate_mps(self.fc)])
                # Sigma from the cell size over sqrt(2*SNR), floored at a
                # tenth of a cell: a 40 dB detection does not really know its
                # delay to 20 cm, and claiming so collapses the gate.
                snr = 10 ** (max(d.snr_db, 0.0) / 10.0)
                k = max(1.0 / np.sqrt(2 * snr), 0.1)
                Rs.append(track.measurement_cov(
                    k * self.resolution()["delay_cell_m"] / C, k * f_cell, self.fc))
        elif (np.isfinite(cr.peak_delay_s)
              and cr.snr_db > self.min_coarse_snr_db
              and cr.excess_m() >= self.exclude_delay_s * C
              and self._outside_clutter_locus(
                  cr.peak_delay_s, cr.doppler_unamb_hz)):
            # The coarse fallback keeps a track alive through a rejected or
            # empty fine stage, but it must apply the same exclusions the fine
            # stage does - direct-path blanking and either the legacy notch or
            # the range-dependent clutter locus - or it confirms tracks on
            # clutter that stage 2 deliberately removed.
            zs.append([cr.excess_m(), cr.range_rate_mps(self.fc)])
            Rs.append(track.measurement_cov(
                cr.sigma_delay_s(), max(cr.doppler_half_interval_hz / 10.0, 1.0),
                self.fc))
        conf = self.tracker.step(float(t), zs, Rs)
        res.tracks = [dict(tid=tr.tid, excess_m=tr.excess_m,
                           range_rate_mps=tr.range_rate_mps, hits=tr.hits,
                           misses=tr.misses,
                           sigma_m=float(tr.sigma()[0]),
                           sigma_mps=float(tr.sigma()[1]))
                      for tr in conf]
