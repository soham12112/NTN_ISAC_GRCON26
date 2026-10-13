#!/usr/bin/env python3
"""Stage 1 of the two-stage ISAC flow: coarse detection from the SSB."""

import numpy as np
import pmt
from gnuradio import gr

from .dwell import DwellBlock, add_repo_py, pack, unpack

add_repo_py()


class ssb_coarse(DwellBlock):
    """SSB matched filter, MTI, range profile, pulse-pair Doppler, tracker.

    The first stage of the paper's Fig. 2 flowgraph. It runs on the SSB, which
    is transmitted regardless of what the scheduler is doing, so this block
    produces an estimate every dwell whether or not there is any PDSCH traffic
    to sense on. That independence is the reason the paper has two stages at
    all, and it is the reason this block - not the fine one - owns the tracker.

    Message ports:

      coarse   out   per-dwell coarse estimate, plus the Gaussian gate that
                     stage 2 should apply (paper eq 9) and the tracker's
                     current state and forecast
      refine   in    refined measurements from ntnisac.grid_sense, fed to the
                     tracker in place of the coarse ones when available

    The refine loop is what closes the coarse-to-fine cycle, and it also
    resolves the coarse stage's Doppler ambiguity. One SSB per 20 ms frame is
    a 50 Hz pulse-pair PRF, unambiguous only to +-25 Hz, and traj_meta puts
    the drone's residual Doppler between -36.7 and +77.1 Hz. Measured, the raw
    coarse Doppler is wrong by exactly one PRF; unwrapped against this
    tracker's forecast it is within 0.03 Hz. There is no processing fix at
    this PRF - only prediction - which is why the tracker lives inside stage 1
    rather than downstream of everything.

    Cost: about 82 ms of a 100 ms dwell in this Python implementation, of
    which 80 ms is the OFDM demodulation and 1.4 ms is the coarse chain
    itself. So stage 1 alone keeps up with the stream; stage 2 does not, which
    is exactly the asymmetry the two-stage architecture is built around.
    """

    # Range-rate uncertainty to seed a track with while the coarse Doppler is
    # ambiguous. Wide enough that no alias is excluded a priori.
    BOOTSTRAP_RATE_MPS = 40.0

    # An unwrap is only trusted once the track's own range-rate sigma is well
    # inside half an alias interval. One alias is PRF*lambda = 7.5 m/s here,
    # so half is 3.75 m/s and this is the 3-sigma point.
    UNWRAP_SIGMA_FRACTION = 1.0 / 6.0

    def __init__(self, wf_dir="", traj_dir="", n_frames=5, t_start=0.0,
                 tau_bias_samples=40.0, pad=2, stride=1, gate_widen=3.0,
                 sigma_accel=3.0, min_coarse_snr_db=12.0,
                 exclude_delay_cells=1.0, notch_cells=2.0):
        from ntn_isac import coarse, fileio, receiver, sensing, track, trajectory

        self._m = dict(coarse=coarse, sensing=sensing, track=track)
        self.bundle = fileio.load(wf_dir)
        self.cfg = self.bundle.cfg
        fs = float(self.cfg["fs_hz"])
        self.n_frames = int(n_frames)
        n_dwell = self.n_frames * int(self.cfg["frame_len_samples"])

        self.traj = trajectory.load(traj_dir)
        self.eph = receiver.EphemerisTracker(self.traj, fs,
                                             tau_bias_samples=tau_bias_samples)
        # One dwell of samples is not enough: the last symbols arrive after
        # the bulk propagation delay. See TwoStage.margin_samples().
        import numpy as _np
        _t = _np.asarray(self.eph.p0.t, float)
        margin = (int(_np.ceil(float(_np.max(self.eph.tau0(_t))) * fs))
                  + int(self.cfg["n_fft"]) + 64)
        DwellBlock.__init__(self, "ssb_coarse", n_dwell, fs, t_start, stride,
                            margin=margin)
        self.receiver = receiver
        self.dwell = (self.bundle if self.n_frames == 1
                      else fileio.TiledBundle(self.bundle, self.n_frames))
        self.n_sym = int(self.dwell.cfg["n_sym_total"])
        self.pad = int(pad)
        self.gate_widen = float(gate_widen)
        self.min_coarse_snr_db = float(min_coarse_snr_db)
        self.fc = float(self.cfg["carrier_hz"])

        self.delay_s, _ = sensing.delay_axis(self.cfg, self.pad,
                                             int(self.cfg["n_sc"]))
        d_cell = 1.0 / (int(self.cfg["n_sc"]) * float(self.cfg["scs_khz"]) * 1e3)
        self.exclude_delay_s = float(exclude_delay_cells) * d_cell
        dop, _ = sensing.doppler_axis(self.cfg, self.pad, self.n_sym)
        self.notch_half_hz = float(notch_cells) * float(dop[1] - dop[0]) * self.pad
        self.tracker = track.MultiTracker(sigma_accel=float(sigma_accel))
        self.d_cell_m = d_cell * 299792458.0
        # One pulse-pair alias, expressed as excess range rate: PRF * lambda.
        self.alias_mps = (1.0 / coarse.rep_interval(self.cfg)) * 299792458.0 / self.fc

        self.port_out = pmt.intern("coarse")
        self.port_in = pmt.intern("refine")
        self.message_port_register_out(self.port_out)
        self.message_port_register_in(self.port_in)
        self.set_msg_handler(self.port_in, self._on_refine)

    # -- the fine stage reporting back ---------------------------------------
    def _on_refine(self, msg):
        """Take stage 2's measurements into the tracker.

        Stage 2's estimates are better than stage 1's by construction - full
        occupied bandwidth instead of the SSB's 240 subcarriers, and a Doppler
        FFT over the whole dwell instead of a 50 Hz pulse pair - so when they
        arrive they replace rather than supplement.
        """
        kind, d = unpack(msg)
        if not d:
            gr.log.warn("ssb_coarse: unreadable refine message")
            return
        try:
            t = float(d["t"])
            zs, ss = d["z"], d["sigma"]
            if not zs or not np.isfinite(t):
                return
            track = self._m["track"]
            Rs = [track.measurement_cov(float(s[0]), float(s[1]), self.fc)
                  for s in ss]
            self.tracker.step(t, [[float(z[0]), float(z[1])] for z in zs], Rs)
        except Exception as exc:                       # noqa: BLE001
            gr.log.warn(f"ssb_coarse: bad refine message: {exc}")

    # -- per dwell ------------------------------------------------------------
    def on_dwell(self, y, t0, index):
        coarse = self._m["coarse"]

        best = self.tracker.best()
        tau_pred = sig_pred = f_pred = None
        rate_known = False
        if best is not None:
            tau_pred, sig_pred = best.gate_delay_s(self.gate_widen,
                                                   floor_m=self.d_cell_m)
            # Only offer a Doppler prediction once the range rate is actually
            # determined. Offering one before that does active harm: a track
            # seeded at zero rate predicts zero Doppler, the unwrap then
            # snaps the measurement to the alias nearest zero - which is the
            # wrong one - and the result looks confident. Measured symptom: a
            # permanent +50.00 Hz error with a track stuck at h1.
            rate_known = (float(best.sigma()[1])
                          < self.UNWRAP_SIGMA_FRACTION * self.alias_mps)
            if rate_known:
                f_pred = best.doppler_hz(self.fc)

        Y, tm = self.receiver.demod_tracked(y, self.cfg, self.eph, t0,
                                            n_sym=self.n_sym)
        cr = coarse.process(Y, self.dwell, self.dwell.cfg, self.n_frames,
                            pad=self.pad, tau_pred=tau_pred,
                            sigma_pred_s=sig_pred, f_pred_hz=f_pred,
                            min_excess_s=self.exclude_delay_s)

        # Feed the tracker from the coarse stage when the fine stage has not.
        # Two exclusions, matching stage 2's, so the fallback cannot confirm a
        # track on something stage 2 removes by design: the illuminator
        # (blanked in delay - measured, its 18-21 dB MTI residue otherwise
        # confirms a track on a target-absent dwell) and static clutter
        # (inside the zero-Doppler notch).
        #
        # The Doppler test is applied only once there is a prediction to
        # unwrap against. Applying it to the raw ambiguous value deadlocks:
        # the drone's true -31.5 Hz aliases to +18.5 Hz, which is inside the
        # +-20 Hz notch, so the measurement is refused, so no track forms, so
        # nothing ever unwraps it - measured as a permanent +50.00 Hz error.
        track = self._m["track"]
        usable = (np.isfinite(cr.peak_delay_s)
                  and cr.snr_db > self.min_coarse_snr_db
                  and cr.excess_m() >= self.exclude_delay_s * 299792458.0)
        # The static-clutter test can only be applied to a Doppler that has
        # been unwrapped, and until the rate is known none has. Applying it to
        # the raw alias deadlocks the bootstrap: the drone's true -31.5 Hz
        # aliases to +18.5 Hz, inside the +-20 Hz notch, so the measurement is
        # refused, so no rate is ever learned, so nothing ever unwraps it.
        if rate_known:
            usable = usable and abs(cr.doppler_unamb_hz) > self.notch_half_hz

        # The tracker is stepped every dwell, with an empty measurement list
        # when nothing is usable. Skipping the step entirely leaves existing
        # tracks unpredicted and, worse, never increments their miss count -
        # measured on a target-absent run, one marginal coarse peak seeded a
        # track that then sat at h1 for the whole pass because no later dwell
        # ever aged it out.
        zs, Rs = [], []
        if usable and (best is None or best.t < t0 - 1e-9):
            if not rate_known:
                # Bootstrap, and it keeps running until the rate converges.
                # The ambiguous Doppler carries no usable rate information, so
                # the rate is measured as zero with a variance wide enough to
                # cover any alias - which makes its Kalman gain negligible, so
                # the filter recovers the rate from the *sequence* of delay
                # residuals through the cross-covariance instead. At the
                # measurement sigma of ~1.2 m and a 100 ms dwell that converges
                # inside the alias interval after a handful of dwells, and
                # only then does the unwrap have something to work with.
                zs = [[cr.excess_m(), 0.0]]
                Rs = [track.measurement_cov(
                    cr.sigma_delay_meas_s(),
                    self.BOOTSTRAP_RATE_MPS * self.fc / 299792458.0, self.fc)]
            else:
                zs = [[cr.excess_m(), cr.range_rate_mps(self.fc)]]
                Rs = [track.measurement_cov(
                    cr.sigma_delay_meas_s(),
                    max(cr.doppler_half_interval_hz / 10.0, 1.0), self.fc)]
        self.tracker.step(t0, zs, Rs)

        self.message_port_pub(self.port_out,
                              pack("coarse", self._payload(cr, t0, index, tm)))

    def _payload(self, cr, t0, index, tm):
        best = self.tracker.best()
        gate = dict(tau_s=cr.peak_delay_s, sigma_s=cr.sigma_delay_s(),
                    widen=self.gate_widen, valid=bool(np.isfinite(cr.peak_delay_s)))
        out = dict(
            t=t0, dwell_index=index, dwell_s=self.n_dwell / self.fs,
            n_frames=self.n_frames, stage=1,
            excess_m=cr.excess_m(), snr_db=cr.snr_db,
            doppler_hz=cr.doppler_hz, doppler_unamb_hz=cr.doppler_unamb_hz,
            half_interval_hz=cr.doppler_half_interval_hz,
            coherence=cr.coherence, mti_notch_db=cr.mti_notch_db,
            n_rep=cr.n_rep, prf_hz=cr.prf_hz,
            sigma_delay_m=cr.sigma_delay_s() * 299792458.0,
            range_rate_mps=cr.range_rate_mps(self.fc),
            gate=gate, tracking=tm,
            n_processed=self.n_processed, n_skipped=self.n_skipped)
        if best is not None:
            xs, Ps = best.forecast(self.n_dwell / self.fs, 5, 3.0)
            out["confirmed"] = bool(best.confirmed)
            out["rate_known"] = bool(
                float(best.sigma()[1]) < self.UNWRAP_SIGMA_FRACTION * self.alias_mps)
            out["track"] = dict(
                tid=best.tid, excess_m=best.excess_m,
                range_rate_mps=best.range_rate_mps,
                sigma_m=float(best.sigma()[0]), sigma_mps=float(best.sigma()[1]),
                hits=best.hits, misses=best.misses, confirmed=best.confirmed,
                doppler_hz=best.doppler_hz(self.fc),
                forecast_m=[float(x[0]) for x in xs],
                forecast_sigma_m=[float(np.sqrt(P[0, 0])) for P in Ps])
        return out
