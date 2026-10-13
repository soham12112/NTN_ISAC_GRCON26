#!/usr/bin/env python3
"""The over-the-air receive path: synchronise, cancel the direct path, read
the micro-Doppler of whatever physical object is between the two radios.

This block exists because the conducted chain cannot be pointed at real RF
without two things it does not have.

**It does not know tau0(t).** `ssb_coarse` and `grid_sense` build an
`EphemerisTracker`, which *commands* the pass. That is right over a cable
driven by one process and wrong across two laptops, where the receiver knows
neither when the transmitter started nor how far its reference has drifted.
`ota.OtaTracker` replaces it behind the identical interface, estimating a
constant offset and one fractional clock error on top of the commanded
trajectory.

**A real target is not resolvable in delay.** At 4.5 MHz the delay cell is
66.6 m and an object a metre off a five-metre baseline has 0.39 m of bistatic
excess path - six thousandths of a cell. No bandwidth reachable here changes
that. So the physical target is measured in Doppler alone, at the direct
path's own delay, which makes cancelling the direct path the whole problem
rather than a refinement. findings.md sections 4 and 11 both already asked for
CLEAN; this is where it became load-bearing. Measured on the simulated link
(tests/verify_ota.py V19), a -40 dB co-delay echo reads +0.8 dB against a
target-absent run without cancellation and +54.0 dB with it.

What this block does not do is replace the emulated-target path. `ssb_coarse`
and `grid_sense` still handle the synthetic drone at 640 m, where there is
exact truth, and the good demo runs both at once: a commanded target hundreds
of metres out with known ground truth, and a real one at zero excess path, in
the same capture. Give those two blocks the `tau_bias` this block converges on
and they work over the air unchanged - it is the calibration constant handoff
section 9's V4 asks to be recorded rather than eliminated.
"""

import time

import numpy as np
import pmt
from gnuradio import gr

from .dwell import DwellBlock, add_repo_py, pack, unpack

add_repo_py()


class ota_sense(DwellBlock):
    def __init__(self, wf_dir="", traj_dir="", n_frames=5, t_start=0.0,
                 tau_bias=40.0, variant="payload-aided", pad=2, stride=1,
                 window="blackmanharris", gps_locked=False,
                 n_clean=2, clean_delay_ns=200.0, clean_doppler_hz=15.0,
                 micro_frames=50, micro_every=10, rot_min_hz=3.0,
                 rot_max_hz=40.0, snr_db=10.0, cfo_max_hz=12e3,
                 publish_profile=True, link_metrics=False):
        from ntn_isac import clean as _clean
        from ntn_isac import fileio, ota, sensing, trajectory, udoppler

        self._m = dict(clean=_clean, fileio=fileio, ota=ota, sensing=sensing,
                       udoppler=udoppler)

        self.b = fileio.load(wf_dir)
        self.cfg = self.b.cfg
        self.traj = trajectory.load(traj_dir)
        fs = self.b.fs
        self.n_frames = int(n_frames)
        self.tb = fileio.TiledBundle(self.b, self.n_frames)
        n_dwell = self.n_frames * int(self.cfg["frame_len_samples"])

        # Same sliding-window margin the other sensing blocks use: symbol s of
        # a dwell arrives at index s + tau0*fs, so a slice exactly one dwell
        # long cannot supply the last symbols. findings.md section 15.
        # tau0(t) = tau_leo(t) + d0, and this covered only tau_leo - the
        # commanded excess over r_nominal, plus the bias. d0 is the term
        # _acquire exists to find: the fixed delay through DAC, filters, cable,
        # ADC and pipeline, plus however far apart the two ends actually
        # started. It is unknown when the buffer is sized and it is not small -
        # measured at 32115 samples (4.2 ms) on the first conducted capture,
        # against a margin of 616 - so every dwell failed in demodulate.
        #
        # _acquire resolves d0 only modulo a frame and takes the non-negative
        # representative, so 0 <= d0 < one frame and that is the bound. Carried
        # as margin rather than re-sizing the buffer after acquisition: it
        # costs 20% of a dwell and nothing else, and a buffer that changes
        # length once the graph is running is worse. The same slack absorbs the
        # delay walk within a run, which this figure - taken once at t_start -
        # would otherwise under-cover by ~180 samples per second of capture.
        self.d0_bound = int(self.cfg["frame_len_samples"])
        self.margin = int(np.ceil(
            (self.traj.paths[0].range_at(np.array([t_start]))[0]
             - self.traj.r_nominal) / 299792458.0 * fs)) + \
            int(tau_bias) + int(self.cfg["n_fft"]) + 64 + self.d0_bound

        DwellBlock.__init__(self, "ota_sense", n_dwell, fs, t_start,
                            int(stride), margin=self.margin)

        self.variant = variant
        self.pad = int(pad)
        self.window = window
        self.n_clean = int(n_clean)
        self.clean_delay_s = float(clean_delay_ns) * 1e-9
        self.clean_doppler_hz = float(clean_doppler_hz)
        self.micro_frames = int(micro_frames)
        self.micro_every = max(int(micro_every), 1)
        self.rot_search = (float(rot_min_hz), float(rot_max_hz))
        self.snr_db = float(snr_db)
        self.cfo_max_hz = float(cfo_max_hz)
        self.publish_profile = bool(publish_profile)
        self.link_metrics = bool(link_metrics)

        self.trk = ota.OtaTracker(self.traj, fs, tau_bias_samples=tau_bias,
                                  gps_locked=bool(gps_locked))
        self.locked = False
        self._acq_peak_db = float("nan")
        # Long-dwell buffer for the rotation rate. The tip-speed edge is
        # readable from the 100 ms sensing dwell, but the line spacing is not:
        # 100 ms gives 5 Hz Doppler cells against a 7 rev/s blade's 7 Hz
        # spacing. Measured, a 1 s dwell recovers 7.00 and 11.00 exactly while
        # 100 ms and 500 ms both return a harmonic. So micro-Doppler runs on
        # its own, longer accumulation.
        self._micro_buf = []
        self._micro_n = 0

        for p in ("ota", "micro", "state"):
            self.message_port_register_out(pmt.intern(p))
        self.message_port_register_in(pmt.intern("ctrl"))
        self.set_msg_handler(pmt.intern("ctrl"), self._ctrl)

    # ------------------------------------------------------------------ ctrl

    def _ctrl(self, msg):
        """Runtime control: re-acquire, or force a tau_bias calibration."""
        kind, d = unpack(msg)
        if not isinstance(d, dict):
            return
        if d.get("reacquire"):
            self.locked = False
        if "tau_bias_samples" in d:
            self.trk.bias = float(d["tau_bias_samples"]) / self.fs

    # ----------------------------------------------------------------- sync

    def _acquire(self, y, t0):
        """Coarse (lag, carrier offset), after removing the commanded pass.

        This is not blind acquisition and must not be built as if it were. The
        receiver has traj.csv, so the pass's own Doppler - up to 54.9 kHz at
        2.478 GHz, with 678 Hz/s of rate - is known and is removed here as
        feedforward. What is left is the clock offset alone, which two
        free-running B210 TCXOs cap at about 10 kHz, so the search is +-12 kHz
        rather than +-60 kHz and the grid is 11 points rather than 50.

        Skipping the de-ramp appears to work and is not: at t = 300 s the pass
        is at closest approach and its Doppler passes through zero, so a raw
        correlation locks. At t = 500 s, which is the epoch the demo actually
        uses, the same code reported a 4.7 dB correlation peak and never
        locked, because the signal was 40 kHz outside the search.
        """
        from ntn_isac import receiver
        ota = self._m["ota"]
        n_frame = int(self.cfg["frame_len_samples"])
        if y.size < n_frame:
            return False
        t = t0 + np.arange(n_frame) / self.fs
        yd = receiver.deramp(y[:n_frame], t, self.trk.tau_leo(t), self.trk.fc)
        acq = ota.acquire(yd, self.b.iq(), self.cfg,
                          cfo_max_hz=self.cfo_max_hz)
        self._acq_peak_db = acq.peak_db
        if not acq.locked:
            return False

        # Seed d0 from the lag, against what the commanded trajectory predicts.
        #
        # This is not optional on real hardware, and it is invisible in
        # simulation. The per-dwell tracker reads residual delay as the phase
        # slope of the channel estimate across subcarriers, which wraps at
        # +-n_fft/2 = +-256 samples. Anything larger than that it cannot see,
        # and a real link has a fixed delay through DAC, filters, cable, air,
        # ADC and pipeline that is comfortably hundreds of samples. Left at
        # zero, the tracker would silently lock to the wrong integer symbol.
        #
        # Taken modulo a frame because the waveform is periodic and nothing in
        # the signal distinguishes one repetition from the next. That ambiguity
        # is real - 20 ms of it - and it is why the two flowgraphs share an
        # epoch rather than letting the receiver guess.
        pred = float(self.trk.tau0(np.array([t0]))[0]) * self.fs
        err = acq.lag_samples - pred
        # Non-negative representative, not the symmetric one. d0 is only
        # defined modulo a frame and the transmitted stream is exactly
        # frame-periodic (`loopable`, file_source repeat), so d0 and
        # d0 + n_frame address identical signal - but a negative d0 makes
        # demodulate read before the start of the slice, which is pre-roll and
        # no amount of trailing margin can supply it. Measured: a capture at
        # t=500 wrapped to d0 = -41.7k and every dwell failed on i0 < 0.
        err = err % n_frame
        self.trk.t_ref = float(t0)
        self.trk.eps = 0.0 if self.trk.gps_locked else acq.cfo_hz / self.trk.fc
        self.trk.d0 = err / self.fs
        self._acq_lag = int(acq.lag_samples)
        self._acq_d0 = float(err)
        self.locked = True
        return True

    # ------------------------------------------------------------- payloads

    @staticmethod
    def _profile(res, ref_delay_s, noise_ref, fd_limit=500.0):
        """Level against Doppler at the direct path's delay, dB over noise."""
        from ntn_isac import udoppler
        fd, lvl = udoppler.doppler_profile(res, ref_delay_s)
        k = np.abs(fd) <= fd_limit
        return dict(
            doppler_hz=fd[k].astype(np.float32).tolist(),
            level_db=(20 * np.log10(np.maximum(lvl[k], 1e-30)
                                    / max(noise_ref, 1e-30))
                      ).astype(np.float32).tolist())

    def _crop(self, res, ref, fd_limit=260.0, half_cells=10):
        """A small delay-Doppler image around the direct path, dB over noise.

        Cropped rather than sent whole for the same reason grid_sense crops
        its own: the full map is 2800 x 600 and a display cannot use it, while
        a payload that large is what broke the message transport the first
        time (findings.md section 15). Referenced to the UNcancelled map's
        noise, so before and after share a colour scale - a cancelled map
        normalised by its own median looks improved even when nothing moved.
        """
        ds, fd = res.delay_s, res.doppler_hz
        j = int(np.argmin(np.abs(ds - ref.peak_delay_s)))
        j0, j1 = max(j - half_cells, 0), min(j + half_cells + 1, res.M.shape[1])
        k = np.abs(fd) <= fd_limit
        sub = res.M[np.ix_(k, np.arange(j0, j1))]
        db = 20 * np.log10(np.maximum(sub, 1e-30) / max(ref.noise, 1e-30))
        C = 299792458.0
        return dict(
            db=db.astype(np.float32).ravel().tolist(),
            shape=[int(db.shape[0]), int(db.shape[1])],
            doppler_hz=[float(fd[k][0]), float(fd[k][-1])],
            excess_m=[float((j0 - j) * (ds[1] - ds[0]) * C),
                      float((j1 - 1 - j) * (ds[1] - ds[0]) * C)])

    # ---------------------------------------------------------------- dwell

    def on_dwell(self, y, t0, index):
        m = self._m
        wall = time.time()

        if not self.locked and not self._acquire(y, t0):
            self.message_port_pub(pmt.intern("state"), pack("ota_state", dict(
                t=t0, locked=False, acq_peak_db=self._acq_peak_db)))
            return

        from ntn_isac import receiver
        n_sym = self.n_frames * int(self.cfg["n_sym_total"])
        Y, st = receiver.demod_tracked(y, self.cfg, self.trk, t0, n_sym=n_sym)

        H = m["sensing"].ls_channel_estimate(Y, self.tb.ref_mask,
                                             self.tb.ref_grid(),
                                             self.tb.occ_mask)
        d_res, f_res, a_d, a_f = m["ota"].residual_from_H(H, self.tb.occ_mask,
                                                          self.cfg)
        self.trk.update(d_res, f_res, t0)

        nv = m["sensing"].estimate_noise_var(Y, H, self.tb.ref_mask,
                                             self.tb.ref_grid())
        X, W = m["sensing"].build_variant(Y, self.tb, self.variant, H=H,
                                          noise_var=nv)
        Z = m["sensing"].form_z(Y, X, W)
        w_sum = float(W.sum())

        before = m["sensing"].map_from_z(Z, w_sum, self.cfg, self.pad,
                                         self.window)
        # Centre the search on where the direct path actually is, not on where
        # the tracker believes it should be. The two agree once the loop has
        # pulled in, and until then they do not: with the window pinned to zero
        # Doppler, a direct path still tens of hertz out during acquisition
        # fell outside it and CLEAN reported 0.0 dB of cancellation on every
        # dwell - a canceller that quietly does nothing is worse than one that
        # fails, because the map still looks plausible.
        f0 = float(before.peak_doppler_hz)
        d0 = float(before.peak_delay_s)
        Zc, comps, maps = m["clean"].clean(
            Z, W, self.cfg, n_components=self.n_clean, pad=self.pad,
            window=self.window,
            search=(d0 - self.clean_delay_s, d0 + self.clean_delay_s),
            search_doppler=(f0 - self.clean_doppler_hz,
                            f0 + self.clean_doppler_hz))
        can = m["clean"].cancellation_db(maps["before"], maps["after"],
                                         before.peak_delay_s,
                                         before.peak_doppler_hz)

        # Micro-Doppler on the sensing dwell gives the tip-speed edge; the
        # rotation rate needs the long accumulation below.
        md = m["udoppler"].analyse(
            maps["after"], self.cfg, before.peak_delay_s,
            noise_ref=before.noise, fc=self.b.fc, snr_db=self.snr_db,
            rot_search_hz=self.rot_search)

        out = dict(
            t0=t0, index=int(index), variant=self.variant,
            wall_utc=wall, latency_ms=dict(total=1e3 * (time.time() - wall)),
            locked=True, acq_peak_db=self._acq_peak_db,
            tracker=self.trk.state(),
            residual=dict(delay_samples=d_res, doppler_hz=f_res,
                          delay_corr=a_d, doppler_corr=a_f),
            direct=dict(pk_nf_db=before.pk_nf_db,
                        delay_s=before.peak_delay_s,
                        doppler_hz=before.peak_doppler_hz, **can),
            clean=[c.as_dict() for c in comps],
            micro=md.as_dict(),
        )
        if self.publish_profile:
            # Both profiles, not just the cancelled one. A display that shows
            # only the result cannot show the thing this whole path exists to
            # do: the target is not hiding in the noise, it is hiding under the
            # direct path, and that is only visible as a before-and-after.
            out["profile"] = self._profile(maps["after"], before.peak_delay_s,
                                           before.noise)
            out["profile_before"] = self._profile(maps["before"],
                                                  before.peak_delay_s,
                                                  before.noise)
            out["map_before"] = self._crop(maps["before"], before)
            out["map_after"] = self._crop(maps["after"], before)
        if self.link_metrics:
            out["link"] = m["sensing"].link_metrics(
                Y, self.tb, H=H, noise_var=nv,
                dwell_s=self.n_dwell / self.fs)
        self.message_port_pub(pmt.intern("ota"), pack("ota", out))
        # Merged, not splatted: the tracker's state already carries `t` from
        # its last update, and dict(t=..., **state) is a TypeError rather than
        # a later-wins update.
        stm = dict(self.trk.state())
        stm.update(t=t0, locked=True, index=int(index))
        self.message_port_pub(pmt.intern("state"), pack("ota_state", stm))

        # -- the long accumulation ------------------------------------------
        self._micro_buf.append(Zc)
        self._micro_n += 1
        if self._micro_n >= self.micro_every and \
                len(self._micro_buf) * self.n_frames >= self.micro_frames:
            self._publish_micro(t0, w_sum)
            self._micro_buf, self._micro_n = [], 0

    def _publish_micro(self, t0, w_sum):
        """Rotation rate from a dwell long enough to resolve the line spacing.

        The cancelled Z of consecutive dwells is concatenated along the symbol
        axis, which is legitimate only because the dwell boundary is a
        transmit-time boundary: `DwellBlock` advances by exactly one dwell on
        the transmit clock and `demod_tracked` does the retiming inside, so
        symbol l of dwell k+1 follows symbol l_max of dwell k with no gap.
        """
        m = self._m
        Z = np.concatenate(self._micro_buf, axis=0)
        cfg = dict(self.cfg)
        res = m["sensing"].map_from_z(Z, w_sum, cfg, self.pad, self.window)
        md = m["udoppler"].analyse(res, cfg, res.delay_s[0], fc=self.b.fc,
                                   snr_db=self.snr_db,
                                   rot_search_hz=self.rot_search)
        d = md.as_dict()
        d.update(t0=t0, n_frames=int(Z.shape[0] / int(cfg["n_sym_total"])),
                 dwell_s=float(Z.shape[0]) / int(cfg["n_sym_total"])
                 * int(cfg["frame_len_samples"]) / self.fs)
        if self.publish_profile:
            keep = np.abs(md.doppler_hz) <= 500.0
            d["profile"] = dict(
                doppler_hz=md.doppler_hz[keep].astype(np.float32).tolist(),
                level_db=md.level_db[keep].astype(np.float32).tolist())
        self.message_port_pub(pmt.intern("micro"), pack("micro", d))
