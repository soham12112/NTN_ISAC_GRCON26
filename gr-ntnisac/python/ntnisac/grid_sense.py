#!/usr/bin/env python3
"""Stage 2 of the two-stage ISAC flow: fine sensing on the full grid."""

import numpy as np
import pmt
from gnuradio import gr

from .dwell import DwellBlock, add_repo_py, pack, unpack

add_repo_py()


class grid_sense(DwellBlock):
    """Gated delay-Doppler sensing on the PDSCH grid, with CFAR and clustering.

    The second stage of the paper's Fig. 2 flowgraph. Reconstructing data
    symbols as a passive-radar reference is prior art; this block tests the
    project's contribution, which is how an NR scheduler changes that sensing
    aperture from one dwell to the next. `variant` selects which reference:

      reference-only   W = ref_mask   X_est = the known reference values
      payload-aided    W = occ_mask   X_est = references + sliced decisions
      genie            W = occ_mask   X_est = tx_grid

    One code path, one flag - handoff section 7. The genie branch exists to
    preempt any accusation of cheating in the payload-aided one.

    Message ports:

      coarse   in    stage 1's estimate and Gaussian gate (paper eq 9)
      sense    out   the fine result: map metrics, detections, resource gate
      refine   out   measurements for stage 1's tracker, closing the loop

    Two departures from the paper's section IV-B worth knowing before reading
    the payload. The map is delay-Doppler, not range-AoA, because this block
    consumes one receive channel and there is no spatial aperture to beamform
    over. The clutter null is applied in the Doppler domain rather than as an
    adjacent-symbol difference: at a 71.35 us symbol period an
    adjacent-symbol MTI is a high-pass with its first null at 14 kHz, which
    would put every target in this scenario - tens of hertz - 37 dB down.

    `extra_variants` maps additional variants from the *same* dwell, which is
    what handoff section 8's two side-by-side maps require: two maps are only
    a comparison if they come from identical received samples. The channel
    estimate is shared across all of them, so a second map costs a map
    (~100 ms) rather than a map plus a channel estimate.

    Cost: about 400 ms of a 100 ms dwell here, 600 ms with a second variant
    and the link metrics, so `stride` exists. Set it to 4 or more for a live
    flowgraph and the block processes one dwell in four by
    decision rather than dropping buffers at random. Replay and the offline
    campaign run at stride 1.
    """

    def __init__(self, wf_dir="", traj_dir="", n_frames=5, t_start=0.0,
                 tau_bias_samples=40.0, variant="payload-aided", pad=2,
                 stride=4, n_false_alarms=1.0, notch_cells=2.0,
                 exclude_delay_cells=1.0, min_det_snr_db=6.0,
                 min_renyi_bits=0.0, max_fragmentation=-1.0,
                 suppress_sidebands=True, publish_maps=False,
                 extra_variants="", link_metrics=False,
                 map_doppler_hz=250.0, clutter_locus=False,
                 clutter_max_range_m=250.0, clutter_guard_bins=1.0,
                 clean_components=0, clean_delay_cells=1.0,
                 clean_doppler_hz=20.0):
        from ntn_isac import fileio, receiver, trajectory, twostage

        self.bundle = fileio.load(wf_dir)
        self.cfg = self.bundle.cfg
        fs = float(self.cfg["fs_hz"])
        n_dwell = int(n_frames) * int(self.cfg["frame_len_samples"])
        self.traj = trajectory.load(traj_dir)
        eph = receiver.EphemerisTracker(self.traj, fs,
                                        tau_bias_samples=tau_bias_samples)
        # `max_fragmentation` is a float on the GRC side because GRC has no
        # optional-float type; negative means "no pattern gate", which is the
        # paper's own criterion.
        frag = None if float(max_fragmentation) < 0 else float(max_fragmentation)
        # Comma-separated on the GRC side because GRC has no list type.
        extras = [v.strip() for v in str(extra_variants).split(",") if v.strip()]
        self.ts = twostage.TwoStage(
            self.bundle, eph, n_frames=int(n_frames), variant=variant,
            pad=int(pad), n_false_alarms=float(n_false_alarms),
            notch_cells=float(notch_cells),
            exclude_delay_cells=float(exclude_delay_cells),
            min_det_snr_db=float(min_det_snr_db),
            min_renyi_bits=float(min_renyi_bits), max_fragmentation=frag,
            suppress_sidebands=bool(suppress_sidebands),
            extra_variants=extras, export_maps=bool(publish_maps),
            link_metrics=bool(link_metrics),
            map_doppler_hz=float(map_doppler_hz),
            clutter_locus=bool(clutter_locus),
            clutter_max_range_m=float(clutter_max_range_m),
            clutter_guard_bins=float(clutter_guard_bins),
            clean_components=int(clean_components),
            clean_delay_cells=float(clean_delay_cells),
            clean_doppler_hz=float(clean_doppler_hz))
        DwellBlock.__init__(self, "grid_sense", n_dwell, fs, t_start, stride,
                            margin=self.ts.margin_samples())
        self.publish_maps = bool(publish_maps)
        self.fc = float(self.cfg["carrier_hz"])
        self._gate = None

        self.port_in = pmt.intern("coarse")
        self.port_sense = pmt.intern("sense")
        self.port_refine = pmt.intern("refine")
        self.message_port_register_in(self.port_in)
        self.message_port_register_out(self.port_sense)
        self.message_port_register_out(self.port_refine)
        self.set_msg_handler(self.port_in, self._on_coarse)

    def resolution(self):
        return self.ts.resolution()

    # -- stage 1's gate arriving ---------------------------------------------
    def _on_coarse(self, msg):
        """Latch the most recent Gaussian gate from stage 1.

        Latched rather than queued. If stage 2 is running at stride 4 there
        are three coarse messages it will never use, and the useful one is
        always the newest: a gate is a prediction, and a stale prediction is
        worse than none once the target has walked a delay cell, which at
        170 samples/s of delay walk takes about 10 ms.
        """
        kind, d = unpack(msg)
        if not d:
            gr.log.warn("grid_sense: unreadable coarse message")
            return
        try:
            g = d.get("gate") or {}
            if g.get("valid") and np.isfinite(g["tau_s"]) and np.isfinite(g["sigma_s"]):
                self._gate = (float(g["tau_s"]), float(g["sigma_s"]))
        except Exception as exc:                       # noqa: BLE001
            gr.log.warn(f"grid_sense: bad coarse message: {exc}")

    # -- per dwell ------------------------------------------------------------
    def on_dwell(self, y, t0, index):
        # Use stage 1's gate when one has arrived. Without this the block
        # would gate itself from its own tracker and the "coarse" port would
        # be decorative - the flow would be two-stage but not coarse-to-fine.
        r = self.ts.process(y, t0, tracker_t=t0, gate=self._gate)
        payload = r.summary()
        payload.update(stage=2, dwell_index=index,
                       n_processed=self.n_processed, n_skipped=self.n_skipped,
                       stride=self.stride,
                       coarse_gate_m=(self._gate[0] * 299792458.0
                                      if self._gate else None))
        # Maps and link metrics are carried by TwoStage.summary() when the
        # processor was asked for them; nothing extra to do here. They are off
        # by default because a map costs ~100 ms and the payload grows to
        # ~170 kB - both fine for a dashboard at a stride, wasteful otherwise.
        self.message_port_pub(self.port_sense, pack("sense", payload))

        if r.detections:
            res = self.ts.resolution()
            f_cell = res["doppler_cell_hz"]
            zs, sg = [], []
            for d in r.detections:
                snr = 10 ** (max(d.snr_db, 0.0) / 10.0)
                k = max(1.0 / np.sqrt(2 * snr), 0.1)
                zs.append([d.excess_m, d.range_rate_mps(self.fc)])
                sg.append([k * res["delay_cell_m"] / 299792458.0, k * f_cell])
            self.message_port_pub(self.port_refine, pack(
                "refine", dict(t=float(t0), z=zs, sigma=sg)))
