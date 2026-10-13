#!/usr/bin/env /usr/bin/python3
"""The two-stage ISAC flowgraph (paper Fig. 2), as a GNU Radio graph.

    File Source (tx_iq.cf32, repeat)
        |
    LEO Channel  (C++, ntnisac.leo_channel)       <- traj.csv
        |
    Es/N0 Noise  (ntnisac.esn0_noise)
        |
    LEO Truth    (ntnisac.leo_truth) --msg "truth"------+
        |                                               |
        +--> SSB Coarse   (stage 1) --msg "coarse"--+   |
        |         ^                                 |   |
        |         | msg "refine"                    v   v
        +--> Grid Sense   (stage 2) --msg "sense"--> ZMQ PUB --> dashboard
                  |                                 |
                  +---------------------------------+

Three sources, one channel, two stages, one publisher. The message loop
between the stages is the coarse-to-fine cycle: stage 1 hands stage 2 a
Gaussian range gate, stage 2 hands back refined measurements for stage 1's
tracker, and that is also what resolves stage 1's 50 Hz Doppler ambiguity.

Modes:

  --mode sim      file -> channel -> noise -> sensing, no radios at all.
                  This is the mode that runs today and the one the offline
                  campaign uses.
  --mode replay   read a recorded capture instead of the channel, so the
                  identical processing chain runs on real RF. Handoff
                  section 8 wants this by day 12.
  --mode live     USRP source. Needs the two-radio cabled path of section 9,
                  which is written but has not yet been exercised on the
                  available B210 pair.

Nothing has been transmitted. In sim and replay this graph touches no radio.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pmt
from gnuradio import blocks, gr, ntnisac

import common

try:
    from gnuradio import zeromq
    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False


class msg_printer(gr.sync_block):
    """Terminal sink for the sensing messages, and the scorer.

    Exists because the dashboard does not, and because a run with no console
    output is impossible to debug. It also scores against the truth messages,
    which is the only reason the truth port is on the graph at all in sim mode.
    """

    def __init__(self, quiet=False):
        gr.sync_block.__init__(self, name="msg_printer", in_sig=None, out_sig=None)
        self.quiet = quiet
        self.truth = None
        self.log = []
        for p in ("coarse", "sense", "truth"):
            self.message_port_register_in(pmt.intern(p))
        self.set_msg_handler(pmt.intern("coarse"), self._coarse)
        self.set_msg_handler(pmt.intern("sense"), self._sense)
        self.set_msg_handler(pmt.intern("truth"), self._truth)

    @staticmethod
    def _read(msg):
        """Payloads arrive as UTF-8 JSON in a u8vector - see ntnisac.dwell.pack."""
        try:
            v = pmt.dict_ref(msg, pmt.intern("data"), pmt.PMT_NIL)
            if not pmt.is_null(v):
                return json.loads(bytes(pmt.u8vector_elements(v)).decode())
            return pmt.to_python(msg)
        except Exception:                              # noqa: BLE001
            return None

    def _truth(self, msg):
        self.truth = self._read(msg)

    def _drone(self):
        t = self.truth or {}
        return t.get("p2_excess_m"), t.get("p2_residual_doppler_hz")

    def _coarse(self, msg):
        d = self._read(msg)
        if not d:
            return
        self.log.append(("coarse", d))
        if self.quiet:
            return
        m, fd = self._drone()
        err = "" if m is None else f"  err {d['excess_m'] - m:+7.1f} m"
        fde = "" if fd is None else f"  fd err {d['doppler_unamb_hz'] - fd:+7.2f} Hz"
        tk = d.get("track")
        # The confirmation flag is printed because an unconfirmed track is
        # shown too: on a target-absent run a marginal coarse peak can seed
        # one, and a line that looked identical to a real track was
        # misleading.
        trk = ("" if not tk else
               f"  track {tk['tid']} {tk['excess_m']:.1f}+-{tk['sigma_m']:.1f} m "
               f"{tk['range_rate_mps']:+.2f} m/s h{tk['hits']}"
               f"{'' if tk.get('confirmed') else ' (tentative)'}")
        print(f"[1] t={d['t']:8.3f} {d['excess_m']:8.1f} m  SNR {d['snr_db']:5.1f} dB "
              f"notch {d['mti_notch_db']:5.1f} dB{err}{fde}{trk}")

    def _sense(self, msg):
        d = self._read(msg)
        if not d:
            return
        self.log.append(("sense", d))
        if self.quiet:
            return
        m, fd = self._drone()
        lat = (d.get("latency_ms") or {}).get("total")
        head = (f"[2] t={d['t0']:8.3f} {d['variant']:<14} admitted={d['admitted']} "
                f"{len(d.get('detections') or [])} det"
                + (f" lat {lat:.0f} ms" if lat is not None else ""))
        if d.get("fine"):
            head += f"  pk/nf {d['fine']['pk_nf_db']:.1f} dB"
        print(head)
        for x in (d.get("detections") or []):
            err = "" if m is None else f"  err {x['excess_m'] - m:+7.2f} m"
            fde = "" if fd is None else f" / {x['doppler_hz'] - fd:+6.2f} Hz"
            print(f"      {x['excess_m']:8.1f} m {x['doppler_hz']:+8.2f} Hz "
                  f"SNR {x['snr_db']:5.1f} dB{err}{fde}")
        for t_ in (d.get("tracks") or []):
            print(f"      track {t_['tid']}: {t_['excess_m']:.1f}+-{t_['sigma_m']:.1f} m "
                  f"{t_['range_rate_mps']:+.2f}+-{t_['sigma_mps']:.2f} m/s  h{t_['hits']}")


class twostage(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "NTN ISAC two-stage")
        cfg = common.load_cfg(a.wf)
        meta = common.load_traj_meta(a.traj)
        fs = float(cfg["fs_hz"])
        self.cfg, self.fs = cfg, fs

        # Occupied subcarriers per symbol, for the Es/N0 calibration. Taken
        # from the mask rather than from config's frame total, which is larger
        # by the symbol count.
        from ntn_isac import fileio
        b = fileio.load(a.wf)
        per_sym = b.occ_mask.sum(axis=1)
        n_occ = int(np.median(per_sym[per_sym > 0]))

        # ---- source -------------------------------------------------------
        if a.mode == "replay":
            src = blocks.file_source(gr.sizeof_gr_complex, a.capture, False)
            chain = [src]
            if a.skip:
                chain.append(blocks.skiphead(gr.sizeof_gr_complex,
                                             int(a.skip * fs)))
            print(f"REPLAY from {a.capture}")
        elif a.mode == "live":
            from gnuradio import uhd
            dev = common.usrp_args(a.rx_args, a.mcr)
            src = uhd.usrp_source(dev, uhd.stream_args("fc32", channels=[0]))
            src.set_samp_rate(fs)
            src.set_center_freq(uhd.tune_request(a.freq or float(cfg["carrier_hz"])), 0)
            src.set_gain(a.rx_gain, 0)
            src.set_antenna(a.rx_ant, 0)
            chain = [src]
            print("LIVE from USRP - this needs the two-radio cabled path")
        else:
            src = blocks.file_source(
                gr.sizeof_gr_complex, os.path.join(a.wf, "tx_iq.cf32"),
                bool(cfg.get("loopable", False)))
            self.chan = ntnisac.leo_channel(
                os.path.join(a.traj, "traj.csv"), fs,
                fc=float(cfg["carrier_hz"]),
                r_nominal=float(meta["r_nominal_m"]),
                tau_bias_samples=a.tau_bias, delay_line=131072,
                interp_taps=a.interp_taps, t_start=a.t_start)
            for p in a.disable:
                self.chan.set_enable(int(p), False)
            for spec in a.gain:
                pid, g = spec.split(":")
                self.chan.set_gain_db(int(pid), float(g))
            self.noise = ntnisac.esn0_noise(a.esn0_db, int(cfg["n_fft"]), n_occ,
                                            -1.0, int(0.1 * fs), a.seed)
            chain = [src, self.chan, self.noise]
            print(f"SIM  pass t0={a.t_start:.1f}s  Es/N0 {a.esn0_db:.1f} dB  "
                  f"paths {[bool(self.chan.enabled(p)) for p in range(self.chan.num_paths())]}")

        self.truth = ntnisac.leo_truth(a.traj, fs, a.t_start, a.truth_ms,
                                       a.tau_bias)
        chain.append(self.truth)
        self.connect(*chain)
        tap = chain[-1]

        # ---- the two stages ------------------------------------------------
        self.coarse = ntnisac.ssb_coarse(
            a.wf, a.traj, a.n_frames, a.t_start, a.tau_bias, a.pad,
            a.coarse_stride, a.gate_widen, a.sigma_accel, a.min_coarse_snr,
            a.exclude_delay_cells, a.notch_cells)
        self.fine = ntnisac.grid_sense(
            a.wf, a.traj, a.n_frames, a.t_start, a.tau_bias, a.variant, a.pad,
            a.stride, a.n_false_alarms, a.notch_cells, a.exclude_delay_cells,
            a.min_det_snr, a.min_renyi_bits, a.max_fragmentation,
            not a.no_comb_reject, a.maps, a.extra_variants, a.link,
            a.map_doppler_hz, a.clutter_locus, a.clutter_max_range_m,
            a.clutter_guard_bins, a.clean_components, a.clean_delay_cells,
            a.clean_doppler_hz)
        self.connect(tap, self.coarse)
        self.connect(tap, self.fine)

        # The Fig. 2 loop, in both directions.
        self.msg_connect(self.coarse, "coarse", self.fine, "coarse")
        self.msg_connect(self.fine, "refine", self.coarse, "refine")

        # ---- sinks ---------------------------------------------------------
        self.printer = msg_printer(a.quiet)
        self.msg_connect(self.coarse, "coarse", self.printer, "coarse")
        self.msg_connect(self.fine, "sense", self.printer, "sense")
        self.msg_connect(self.truth, "truth", self.printer, "truth")

        if a.zmq:
            if not HAVE_ZMQ:
                raise RuntimeError("gr-zeromq not available")
            # One socket per stage, so a dashboard can subscribe to the cheap
            # stage at full rate and the expensive one at whatever rate it
            # manages. Handoff section 8: a display crash must not be able to
            # kill the flowgraph, which is why these are PUB and not REQ.
            self.z1 = zeromq.pub_msg_sink(a.zmq, 100, True)
            self.z2 = zeromq.pub_msg_sink(a.zmq2, 100, True)
            self.msg_connect(self.coarse, "coarse", self.z1, "in")
            self.msg_connect(self.fine, "sense", self.z2, "in")
            msg = f"  ZMQ PUB  stage 1 -> {a.zmq}   stage 2 -> {a.zmq2}"
            if a.zmq3:
                # Truth on its own socket. A dashboard needs it to score
                # anything, and keeping it separate means a consumer that only
                # wants numbers does not have to receive the maps.
                self.z3 = zeromq.pub_msg_sink(a.zmq3, 100, True)
                self.msg_connect(self.truth, "truth", self.z3, "in")
                msg += f"   truth -> {a.zmq3}"
            print(msg)

        if a.seconds:
            self.head = blocks.head(gr.sizeof_gr_complex, int(a.seconds * fs))
            self.sink = blocks.null_sink(gr.sizeof_gr_complex)
            self.connect(tap, self.head, self.sink)

        res = self.fine.resolution()
        print(f"  dwell {res['dwell_s']*1e3:.0f} ms ({a.n_frames} frames), "
              f"stage 1 stride {a.coarse_stride}, stage 2 stride {a.stride}")
        clutter_text = (
            f"clutter locus +-{res['clutter_bound_hz']:.1f} Hz at "
            f"{res['clutter_max_range_m']:.0f} m"
            if res["clutter_locus"] else
            f"notch +-{res['notch_half_hz']:.0f} Hz "
            f"(+-{res['notch_half_mps']:.2f} m/s)")
        print(f"  delay cell {res['delay_cell_m']:.1f} m, Doppler cell "
              f"{res['doppler_cell_hz']:.1f} Hz, {clutter_text}")
        print(f"  coarse PRF {res['coarse_prf_hz']:.0f} Hz -> unambiguous "
              f"+-{res['coarse_unamb_hz']:.0f} Hz; comb {res['frame_rate_hz']:.0f} Hz; "
              f"Pfa {res['pfa']:.2e}")
        if res["clean_components"]:
            print(f"  direct-path CLEAN: {res['clean_components']} component(s)")
        print(f"  Es/N0 reference: {n_occ} occupied subcarriers per symbol")


def main():
    p = common.base_parser(__doc__)
    p.add_argument("--mode", choices=("sim", "replay", "live"), default="sim")
    p.add_argument("--capture", default="", help="replay mode: capture file")
    p.add_argument("--seconds", type=float, default=2.0,
                   help="0 runs until the source ends")
    p.add_argument("--skip", type=float, default=0.0)
    p.add_argument("--t-start", type=float, default=500.0, dest="t_start",
                   help="trajectory time of the first sample; the default sits "
                        "in traj_meta's last well-separated window")
    p.add_argument("--n-frames", type=int, default=5, dest="n_frames")
    p.add_argument("--tau-bias", type=float, default=40.0, dest="tau_bias")
    p.add_argument("--interp-taps", type=int, default=16, dest="interp_taps")
    p.add_argument("--esn0-db", type=float, default=30.0, dest="esn0_db")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable", nargs="*", default=[],
                   help="path indices to switch off, for target-absent runs")
    p.add_argument("--gain", nargs="*", default=[],
                   help="PID:dB echo-to-direct overrides, e.g. 2:-30")
    p.add_argument("--variant", default="payload-aided",
                   choices=("reference-only", "payload-aided", "genie"))
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--stride", type=int, default=1, help="stage 2 dwell stride")
    p.add_argument("--coarse-stride", type=int, default=1, dest="coarse_stride")
    p.add_argument("--n-false-alarms", type=float, default=1.0, dest="n_false_alarms")
    p.add_argument("--notch-cells", type=float, default=2.0, dest="notch_cells")
    p.add_argument("--clutter-locus", action="store_true",
                   help="replace the zero-Doppler notch with the conservative "
                        "ephemeris-driven stationary-clutter envelope")
    p.add_argument("--clutter-max-range-m", type=float, default=250.0,
                   dest="clutter_max_range_m",
                   help="maximum receiver range containing stationary clutter")
    p.add_argument("--clutter-guard-bins", type=float, default=1.0,
                   dest="clutter_guard_bins")
    p.add_argument("--clean-components", type=int, default=0,
                   dest="clean_components",
                   help="direct-path CLEAN components; use 1 for realistic "
                        "weak-target measurements")
    p.add_argument("--clean-delay-cells", type=float, default=1.0,
                   dest="clean_delay_cells")
    p.add_argument("--clean-doppler-hz", type=float, default=20.0,
                   dest="clean_doppler_hz")
    p.add_argument("--exclude-delay-cells", type=float, default=1.0,
                   dest="exclude_delay_cells")
    p.add_argument("--min-det-snr", type=float, default=6.0, dest="min_det_snr")
    p.add_argument("--min-coarse-snr", type=float, default=12.0, dest="min_coarse_snr")
    p.add_argument("--min-renyi-bits", type=float, default=0.0, dest="min_renyi_bits")
    p.add_argument("--max-fragmentation", type=float, default=-1.0,
                   dest="max_fragmentation")
    p.add_argument("--no-comb-reject", action="store_true", dest="no_comb_reject")
    p.add_argument("--maps", action="store_true",
                   help="publish cropped delay-Doppler maps (for the dashboard)")
    p.add_argument("--extra-variants", default="", dest="extra_variants",
                   help="comma-separated extra variants mapped from the same "
                        "dwell, e.g. reference-only")
    p.add_argument("--link", action="store_true",
                   help="compute EVM, uncoded SER and offered throughput")
    p.add_argument("--map-doppler-hz", type=float, default=250.0,
                   dest="map_doppler_hz")
    p.add_argument("--gate-widen", type=float, default=3.0, dest="gate_widen")
    p.add_argument("--sigma-accel", type=float, default=3.0, dest="sigma_accel")
    p.add_argument("--truth-ms", type=float, default=100.0, dest="truth_ms")
    p.add_argument("--zmq", default="", help="stage 1 PUB endpoint")
    p.add_argument("--zmq2", default="tcp://127.0.0.1:5556",
                   help="stage 2 PUB endpoint")
    p.add_argument("--zmq3", default="tcp://127.0.0.1:5557",
                   help="commanded-truth PUB endpoint")
    p.add_argument("--rx-args", default="", dest="rx_args")
    p.add_argument("--rx-gain", type=float, default=30.0, dest="rx_gain")
    p.add_argument("--rx-ant", default="RX2", dest="rx_ant")
    p.add_argument("--json", default="", help="write the message log here")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    if a.mode == "replay" and not a.capture:
        p.error("--mode replay needs --capture")

    tb = twostage(a)
    t0 = time.time()
    tb.run()
    wall = time.time() - t0
    n1 = sum(1 for k, _ in tb.printer.log if k == "coarse")
    n2 = sum(1 for k, _ in tb.printer.log if k == "sense")
    print(f"\n{wall:.1f} s wall for {a.seconds or 0:.1f} s of signal: "
          f"{n1} coarse dwells, {n2} fine dwells")
    if a.json:
        with open(a.json, "w") as f:
            json.dump([dict(kind=k, **d) for k, d in tb.printer.log], f, indent=1)
        print(f"  log -> {a.json}")


if __name__ == "__main__":
    main()
