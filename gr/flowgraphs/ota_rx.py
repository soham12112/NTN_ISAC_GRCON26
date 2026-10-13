#!/usr/bin/env python3
"""RX laptop: USRP -> OTA sense (+ the emulated-target stages) -> ZMQ.

    UHD Source (start time = shared epoch)   or File Source (replay / sim)
        |
        +--> OTA Sense    (sync, CLEAN, micro-Doppler) --msg "ota","micro","state"
        |
        +--> SSB Coarse   (stage 1) --msg "coarse"--+
        |         ^                                 |
        |         | msg "refine"                    v
        +--> Grid Sense   (stage 2) --msg "sense"--> ZMQ PUB --> dashboard
        |
        +--> File Sink    (optional raw capture)

Two sensing paths on one stream, because there are two targets and they need
different processing:

  the emulated drone   640 m of excess path, exact commanded truth, resolved in
                       delay. ssb_coarse and grid_sense, unchanged.
  the physical object  zero excess path to within 0.006 of a delay cell, no
                       commanded truth, resolvable only in Doppler after the
                       direct path is cancelled. ota_sense.

Running both on the same samples is the demo: a target whose answer is known
exactly validates the chain, and a target that is actually there proves it is
not validating itself. Neither alone does both.

## Modes

  --mode live    the USRP, started on the epoch ota_tx.py printed.
  --mode replay  a recorded capture through the identical blocks.
  --mode sim     no radio: the emulated pass plus an optional rotating target
                 and clock offset, from py/ntn_isac/otasim.py. This is how to
                 check the graph before touching hardware, and it is what
                 tests/verify_ota.py measures.

## The one number the two laptops must share

`--epoch` and `--t-start` together say "trajectory time t_start happened at
absolute time epoch". ota_tx.py prints the command line that carries them
across. With both GPSDOs locked that identity is good to tens of nanoseconds;
free-running, it is good to however well the operator started the two programs,
and ota_sense's acquisition absorbs the rest into its d0.
"""
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pmt
# gnuradio.uhd must be imported before gnuradio.gr and .blocks. On the
# Homebrew macOS build, importing it second segfaults the interpreter
# inside usrp_source construction - exit 139, no exception, no output.
# Harmless on Linux, where the order does not matter.
from gnuradio import uhd            # noqa: F401  (ordering)
from gnuradio import blocks, gr, ntnisac

import common
import gpssync

try:
    from gnuradio import zeromq
    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False


class printer(gr.sync_block):
    """Terminal sink. A run with no console output cannot be debugged, and the
    dashboard is a separate process that may not be there."""

    def __init__(self, quiet=False):
        gr.sync_block.__init__(self, name="printer", in_sig=None, out_sig=None)
        self.quiet = quiet
        self.log = []
        # Named methods, not lambdas or partials: GNU Radio's Python gateway
        # resolves a handler by looking its __name__ up on the block, so a
        # lambda fails at runtime with "no attribute '<lambda>'" rather than
        # at connect time.
        self.message_port_register_in(pmt.intern("ota"))
        self.set_msg_handler(pmt.intern("ota"), self.on_ota)
        self.message_port_register_in(pmt.intern("micro"))
        self.set_msg_handler(pmt.intern("micro"), self.on_micro)
        self.message_port_register_in(pmt.intern("state"))
        self.set_msg_handler(pmt.intern("state"), self.on_state)
        self.message_port_register_in(pmt.intern("sense"))
        self.set_msg_handler(pmt.intern("sense"), self.on_sense)

    def on_ota(self, msg):
        self._on("ota", msg)

    def on_micro(self, msg):
        self._on("micro", msg)

    def on_state(self, msg):
        self._on("state", msg)

    def on_sense(self, msg):
        self._on("sense", msg)

    @staticmethod
    def _read(msg):
        try:
            v = pmt.dict_ref(msg, pmt.intern("data"), pmt.PMT_NIL)
            if not pmt.is_null(v):
                return json.loads(bytes(pmt.u8vector_elements(v)).decode())
        except Exception:                              # noqa: BLE001
            pass
        return None

    def _on(self, kind, msg):
        d = self._read(msg)
        if not d:
            return
        self.log.append((kind, d))
        if self.quiet:
            return
        if kind == "ota":
            tk, dr, mi = d["tracker"], d["direct"], d["micro"]
            print(f"[ota] t={d['t0']:8.3f}  direct {dr['pk_nf_db']:5.1f} dB "
                  f"-> cancelled {dr['cancelled_db']:5.1f} dB   "
                  f"eps {tk.get('eps_ppm', 0.0):+.3f} ppm  "
                  f"resid {d['residual']['delay_samples']:+6.2f} sa "
                  f"{d['residual']['doppler_hz']:+7.2f} Hz")
            if np.isfinite(mi.get("extent_hz", np.nan)):
                print(f"      micro: edge +-{mi['extent_hz']:.1f} Hz "
                      f"({mi['tip_speed_mps']:.2f} m/s)  peak "
                      f"{mi['peak_hz']:+.1f} Hz at {mi['peak_snr_db']:.1f} dB  "
                      f"{mi['n_lines']} lines")
        elif kind == "micro":
            r = d.get("rotation_hz")
            rs = "unresolved" if r is None or not np.isfinite(r) else \
                 f"{r:.2f} rev/s ({r*60:.0f} RPM)"
            print(f"[micro] {d['dwell_s']*1e3:.0f} ms dwell: rotation {rs}  "
                  f"edge +-{d['extent_hz']:.1f} Hz  "
                  f"radius {100*d['radius_m']:.1f} cm  conf {d['rotation_conf']:.2f}")
        elif kind == "state" and not d.get("locked"):
            print(f"[sync] t={d['t']:8.3f} NOT LOCKED  "
                  f"acq peak {d.get('acq_peak_db', float('nan')):.1f} dB")
        elif kind == "sense":
            n = len(d.get("detections") or [])
            print(f"[2] t={d['t0']:8.3f} {d['variant']:<14} {n} det"
                  + (f"  pk/nf {d['fine']['pk_nf_db']:.1f} dB"
                     if d.get("fine") else ""))
            for x in (d.get("detections") or []):
                print(f"      {x['excess_m']:8.1f} m {x['doppler_hz']:+8.2f} Hz "
                      f"SNR {x['snr_db']:5.1f} dB")


class ota_rx(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "NTN ISAC OTA receive")
        cfg = common.load_cfg(a.wf)
        fs = float(cfg["fs_hz"])
        self.cfg, self.fs = cfg, fs
        self.clock = {}
        self.epoch = a.epoch

        if a.mode == "live":
            from gnuradio import uhd
            src = uhd.usrp_source(common.usrp_args(a.rx_args, a.mcr),
                                  uhd.stream_args(cpu_format="fc32",
                                                  channels=[0]))
            src.set_samp_rate(fs)
            src.set_center_freq(
                uhd.tune_request(a.freq or float(cfg["carrier_hz"])), 0)
            src.set_gain(a.rx_gain, 0)
            src.set_antenna(a.rx_ant, 0)
            if a.bandwidth:
                src.set_bandwidth(a.bandwidth, 0)
            self.clock = gpssync.configure(src, a.clock, a.gps_timeout)
            if a.epoch:
                # Start on the same absolute second the transmitter did, so
                # the first sample here is the first sample there plus the
                # propagation delay. Without this the receiver would have to
                # search a whole pass for its epoch instead of a frame.
                src.set_start_time(uhd.time_spec(a.epoch))
            chain = [src]
            actual = src.get_samp_rate()
            if abs(actual - fs) > 1.0:
                raise RuntimeError(f"USRP set {actual} S/s, not {fs}")
            print(f"RX  {(a.freq or cfg['carrier_hz'])/1e6:.3f} MHz  "
                  f"{actual/1e6:.4f} MS/s  gain {a.rx_gain} dB  ant {a.rx_ant}")
            print(gpssync.describe(self.clock, a.epoch, a.t_start))
            if not gpssync.disciplined(self.clock) and not a.no_emulated:
                # ssb_coarse and grid_sense read --epoch as trajectory time and
                # have no clock estimator. Free running the two devices' zeros
                # are the operator's launch times, so their commanded truth is
                # out by seconds - and 1 ppm alone already takes grid_sense from
                # 91 dB of peak-to-noise to 62 dB. They will report detections
                # either way, which is the failure mode that costs the most time.
                print("WARNING: the clock is free running and the emulated "
                      "stages are on. SSB Coarse and Grid Sense will report "
                      "confident nonsense. Add --no-emulated, or use "
                      "--clock external on both laptops.")
        elif a.mode == "replay":
            src = blocks.file_source(gr.sizeof_gr_complex, a.capture, False)
            chain = [src]
            if a.skip:
                chain.append(blocks.skiphead(gr.sizeof_gr_complex,
                                             int(a.skip * fs)))
            print(f"REPLAY {a.capture}")
        else:
            from ntn_isac import otasim
            blade = None
            if a.blade_gain_db > -900:
                blade = dict(excess_m=a.blade_excess_m,
                             radius_m=a.blade_radius_m,
                             rot_hz=a.blade_rot_hz, gain_db=a.blade_gain_db)
            y, b, tr, _, truth = otasim.simulate(
                a.wf, a.traj, seconds=a.seconds or 3.0, t_start=a.t_start,
                eps_ppm=a.sim_eps_ppm, blade=blade, esn0_db=a.esn0_db,
                disable=[int(x) for x in a.disable])
            src = blocks.vector_source_c(y.tolist(), False)
            chain = [src]
            self.sim_truth = truth
            print(f"SIM  pass t0={a.t_start:.1f}s  Es/N0 {a.esn0_db:.1f} dB  "
                  f"clock {a.sim_eps_ppm:+.2f} ppm  "
                  f"bias {truth['bias_samples']:.0f} samples")
            if blade:
                from ntn_isac import udoppler
                print(f"     blade r={blade['radius_m']} m at "
                      f"{blade['rot_hz']} rev/s, echo {blade['gain_db']} dB, "
                      f"predicted edge "
                      f"+-{udoppler.predicted_extent_hz(b.fc, blade['radius_m'], blade['rot_hz']):.1f} Hz")
            if not a.tau_bias_set:
                # In sim the cascaded stages' interpolator biases are known, so
                # hand them over rather than making acquisition rediscover a
                # number the simulation chose. On real RF this is V4's measured
                # calibration constant instead.
                a.tau_bias = float(truth["bias_samples"])

        if len(chain) > 1:
            self.connect(*chain)
        tap = chain[-1]
        self.chain = chain

        if a.seconds and a.mode != "sim":
            self.head = blocks.head(gr.sizeof_gr_complex, int(a.seconds * fs))
            self.connect(tap, self.head)
            tap = self.head

        # `gps_locked` on the block means "eps is negligible, do not estimate
        # it", which is equally true of a cabled 10 MHz. Testing gps_locked
        # directly would put the external mode back into free-running search.
        self.ota = ntnisac.ota_sense(
            a.wf, a.traj, a.n_frames, a.t_start, a.tau_bias, a.variant, a.pad,
            a.stride, a.window, gpssync.disciplined(self.clock),
            a.n_clean, a.clean_delay_ns, a.clean_doppler_hz, a.micro_frames,
            a.micro_every, a.rot_min_hz, a.rot_max_hz, a.snr_db, a.cfo_max_hz,
            True, a.link)
        self.connect(tap, self.ota)

        self.printer = printer(a.quiet)
        for p in ("ota", "micro", "state"):
            self.msg_connect(self.ota, p, self.printer, p)

        # The emulated-target stages, on the same samples. Their delay bias is
        # the same calibration constant ota_sense converges on.
        self.coarse = self.fine = None
        if not a.no_emulated:
            self.coarse = ntnisac.ssb_coarse(
                a.wf, a.traj, a.n_frames, a.t_start, a.tau_bias, a.pad, 1,
                3.0, 3.0, 12.0, 1.0, 2.0)
            self.fine = ntnisac.grid_sense(
                a.wf, a.traj, a.n_frames, a.t_start, a.tau_bias, a.variant,
                a.pad, a.stride, 1.0, 2.0, 1.0, 6.0, 0.0, -1.0, True,
                a.maps, a.extra_variants, a.link, 250.0,
                a.clutter_locus, a.clutter_max_range_m, 1.0,
                a.grid_clean_components, 1.0, a.clean_doppler_hz)
            self.connect(tap, self.coarse)
            self.connect(tap, self.fine)
            self.msg_connect(self.coarse, "coarse", self.fine, "coarse")
            self.msg_connect(self.fine, "refine", self.coarse, "refine")
            self.msg_connect(self.fine, "sense", self.printer, "sense")

        if a.capture_out:
            self.cap = blocks.file_sink(gr.sizeof_gr_complex, a.capture_out,
                                        False)
            self.connect(tap, self.cap)
            print(f"    capturing raw IQ -> {a.capture_out}")
        else:
            self.connect(tap, blocks.null_sink(gr.sizeof_gr_complex))

        if a.zmq:
            if not HAVE_ZMQ:
                raise RuntimeError("gr-zeromq not available")
            self.z1 = zeromq.pub_msg_sink(a.zmq, 100, True)
            self.z2 = zeromq.pub_msg_sink(a.zmq2, 100, True)
            self.msg_connect(self.ota, "ota", self.z1, "in")
            self.msg_connect(self.ota, "micro", self.z2, "in")
            if self.fine is not None:
                self.msg_connect(self.fine, "sense", self.z2, "in")
            print(f"    ZMQ PUB  ota -> {a.zmq}   micro/sense -> {a.zmq2}")

        print(f"    dwell {a.n_frames*20} ms, micro-Doppler dwell "
              f"{a.micro_frames*20} ms, window {a.window}, "
              f"tau_bias {a.tau_bias:.0f} samples")

    def sidecar(self, path, a):
        d = dict(kind="ntn-isac-ota", fs_hz=self.fs,
                 carrier_hz=float(self.cfg["carrier_hz"]),
                 mode=a.mode, epoch=self.epoch, t_start=a.t_start,
                 tau_bias=a.tau_bias, clock=self.clock,
                 waveform=os.path.abspath(a.wf),
                 trajectory=os.path.abspath(a.traj),
                 captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 note=a.note)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        return path


def main():
    p = common.base_parser(__doc__.split("##")[0])
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.add_argument("--mode", choices=("live", "replay", "sim"), default="sim")
    p.add_argument("--capture", default="", help="replay: input .cf32")
    p.add_argument("--capture-out", default="", dest="capture_out",
                   help="also write raw IQ here, for the offline campaign")
    p.add_argument("--seconds", type=float, default=3.0, help="0 = forever")
    p.add_argument("--skip", type=float, default=0.0)
    p.add_argument("--t-start", type=float, default=300.0, dest="t_start")
    p.add_argument("--epoch", type=float, default=0.0,
                   help="absolute time the transmitter started; ota_tx prints it")
    p.add_argument("--n-frames", type=int, default=5, dest="n_frames")
    p.add_argument("--tau-bias", type=float, default=40.0, dest="tau_bias")
    p.add_argument("--variant", default="payload-aided",
                   choices=("reference-only", "payload-aided", "genie"))
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--window", default="blackmanharris",
                   choices=("hann", "blackmanharris"))
    p.add_argument("--n-clean", type=int, default=2, dest="n_clean")
    p.add_argument("--clean-delay-ns", type=float, default=200.0,
                   dest="clean_delay_ns")
    p.add_argument("--clean-doppler-hz", type=float, default=15.0,
                   dest="clean_doppler_hz")
    p.add_argument("--micro-frames", type=int, default=50, dest="micro_frames",
                   help="frames per micro-Doppler dwell; 50 = 1 s, which is "
                        "what resolves a 7 rev/s line spacing")
    p.add_argument("--micro-every", type=int, default=10, dest="micro_every")
    p.add_argument("--rot-min-hz", type=float, default=3.0, dest="rot_min_hz")
    p.add_argument("--rot-max-hz", type=float, default=40.0, dest="rot_max_hz")
    p.add_argument("--snr-db", type=float, default=10.0, dest="snr_db")
    p.add_argument("--cfo-max-hz", type=float, default=12e3, dest="cfo_max_hz")
    p.add_argument("--no-emulated", action="store_true", dest="no_emulated",
                   help="skip ssb_coarse/grid_sense and sense only the "
                        "physical target")
    p.add_argument("--maps", action="store_true")
    p.add_argument("--extra-variants", default="reference-only",
                   dest="extra_variants",
                   help="same-dwell Grid Sense comparison variants")
    p.add_argument("--no-clutter-locus", action="store_false",
                   dest="clutter_locus", default=True,
                   help="restore the legacy zero-Doppler notch")
    p.add_argument("--clutter-max-range-m", type=float, default=250.0,
                   dest="clutter_max_range_m")
    p.add_argument("--grid-clean-components", type=int, default=1,
                   dest="grid_clean_components")
    p.add_argument("--link", action="store_true",
                   help="compute EVM, uncoded SER and offered throughput")
    p.add_argument("--rx-args", default="", dest="rx_args")
    p.add_argument("--rx-gain", type=float, default=40.0, dest="rx_gain")
    p.add_argument("--rx-ant", default="RX2", dest="rx_ant")
    p.add_argument("--bandwidth", type=float, default=0.0)
    p.add_argument("--clock", choices=("gpsdo", "external", "internal", "try"),
                   default="try",
                   help="external is a cabled 10 MHz/PPS into both radios and "
                        "is what the B210 pair has; must match the transmitter")
    p.add_argument("--gps-timeout", type=float, default=120.0,
                   dest="gps_timeout")
    p.add_argument("--zmq", default="", help="ota PUB endpoint")
    p.add_argument("--zmq2", default="tcp://127.0.0.1:5556")
    p.add_argument("--json", default="", help="write the message log here")
    p.add_argument("--note", default="")
    p.add_argument("--quiet", action="store_true")
    # sim-only
    p.add_argument("--esn0-db", type=float, default=45.0, dest="esn0_db")
    p.add_argument("--sim-eps-ppm", type=float, default=0.0, dest="sim_eps_ppm")
    p.add_argument("--disable", nargs="*", default=["1", "2"],
                   help="emulated paths to switch off in sim mode")
    p.add_argument("--blade-gain-db", type=float, default=-30.0,
                   dest="blade_gain_db", help="-999 for no physical target")
    p.add_argument("--blade-radius-m", type=float, default=0.25,
                   dest="blade_radius_m")
    p.add_argument("--blade-rot-hz", type=float, default=7.0,
                   dest="blade_rot_hz")
    p.add_argument("--blade-excess-m", type=float, default=0.4,
                   dest="blade_excess_m")
    a = p.parse_args()
    a.tau_bias_set = any(x.startswith("--tau-bias") for x in sys.argv)

    if a.mode == "replay" and not a.capture:
        p.error("--mode replay needs --capture")
    if a.mode == "live" and not a.epoch:
        print("WARNING: --mode live without --epoch. The receiver will have to "
              "find the transmitter's frame alignment by search, and the "
              "trajectory time of a sample is then only as good as the "
              "operator's stopwatch.")

    tb = ota_rx(a)
    t0 = time.time()

    def stop(*_):
        tb.stop()
    signal.signal(signal.SIGINT, stop)
    try:
        tb.run()
    except KeyboardInterrupt:
        tb.stop()
        tb.wait()
    wall = time.time() - t0
    n = {k: sum(1 for kk, _ in tb.printer.log if kk == k)
         for k in ("ota", "micro", "sense")}
    print(f"\n{wall:.1f} s wall: {n['ota']} ota dwells, {n['micro']} "
          f"micro-Doppler dwells, {n['sense']} emulated-target dwells")
    if a.capture_out:
        print("    sidecar ->",
              tb.sidecar(os.path.splitext(a.capture_out)[0] + ".json", a))
    if a.json:
        with open(a.json, "w") as f:
            json.dump([dict(kind=k, **d) for k, d in tb.printer.log], f, indent=1)
        print(f"    log -> {a.json}")


if __name__ == "__main__":
    main()
