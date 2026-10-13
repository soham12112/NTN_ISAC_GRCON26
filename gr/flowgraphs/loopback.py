#!/usr/bin/env /usr/bin/python3
"""One-process cabled loopback: transmit and capture together.

Needed when a single B210 does both ends, because two processes cannot share
one device. With two radios, run tx_usrp.py and rx_capture.py separately and
pass --tx-args / --rx-args to pick them by serial.

Read this before trusting a single-radio result: one B210 means one LO and one
reference clock for both ends, so the clock-offset Doppler bias and the
range-rate error floor that V4 and V5 exist to measure are identically zero.
That is the flattering configuration. It validates the RF path, the cabling and
the receive chain, but it is not the error floor, and a slide must say so.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

# gnuradio.uhd must be imported before gnuradio.gr and .blocks. On the
# Homebrew macOS build, importing it second segfaults the interpreter
# inside usrp_source construction - exit 139, no exception, no output.
# Harmless on Linux, where the order does not matter.
from gnuradio import uhd
from gnuradio import blocks, gr, ntnisac

import common
import gpssync


class loopback(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "NTN ISAC cabled loopback")
        cfg = common.load_cfg(args.wf)
        meta = common.load_traj_meta(args.traj)
        fs = float(cfg["fs_hz"])
        freq = args.freq if args.freq else float(cfg["carrier_hz"])
        self.single = not args.rx_args or args.rx_args == args.tx_args

        dev = common.usrp_args(args.tx_args, args.mcr)
        self.usrp_tx = uhd.usrp_sink(dev, uhd.stream_args("fc32", channels=[0]))
        self.clock = gpssync.configure(self.usrp_tx, args.clock, args.gps_timeout)
        self.usrp_tx.set_samp_rate(fs)
        self.usrp_tx.set_center_freq(uhd.tune_request(freq), 0)
        self.usrp_tx.set_gain(args.tx_gain, 0)
        self.usrp_tx.set_antenna(args.tx_ant, 0)

        rdev = dev if self.single else common.usrp_args(args.rx_args, args.mcr)
        self.usrp_rx = uhd.usrp_source(rdev, uhd.stream_args("fc32", channels=[0]))
        self.clock_rx = self.clock if self.single else gpssync.configure(
            self.usrp_rx, args.clock, args.gps_timeout)
        self.usrp_rx.set_samp_rate(fs)
        self.usrp_rx.set_center_freq(uhd.tune_request(freq), 0)
        self.usrp_rx.set_gain(args.rx_gain, 0)
        self.usrp_rx.set_antenna(args.rx_ant, 0)

        self.src = blocks.file_source(
            gr.sizeof_gr_complex, os.path.join(args.wf, "tx_iq.cf32"),
            bool(cfg.get("loopable", False)))
        self.chan = ntnisac.leo_channel(
            os.path.join(args.traj, "traj.csv"), fs, fc=float(cfg["carrier_hz"]),
            r_nominal=float(meta["r_nominal_m"]), tau_bias_samples=args.tau_bias,
            delay_line=131072, interp_taps=args.interp_taps, t_start=args.t_start)
        if args.only_direct:
            for p in range(1, self.chan.num_paths()):
                self.chan.set_enable(p, False)
        self.amp = blocks.multiply_const_cc(args.amp)

        # Bound the transmit chain too. tx_iq repeats forever, so without this
        # the transmit blocks never signal done and top_block.run() blocks
        # after the capture has finished - the graph hangs with the carrier
        # still up until something kills it. The margin keeps the transmitter
        # alive past the last sample the receiver wants, so the capture never
        # ends on a dead carrier.
        n_tx = int((args.skip + args.seconds + args.tx_margin) * fs)
        self.tx_head = blocks.head(gr.sizeof_gr_complex, n_tx)
        self.connect(self.src, self.chan, self.amp, self.tx_head, self.usrp_tx)

        n_skip = int(args.skip * fs)
        n_keep = int(args.seconds * fs)
        self.skip = blocks.skiphead(gr.sizeof_gr_complex, n_skip)
        self.head = blocks.head(gr.sizeof_gr_complex, n_keep)
        self.fsink = blocks.file_sink(gr.sizeof_gr_complex, args.out, False)
        self.connect(self.usrp_rx, self.skip, self.head, self.fsink)

        # With a disciplined clock, start both ends on the same absolute
        # second. Without it the transmitter and the receiver begin whenever
        # the scheduler reaches them, and that unknown offset is
        # indistinguishable from delay through the cable - which is the one
        # quantity this rung exists to measure.
        self.epoch = None
        if args.timed_start and gpssync.disciplined(self.clock):
            self.epoch = gpssync.next_epoch(self.usrp_tx, lead_s=args.start_lead,
                                            align_s=1.0)
            self.usrp_tx.set_start_time(uhd.time_spec(self.epoch))
            self.usrp_rx.set_start_time(uhd.time_spec(self.epoch))
            print(f"  timed start: both ends armed for {self.epoch:.0f} "
                  f"(GPS time), so the TX/RX start offset is zero by "
                  f"construction")
        elif args.timed_start:
            print("  timed start requested but the clock is free running; "
                  "skipped (the start offset stays unknown)")

        got_tx = self.usrp_tx.get_samp_rate()
        got_rx = self.usrp_rx.get_samp_rate()
        if abs(got_tx - fs) > 1.0 or abs(got_rx - fs) > 1.0:
            raise RuntimeError(f"USRP rates {got_tx}/{got_rx}, wanted {fs}")

        self.args, self.fs, self.n_keep, self.freq = args, fs, n_keep, freq
        print(f"{'SINGLE-RADIO' if self.single else 'TWO-RADIO'} loopback  "
              f"{freq/1e9:.4f} GHz  {got_tx/1e6:.4f} MS/s")
        print(f"  TX gain {args.tx_gain} dB ant {args.tx_ant} | "
              f"RX gain {args.rx_gain} dB ant {args.rx_ant}")
        lock = "disciplined" if gpssync.disciplined(self.clock) else "free running"
        print(f"  clock {self.clock.get('requested')}: {lock}"
              + (f", gps_time {self.clock['gps_time']}"
                 if self.clock.get("gps_time") else ""))
        print(f"  pass t0={args.t_start:.1f}s, paths "
              f"{[bool(self.chan.enabled(p)) for p in range(self.chan.num_paths())]}")
        if self.single:
            print("  NOTE one radio: TX and RX share an LO and a clock, so the "
                  "Doppler bias\n       and range-rate floor V4/V5 measure are "
                  "zero by construction.")

    def write_meta(self):
        a = self.args
        m = dict(kind="ntn-isac-capture", mode="loopback",
                 single_radio=self.single, file=os.path.basename(a.out),
                 fs_hz=self.fs, carrier_hz=self.freq, tx_gain_db=a.tx_gain,
                 rx_gain_db=a.rx_gain, tx_antenna=a.tx_ant, rx_antenna=a.rx_ant,
                 amp=a.amp, master_clock_rate=a.mcr, skip_s=a.skip,
                 seconds=a.seconds, n_samples=self.n_keep,
                 waveform=os.path.abspath(a.wf), trajectory=os.path.abspath(a.traj),
                 t_start_s=a.t_start, tau_bias_samples=a.tau_bias,
                 interp_taps=a.interp_taps, only_direct=a.only_direct,
                 clock=self.clock, clock_rx=self.clock_rx, epoch=self.epoch,
                 captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 note=a.note)
        side = os.path.splitext(a.out)[0] + ".json"
        with open(side, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  sidecar -> {side}")


def main():
    p = common.base_parser(__doc__)
    p.add_argument("--tx-args", default="")
    p.add_argument("--rx-args", default="")
    p.add_argument("--tx-gain", type=float, default=50.0)
    p.add_argument("--rx-gain", type=float, default=30.0)
    p.add_argument("--tx-ant", default="TX/RX")
    p.add_argument("--rx-ant", default="RX2")
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--skip", type=float, default=1.0)
    p.add_argument("--tx-margin", type=float, default=3.0, dest="tx_margin",
                   help="keep transmitting this long past the last captured "
                        "sample, so the capture never ends on a dead carrier")
    p.add_argument("--t-start", type=float, default=0.0, dest="t_start")
    p.add_argument("--tau-bias", type=float, default=16.0, dest="tau_bias")
    p.add_argument("--interp-taps", type=int, default=16, dest="interp_taps")
    p.add_argument("--only-direct", action="store_true", dest="only_direct",
                   help="disable the scatterers, for V4 delay/Doppler calibration")
    p.add_argument("--note", default="")
    p.add_argument("--clock", choices=("gpsdo", "external", "internal", "try"),
                   default="internal",
                   help="reference and PPS; 'try' prefers a GPSDO and falls "
                        "back to free-running")
    p.add_argument("--gps-timeout", type=float, default=120.0, dest="gps_timeout")
    p.add_argument("--timed-start", action="store_true", dest="timed_start",
                   help="with a disciplined clock, start TX and RX on the same "
                        "absolute second so their offset is known to be zero")
    p.add_argument("--start-lead", type=float, default=3.0, dest="start_lead",
                   help="how far ahead to schedule that start")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tb = loopback(args)
    t0 = time.time()
    tb.run()
    tb.write_meta()
    print(f"  captured {args.seconds:.2f} s in {time.time()-t0:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
