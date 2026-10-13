#!/usr/bin/env /usr/bin/python3
"""Receive flowgraph: USRP -> file, with a metadata sidecar.

    uhd_source -> skip_head -> head -> file_sink

Deliberately dumb. Everything the handoff's section 7 receive chain does -
acquisition, de-ramp, de-walk, OFDM demodulation, channel estimation,
equalisation, slicing, the three sensing branches - runs on the captured file,
so the identical code path serves live capture, replay from file (section 8)
and the offline campaign (section 10). The sidecar records what the radio was
actually set to, because a capture whose sample rate you have to guess is
worthless a week later.

Being dumb is the point on the two-host path. `ota_rx.py --mode live` runs the
sensing chain against the radio in real time, and on the macOS receiver that
chain is about 2.3x slower than real time: the flowgraph backpressures the
source, the device overflows, and the file it writes is spliced. Level, PAPR
and spectrum all survive that - every sample kept is perfect - so the capture
looks clean while every correlation-based number is quietly wrong. This graph
keeps up, so `--clock` and `--epoch` here, then `ota_rx.py --mode replay`, is
how to take a timed two-host capture.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# gnuradio.uhd must be imported before gnuradio.gr and .blocks. On the
# Homebrew macOS build, importing it second segfaults the interpreter
# inside usrp_source construction - exit 139, no exception, no output.
# Harmless on Linux, where the order does not matter.
from gnuradio import uhd
from gnuradio import blocks, gr

import common
import gpssync


class rx_capture(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "NTN ISAC capture")
        cfg = common.load_cfg(args.wf)
        fs = float(cfg["fs_hz"])
        freq = args.freq if args.freq else float(cfg["carrier_hz"])

        self.source = uhd.usrp_source(
            common.usrp_args(args.rx_args, args.mcr),
            uhd.stream_args(cpu_format="fc32", channels=[0]))
        self.source.set_samp_rate(fs)
        self.source.set_center_freq(uhd.tune_request(freq), 0)
        self.source.set_gain(args.rx_gain, 0)
        self.source.set_antenna(args.rx_ant, 0)
        if args.bandwidth:
            self.source.set_bandwidth(args.bandwidth, 0)

        self.clock = gpssync.configure(self.source, args.clock,
                                       args.gps_timeout)
        if args.epoch:
            # The absolute second the transmitter emitted its first sample, as
            # printed by ota_tx.py. Under gpsdo both hosts have put their
            # device clock on GPS time, so this is a shared number and not
            # merely a shared edge.
            self.source.set_start_time(uhd.time_spec(args.epoch))

        n_skip = int(args.skip * fs)
        n_keep = int(args.seconds * fs)
        self.skip = blocks.skiphead(gr.sizeof_gr_complex, n_skip)
        self.head = blocks.head(gr.sizeof_gr_complex, n_keep)
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, args.out, False)
        self.sink.set_unbuffered(False)
        self.connect(self.source, self.skip, self.head, self.sink)

        actual = self.source.get_samp_rate()
        if abs(actual - fs) > 1.0:
            raise RuntimeError(
                f"USRP set {actual} S/s, not {fs}; check --mcr divides evenly")

        self.meta = dict(
            kind="ntn-isac-capture",
            file=os.path.basename(args.out),
            fs_hz=actual,
            carrier_hz=freq,
            rx_gain_db=args.rx_gain,
            rx_antenna=args.rx_ant,
            rx_args=args.rx_args,
            master_clock_rate=args.mcr,
            bandwidth_hz=args.bandwidth or 0.0,
            skip_s=args.skip,
            seconds=args.seconds,
            n_samples=n_keep,
            waveform=os.path.abspath(args.wf),
            trajectory=os.path.abspath(args.traj),
            t_start_s=args.t_start,
            tau_bias_samples=args.tau_bias,
            epoch=float(args.epoch),
            clock=self.clock,
            note=args.note,
        )
        print(f"RX  {freq/1e9:.4f} GHz  {actual/1e6:.4f} MS/s  gain {args.rx_gain} dB  "
              f"ant {args.rx_ant}")
        print(f"    skip {args.skip:.2f} s, keep {args.seconds:.2f} s "
              f"({n_keep} samples, {n_keep*8/1e6:.1f} MB) -> {args.out}")
        if args.epoch:
            print(gpssync.describe(self.clock, args.epoch, args.t_start))
        else:
            print(f"    clock {self.clock.get('requested')}, untimed start")

    def finish(self):
        self.meta["captured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        side = os.path.splitext(self.meta_path)[0] + ".json"
        with open(side, "w") as f:
            json.dump(self.meta, f, indent=2)
        print(f"    sidecar -> {side}")


def main():
    p = common.base_parser(__doc__)
    p.add_argument("--rx-args", default="", help="UHD device args for the receiver")
    p.add_argument("--rx-gain", type=float, default=30.0)
    p.add_argument("--rx-ant", default="RX2")
    p.add_argument("--bandwidth", type=float, default=0.0)
    p.add_argument("--out", required=True, help="output .cf32 path")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--skip", type=float, default=None,
                   help="discard this much before capturing, past AGC/tune "
                        "settling. Defaults to 0.5 s untimed, 0 with --epoch, "
                        "where the epoch already defines the first sample")
    p.add_argument("--t-start", type=float, default=0.0, dest="t_start",
                   help="pass time the transmitter was started at, recorded only")
    p.add_argument("--tau-bias", type=float, default=16.0, dest="tau_bias")
    p.add_argument("--epoch", type=float, default=0.0,
                   help="absolute time the transmitter started, as printed by "
                        "ota_tx.py; the capture opens on it. Needs --clock "
                        "gpsdo or external on both hosts to be a shared number")
    p.add_argument("--clock", choices=("gpsdo", "external", "internal", "try"),
                   default="try",
                   help="gpsdo is the two-B200 setup; external is a cabled "
                        "10 MHz/PPS into both radios")
    p.add_argument("--gps-timeout", type=float, default=120.0,
                   dest="gps_timeout")
    p.add_argument("--note", default="")
    args = p.parse_args()

    if args.skip is None:
        args.skip = 0.0 if args.epoch else 0.5
    elif args.epoch and args.skip:
        # skiphead discards from the first sample, which under a timed start is
        # the epoch itself - so the capture would silently open args.skip
        # seconds into the pass while the sidecar still claims the epoch.
        print(f"WARNING: --skip {args.skip} with --epoch throws away the first "
              f"{args.skip} s after the epoch. The capture will start at "
              f"{args.epoch + args.skip:.1f}, not {args.epoch:.1f}.")
    if args.epoch and args.clock == "internal":
        print("WARNING: --epoch with --clock internal. Device time is zeroed "
              "at launch, so the epoch is not a shared number and the capture "
              "will open at the wrong moment.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tb = rx_capture(args)
    tb.meta_path = args.out
    t0 = time.time()
    tb.run()
    tb.finish()
    print(f"    done in {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
