#!/usr/bin/env /usr/bin/python3
"""Transmit flowgraph: waveform file -> LEO channel -> USRP.

    file_source(repeat) -> ntnisac.leo_channel -> multiply_const -> uhd_sink

The channel is applied digitally on the transmit side, before the DAC
(handoff section 2). File Source repeat is driven by config.json's `loopable`:
true for the MATLAB generator, false for an OCUDU capture, which is not
periodic.

Conducted only. Cable radio 1's TX/RX port to radio 2's RX2 port through about
40 dB of attenuation before running this.
"""
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

# gnuradio.uhd must be imported before gnuradio.gr and .blocks. On the
# Homebrew macOS build, importing it second segfaults the interpreter
# inside usrp_source construction - exit 139, no exception, no output.
# Harmless on Linux, where the order does not matter.
from gnuradio import uhd
from gnuradio import blocks, gr, ntnisac

import common


class tx_usrp(gr.top_block):
    def __init__(self, args):
        gr.top_block.__init__(self, "NTN ISAC transmit")
        cfg = common.load_cfg(args.wf)
        meta = common.load_traj_meta(args.traj)
        fs = float(cfg["fs_hz"])
        fc = float(cfg["carrier_hz"])
        freq = args.freq if args.freq else fc
        self.cfg, self.fs = cfg, fs

        self.src = blocks.file_source(
            gr.sizeof_gr_complex, os.path.join(args.wf, "tx_iq.cf32"),
            bool(cfg.get("loopable", False)))
        self.chan = ntnisac.leo_channel(
            os.path.join(args.traj, "traj.csv"), fs, fc,
            float(meta["r_nominal_m"]), args.tau_bias, args.delay_line,
            args.interp_taps, args.t_start)
        self.amp = blocks.multiply_const_cc(args.amp)
        self.sink = uhd.usrp_sink(
            common.usrp_args(args.tx_args, args.mcr),
            uhd.stream_args(cpu_format="fc32", channels=[0]))
        self.sink.set_samp_rate(fs)
        self.sink.set_center_freq(uhd.tune_request(freq), 0)
        self.sink.set_gain(args.tx_gain, 0)
        self.sink.set_antenna(args.tx_ant, 0)

        self.connect(self.src, self.chan, self.amp, self.sink)

        actual = self.sink.get_samp_rate()
        if abs(actual - fs) > 1.0:
            raise RuntimeError(
                f"USRP set {actual} S/s, not {fs}; check --mcr divides evenly")

        print(f"TX  {freq/1e9:.4f} GHz  {actual/1e6:.4f} MS/s  gain {args.tx_gain} dB  "
              f"ant {args.tx_ant}  amp {args.amp}")
        print(f"    waveform {args.wf} (repeat={bool(cfg.get('loopable'))})")
        print(f"    pass t0={args.t_start:.1f}s  r_nominal={meta['r_nominal_m']/1e3:.1f} km  "
              f"{self.chan.num_paths()} paths, {args.interp_taps}-tap interpolator")

    def report(self):
        c = self.chan
        parts = [f"t={c.t_now():7.2f}s"]
        for p in range(c.num_paths()):
            parts.append(f"p{p}: {c.tap_delay_samples(p):9.2f}sa "
                         f"{c.tap_doppler_hz(p):+8.1f}Hz {c.tap_gain_db(p):+6.1f}dB")
        return "  ".join(parts)


def main():
    p = common.base_parser(__doc__)
    p.add_argument("--tx-args", default="", help="UHD device args for the transmitter")
    p.add_argument("--tx-gain", type=float, default=50.0)
    p.add_argument("--tx-ant", default="TX/RX")
    p.add_argument("--amp", type=float, default=1.0,
                   help="scalar before the DAC; tx_iq is already RMS ~0.1")
    p.add_argument("--t-start", type=float, default=0.0, dest="t_start")
    p.add_argument("--tau-bias", type=float, default=16.0, dest="tau_bias")
    p.add_argument("--delay-line", type=int, default=131072, dest="delay_line")
    p.add_argument("--interp-taps", type=int, default=16, dest="interp_taps")
    p.add_argument("--jump", type=float, default=None,
                   help="jump to this time in the pass after starting")
    args = p.parse_args()

    tb = tx_usrp(args)
    tb.start()
    if args.jump is not None:
        tb.chan.set_time(args.jump)

    def stop(*_):
        tb.stop()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        import time
        while True:
            time.sleep(2.0)
            print(tb.report())
    except KeyboardInterrupt:
        pass
    finally:
        tb.stop()
        tb.wait()


if __name__ == "__main__":
    main()
