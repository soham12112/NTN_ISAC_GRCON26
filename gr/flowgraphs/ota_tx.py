#!/usr/bin/env python3
"""TX laptop: waveform file -> LEO channel -> USRP, started on a shared epoch.

    File Source (tx_iq.cf32, repeat)
        |
    LEO Channel  (C++, ntnisac.leo_channel)   <- traj.csv
        |
    Multiply Const
        |
    UHD Sink  (start time = shared epoch)

This is `tx_usrp.py` with the two things the two-laptop setup needs: a scheduled
start, so the receiver knows which trajectory time each sample belongs to when
the commanded emulated path is enabled, and a 2.4 GHz ISM default instead of
2.0 GHz.

The channel is still applied digitally on the transmit side, before the DAC
(handoff section 2). Nothing about the emulation changes going over the air -
what changes is that a second machine has to agree with this one about when
t = t_start happened.

## Which epoch of the pass to transmit

--t-start 300 is closest approach on this trajectory, where the range rate
passes through zero. Doppler is near zero there and its rate is at its largest.
That is the easy epoch for bringing a link up and the honest one to measure
acquisition on, because nothing is being avoided.

--t-start 500 is the epoch the conducted demo uses: the emulated drone sits at
about 640 m of excess path with -39.0 Hz of residual Doppler at 2478 MHz, clear of the MTI
blind speed. Use it once the link closes, because that is where the synthetic
target has exact truth.

Do not compress the trajectory to make acquisition easier. Handoff section 6 is
explicit that it makes the Doppler rate unphysical and invalidates the NTN
claim, and there is no need: a high-elevation epoch is physical and already
slow.

## Power and legality

At 2478 MHz under FCC part 15.247 this is a digital modulation system - the
6 dB bandwidth is far above the 500 kHz floor - so it is licence-free up to
30 dBm conducted with a 6 dBi antenna. A B210 delivers about 10 dBm at these
gains, which is 20 dB of headroom. tx_iq is scaled to RMS 0.1 with a 10.6 dB
peak-to-average ratio; --amp above about 2.5 will clip the DAC, and clipping
quietly ruins every sensing measurement downstream because it is a nonlinearity
the receiver's reference does not contain.
"""
import argparse
import os
import signal
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


class ota_tx(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "NTN ISAC OTA transmit")
        cfg = common.load_cfg(a.wf)
        meta = common.load_traj_meta(a.traj)
        fs = float(cfg["fs_hz"])
        fc = float(cfg["carrier_hz"])
        freq = a.freq if a.freq else fc
        self.cfg, self.fs = cfg, fs

        self.src = blocks.file_source(
            gr.sizeof_gr_complex, os.path.join(a.wf, "tx_iq.cf32"),
            bool(cfg.get("loopable", False)))
        self.chan = ntnisac.leo_channel(
            os.path.join(a.traj, "traj.csv"), fs, fc,
            float(meta["r_nominal_m"]), a.tau_bias, a.delay_line,
            a.interp_taps, a.t_start)
        self.amp = blocks.multiply_const_cc(a.amp)
        self.sink = uhd.usrp_sink(common.usrp_args(a.tx_args, a.mcr),
                                  uhd.stream_args(cpu_format="fc32",
                                                  channels=[0]))
        self.sink.set_samp_rate(fs)
        self.sink.set_center_freq(uhd.tune_request(freq), 0)
        self.sink.set_gain(a.tx_gain, 0)
        self.sink.set_antenna(a.tx_ant, 0)

        self.clock = gpssync.configure(self.sink, a.clock, a.gps_timeout)
        self.epoch = (a.epoch if a.epoch else
                      gpssync.next_epoch(self.sink, a.lead, a.align,
                                         self.clock.get("gps_time")))
        # The sink emits its first sample at `epoch`, and that sample is
        # trajectory time t_start by definition. Everything the receiver does
        # hangs off this one identity.
        self.sink.set_start_time(uhd.time_spec(self.epoch))

        self.connect(self.src, self.chan, self.amp, self.sink)

        actual = self.sink.get_samp_rate()
        if abs(actual - fs) > 1.0:
            raise RuntimeError(
                f"USRP set {actual} S/s, not {fs}; check --mcr divides evenly")

        print(f"TX  {freq/1e6:.3f} MHz  {actual/1e6:.4f} MS/s  "
              f"gain {a.tx_gain} dB  ant {a.tx_ant}  amp {a.amp}")
        print(f"    waveform {a.wf} (repeat={bool(cfg.get('loopable'))})")
        print(gpssync.describe(self.clock, self.epoch, a.t_start))
        print(f"    pass t0={a.t_start:.1f}s  "
              f"r_nominal={meta['r_nominal_m']/1e3:.1f} km  "
              f"{self.chan.num_paths()} paths, {a.interp_taps}-tap interpolator")
        print()
        # Echo the mode this side actually ended up in, not the one requested:
        # `--clock try` can fall back to internal, and the receiver pasting
        # `--clock gpsdo` after that would fail on a radio that has no GPSDO.
        mode = self.clock.get("requested", "internal")
        # Free running, --epoch is device time and the two devices' zeros are
        # the operator's launch times, so the commanded stages cannot be scored.
        emu = "" if gpssync.disciplined(self.clock) else " --no-emulated"
        print("    On the receive laptop, run:")
        print(f"      $NTN_PY gr/flowgraphs/ota_rx.py --wf {a.wf} "
              f"--traj {a.traj} \\\n"
              f"          --t-start {a.t_start} --epoch {self.epoch:.1f} "
              f"--clock {mode} --tau-bias <40 + measured>{emu}")

    def report(self):
        c = self.chan
        parts = [f"t={c.t_now():7.2f}s"]
        for p in range(c.num_paths()):
            parts.append(f"p{p}: {c.tap_delay_samples(p):9.2f}sa "
                         f"{c.tap_doppler_hz(p):+8.1f}Hz "
                         f"{c.tap_gain_db(p):+6.1f}dB")
        return "  ".join(parts)


def main():
    p = common.base_parser(__doc__.split("##")[0])
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.add_argument("--tx-args", default="", help="UHD device args")
    p.add_argument("--tx-gain", type=float, default=70.0)
    p.add_argument("--tx-ant", default="TX/RX")
    p.add_argument("--amp", type=float, default=1.0,
                   help="scalar before the DAC; tx_iq is already RMS ~0.1 and "
                        "above ~2.5 will clip")
    p.add_argument("--t-start", type=float, default=300.0, dest="t_start",
                   help="trajectory time of the first transmitted sample; 300 "
                        "is closest approach, where Doppler passes through zero")
    p.add_argument("--tau-bias", type=float, default=40.0, dest="tau_bias")
    p.add_argument("--delay-line", type=int, default=131072, dest="delay_line")
    p.add_argument("--interp-taps", type=int, default=16, dest="interp_taps")
    p.add_argument("--clock", choices=("gpsdo", "external", "internal", "try"),
                   default="try",
                   help="gpsdo uses the onboard GPSDO (the paper's two B200s); "
                        "external is a cabled 10 MHz/PPS into both radios; "
                        "try falls back to internal")
    p.add_argument("--gps-timeout", type=float, default=120.0,
                   dest="gps_timeout")
    p.add_argument("--epoch", type=float, default=0.0,
                   help="absolute start time; default is the next --align "
                        "boundary at least --lead away")
    p.add_argument("--lead", type=float, default=15.0)
    p.add_argument("--align", type=float, default=10.0)
    a = p.parse_args()

    tb = ota_tx(a)
    tb.start()

    def stop(*_):
        tb.stop()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
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
