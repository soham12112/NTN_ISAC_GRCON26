#!/usr/bin/env python3
"""Readable console output for this module's messages.

GNU Radio's own Message Debug is the obvious block to hang on these ports and
it is useless here. Every payload in this module travels as UTF-8 JSON inside a
u8vector - `dwell.pack` explains why: a PMT symbol cannot carry more than 65535
characters, because pmt::serialize writes its length in two bytes, and the
moment a payload carried two delay-Doppler maps it died on the wire as
"pmt::deserialize: very long symbol". Message Debug prints a u8vector as its
decimal bytes, so a single dwell arrives as several thousand numbers.

This decodes the payload and prints the few fields that matter per message
kind, which is what makes the bench flowgraph usable on the canvas rather than
only from the Python driver.
"""

import json

import numpy as np
import pmt
from gnuradio import gr

from .dwell import unpack


def _f(d, *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


class msg_print(gr.sync_block):
    def __init__(self, tag="", verbose=False):
        gr.sync_block.__init__(self, name="msg_print", in_sig=None,
                               out_sig=None)
        self.tag = str(tag)
        self.verbose = bool(verbose)
        self.n = 0
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle)

    def handle(self, msg):
        kind, d = unpack(msg)
        if not isinstance(d, dict):
            return
        self.n += 1
        pre = f"[{self.tag or kind}]"
        try:
            line = self.format(kind, d)
        except Exception as exc:                       # noqa: BLE001
            line = f"unprintable {kind}: {exc}"
        if line:
            print(f"{pre} {line}", flush=True)
        if self.verbose:
            small = {k: v for k, v in d.items()
                     if not isinstance(v, (dict, list))}
            print(f"    {json.dumps(small, default=str)}", flush=True)

    def format(self, kind, d):
        if kind == "ota":
            dr, tk, mi = d.get("direct", {}), d.get("tracker", {}), d.get("micro", {})
            s = (f"t={_f(d,'t0'):8.3f}  direct {_f(dr,'pk_nf_db'):5.1f} dB "
                 f"-> cancelled {_f(dr,'cancelled_db'):5.1f} dB   "
                 f"eps {_f(tk,'eps_ppm',default=0.0):+.3f} ppm  "
                 f"resid {_f(d,'residual','delay_samples'):+6.2f} sa "
                 f"{_f(d,'residual','doppler_hz'):+7.2f} Hz")
            e = _f(mi, "extent_hz")
            if np.isfinite(e):
                s += (f"\n      micro: edge +-{e:.1f} Hz "
                      f"({_f(mi,'tip_speed_mps'):.2f} m/s), strongest line "
                      f"{_f(mi,'peak_hz'):+.1f} Hz at "
                      f"{_f(mi,'peak_snr_db'):.1f} dB, "
                      f"{int(_f(mi,'n_lines',default=0))} lines")
            lk = d.get("link")
            if lk:
                s += (f"\n      link: EVM {100*_f(lk,'evm'):.2f}%  "
                      f"offered {_f(lk,'offered_mbps'):.2f} Mb/s")
            return s

        if kind == "micro":
            r = _f(d, "rotation_hz")
            rs = ("unresolved" if not np.isfinite(r)
                  else f"{r:.2f} rev/s = {r*60:.0f} RPM")
            return (f"{1e3*_f(d,'dwell_s'):.0f} ms dwell: rotation {rs}  "
                    f"edge +-{_f(d,'extent_hz'):.1f} Hz  "
                    f"radius {100*_f(d,'radius_m'):.1f} cm  "
                    f"conf {_f(d,'rotation_conf'):.2f}")

        if kind == "ota_state":
            if not d.get("locked", False):
                return (f"t={_f(d,'t'):8.3f} NOT LOCKED  acq peak "
                        f"{_f(d,'acq_peak_db'):.1f} dB")
            return None                    # locked state is on the ota port

        if kind == "sense":
            dets = d.get("detections") or []
            s = (f"t={_f(d,'t0'):8.3f} {d.get('variant','')} "
                 f"admitted={d.get('admitted')} {len(dets)} det")
            if d.get("fine"):
                s += f"  pk/nf {_f(d,'fine','pk_nf_db'):.1f} dB"
            for x in dets:
                s += (f"\n      {_f(x,'excess_m'):8.1f} m "
                      f"{_f(x,'doppler_hz'):+8.2f} Hz  SNR "
                      f"{_f(x,'snr_db'):5.1f} dB")
            for t in (d.get("tracks") or []):
                s += (f"\n      track {t.get('tid')}: {_f(t,'excess_m'):.1f} "
                      f"+-{_f(t,'sigma_m'):.1f} m  "
                      f"{_f(t,'range_rate_mps'):+.2f} m/s  h{t.get('hits')}")
            return s

        if kind == "coarse":
            return (f"t={_f(d,'t'):8.3f} {_f(d,'excess_m'):8.1f} m  SNR "
                    f"{_f(d,'snr_db'):5.1f} dB  notch "
                    f"{_f(d,'mti_notch_db'):5.1f} dB")

        if kind == "truth":
            return (f"t={_f(d,'t_s'):8.3f}  "
                    + "  ".join(
                        f"p{i}: {d[f'p{i}_excess_m']:7.1f} m "
                        f"{d[f'p{i}_residual_doppler_hz']:+7.2f} Hz"
                        for i in range(8) if f"p{i}_excess_m" in d))

        return json.dumps({k: v for k, v in d.items()
                           if not isinstance(v, (dict, list))}, default=str)

    def work(self, input_items, output_items):
        return 0
