#!/usr/bin/env python3
"""Commanded trajectory truth, published as messages on the sample clock."""

import time

import numpy as np
import pmt
from gnuradio import gr

from .dwell import add_repo_py, pack

add_repo_py()


class leo_truth(gr.sync_block):
    """Publish per-path commanded truth for the emulated pass.

    The C++ leo_channel exposes tap_delay_samples, tap_doppler_hz,
    tap_excess_m and tap_gain_db as getters, which is fine for a Python driver
    script and useless inside a flowgraph: nothing downstream can see them and
    nothing records them alongside a capture. This block turns them into a
    message stream, which is what a dashboard and an offline scorer both need.

    "There is no physical reflector anywhere in this setup, and that is
    deliberate, because it gives exact truth" - handoff section 1. This block
    is where that exactness is delivered, so it reads the same trajectory
    table through the same not-a-knot spline the channel uses. It must not
    grow its own interpolator.

    It counts samples on a pass-through input rather than running on a timer,
    so its clock is the flowgraph's sample clock. A wall-clock timer would
    drift against the emulated pass whenever the graph is not running at
    exactly 1x - during a file-driven replay it is not - and the truth would
    silently disagree with the capture it is labelling.
    """

    def __init__(self, traj_dir="", fs=7.68e6, t_start=0.0, period_ms=100.0,
                 tau_bias_samples=40.0, port="truth"):
        gr.sync_block.__init__(self, name="leo_truth",
                               in_sig=[np.complex64], out_sig=[np.complex64])
        from ntn_isac import trajectory

        self.fs = float(fs)
        self.t_start = float(t_start)
        self.bias = float(tau_bias_samples) / self.fs
        self.traj = trajectory.load(traj_dir)
        self.r_nominal = self.traj.r_nominal
        self.fc = self.traj.fc
        self.period = max(int(round(period_ms * 1e-3 * self.fs)), 1)
        self.port = pmt.intern(port)
        self.message_port_register_out(self.port)
        self._n = 0
        self._next = 0

    def t_now(self):
        return self.t_start + self._n / self.fs

    def state(self, t):
        """Per-path truth at trajectory time t. Same formulas as
        leo_channel.LeoChannel.tap_state, and the same splines."""
        eps = 1e-3
        C = 299792458.0
        r0 = float(self.traj.paths[0].range_at(np.array([t]))[0])
        out = []
        self._r0 = r0
        for p in self.traj.paths:
            R = p.range_at(np.array([t - eps, t, t + eps]))
            tau = (R[1] - self.r_nominal) / C + self.bias
            out.append(dict(
                pid=int(p.pid), name=p.name, delay_samples=tau * self.fs,
                excess_m=float(R[1] - r0),
                doppler_hz=float(-self.fc / C * (R[2] - R[0]) / (2 * eps)),
                gain_db=float(20 * np.log10(
                    max(float(p.gain_at(np.array([t]))[0]), 1e-30))),
                present=bool(p.present(np.array([t]))[0])))
        return out

    def _publish(self):
        """One message per period, in the same envelope the sensing blocks use.

        A plain PMT dict of scalars would serialise safely too, but then a
        consumer needs two decoders for three sockets. Everything on the wire
        from this module is a u8vector of UTF-8 JSON under the "data" key.
        """
        t = self.t_now()
        paths = self.state(t)
        d = dict(t_s=t, n_abs=int(self._n), wall_utc=time.time(),
                 r_nominal_m=float(self.r_nominal),
                 slant_range_m=float(self._r0),
                 carrier_hz=float(self.fc))
        # Residual quantities are published alongside the absolute ones rather
        # than left for every consumer to recompute and get the sign wrong.
        for p in paths:
            pre = f"p{p['pid']}_"
            d[pre + "name"] = p["name"]
            for k in ("delay_samples", "excess_m", "doppler_hz", "gain_db"):
                d[pre + k] = float(p[k])
            d[pre + "present"] = bool(p["present"])
            d[pre + "residual_doppler_hz"] = float(
                p["doppler_hz"] - paths[0]["doppler_hz"])
        self.message_port_pub(self.port, pack("truth", d))

    def work(self, input_items, output_items):
        n = len(input_items[0])
        output_items[0][:] = input_items[0]
        end = self._n + n
        while self._next < end:
            # Publish at the boundary, not at the block edge, so the message
            # cadence is exactly `period` samples regardless of buffer sizes.
            self._n = self._next
            self._publish()
            self._next += self.period
        self._n = end
        return n
