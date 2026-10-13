#!/usr/bin/env /usr/bin/python3
"""PyQtGraph dashboard for the two-stage ISAC flow (handoff section 8).

A separate process fed over ZMQ, so that a display crash cannot kill the
flowgraph - which is section 2's requirement and the reason the sinks in the
graph are PUB rather than REQ. This process can also *launch* the flowgraph
(the default) so a demo is one command; `--attach` subscribes to one that is
already running.

Layout, following section 8:

    pass state strip     range, range rate, Doppler, lock state
    two delay-Doppler maps side by side, one shared colour scale, each
        labelled with its own peak-to-noise-floor in dB
    metric cards         EVM, uncoded SER, offered throughput, CPI latency,
                         delay error, range-rate error, detection rate,
                         PDSCH load
    error history        estimated minus commanded, delay and range rate
    occupancy control    the demo's best moment, see below

Three places this deviates from section 8, all stated on the display itself
rather than only here:

**No BLER.** It needs LDPC decoding, which section 12 forbids in the live
receive path. The card shows the uncoded symbol error rate against tx_grid
instead and is labelled "SER (uncoded)". An uncoded SER presented as a BLER
would overstate the link by orders of magnitude at any useful operating point.

**No elevation.** Section 8 asks for it in the pass strip. `traj.csv` carries
ranges and gains only - that is the file interface of section 3 - and
`traj_meta.json` has just the min and max over the pass. Elevation per dwell
would need `gen_trajectory.m` to emit it, which is a one-column change to a
frozen contract and was not worth making unilaterally. Range, range rate and
Doppler are all present and all derived from the commanded trajectory.

**The occupancy control is a switch, not a drag.** Section 8 wants a slider
from 100% to 10% that collapses the payload-aided map back onto the
reference-only one. The physics requires the *transmitted* waveform to change:
findings.md section 4 shows the effect is dominated by the thinned-aperture
sidelobe pedestal, not by processing gain, so simply masking resource elements
in the receiver would reproduce the gain change and hide the pedestal change -
which is to say it would show the wrong result convincingly. Each occupancy is
a different waveform bundle, so the buttons relaunch the flowgraph. It takes
about two seconds and it is the honest version of the demo.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

import numpy as np
import pmt
import pyqtgraph as pg
import zmq
from pyqtgraph.Qt import QtCore, QtWidgets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# label, bundle directory, and what the bundle is for the caption
OCCUPANCY = [
    ("100%", "data/wf_occ100", "filled aperture"),
    ("50% rand", "data/wf_occ050", "thinned, full bandwidth"),
    ("50% cont", "data/wf_cont050", "filled but narrow"),
    ("10% rand", "data/wf_occ010", "sparse"),
]


class Sub:
    """Non-blocking ZMQ SUB that decodes the blocks' PMT-wrapped JSON."""

    def __init__(self, ctx, endpoint):
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.connect(endpoint)

    def drain(self, limit=32):
        """Every message waiting, oldest first. Bounded so a backlog cannot
        stall the GUI thread: the maps are large and a slow consumer that
        tries to catch up completely never redraws."""
        out = []
        for _ in range(limit):
            try:
                raw = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = pmt.deserialize_str(raw)
                # The payload is a u8vector, not a symbol: pmt::serialize
                # caps a symbol at 65535 characters and a dwell carrying two
                # maps is ~166 kB. See ntnisac.dwell.pack.
                v = pmt.dict_ref(msg, pmt.intern("data"), pmt.PMT_NIL)
                if pmt.is_null(v):
                    continue
                out.append(json.loads(bytes(pmt.u8vector_elements(v)).decode()))
            except Exception:                          # noqa: BLE001
                continue
        return out


class Card(QtWidgets.QFrame):
    """One metric: a big value, a unit, a caption."""

    def __init__(self, title, unit="", note=""):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("QFrame{background:#1b1b1b;border:1px solid #333;"
                           "border-radius:4px;}")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(1)
        t = QtWidgets.QLabel(title)
        t.setStyleSheet("color:#9aa;font-size:10pt;")
        self.val = QtWidgets.QLabel("--")
        self.val.setStyleSheet("color:#e8e8e8;font-size:19pt;font-weight:600;")
        self.sub = QtWidgets.QLabel(note or unit)
        self.sub.setStyleSheet("color:#667;font-size:8pt;")
        for w in (t, self.val, self.sub):
            lay.addWidget(w)
        self.unit = unit

    def set(self, value, fmt="{:.2f}", note=None, colour=None):
        self.val.setText("--" if value is None else
                         (fmt.format(value) + (f" {self.unit}" if self.unit else "")))
        if note is not None:
            self.sub.setText(note)
        self.val.setStyleSheet(f"color:{colour or '#e8e8e8'};font-size:19pt;"
                               "font-weight:600;")


class MapPane:
    """One delay-Doppler map, with its own peak/noise annotation."""

    def __init__(self, layout, title, row, col, cmap):
        self.plot = layout.addPlot(row=row, col=col, title=title)
        self.plot.setLabel("bottom", "excess path", units="m")
        self.plot.setLabel("left", "residual Doppler", units="Hz")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.img = pg.ImageItem()
        self.img.setColorMap(cmap)
        self.plot.addItem(self.img)
        self.label = pg.TextItem(anchor=(0, 0), color="#ffd")
        self.plot.addItem(self.label)
        self.truth = pg.ScatterPlotItem(size=15, pen=pg.mkPen("#ff4444", width=2),
                                        brush=None, symbol="o")
        self.det = pg.ScatterPlotItem(size=11, pen=pg.mkPen("#44ff88", width=2),
                                      brush=None, symbol="x")
        self.plot.addItem(self.truth)
        self.plot.addItem(self.det)
        self.have = False

    def update(self, payload, axes, levels):
        a = np.asarray(payload["db"], np.float32).reshape(payload["shape"])
        d, f = axes["delay_m"], axes["doppler_hz"]
        self.img.setImage(a, levels=levels, autoLevels=False)
        self.img.setRect(QtCore.QRectF(d[0], f[0], d[-1] - d[0], f[-1] - f[0]))
        self.label.setText(f"peak/noise {payload['pk_nf_db']:.1f} dB\n"
                           f"sum(W) {payload['w_sum']:.0f}")
        self.label.setPos(d[0], f[-1])
        if not self.have:
            self.plot.setXRange(d[0], d[-1], padding=0)
            self.plot.setYRange(f[0], f[-1], padding=0)
            self.have = True

    def marks(self, truth_pts, det_pts):
        self.truth.setData([p[0] for p in truth_pts], [p[1] for p in truth_pts])
        self.det.setData([p[0] for p in det_pts], [p[1] for p in det_pts])


class Dashboard(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("NTN ISAC - two-stage coarse-to-fine")
        self.resize(1500, 950)
        self.setStyleSheet("background:#111;color:#ddd;")

        self.ctx = zmq.Context()
        self.s1 = Sub(self.ctx, args.zmq1)
        self.s2 = Sub(self.ctx, args.zmq2)
        self.s3 = Sub(self.ctx, args.zmq3)

        # A short history rather than just the latest sample. Truth is
        # published on the sample clock and a fine result arrives ~600 ms of
        # sample time after its dwell began, so comparing the newest truth
        # against the newest result mismatches them by up to a dwell. With the
        # drone's excess path changing at ~4.7 m/s that showed up as a delay
        # error drifting steadily from -1.2 m to +1.0 m over four seconds -
        # a scoring artefact that looks exactly like a real estimator bias.
        self.truth = None
        self.truth_hist = deque(maxlen=600)
        self.proc = None
        self.bundle = args.wf
        self.hist_t = deque(maxlen=400)
        self.hist_dr = deque(maxlen=400)
        self.hist_dv = deque(maxlen=400)
        self.n_fine = 0
        self.n_hit = 0
        self.n_fa = 0
        self.levels = (0.0, 60.0)
        self.blind = None
        self.coarse_err = None
        self._last_map = 0.0

        self._build()
        if args.attach:
            # The buttons relaunch a flowgraph this process does not own, so
            # they cannot work here. Disabled rather than left live and inert:
            # a dead control in a demo reads as a broken dashboard.
            for b in self.buttons.values():
                b.setEnabled(False)
            self.status.setText("attached - occupancy buttons need launch mode")
        else:
            self.launch(self.bundle)

        # Section 8 asks for numbers at 10 Hz and maps at 5-10 Hz. The poll
        # runs at 20 Hz and the map redraw is throttled separately, because
        # stage 2 only produces a map every ~600 ms at stride 1 and redrawing
        # a 43x99 image is not what costs anything - decoding a 170 kB JSON
        # payload is.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)

    # ---- construction ---------------------------------------------------
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        # pass state strip
        strip = QtWidgets.QHBoxLayout()
        self.pass_lbl = {}
        for k, t in (("range", "slant range"), ("rate", "range rate"),
                     ("dopp", "Doppler"), ("elev", "elevation"),
                     ("lock", "track"), ("bundle", "waveform")):
            box = QtWidgets.QVBoxLayout()
            a = QtWidgets.QLabel(t)
            a.setStyleSheet("color:#778;font-size:8pt;")
            b = QtWidgets.QLabel("--")
            b.setStyleSheet("color:#dde;font-size:12pt;font-weight:600;")
            box.addWidget(a)
            box.addWidget(b)
            self.pass_lbl[k] = b
            strip.addLayout(box)
        strip.addStretch(1)
        root.addLayout(strip)

        # maps
        cmap = pg.colormap.get("viridis", source="matplotlib")
        gl = pg.GraphicsLayoutWidget()
        gl.setBackground("#111")
        self.panes = {
            "reference-only": MapPane(gl, "reference-only  (W = ref_mask)",
                                      0, 0, cmap),
            "payload-aided": MapPane(gl, "payload-aided  (W = occ_mask)",
                                     0, 1, cmap),
        }
        self.cbar = pg.ColorBarItem(values=self.levels, colorMap=cmap,
                                    label="dB over own noise floor")
        gl.addItem(self.cbar, row=0, col=2)
        gl.ci.layout.setColumnStretchFactor(0, 10)
        gl.ci.layout.setColumnStretchFactor(1, 10)
        gl.ci.layout.setColumnStretchFactor(2, 1)
        root.addWidget(gl, stretch=5)

        note = QtWidgets.QLabel(
            "Shared colour scale, in dB over each map's own noise floor - "
            "handoff section 7 normalises each map by sqrt(sum W), so absolute "
            "magnitudes are not comparable between variants but "
            "dB-over-own-noise is.   ○ commanded truth   ✕ detection")
        note.setStyleSheet("color:#778;font-size:8pt;")
        root.addWidget(note)

        # metric cards
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        self.cards = {
            "evm": Card("EVM", "%", "data REs, best-fit reference"),
            "ser": Card("SER (uncoded)", "", "vs tx_grid - NOT BLER, see below"),
            "tput": Card("offered", "Mb/s", "from allocation, not delivered"),
            "lat": Card("CPI latency", "ms", "dwell is 100 ms"),
            "derr": Card("delay error", "m", "cell is 66.6 m"),
            "verr": Card("range-rate error", "m/s", "vs commanded"),
            "pd": Card("detection rate", "", "fine dwells with the target"),
            "load": Card("PDSCH load", "%", "per-symbol occupancy"),
        }
        for i, k in enumerate(("evm", "ser", "tput", "lat",
                               "derr", "verr", "pd", "load")):
            grid.addWidget(self.cards[k], 0, i)
        root.addLayout(grid)

        bler = QtWidgets.QLabel(
            "BLER is absent deliberately: it needs LDPC decoding, which "
            "handoff section 12 forbids in the live receive path. The uncoded "
            "SER above is a real measurement one layer down, not a substitute "
            "figure of merit.    Elevation is absent because traj.csv carries "
            "ranges only - it would need a gen_trajectory column.")
        bler.setStyleSheet("color:#886;font-size:8pt;")
        bler.setWordWrap(True)
        root.addWidget(bler)

        # error history
        self.hist = pg.PlotWidget()
        self.hist.setBackground("#151515")
        self.hist.setLabel("bottom", "pass time", units="s")
        self.hist.setLabel("left", "estimated - commanded")
        self.hist.addLegend(offset=(-10, 10))
        self.hist.showGrid(x=True, y=True, alpha=0.25)
        self.c_dr = self.hist.plot(pen=pg.mkPen("#5cf", width=2), name="delay error (m)")
        self.c_dv = self.hist.plot(pen=pg.mkPen("#fc6", width=2),
                                   name="range-rate error (m/s)")
        root.addWidget(self.hist, stretch=3)

        # occupancy control
        ctl = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel("PDSCH occupancy:")
        lab.setStyleSheet("color:#9aa;font-size:10pt;")
        ctl.addWidget(lab)
        self.buttons = {}
        for name, path, cap in OCCUPANCY:
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            btn.setToolTip(f"{path} - {cap}")
            btn.setStyleSheet(
                "QPushButton{background:#232323;border:1px solid #3a3a3a;"
                "padding:5px 12px;color:#ccc;border-radius:3px;}"
                "QPushButton:checked{background:#2b4a6b;color:#fff;"
                "border:1px solid #5b8;}")
            btn.clicked.connect(lambda _, p=path: self.launch(p))
            ctl.addWidget(btn)
            self.buttons[path] = btn
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color:#7a8;font-size:9pt;")
        ctl.addWidget(self.status)
        ctl.addStretch(1)
        hint = QtWidgets.QLabel(
            "switching relaunches the flowgraph: the occupancy effect is "
            "dominated by the transmitted aperture, so masking in the receiver "
            "would show the wrong result")
        hint.setStyleSheet("color:#667;font-size:8pt;")
        ctl.addWidget(hint)
        root.addLayout(ctl)

    # ---- flowgraph subprocess -------------------------------------------
    def launch(self, bundle):
        if self.args.attach:
            return
        self.stop_proc()
        self.bundle = bundle
        for p, b in self.buttons.items():
            b.setChecked(p == bundle)
        cmd = [sys.executable, os.path.join(ROOT, "gr", "flowgraphs", "twostage.py"),
               "--wf", bundle, "--traj", self.args.traj,
               "--t-start", str(self.args.t_start),
               "--n-frames", str(self.args.n_frames),
               "--esn0-db", str(self.args.esn0_db),
               "--stride", str(self.args.stride),
               "--seconds", str(self.args.seconds),
               "--variant", "payload-aided",
               "--extra-variants", "reference-only",
               "--maps", "--link", "--quiet",
               "--zmq", self.args.zmq1, "--zmq2", self.args.zmq2]
        if self.args.mode == "replay":
            cmd += ["--mode", "replay", "--capture", self.args.capture]
        # Reset the derived state: a new bundle is a new experiment and
        # carrying the old detection rate across would misreport it.
        self.hist_t.clear(); self.hist_dr.clear(); self.hist_dv.clear()
        self.n_fine = self.n_hit = self.n_fa = 0
        for pane in self.panes.values():
            pane.have = False
        self.proc = subprocess.Popen(cmd, cwd=ROOT,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self.status.setText(f"running {os.path.basename(bundle)}")

    def stop_proc(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def closeEvent(self, ev):
        self.stop_proc()
        super().closeEvent(ev)

    # ---- polling ---------------------------------------------------------
    def drone_truth(self, at=None):
        """Commanded drone excess and residual Doppler, interpolated to `at`.

        `at` should be the middle of the dwell being scored: a coherent
        integration reports the average over its own span, not the value at
        its first sample.
        """
        if at is None or len(self.truth_hist) < 2:
            t = self.truth or {}
            return t.get("p2_excess_m"), t.get("p2_residual_doppler_hz")
        ts = np.array([h[0] for h in self.truth_hist])
        ex = np.array([h[1] for h in self.truth_hist])
        fd = np.array([h[2] for h in self.truth_hist])
        if at < ts[0] or at > ts[-1]:
            # Outside the buffered span, so an interpolation would silently
            # become an extrapolation. Report no truth instead.
            return None, None
        return float(np.interp(at, ts, ex)), float(np.interp(at, ts, fd))

    def poll(self):
        for d in self.s3.drain():
            self.truth = d
            if "p2_excess_m" in d:
                self.truth_hist.append((d["t_s"], d["p2_excess_m"],
                                        d.get("p2_residual_doppler_hz", 0.0)))
        for d in self.s1.drain():
            self.on_coarse(d)
        msgs = self.s2.drain(limit=4)
        if msgs:
            self.on_fine(msgs[-1])          # newest only; maps are the payload
        if self.proc and self.proc.poll() is not None:
            self.status.setText(f"flowgraph exited ({self.proc.returncode})")

    def on_coarse(self, d):
        tk = d.get("track")
        mid = d["t"] + 0.5 * float(d.get("dwell_s") or 0.0)
        tm, _tf = self.drone_truth(mid)
        if tm is not None:
            self.coarse_err = d["excess_m"] - tm
        if tk:
            self.pass_lbl["lock"].setText(
                f"{'CONFIRMED' if tk.get('confirmed') else 'tentative'} "
                f"h{tk['hits']}")
            self.pass_lbl["lock"].setStyleSheet(
                "font-size:12pt;font-weight:600;color:"
                + ("#6f8" if tk.get("confirmed") else "#fc6"))
        t = self.truth or {}
        if "slant_range_m" in t:
            self.pass_lbl["range"].setText(f"{t['slant_range_m']/1e3:.1f} km")
        if "p0_doppler_hz" in t:
            lam = 299792458.0 / float(t.get("carrier_hz", 2e9))
            self.pass_lbl["dopp"].setText(f"{t['p0_doppler_hz']/1e3:+.2f} kHz")
            self.pass_lbl["rate"].setText(
                f"{-t['p0_doppler_hz']*lam:+.0f} m/s")
        self.pass_lbl["elev"].setText("n/a")
        self.pass_lbl["bundle"].setText(os.path.basename(self.bundle))

    def on_fine(self, d):
        axes = d.get("map_axes")
        maps = d.get("maps") or {}
        if axes and maps:
            # Shared levels across both panes, from the strongest map present:
            # section 8 requires one colour scale so the collapse of the
            # payload-aided map onto the reference-only one is visible as a
            # change in the image, not absorbed by two autoscales.
            top = max(float(np.max(np.asarray(m["db"]))) for m in maps.values())
            self.levels = (max(top - 55.0, 0.0), top)
            self.cbar.setLevels(self.levels)
            for name, pane in self.panes.items():
                if name in maps:
                    pane.update(maps[name], axes, self.levels)

        # Score against the middle of this dwell, not against whatever truth
        # happened to arrive most recently.
        mid = d["t0"] + 0.5 * float(d.get("dwell_s") or 0.0)
        tm, tf = self.drone_truth(mid)
        dets = d.get("detections") or []
        tp = [(tm, tf)] if tm is not None else []
        dp = [(x["excess_m"], x["doppler_hz"]) for x in dets]
        for pane in self.panes.values():
            pane.marks(tp, dp)

        L = d.get("link") or {}
        self.cards["evm"].set(100 * L["evm"] if "evm" in L else None, "{:.2f}")
        if "ser_uncoded" in L:
            s = L["ser_uncoded"]
            self.cards["ser"].set(s, "{:.2e}" if s else "{:.0f}",
                                  note=("vs tx_grid - NOT BLER"
                                        if L.get("slicer_valid") else
                                        "INVALID: slicer needs QPSK"))
        self.cards["tput"].set(L.get("offered_mbps"), "{:.2f}")
        lat = (d.get("latency_ms") or {}).get("total")
        self.cards["lat"].set(lat, "{:.0f}",
                              colour="#f88" if lat and lat > 1000 else None)

        # The notch plus its guard band is a real blind velocity band and the
        # display has to say so, or a non-detection inside it looks like a
        # miss rather than a documented limitation.
        self.blind = (d.get("resolution") or {}).get("blind_mps", self.blind)

        r = d.get("resource") or {}
        if "per_symbol_fraction" in r:
            self.cards["load"].set(100 * r["per_symbol_fraction"], "{:.0f}",
                                   note=f"H2 {r.get('renyi_bits_per_symbol', 0):.2f} b"
                                        f", frag {r.get('run_fragmentation', 0):.2f}")

        # Detection accounting against commanded truth. A "hit" is a detection
        # within two delay cells of the drone; anything else on a
        # target-present dwell is a false alarm. Reported as a running rate
        # rather than a probability - one bundle at one echo ratio is not a Pd
        # curve, and section 10's sweep is what produces one.
        self.n_fine += 1
        if tm is not None:
            # Associate on both axes, then take the STRONGEST survivor.
            # Nearest-in-delay alone picked the wrong one whenever a second
            # detection shared the target's delay - which the notch-edge
            # residue did - and reported its Doppler error as the target's:
            # -8.5 m/s instead of -0.02 m/s, on a dwell whose actual estimate
            # was correct.
            near = [x for x in dets
                    if abs(x["excess_m"] - tm) < 133.0
                    and (tf is None or abs(x["doppler_hz"] - tf) < 30.0)]
            if near:
                self.n_hit += 1
                best = max(near, key=lambda x: x.get("snr_db", 0.0))
                dr = best["excess_m"] - tm
                lam = 299792458.0 / float((self.truth or {}).get("carrier_hz", 2e9))
                dv = (tf - best["doppler_hz"]) * lam
                self.hist_t.append(d["t0"])
                self.hist_dr.append(dr)
                self.hist_dv.append(dv)
                self.cards["derr"].set(dr, "{:+.2f}")
                self.cards["verr"].set(dv, "{:+.3f}")
            else:
                self.cards["derr"].set(None)
                self.cards["verr"].set(None)
            self.n_fa += max(len(dets) - len(near), 0)
        if self.blind is not None:
            self.cards["verr"].sub.setText(
                f"vs commanded; blind +-{self.blind:.1f} m/s")
        self.cards["pd"].set(self.n_hit / max(self.n_fine, 1), "{:.2f}",
                             note=f"{self.n_hit}/{self.n_fine} dwells, "
                                  f"{self.n_fa} other detections")
        if self.hist_t:
            self.c_dr.setData(list(self.hist_t), list(self.hist_dr))
            self.c_dv.setData(list(self.hist_t), list(self.hist_dv))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wf", default="data/wf_occ100")
    p.add_argument("--traj", default="data/traj")
    p.add_argument("--t-start", type=float, default=500.0, dest="t_start")
    p.add_argument("--n-frames", type=int, default=5, dest="n_frames")
    p.add_argument("--esn0-db", type=float, default=30.0, dest="esn0_db")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seconds", type=float, default=60.0,
                   help="signal seconds per launch")
    p.add_argument("--mode", choices=("sim", "replay"), default="sim")
    p.add_argument("--capture", default="")
    p.add_argument("--attach", action="store_true",
                   help="subscribe to an already-running flowgraph instead of "
                        "launching one; the occupancy buttons then do nothing")
    p.add_argument("--zmq1", default="tcp://127.0.0.1:5555")
    p.add_argument("--zmq2", default="tcp://127.0.0.1:5556")
    p.add_argument("--zmq3", default="tcp://127.0.0.1:5557")
    p.add_argument("--screenshot", default="",
                   help="render to this PNG after --screenshot-after seconds "
                        "and exit; for headless verification")
    p.add_argument("--screenshot-after", type=float, default=25.0,
                   dest="shot_after")
    a = p.parse_args()

    # Deliberately NOT imageAxisOrder="row-major". map_roi returns
    # [delay, doppler], and pyqtgraph's default column-major ordering maps
    # array[i, j] to x = i, y = j - so delay lands on x and Doppler on y with
    # no transpose. Setting row-major here silently swaps the two axes, and a
    # delay-Doppler map with the axes exchanged still looks plausible.
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    w = Dashboard(a)
    w.show()

    if a.screenshot:
        def shoot():
            w.grab().save(a.screenshot)
            print(f"screenshot -> {a.screenshot}")
            w.stop_proc()
            app.quit()
        QtCore.QTimer.singleShot(int(a.shot_after * 1000), shoot)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
