#!/usr/bin/env python3
"""Live display for the over-the-air sensing path.

    $NTN_PY gr/ota_dashboard.py                 # attach to a running flowgraph
    $NTN_PY gr/ota_dashboard.py --launch bench  # start the bench graph too

A separate process fed over ZMQ, deliberately: a display that crashes must not
be able to take the flowgraph - and with it a capture that cannot be repeated -
down with it.

`gr/dashboard.py` is the display for the emulated-target two-stage flow, and it
stays. This one shows the thing that flow cannot: what happens at the direct
path's own delay cell, which is where a physical object between the two radios
lives and where a delay-Doppler map on its own tells you nothing.

## What the panels are for

Everything here is arranged around one claim, which is that over the air the
target is not hiding in the noise - it is hiding under the direct path.

**Delay-Doppler, before and after.** Two images, cropped to a few delay cells
around the direct path, on ONE colour scale. The scale is fixed by the
uncancelled map's noise floor, so a cancelled map cannot look better merely by
being renormalised. Cancellation that only shows up when each image is scaled
to itself is not cancellation, it is rescaling.

**The Doppler cut.** The same information as a line, which is easier to read
because the whole measurement is one-dimensional: a physical target at 0.006 of
a delay cell has no delay information to show. Both traces are drawn, and the
distance between them at the target's Doppler is the thing the over-the-air
path exists to produce.

**Cancellation history.** How far the direct path came down, per dwell.
Measured locally, in a box around where the direct path was, not as a global
change in the map.

**Micro-Doppler.** The rotation rate needs a one-second accumulation to resolve
the line spacing, so it arrives an order of magnitude slower than everything
else and gets its own card. Harmonics of the recovered rate are drawn over the
Doppler cut, which is the quickest way to see whether the estimate is the
fundamental or one of its harmonics.

**Lock state.** Before anything else is meaningful the receiver has to have
acquired. An unlocked run still draws plausible-looking maps.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
import zmq
from pyqtgraph.Qt import QtCore, QtWidgets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Sub:
    """Non-blocking ZMQ SUB that decodes the blocks' PMT-wrapped JSON.

    Payloads are UTF-8 JSON inside a PMT u8vector - see ntnisac.dwell.pack for
    why they cannot be PMT symbols. Rather than depend on pmt here, the JSON is
    found by looking for the first '{' and the last '}', which is robust to the
    PMT framing around it and costs nothing.
    """

    def __init__(self, ctx, endpoint):
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.connect(endpoint)

    def drain(self, limit=32):
        out = []
        for _ in range(limit):
            try:
                raw = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            i, j = raw.find(b"{"), raw.rfind(b"}")
            if i < 0 or j <= i:
                continue
            try:
                out.append(json.loads(raw[i:j + 1].decode("utf-8", "replace")))
            except Exception:                          # noqa: BLE001
                continue
        return out


class Card(QtWidgets.QFrame):
    def __init__(self, title, unit=""):
        super().__init__()
        self.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame{background:#161a1f;border-radius:6px;}")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(9, 5, 9, 6)
        v.setSpacing(1)
        t = QtWidgets.QLabel(title)
        t.setStyleSheet("color:#8b98a8;font-size:10px;")
        self.val = QtWidgets.QLabel("--")
        self.val.setStyleSheet("color:#e8eef5;font-size:19px;font-weight:600;")
        self.note = QtWidgets.QLabel(unit)
        self.note.setStyleSheet("color:#6f7d8c;font-size:9px;")
        for w in (t, self.val, self.note):
            v.addWidget(w)

    def set(self, text, note=None, colour="#e8eef5"):
        self.val.setText(text)
        self.val.setStyleSheet(
            f"color:{colour};font-size:19px;font-weight:600;")
        if note is not None:
            self.note.setText(note)


class MapPane:
    def __init__(self, layout, title, col):
        self.plot = layout.addPlot(row=0, col=col, title=title)
        self.plot.setLabel("bottom", "excess path", units="m")
        self.plot.setLabel("left", "Doppler", units="Hz")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.plot.setMouseEnabled(False, False)

    def update(self, m, levels):
        if not m:
            return
        h, w = m["shape"]
        a = np.asarray(m["db"], np.float32).reshape(h, w)
        x0, x1 = m["excess_m"]
        y0, y1 = m["doppler_hz"]
        self.img.setImage(a.T, levels=levels, autoLevels=False)
        self.img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))


class Dashboard(QtWidgets.QWidget):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.proc = None
        self.setWindowTitle("NTN ISAC - over-the-air sensing")
        self.resize(1500, 880)
        self.setStyleSheet("background:#0d1014;color:#e8eef5;")
        pg.setConfigOptions(antialias=True, background="#0d1014",
                            foreground="#8b98a8")

        ctx = zmq.Context()
        self.subs = [Sub(ctx, e) for e in (a.zmq, a.zmq2, a.zmq3)]
        self.cancel_hist = deque(maxlen=240)
        self.t_hist = deque(maxlen=240)
        self.levels = (0.0, 90.0)
        self.locked = False
        self.n_dwell = 0
        self.last_micro = None

        self._build()
        if a.launch:
            self.launch(a.launch)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(80)

    # ------------------------------------------------------------------ ui

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        hdr = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(
            "Over-the-air sensing  -  the target is in the direct path's own "
            "delay cell")
        self.title.setStyleSheet("font-size:14px;font-weight:600;")
        hdr.addWidget(self.title)
        hdr.addStretch(1)
        self.lock = QtWidgets.QLabel("WAITING")
        self.lock.setStyleSheet("color:#ffb454;font-size:13px;font-weight:700;")
        hdr.addWidget(self.lock)
        root.addLayout(hdr)

        cards = QtWidgets.QHBoxLayout()
        cards.setSpacing(8)
        self.cards = {}
        for k, (t, u) in dict(
                direct=("Direct path", "dB over noise"),
                cancelled=("CLEAN removed", "dB, measured locally"),
                rot=("Rotation rate", "needs a 1 s dwell"),
                edge=("Doppler edge", "tip speed"),
                radius=("Implied radius", "from edge and rate"),
                eps=("Clock offset", "ppm between the radios"),
                resid=("Tracking residual", "samples / Hz"),
                dwells=("Dwells", "processed"),
        ).items():
            c = Card(t, u)
            self.cards[k] = c
            cards.addWidget(c)
        root.addLayout(cards)

        maps = pg.GraphicsLayoutWidget()
        maps.setFixedHeight(300)
        self.before = MapPane(maps, "Before CLEAN - the direct path", 0)
        self.after = MapPane(maps, "After CLEAN - same colour scale", 1)
        root.addWidget(maps)

        lower = pg.GraphicsLayoutWidget()
        self.cut = lower.addPlot(row=0, col=0, colspan=2,
                                 title="Doppler at the direct path's delay")
        self.cut.setLabel("bottom", "Doppler", units="Hz")
        self.cut.setLabel("left", "dB over noise")
        self.cut.addLegend(offset=(-10, 10), labelTextSize="8pt")
        self.cut.showGrid(x=True, y=True, alpha=0.2)
        self.tr_before = self.cut.plot(pen=pg.mkPen("#55606d", width=1),
                                       name="before CLEAN")
        self.tr_after = self.cut.plot(pen=pg.mkPen("#4ea3ff", width=1.4),
                                      name="after CLEAN")
        self.tr_micro = self.cut.plot(pen=pg.mkPen("#ffd166", width=1.2),
                                      name="1 s micro-Doppler dwell")
        self.harm = []

        self.hist = lower.addPlot(row=1, col=0, colspan=2,
                                  title="Direct-path cancellation per dwell")
        self.hist.setLabel("bottom", "pass time", units="s")
        self.hist.setLabel("left", "dB removed")
        self.hist.showGrid(x=True, y=True, alpha=0.2)
        self.tr_hist = self.hist.plot(pen=pg.mkPen("#6ee7a0", width=1.4))
        root.addWidget(lower)

        foot = QtWidgets.QLabel(
            "Both maps share one colour scale, fixed by the uncancelled map's "
            "noise floor. A cancelled map normalised to itself looks improved "
            "when nothing has moved.")
        foot.setStyleSheet("color:#6f7d8c;font-size:10px;")
        root.addWidget(foot)

    # -------------------------------------------------------------- launch

    def launch(self, which):
        self.stop_proc()
        py = os.environ.get("NTN_PY", sys.executable)
        if which == "bench":
            cmd = [py, "-u", os.path.join(ROOT, "gr", "flowgraphs", "ota_rx.py"),
                   "--mode", "sim", "--seconds", "0",
                   "--wf", self.a.wf, "--traj", self.a.traj,
                   "--t-start", str(self.a.t_start),
                   "--seconds", str(self.a.seconds),
                   "--blade-rot-hz", str(self.a.rot_hz),
                   "--blade-gain-db", str(self.a.gain_db),
                   "--no-emulated", "--quiet",
                   "--zmq", self.a.zmq, "--zmq2", self.a.zmq2]
        else:
            cmd = [py, "-u", which]
        self.proc = subprocess.Popen(cmd, cwd=ROOT)
        self.title.setText(f"Over-the-air sensing  -  launched {which}")

    def stop_proc(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=4)
            except Exception:                          # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def closeEvent(self, ev):
        self.stop_proc()
        ev.accept()

    # ---------------------------------------------------------------- poll

    def poll(self):
        for s in self.subs:
            for d in s.drain():
                k = self.kind(d)
                if k == "ota":
                    self.on_ota(d)
                elif k == "micro":
                    self.on_micro(d)
                elif k == "state":
                    self.on_state(d)

    @staticmethod
    def kind(d):
        if "direct" in d and "tracker" in d:
            return "ota"
        if "rotation_hz" in d and "dwell_s" in d:
            return "micro"
        if "locked" in d:
            return "state"
        return None

    def on_state(self, d):
        if not d.get("locked", True):
            self.locked = False
            self.lock.setText(
                f"NOT LOCKED   acq peak {d.get('acq_peak_db', float('nan')):.1f} dB")
            self.lock.setStyleSheet(
                "color:#ff6b6b;font-size:13px;font-weight:700;")

    def on_ota(self, d):
        self.n_dwell += 1
        if not self.locked:
            self.locked = True
            self.lock.setText("LOCKED")
            self.lock.setStyleSheet(
                "color:#6ee7a0;font-size:13px;font-weight:700;")

        dr, tk, mi = d["direct"], d.get("tracker", {}), d.get("micro", {})
        pk = float(dr.get("pk_nf_db", float("nan")))
        cd = float(dr.get("cancelled_db", float("nan")))
        self.levels = (0.0, max(pk, 20.0))
        self.cards["direct"].set(f"{pk:.1f}")
        # Below about 10 dB the canceller has not found the direct path, which
        # usually means the tracker has not settled rather than that CLEAN
        # failed. Worth colouring, because the maps still look plausible.
        self.cards["cancelled"].set(
            f"{cd:.1f}", colour="#6ee7a0" if cd > 20 else "#ffb454")
        self.cards["eps"].set(f"{tk.get('eps_ppm', 0.0):+.3f}")
        self.cards["resid"].set(
            f"{d['residual']['delay_samples']:+.2f} / "
            f"{d['residual']['doppler_hz']:+.2f}")
        self.cards["dwells"].set(str(self.n_dwell))
        e = mi.get("extent_hz")
        if e is not None and np.isfinite(e):
            self.cards["edge"].set(f"{e:.0f} Hz",
                                   f"{mi.get('tip_speed_mps', 0):.1f} m/s")

        self.before.update(d.get("map_before"), self.levels)
        self.after.update(d.get("map_after"), self.levels)

        pb, pa = d.get("profile_before"), d.get("profile")
        if pb:
            self.tr_before.setData(pb["doppler_hz"], pb["level_db"])
        if pa:
            self.tr_after.setData(pa["doppler_hz"], pa["level_db"])

        self.t_hist.append(float(d.get("t0", self.n_dwell)))
        self.cancel_hist.append(cd)
        self.tr_hist.setData(list(self.t_hist), list(self.cancel_hist))

    def on_micro(self, d):
        self.last_micro = d
        r = d.get("rotation_hz")
        ok = r is not None and np.isfinite(r)
        self.cards["rot"].set(
            f"{r*60:.0f} RPM" if ok else "unresolved",
            f"{r:.2f} rev/s, conf {d.get('rotation_conf', 0):.2f}" if ok
            else "dwell too short to resolve the line spacing",
            colour="#6ee7a0" if ok else "#ffb454")
        rad = d.get("radius_m")
        if rad is not None and np.isfinite(rad):
            self.cards["radius"].set(f"{rad*100:.1f} cm")
        p = d.get("profile")
        if p:
            self.tr_micro.setData(p["doppler_hz"], p["level_db"])

        for h in self.harm:
            self.cut.removeItem(h)
        self.harm = []
        if ok and r > 0:
            # Harmonics of the recovered rate, drawn over the cut. The quickest
            # way to see whether the estimate is the fundamental or one of its
            # harmonics: if every other line is empty, it is a harmonic.
            for k in range(1, int(500.0 / r) + 1):
                for s in (-1, 1):
                    ln = pg.InfiniteLine(pos=s * k * r, angle=90,
                                         pen=pg.mkPen("#ffd166", width=0.5,
                                                      style=QtCore.Qt.PenStyle.DotLine))
                    self.cut.addItem(ln)
                    self.harm.append(ln)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zmq", default="tcp://127.0.0.1:5555")
    p.add_argument("--zmq2", default="tcp://127.0.0.1:5556")
    p.add_argument("--zmq3", default="tcp://127.0.0.1:5557")
    p.add_argument("--launch", default="",
                   help="'bench' to start the simulated flowgraph too, or a "
                        "path to a flowgraph to run")
    p.add_argument("--wf", default=os.path.join(ROOT, "data", "wf_ota"))
    # traj_ota, not traj_bench: in sim mode ota_rx adds its own rotating
    # target on top of the trajectory, so a trajectory that already contains
    # one would put two in the same delay cell.
    p.add_argument("--traj", default=os.path.join(ROOT, "data", "traj_ota"))
    # Short on purpose. In sim mode the whole signal is generated before the
    # flowgraph starts, so --seconds is dead time at the front: 120 s of signal
    # is over a minute of a blank dashboard before the first dwell arrives.
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--rot-hz", type=float, default=7.0, dest="rot_hz")
    p.add_argument("--gain-db", type=float, default=-30.0, dest="gain_db")
    p.add_argument("--t-start", type=float, default=300.0, dest="t_start")
    a = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = Dashboard(a)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
