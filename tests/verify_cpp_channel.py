#!/usr/bin/env python3
"""The C++ block must be the Python reference, not merely similar.

Offline analysis, the V1-V3 checkpoints and the section 10 campaign all run the
Python channel; the live flowgraph runs the C++ one. If they diverge, a result
measured offline does not describe what went over the cable.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import numpy as np
from gnuradio import blocks, gr
from gnuradio import ntnisac

from ntn_isac import fileio, leo_channel, trajectory

root = os.path.join(os.path.dirname(__file__), "..", "data")
CSV = os.path.join(root, "traj", "traj.csv")
b = fileio.load(os.path.join(root, "wf"))
tj = trajectory.load(os.path.join(root, "traj"))
fs, fc, frame = b.fs, b.fc, b.cfg["frame_len_samples"]
BIAS, TAPS, T0 = 16.0, 16, 60.0
C = 299792458.0
tx = np.tile(b.iq(), 3)

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

def make_cpp(taps=TAPS, t0=T0, r_nom=None):
    # Pass r_nominal explicitly rather than letting the block take the minimum
    # over the CSV: both routes now agree, but the flowgraph reads traj_meta
    # anyway and being explicit keeps offline and live bit-identical.
    return ntnisac.leo_channel(CSV, fs, fc,
                               tj.r_nominal if r_nom is None else r_nom,
                               BIAS, 131072, taps, t0)


def run_cpp(x, taps=TAPS, t0=T0, tune=None):
    tb = gr.top_block()
    src = blocks.vector_source_c(x.tolist(), False)
    ch = make_cpp(taps, t0)
    snk = blocks.vector_sink_c()
    if tune:
        tune(ch)
    tb.connect(src, ch, snk)
    tb.run()
    return np.array(snk.data(), np.complex64), 0.0, ch


def bench_cpp(n, taps):
    """Throughput with trivial endpoints, so the number is the block's.

    vector_source/vector_sink move data through Python-owned std::vectors and
    dominate a short run; null_source and null_sink do not.
    """
    tb = gr.top_block()
    src = blocks.null_source(gr.sizeof_gr_complex)
    hd = blocks.head(gr.sizeof_gr_complex, n)
    ch = make_cpp(taps)
    snk = blocks.null_sink(gr.sizeof_gr_complex)
    tb.connect(src, hd, ch, snk)
    t_a = time.perf_counter()
    tb.run()
    return n / (time.perf_counter() - t_a)

def run_py(x, taps=TAPS, t0=T0, kind="sinc", **kw):
    ch = leo_channel.from_trajectory(tj, fs, t_start_s=t0, interp=kind,
                                     interp_taps=taps, tau_bias_samples=BIAS, **kw)
    t_a = time.perf_counter()
    y = ch.process(x)
    return y, time.perf_counter() - t_a, ch

print("construction and geometry")
_, _, cpp0 = run_cpp(tx[:1024])
check("r_nominal agrees", abs(cpp0.r_nominal() - tj.r_nominal) < 1e-6,
      f"cpp {cpp0.r_nominal():.6f} vs py {tj.r_nominal:.6f}")
check("num_paths", cpp0.num_paths() == len(tj.paths), f"{cpp0.num_paths()}")
check("duration", abs(cpp0.duration() - tj.duration) < 1e-9, f"{cpp0.duration()} s")

print("\nspline: C++ not-a-knot vs scipy CubicSpline, off-knot samples")
_, _, c = run_cpp(tx[:64], t0=0.0)
tt = np.linspace(0.0, tj.duration, 101)
for p in range(3):
    d_cpp = []
    for t in tt:
        c.set_time(float(t))
        d_cpp.append(c.tap_delay_samples(p))
    d_cpp = np.array(d_cpp)
    d_py = ((tj.paths[p].range_at(tt) - tj.r_nominal) / C + BIAS / fs) * fs
    err = np.abs(d_cpp - d_py).max() / fs * C
    check(f"path {p} range agrees", err < 1e-8, f"max {err:.3e} m")

print("\nwaveform equivalence, 60 ms through the real pass")
y_c, t_c, _ = run_cpp(tx)
y_p, t_p, _ = run_py(tx)
n = min(y_c.size, y_p.size)
rel = np.linalg.norm(y_c[:n] - y_p[:n]) / np.linalg.norm(y_p[:n])
check("same length", y_c.size == y_p.size, f"cpp {y_c.size} py {y_p.size}")
check("C++ == Python (16-tap sinc)", rel < 3e-6, f"rel-err {rel:.3e}")

print("\ncubic Farrow path (interp_taps=4)")
y_c4, _, _ = run_cpp(tx, taps=4)
y_p4, _, _ = run_py(tx, kind="lagrange", taps=4)
n = min(y_c4.size, y_p4.size)
rel4 = np.linalg.norm(y_c4[:n] - y_p4[:n]) / np.linalg.norm(y_p4[:n])
check("C++ Farrow == Python Farrow", rel4 < 3e-6, f"rel-err {rel4:.3e}")

print("\ntrajectory boundary")
edge_s = 1e-3
y_edge, _, _ = run_cpp(tx, t0=tj.duration - edge_s)
want_edge = int(np.floor(edge_s * fs))
check("finite pass stops instead of extrapolating",
      abs(y_edge.size - want_edge) <= 1,
      f"{y_edge.size} samples, expected {want_edge}")

print("\nruntime controls")
y_off, _, _ = run_cpp(tx[:frame], tune=lambda c: [c.set_enable(1, False),
                                                  c.set_enable(2, False)])
y_ref, _, _ = run_py(tx[:frame], enable=[True, False, False])
n = min(y_off.size, y_ref.size)
r_en = np.linalg.norm(y_off[:n] - y_ref[:n]) / np.linalg.norm(y_ref[:n])
check("set_enable matches python enable", r_en < 3e-6, f"rel-err {r_en:.3e}")
y_g, _, _ = run_cpp(tx[:frame], tune=lambda c: [c.set_enable(2, False),
                                                c.set_gain_db(1, -30.0)])
y_gr, _, _ = run_py(tx[:frame], enable=[True, True, False],
                    gain_override_db=[None, -30.0, None])
n = min(y_g.size, y_gr.size)
r_g = np.linalg.norm(y_g[:n] - y_gr[:n]) / np.linalg.norm(y_gr[:n])
check("set_gain_db matches python override", r_g < 3e-6, f"rel-err {r_g:.3e}")

print("\nthroughput, 3 paths, target 7.68 MS/s")
N = 4 * 1024 * 1024
for taps in (4, 8, 12, 16, 24):
    r = bench_cpp(N, taps)
    kind = "lagrange" if taps == 4 else "sinc"
    _, dtp, _ = run_py(tx, taps=taps, kind=kind)
    rp = tx.size / dtp
    flag = "OK " if r > fs else "SLOW"
    print(f"  {taps:2d}-tap: {r/1e6:6.2f} MS/s  {r/fs:5.2f}x real time  {flag}"
          f"   (python {rp/1e6:5.2f} MS/s, speedup {r/rp:5.1f}x)")

print("\nOK" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
