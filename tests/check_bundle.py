#!/usr/bin/env python3
"""Contract test for the generator file interface.

Verifies, from Python only, that the bundle in `data/wf` is self-consistent:
the grid really is the OFDM demodulation of the IQ, the masks agree with the
grid, and the sparse reference symbols match the dense grid.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import numpy as np
from ntn_isac import fileio, ofdm

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "wf")

b = fileio.load(path)
cfg = b.cfg
iq = b.iq()
grid = np.asarray(b.grid())
ok = True

def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

print(f"bundle: {os.path.realpath(path)}  generator={cfg['generator']}")
print(f"  fs={cfg['fs_hz']/1e6:.3f} MS/s  fc={cfg['carrier_hz']/1e9:.3f} GHz  "
      f"grid={cfg['grid_shape']}  loopable={cfg['loopable']}")

check("iq length == frame_len_samples", iq.size == cfg["frame_len_samples"],
      f"{iq.size}")
check("iq length == sum of symbol lengths",
      iq.size == cfg["n_sym_total"] // cfg["n_sym_slot"] *
      (sum(cfg["cp_lengths"]) + cfg["n_sym_slot"] * cfg["n_fft"]))
rms, pk = float(np.sqrt(np.mean(abs(iq) ** 2))), float(abs(iq).max())
check("0.05 < rms < 0.2", 0.05 < rms < 0.2, f"rms={rms:.4f}")
check("peak < 0.9", pk < 0.9, f"peak={pk:.4f}  PAPR={20*np.log10(pk/rms):.2f} dB")

g2 = ofdm.demodulate(iq, cfg)
err = np.linalg.norm(g2 - grid) / np.linalg.norm(grid)
check("python demod(tx_iq) == tx_grid", err < 1e-5, f"rel-err={err:.3e}")

x2 = ofdm.modulate(grid, cfg)
err2 = np.linalg.norm(x2 - iq) / np.linalg.norm(iq)
check("python mod(tx_grid) == tx_iq", err2 < 1e-5, f"rel-err={err2:.3e}")

re_rms = float(np.sqrt(np.mean(abs(grid[b.occ_mask]) ** 2)))
leak = float(abs(grid[~b.occ_mask]).max()) if (~b.occ_mask).any() else 0.0
check("grid is exactly zero off occ_mask", leak == 0.0, f"max={leak:.3e}")
check("no empty RE inside occ_mask",
      float(abs(grid[b.occ_mask]).min()) > 1e-6 * re_rms)
check("ref_syms == grid[ref_mask]",
      np.allclose(b.ref_syms, grid[b.ref_mask], rtol=0, atol=1e-6 * re_rms))
check("ref_mask subset of occ_mask", bool(np.all(b.occ_mask[b.ref_mask])))
check("n_ref_re matches config", int(b.ref_mask.sum()) == cfg["n_ref_re"],
      f"{int(b.ref_mask.sum())}")
check("n_occ_re matches config", int(b.occ_mask.sum()) == cfg["n_occ_re"],
      f"{int(b.occ_mask.sum())}")

nss = cfg["n_sym_slot"]
occ_slot = b.occ_mask.reshape(-1, nss, cfg["n_sc"]).sum(axis=(1, 2))
print(f"  occupancy per slot: {occ_slot[:4]} ... {occ_slot[-1]}")
print(f"  reference REs {int(b.ref_mask.sum())} "
      f"({100*b.ref_mask.mean():.2f}% of grid, "
      f"{100*b.ref_mask.sum()/b.occ_mask.sum():.2f}% of occupied)")

print("OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
