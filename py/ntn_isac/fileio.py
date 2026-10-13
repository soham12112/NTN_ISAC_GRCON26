"""Readers for the generator file interface (handoff section 3, as amended).

Nothing here knows anything about NR. That is the whole point: the generator
can be MATLAB or OCUDU and this code does not change.

Files
-----
tx_iq.cf32    interleaved float32 LE, complex baseband at cfg["fs_hz"]
tx_grid.c64   interleaved float32 LE, C order [n_sym_total, n_sc];
              element (l, k) at index l*n_sc + k. Memory-mapped: at 60 s of
              OCUDU capture this is >2 GB and must not be read eagerly.
ref_mask.u8   one byte per RE, same shape and order, 1 = regenerable reference
occ_mask.u8   one byte per RE, 1 = the RE carries transmitted energy
ref_syms.c64  values of the reference REs only, in raster order of ref_mask;
              length == popcount(ref_mask). Sparse so that long captures can
              ship references without shipping the payload.
config.json   parameters and a file manifest
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


def read_cf32(path: str, count: int = -1, offset: int = 0) -> np.ndarray:
    """Interleaved float32 LE -> complex64. `count`/`offset` are in samples."""
    n = -1 if count < 0 else 2 * count
    raw = np.fromfile(path, dtype="<f4", count=n, offset=8 * offset)
    if raw.size % 2:
        raise ValueError(f"{path}: odd float count {raw.size}, not interleaved IQ")
    return raw.view(np.complex64)


def write_cf32(path: str, x: np.ndarray) -> None:
    np.asarray(x, dtype=np.complex64).tofile(path)


def map_grid(path: str, n_sym: int, n_sc: int) -> np.memmap:
    """Memory-map tx_grid.c64 as complex64 [n_sym, n_sc]."""
    want = n_sym * n_sc * 8
    have = os.path.getsize(path)
    if have != want:
        raise ValueError(f"{path}: {have} bytes, expected {want} for [{n_sym},{n_sc}]")
    return np.memmap(path, dtype="<c8", mode="r", shape=(n_sym, n_sc))


def read_mask(path: str, n_sym: int, n_sc: int) -> np.ndarray:
    m = np.fromfile(path, dtype=np.uint8)
    if m.size != n_sym * n_sc:
        raise ValueError(f"{path}: {m.size} bytes, expected {n_sym * n_sc}")
    return m.reshape(n_sym, n_sc).astype(bool)


@dataclass
class Bundle:
    """One generator output directory."""

    path: str
    cfg: dict
    ref_mask: np.ndarray  # bool [n_sym, n_sc]
    occ_mask: np.ndarray  # bool [n_sym, n_sc]
    ref_syms: np.ndarray  # complex64 [n_ref]

    @property
    def n_sym(self) -> int:
        return int(self.cfg["grid_shape"][0])

    @property
    def n_sc(self) -> int:
        return int(self.cfg["grid_shape"][1])

    @property
    def fs(self) -> float:
        return float(self.cfg["fs_hz"])

    @property
    def fc(self) -> float:
        return float(self.cfg["carrier_hz"])

    def iq(self) -> np.ndarray:
        return read_cf32(os.path.join(self.path, "tx_iq.cf32"))

    def grid(self) -> np.memmap:
        """The genie reference. Memory-mapped; never read this live."""
        return map_grid(os.path.join(self.path, "tx_grid.c64"), self.n_sym, self.n_sc)

    def ref_grid(self) -> np.ndarray:
        """Dense [n_sym, n_sc] complex64: reference values, zero elsewhere."""
        g = np.zeros((self.n_sym, self.n_sc), dtype=np.complex64)
        g[self.ref_mask] = self.ref_syms
        return g


class TiledBundle:
    """A bundle repeated over several looped frames, for a CPI longer than the
    waveform file. Duck-types Bundle for everything sensing.build_variant needs.

    Only valid when config["loopable"] is true. An OCUDU capture is not
    periodic, so there the CPI has to come out of one continuous grid instead.
    """

    def __init__(self, b: "Bundle", n_frames: int):
        if n_frames > 1 and not b.cfg.get("loopable", False):
            raise ValueError("cannot tile a non-loopable capture")
        self.base = b
        self.n_frames = int(n_frames)
        self.cfg = dict(b.cfg)
        self.cfg["n_sym_total"] = b.n_sym * self.n_frames
        self.cfg["grid_shape"] = [b.n_sym * self.n_frames, b.n_sc]
        self.ref_mask = np.tile(b.ref_mask, (self.n_frames, 1))
        self.occ_mask = np.tile(b.occ_mask, (self.n_frames, 1))
        self._ref = np.tile(b.ref_grid(), (self.n_frames, 1))

    def ref_grid(self):
        return self._ref

    def grid(self):
        return np.tile(np.asarray(self.base.grid()), (self.n_frames, 1))


def load(path: str) -> Bundle:
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    n_sym, n_sc = (int(v) for v in cfg["grid_shape"])
    ref_mask = read_mask(os.path.join(path, "ref_mask.u8"), n_sym, n_sc)
    occ_mask = read_mask(os.path.join(path, "occ_mask.u8"), n_sym, n_sc)
    ref_syms = read_cf32(os.path.join(path, "ref_syms.c64"))
    n_ref = int(ref_mask.sum())
    if ref_syms.size != n_ref:
        raise ValueError(
            f"ref_syms.c64 has {ref_syms.size} values but ref_mask marks {n_ref} REs"
        )
    if not np.all(occ_mask[ref_mask]):
        raise ValueError("ref_mask marks REs that occ_mask says are empty")
    return Bundle(path=path, cfg=cfg, ref_mask=ref_mask, occ_mask=occ_mask,
                  ref_syms=ref_syms)
