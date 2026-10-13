#!/usr/bin/env python3
"""Shared dwell buffering for the sensing blocks."""

import json
import os
import sys

import numpy as np
import pmt
from gnuradio import gr


def add_repo_py():
    """Put the project's `py/` on sys.path.

    The sensing blocks are thin wrappers over py/ntn_isac, deliberately: the
    DSP is verified offline by tests/verify_twostage.py and must be the same
    code in the flowgraph, or a live result and a replay of the same capture
    can disagree with nothing to point at.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.abspath(os.path.join(here, "..", "..", "..", "py")),
              os.path.abspath(os.path.join(os.getcwd(), "py")),
              os.environ.get("NTN_ROOT") and
              os.path.join(os.environ["NTN_ROOT"], "py")):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


class _Enc(json.JSONEncoder):
    """Make numpy scalars and arrays serialisable."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def pmt_data():
    """Interned key for the payload vector. Cached: interning a symbol on
    every message is measurable at a 20 Hz message rate."""
    global _DATA_KEY
    if _DATA_KEY is None:
        _DATA_KEY = pmt.intern("data")
    return _DATA_KEY


_DATA_KEY = None


def pack(kind, obj):
    """Wrap a JSON-shaped payload as a PMT message.

    The payload travels as a JSON string under the "json" key rather than as a
    PMT dict tree, for one blunt reason: PMT's Python mapping is not a
    round trip for the shapes these payloads have. A one-element PMT cons list
    built with pmt.cons(x, PMT_NIL) comes back from pmt.to_python as the pair
    (x, None) rather than as a list, so a dwell with exactly one detection
    deserialised into something whose [0] was the detection and whose [1] was
    None - which then reached float() and raised. Nested dicts fared no better.

    Both consumers are Python - the sibling sensing block and the PyQtGraph
    dashboard over ZMQ - and handoff section 7 asks for "one dict per coherent
    processing interval over ZMQ", so JSON is the format that was wanted
    anyway. `kind` stays a plain PMT symbol so a GRC message debug sink still
    shows something useful without parsing.

    Non-finite floats are allowed through. NaN means "no estimate"; replacing
    it with 0 would turn that into an estimate of zero, which is the kind of
    thing that reaches a slide. Python's json reads them back.
    """
    b = json.dumps(obj, cls=_Enc, allow_nan=True).encode("utf-8")
    d = pmt.make_dict()
    d = pmt.dict_add(d, pmt.intern("kind"), pmt.intern(str(kind)))
    d = pmt.dict_add(d, pmt_data(), pmt.init_u8vector(len(b), list(b)))
    return d


def unpack(msg):
    """Inverse of `pack`. Returns (kind, payload) or (None, None)."""
    try:
        k = pmt.dict_ref(msg, pmt.intern("kind"), pmt.PMT_NIL)
        kind = None if pmt.is_null(k) else pmt.symbol_to_string(k)
        v = pmt.dict_ref(msg, pmt_data(), pmt.PMT_NIL)
        if pmt.is_null(v):
            return None, None
        return kind, json.loads(bytes(pmt.u8vector_elements(v)).decode("utf-8"))
    except Exception:                                  # noqa: BLE001
        return None, None


class DwellBlock(gr.sync_block):
    """Accumulate whole dwells of samples, then call `on_dwell`.

    The dwell advances by exactly `samples_per_dwell` on the transmit-side
    clock. That is correct even though the delay walks 170 samples/s: the
    receiver's per-symbol retiming happens inside receiver.demod_tracked, so
    the dwell boundary is a transmit-time boundary and does not move. Trying
    to advance the boundary by the walked amount instead would double-count
    the correction.

    `stride` processes one dwell in every `stride` and discards the rest.
    This is how a block whose processing costs more than its dwell keeps up
    with a live stream deliberately rather than by dropping buffers at random:
    an overrun is a decision, taken here, and reported in the payload.
    """

    def __init__(self, name, samples_per_dwell, fs, t_start=0.0, stride=1,
                 in_sig=None, margin=0):
        gr.sync_block.__init__(
            self, name=name,
            in_sig=[np.complex64] if in_sig is None else in_sig,
            out_sig=None)
        self.n_dwell = int(samples_per_dwell)
        self.margin = max(int(margin), 0)
        self.fs = float(fs)
        self.t_start = float(t_start)
        self.stride = max(int(stride), 1)
        # A sliding window, not a chain of disjoint blocks. The window is one
        # dwell plus the margin and it advances by one dwell, so consecutive
        # windows overlap by `margin` - which is what supplies the last
        # symbols of a dwell, since those arrive after the bulk propagation
        # delay. See TwoStage.margin_samples().
        self._win = self.n_dwell + self.margin
        self._buf = np.empty(self._win, np.complex64)
        self._have = 0
        self._start = 0
        self._index = 0
        self.n_processed = 0
        self.n_skipped = 0

    def dwell_t0(self, first_sample_index):
        return self.t_start + first_sample_index / self.fs

    def on_dwell(self, y, t0, index):
        raise NotImplementedError

    def work(self, input_items, output_items):
        x = input_items[0]
        n = len(x)
        i = 0
        while i < n:
            take = min(self._win - self._have, n - i)
            self._buf[self._have:self._have + take] = x[i:i + take]
            self._have += take
            i += take
            if self._have < self._win:
                break
            if self._index % self.stride == 0:
                try:
                    self.on_dwell(self._buf.copy(),
                                  self.dwell_t0(self._start), self._index)
                    self.n_processed += 1
                except Exception as exc:               # noqa: BLE001
                    # A dwell that fails must not take the flowgraph with it.
                    # The transmit side is real RF; losing the graph mid-pass
                    # loses the capture.
                    gr.log.error(f"{self.name()}: dwell {self._index} failed: {exc}")
            else:
                self.n_skipped += 1
            self._index += 1
            # Slide by one dwell, keeping the overlap.
            if self.margin:
                self._buf[:self.margin] = self._buf[self.n_dwell:]
            self._have = self.margin
            self._start += self.n_dwell
        return n
