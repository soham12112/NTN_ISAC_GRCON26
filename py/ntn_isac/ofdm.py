"""CP-OFDM modulation and demodulation matched to the generator.

The convention here is fixed by config.json and must not drift: cyclic prefix
fraction 0 (the FFT window starts at the first sample after the CP) and no
time-domain windowing. Both set the origin of the delay axis, so a change here
silently biases every sensing measurement.
"""

from __future__ import annotations

import numpy as np


def symbol_layout(cfg: dict, n_sym: int = None):
    """Per-symbol CP lengths and absolute sample start offsets.

    Returns (cp, start) with cp[l] the cyclic prefix of symbol l and start[l]
    the index of its first CP sample in the waveform. `n_sym` may exceed one
    file length: the waveform loops, so the CP pattern simply repeats.
    """
    cp_slot = np.asarray(cfg["cp_lengths"], dtype=np.int64)
    n_sym_slot = int(cfg["n_sym_slot"])
    if n_sym is None:
        n_sym = int(cfg["n_sym_total"])
    n_fft = int(cfg["n_fft"])
    if cp_slot.size != n_sym_slot:
        raise ValueError("cp_lengths length does not match n_sym_slot")
    cp = np.tile(cp_slot, -(-n_sym // n_sym_slot))[:n_sym]
    start = np.concatenate(([0], np.cumsum(cp + n_fft)[:-1]))
    return cp, start


def slot_rate_hz(cfg: dict) -> float:
    """Slots per second. Exactly 1000 Hz for 15 kHz SCS, computed not assumed.

    This is the repetition rate of the *aperture*, not of the waveform. The
    generator redraws the PRB set every slot, so the sensing weight W is
    periodic at the slot rate, and any tap in the delay-Doppler map acquires
    sidebands at multiples of it. Measured on a filled grid: a drone tap at
    -31.9 Hz produced a 13 dB spur at +1968.17 Hz, which is 2*1000 - 31.9.
    Consequently the Doppler search must stay inside +-slot_rate/2, where no
    sideband of an in-band tap can fall.
    """
    cp = np.asarray(cfg["cp_lengths"], dtype=np.int64)
    per_slot = int(cfg["n_sym_slot"]) * int(cfg["n_fft"]) + int(cp.sum())
    return float(cfg["fs_hz"]) / per_slot


def subcarrier_bins(n_sc: int, n_fft: int) -> np.ndarray:
    """FFT bin index for each grid subcarrier 0..n_sc-1 (DC-centred mapping)."""
    return (np.arange(n_sc) - n_sc // 2) % n_fft


def demodulate(x: np.ndarray, cfg: dict, cp_fraction: float = None,
               sym_offset: int = 0, n_sym: int = None,
               allow_short: bool = False) -> np.ndarray:
    """Waveform -> resource grid [n_sym, n_sc].

    `sym_offset` shifts the FFT window by a whole number of samples and is how
    the receiver applies its integer timing estimate. It may be a scalar or one
    value per symbol - the latter is what tracks a walking delay.

    Raises if `x` is too short to fill every symbol, unless `allow_short`.
    This used to break out of the loop and return zeros for the remainder,
    which is a silent failure and it hid a real bug for a long time: the
    per-dwell slices handed to receiver.demod_tracked were exactly one dwell
    long and did not include the bulk propagation delay, so the last symbols
    of every dwell were quietly zero. At t = 500 s the delay is 23664 samples
    and 44 of 1400 symbols were lost - 3.1% - which cost only 0.14 dB of
    coherent gain and so never showed up in any sensing check. It showed up as
    a 3.3% symbol error rate on a noiseless channel the first time anything
    measured the *data*. At the start of the pass the delay is 40386 samples,
    which is 26% of a one-frame dwell.
    """
    n_fft = int(cfg["n_fft"])
    n_sc = int(cfg["n_sc"])
    if cp_fraction is None:
        cp_fraction = float(cfg.get("cp_fraction", 0.0))
    if n_sym is None:
        n_sym = int(cfg["n_sym_total"])
    cp, start = symbol_layout(cfg, n_sym)
    bins = subcarrier_bins(n_sc, n_fft)

    off = np.broadcast_to(np.asarray(sym_offset, dtype=np.int64), (n_sym,))
    grid = np.zeros((n_sym, n_sc), dtype=np.complex64)
    n_filled = 0
    for l in range(n_sym):
        back = int(round(cp_fraction * cp[l]))
        i0 = start[l] + cp[l] - back + off[l]
        if i0 < 0 or i0 + n_fft > x.size:
            if not allow_short:
                raise ValueError(
                    f"demodulate: input has {x.size} samples but symbol {l} of "
                    f"{n_sym} needs samples {i0}..{i0 + n_fft}. The slice must "
                    f"cover the dwell plus the bulk propagation delay plus one "
                    f"FFT - see TwoStage.margin_samples().")
            break
        # Scaling matches nrOFDMModulate/Demodulate: plain fft/ifft, no 1/sqrt(N).
        blk = np.fft.fft(x[i0:i0 + n_fft])
        if back:
            # A window that starts `back` samples early sees the symbol
            # cyclically delayed, so undo that linear phase and keep the
            # delay origin where cp_fraction=0 would have put it.
            blk *= np.exp(2j * np.pi * back * np.fft.fftfreq(n_fft))
        grid[l] = blk[bins]
        n_filled = l + 1
    demodulate.n_filled = n_filled
    return grid


def modulate(grid: np.ndarray, cfg: dict) -> np.ndarray:
    """Resource grid [n_sym, n_sc] -> waveform. Inverse of demodulate()."""
    n_fft = int(cfg["n_fft"])
    n_sc = int(cfg["n_sc"])
    cp, start = symbol_layout(cfg)
    bins = subcarrier_bins(n_sc, n_fft)
    n_sym = grid.shape[0]
    out = np.zeros(int(start[n_sym - 1] + cp[n_sym - 1] + n_fft), dtype=np.complex64)
    for l in range(n_sym):
        blk = np.zeros(n_fft, dtype=np.complex128)
        blk[bins] = grid[l]
        sym = np.fft.ifft(blk)
        i0 = start[l]
        out[i0:i0 + cp[l]] = sym[n_fft - cp[l]:]
        out[i0 + cp[l]:i0 + cp[l] + n_fft] = sym
    return out
