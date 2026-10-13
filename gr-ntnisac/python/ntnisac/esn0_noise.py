#!/usr/bin/env python3
"""Calibrated AWGN for the Es/N0 sweep of handoff section 10."""

import numpy as np
from gnuradio import gr


class esn0_noise(gr.sync_block):
    """Add complex AWGN at a specified Es/N0, in dB.

    Es is the energy per *occupied* resource element and N0 the noise density
    per resource element after the receiver's FFT. Getting from that to a
    time-domain sample variance is the whole reason this block exists, because
    the conversion involves the FFT size and the per-symbol occupancy and is
    easy to get wrong by 20 dB with a plain noise source and an adder.

    With this codebase's OFDM convention - plain fft/ifft, no 1/sqrt(N), fixed
    by config.json and matched to nrOFDMModulate:

        E|x_t|^2 = (1/n_fft^2) * n_occ * Es          (modulator has ifft's 1/N)
        N0_re    = n_fft * sigma^2                   (demodulator's fft has none)

    so

        sigma^2 = E|x_t|^2 * n_fft / (n_occ * Es/N0)

    where n_occ is occupied subcarriers per symbol, not per frame. Using the
    frame total instead - the tempting reading of config's n_occ_re - would be
    wrong by the symbol count, which is 22 dB here.

    The signal power E|x_t|^2 is measured at this block's input rather than
    taken from config.json, because by this point the waveform has been
    through the LEO channel and its power is the channel's business: path
    gains, the number of enabled paths, and any amplifier ahead of the DAC all
    change it. It is estimated over the first `settle` samples and then held,
    so a sweep is reproducible and a fading path cannot make the noise floor
    breathe. Pass `signal_rms` explicitly to pin it.
    """

    def __init__(self, esn0_db=30.0, n_fft=512, n_occ_per_sym=300,
                 signal_rms=-1.0, settle=200000, seed=0):
        gr.sync_block.__init__(self, name="esn0_noise",
                               in_sig=[np.complex64], out_sig=[np.complex64])
        self.n_fft = int(n_fft)
        self.n_occ = max(int(n_occ_per_sym), 1)
        self.settle = int(settle)
        self._rng = np.random.default_rng(int(seed))
        self._acc = 0.0
        self._n = 0
        self._fixed = float(signal_rms) > 0
        self._p_sig = float(signal_rms) ** 2 if self._fixed else 0.0
        self._sigma = 0.0
        self.set_esn0_db(esn0_db)

    # -- runtime controls, for the offline sweep -----------------------------
    def set_esn0_db(self, db):
        self.esn0_db = float(db)
        self._recompute()

    def set_signal_rms(self, rms):
        self._p_sig = float(rms) ** 2
        self._fixed = True
        self._recompute()

    def sigma(self):
        return self._sigma

    def signal_rms(self):
        return float(np.sqrt(self._p_sig))

    def _recompute(self):
        if self._p_sig <= 0:
            self._sigma = 0.0
            return
        lin = 10.0 ** (self.esn0_db / 10.0)
        var = self._p_sig * self.n_fft / (self.n_occ * lin)
        self._sigma = float(np.sqrt(max(var, 0.0)))

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        n = len(x)

        if not self._fixed and self._n < self.settle:
            take = min(n, self.settle - self._n)
            self._acc += float(np.sum(np.abs(x[:take].astype(np.complex128)) ** 2))
            self._n += take
            self._p_sig = self._acc / max(self._n, 1)
            self._recompute()

        if self._sigma > 0:
            # sigma is the total complex variance, so each of the two
            # quadratures carries half of it.
            s = self._sigma / np.sqrt(2.0)
            w = (self._rng.normal(0.0, s, n) + 1j * self._rng.normal(0.0, s, n))
            y[:] = (x + w.astype(np.complex64))
        else:
            y[:] = x
        return n
