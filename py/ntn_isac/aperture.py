"""Renyi entropy of the sensing aperture, and what it does and does not predict.

Paper section IV-A gates the fine stage on a Renyi entropy criterion: only
downlink connections whose time-frequency occupancy is rich enough are used
for high-resolution sensing, "ensuring that only connections with sufficient
signal richness" are selected. That is the paper's proxy for ambiguity
function quality.

This module implements the gate and then checks the proxy, because
docs/findings.md section 4 already measured the thing the proxy stands in for.

The pedestal law
----------------
A randomly thinned aperture has a sidelobe pedestal set by the *variance* of
the thinning, not by the active count. Per OFDM symbol, with N subcarriers in
the aperture support and a fraction f of them active, the delay response has a
coherent main lobe of amplitude ~ f*N and an incoherent floor of power
~ N*f*(1-f), so

    pedestal / peak  ~  N f (1-f) / (f N)^2  =  (1 - f) / (f N)

With the per-symbol occupancy uniform over n_eff = f*N active subcarriers,
H_2 = log2(n_eff), which gives the entropy form

    pedestal_dBc  ~  -3.0103 * H_2  +  10*log10(1 - 2**H_2 / N)

Both terms fall as occupancy rises, so the entropy is a monotone proxy and the
paper's gate works. tests/entropy_pedestal.py measures the agreement.

Two things the entropy cannot do, and both matter here
-----------------------------------------------------
1. It must be computed *per symbol*. Taken over a whole coherent processing
   interval the union of 20 independently drawn PRB sets touches every
   subcarrier, so the aperture reads as full at 10% occupancy. `support_*`
   quantities below describe the union; `per_symbol_*` describe the aperture
   that actually sets the pedestal.

2. It is blind to allocation *pattern*. findings section 4 measured pattern
   mattering as much as fraction: at 50% occupancy random and contiguous PRB
   sets have identical per-symbol entropy - same active count - but -21.1 dB
   and -34.7 dB of far-out sidelobe. `run_fragmentation` is the cheap addition
   that separates them.

Renyi entropy of order alpha on a normalised occupancy p:

    H_alpha = 1/(1-alpha) * log2( sum_k p_k^alpha )

alpha = 2, the collision entropy, is the default: 2**H_2 is exactly the
inverse participation ratio 1/sum(p^2), which is the effective active count
the pedestal law needs.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHA = 2.0


def _renyi(p: np.ndarray, alpha: float) -> float:
    """H_alpha in bits of an already-normalised, strictly positive p."""
    if p.size == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-9:                # Shannon limit; 1/(1-a) is singular
        return float(-np.sum(p * np.log2(p)))
    return float(np.log2(np.sum(p ** alpha)) / (1.0 - alpha))


def aperture_profile(W: np.ndarray) -> np.ndarray:
    """Per-subcarrier weight summed over symbols: the coherent aperture.

    The delay axis of the sensing map is the Fourier transform of the occupied
    subcarriers, so this vector - not the 2-D mask - sets delay resolution and
    the shape of the direct path's point spread function.
    """
    return np.asarray(W, dtype=np.float64).sum(axis=0)


def support(W: np.ndarray) -> np.ndarray:
    """Boolean: subcarriers carrying weight in at least one symbol."""
    return aperture_profile(W) > 0


def n_span(W: np.ndarray) -> int:
    """Subcarriers in the aperture support. This is the N of the pedestal law."""
    return int(np.count_nonzero(support(W)))


def support_fraction(W: np.ndarray) -> float:
    """Support size over the full subcarrier axis. Union over the whole CPI, so
    it saturates at 1.0 for any per-slot-randomised scheduler - which is why it
    is reported but never used to predict anything."""
    a = aperture_profile(W)
    return float(np.count_nonzero(a) / a.size) if a.size else 0.0


def per_symbol_fraction(W: np.ndarray) -> float:
    """Median active fraction of the support, over symbols that carry anything.

    This is the f of the pedestal law. Two choices worth stating:

    Symbols with no weight are excluded. Including them would report a lower
    occupancy for a waveform that merely has idle symbols, which does not thin
    the aperture at all.

    The median, not the mean, because the symbol population is heterogeneous:
    4 of the 280 symbols in a frame carry the 240-subcarrier SSB rather than
    the 300-subcarrier PDSCH aperture. Averaging pulled a fully allocated
    waveform to f = 0.994, which made the pedestal law predict -46.7 dBc for
    an aperture that has no thinning pedestal at all. The median reports 1.0
    and the law correctly returns -inf.
    """
    Wf = np.asarray(W, dtype=np.float64) > 0
    n = n_span(W)
    if n == 0:
        return 0.0
    per_sym = Wf.sum(axis=1)
    live = per_sym[per_sym > 0]
    if live.size == 0:
        return 0.0
    return float(np.median(live) / n)


def renyi_entropy(W: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """H_alpha of the normalised coherent aperture profile, bits.

    The paper's quantity: entropy of the time-frequency occupancy of the
    allocation. Use this for the gate.
    """
    a = aperture_profile(W)
    tot = a.sum()
    if tot <= 0:
        return 0.0
    p = a / tot
    return _renyi(p[p > 0], alpha)


def renyi_entropy_per_symbol(W: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Median over symbols of H_alpha of that symbol's own occupancy, bits.

    This, not `renyi_entropy`, is what the pedestal law is a function of: the
    thinning is redrawn per slot, so the pedestal is set by the aperture one
    symbol at a time. Median for the reason given in `per_symbol_fraction`.
    """
    Wf = np.abs(np.asarray(W, dtype=np.float64))
    tot = Wf.sum(axis=1)
    live = np.flatnonzero(tot > 0)
    if live.size == 0:
        return 0.0
    hs = [_renyi(Wf[l][Wf[l] > 0] / tot[l], alpha) for l in live]
    return float(np.median(hs))


def effective_subcarriers(W: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """2**H_alpha per symbol: effective active subcarriers in one symbol.

    Exactly the active count for a 0/1 aperture, which is what makes alpha = 2
    the right order for a pedestal prediction.
    """
    return float(2.0 ** renyi_entropy_per_symbol(W, alpha))


def predicted_pedestal_db(W: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Thinning pedestal relative to the main peak, dBc, from the entropy.

    -3.0103*H_2 + 10*log10(1 - 2**H_2/N). Returns -inf for a per-symbol-filled
    aperture, where the random-thinning pedestal genuinely vanishes and the
    measured floor is set by windowing and float32 arithmetic instead
    (findings section 4 measured -62.6 dBc there).
    """
    if n_span(W) == 0:
        return 0.0
    h = renyi_entropy_per_symbol(W, alpha)
    f = per_symbol_fraction(W)
    if f >= 1.0:
        return -np.inf
    return float(-10 * np.log10(2.0) * h + 10 * np.log10(1.0 - f))


def spectral_flatness(W: np.ndarray) -> float:
    """Geometric over arithmetic mean of the aperture profile on its support.

    1.0 when every active subcarrier carries equal total weight. Unlike the
    entropy this responds to unequal weighting, which is what a real
    scheduler's per-slot PRB churn produces across a CPI.
    """
    a = aperture_profile(W)
    a = a[a > 0]
    if a.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(a))) / np.mean(a))


def run_fragmentation(W: np.ndarray) -> float:
    """Contiguous-run count over the largest possible count, in [0, 1].

    The pattern discriminator the entropy lacks. 0 is one solid block - a
    contiguous allocation, so a narrow aperture with low far sidelobes; near 1
    is maximal scattering - full delay resolution paid for with a pedestal.
    Both are realistic schedulers and findings section 4 measured 13.6 dB
    between them at equal occupancy, and therefore at equal entropy.

    Normalised by the *allocation granularity*, not by the subcarrier count.
    NR schedules whole 12-subcarrier PRBs, so even a maximally scattered PRB
    set has runs at least 12 long and can never approach one run per
    subcarrier. Dividing by a subcarrier-granular maximum reported 0.08 for a
    scattered 50% allocation and 0.01 for a contiguous one - a 0.07 spread
    that no threshold can use. Taking the granularity as the shortest observed
    run turns the same two allocations into 0.50 and 0.00.

    Computed per symbol and medianed, for the same reason the entropy is: the
    union over a CPI is not the aperture that sets the pedestal.
    """
    Wf = np.asarray(W, dtype=np.float64) > 0
    live = np.flatnonzero(Wf.sum(axis=1) > 0)
    if live.size == 0:
        return 0.0
    out = []
    for l in live:
        a = Wf[l]
        n_act = int(a.sum())
        if n_act == 0:
            continue
        edges = np.diff(np.concatenate(([0], a.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        n_runs = starts.size
        gran = int((ends - starts).min())          # shortest run = the unit
        n_units = max(n_act // max(gran, 1), 1)
        out.append((n_runs - 1) / max(n_units - 1, 1))
    return float(np.median(out)) if out else 0.0


def describe(W: np.ndarray, alpha: float = DEFAULT_ALPHA) -> dict:
    """Everything the resource gate and the telemetry need, in one dict."""
    return dict(
        alpha=float(alpha),
        renyi_bits=renyi_entropy(W, alpha),
        renyi_bits_per_symbol=renyi_entropy_per_symbol(W, alpha),
        n_eff_sc=effective_subcarriers(W, alpha),
        n_span_sc=n_span(W),
        support_fraction=support_fraction(W),
        per_symbol_fraction=per_symbol_fraction(W),
        predicted_pedestal_db=predicted_pedestal_db(W, alpha),
        spectral_flatness=spectral_flatness(W),
        run_fragmentation=run_fragmentation(W),
    )


def gate(W: np.ndarray, min_bits: float, alpha: float = DEFAULT_ALPHA,
         max_fragmentation: float = None) -> bool:
    """Admit this allocation to the fine stage?

    `min_bits` alone is the paper's criterion. `max_fragmentation` is optional
    and off by default, so the paper's gate is what runs unless a caller asks
    for the stricter one; tests/entropy_pedestal.py is the evidence that the
    stricter one is worth having.
    """
    if renyi_entropy_per_symbol(W, alpha) <= float(min_bits):
        return False
    if max_fragmentation is not None and run_fragmentation(W) > max_fragmentation:
        return False
    return True
