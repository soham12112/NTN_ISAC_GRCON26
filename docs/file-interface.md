# Generator file interface

Handoff section 3, with two additions. Everything upstream of the transmit
flowgraph produces these files; everything downstream consumes them and does
not know who wrote them. Getting this right is what makes the OCUDU swap cheap.

## Unchanged from section 3

| file | content |
|---|---|
| `tx_iq.cf32` | interleaved float32 LE complex baseband at `fs_hz`; RMS ~0.1, peak <0.9 |
| `tx_grid.c64` | interleaved float32 LE grid, C order `[n_sym_total, n_sc]`, element `(l,k)` at `l*n_sc + k` |
| `ref_mask.u8` | one byte per RE, 1 where the RE carries a regenerable reference |
| `config.json` | parameters and a file manifest |

`tx_grid.c64` is for EVM, truth, and the genie bound only. The live receiver
never reads it. It is memory-mapped everywhere in this codebase because a 60 s
OCUDU capture is over 2 GB.

## Addition 1: `ref_syms.c64`

Complex64, length `popcount(ref_mask)`, values of the reference REs in raster
order of the mask.

`ref_mask` alone is not sufficient. The receiver does least-squares channel
estimation, `H = Y / X` on reference REs, so it needs the reference *values*.
Without this file it would have to generate NR PSS, SSS, PBCH-DMRS and PDSCH
DMRS sequences itself, which is exactly the NR-specific knowledge the mask was
introduced to avoid. Shipping the values keeps the receiver generator-agnostic.

Stored sparse rather than dense so long captures stay affordable: references
are about 4-7% of the grid, so this is ~7% of the size of `tx_grid.c64`. That
keeps section 11's "dump only ref_mask and config for longer demo captures"
viable.

It does not leak the payload. It is defined only on reference REs, so the
payload-aided branch still has to slice for everything else.

## Addition 2: `occ_mask.u8`

One byte per RE, 1 where the RE carries any transmitted energy.

The payload-aided weight is not `W = 1` across the allocated bandwidth, because
the allocation varies per slot. Without an explicit occupancy map the receiver
integrates noise over empty REs; at 10% occupancy that is 90% noise, and the
occupancy demo would look catastrophic for the wrong reason.

This is legitimate knowledge for the receiver to have. Section 12 forbids DCI
decoding and section 4 says the allocation comes from `config.json`;
`occ_mask` is just the pixel-accurate form of that. A real receiver could also
recover it blind by per-PRB energy detection.

So the three variants are:

| variant | `W` | `X_est` |
|---|---|---|
| reference-only | `ref_mask` | `ref_syms` |
| payload-aided | `occ_mask` | references + sliced decisions |
| genie | `occ_mask` | `tx_grid` |

## Build the mask structurally, not by thresholding

`nrOFDMDemodulate` leaves ~1e-17 of numerical dust on unallocated REs. A test
like `abs(grid) > 0` reports 100% occupancy on a 50%-occupied grid. Both masks
are built from the toolbox's own resource index lists, and the grid is scrubbed
to exact zeros off `occ_mask` before writing.

## config.json keys beyond section 3

`cp_fraction` (0) and `windowing` (0) pin the OFDM convention: the FFT window
starts at the first sample after the CP and no time-domain windowing is
applied. Both set the origin of the delay axis. With `WindowingPercent = 0` the
demodulate/remodulate round-trip is exact to ~3e-16, so `tx_grid.c64` is a
bit-exact description of `tx_iq.cf32`.

Also added: `tx_scale`, `tx_rms`, `tx_peak`, `papr_db`, `n_ref_re`, `n_occ_re`,
`ncellid`, `ssb`, `n_sym_slot`, and `occupancy.pattern` / `occupancy.prb_sets`.

The MATLAB scaling convention is the plain `fft`/`ifft` pair with no `1/sqrt(N)`
factor, matching `nrOFDMModulate`. `py/ntn_isac/ofdm.py` reproduces it exactly.

## Trajectory files

`traj.csv` header is `t_s,R0_m,G0_db,R1_m,G1_db,R2_m,G2_db` as specified, at
100 Hz. Ranges are written to 1 um so the carrier phase term `fc*tau` is good
to ~7e-6 cycles.

`traj_meta.json` deviates from section 5 on one point. Section 5 says set
`R_nominal` to `R0` at t=0, but a pass that starts at low elevation has its
maximum range at t=0, which makes every tau negative and the delay line
non-causal. `r_nominal_m` is therefore the minimum range over all present
paths; `r0_at_t0_m` records what section 5 asked for. Any constant nominal is
physically equivalent - it is a fixed phase rotation and a fixed range offset,
and it leaves inter-path delays and all Doppler untouched.
