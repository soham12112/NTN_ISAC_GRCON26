# The two-stage flow, and where it departs from the paper

Source: W. Khawaja, A. M. Alam, A. C. Gurbuz, "A Two-Stage 5G NR ISAC
Framework for Coarse-to-Fine UAV Detection and Tracking", IEEE DySPAN 2026.
Equation numbers below are the paper's. The IEEE PDF is not redistributed here.

The paper's idea transfers to this testbed cleanly. Its arithmetic does not,
in four places, and all four are hardware rather than effort. Read section 2
before trusting any number this chain produces.

## 1. What is built

```
   file / capture / USRP
        |
   LEO Channel      ntnisac.leo_channel   (C++, existing)
        |
   Es/N0 Noise      ntnisac.esn0_noise    (new)
        |
   LEO Truth        ntnisac.leo_truth     (new)  --msg--> truth
        |
        +--> SSB Coarse   ntnisac.ssb_coarse   stage 1  --msg "coarse"--+
        |         ^                                                     |
        |         +----------------- msg "refine" -----------------+    |
        |                                                          |    v
        +--> Grid Sense   ntnisac.grid_sense    stage 2  ----------+--> ZMQ
```

`gr-ntnisac/examples/ntnisac_twostage.grc` is that graph on a GRC canvas;
`gr/flowgraphs/twostage.py` is the same thing as a script with sim, replay and
live modes. The DSP itself lives in `py/ntn_isac/` and the blocks are thin
wrappers over it, deliberately: `tests/verify_twostage.py` verifies the
offline path, and a live result that disagreed with a replay of the same
capture would have nothing to point at.

The message loop runs **both ways**, and that is the whole point. Stage 1
publishes a Gaussian range gate (eq 9); stage 2 returns refined measurements
which go into stage 1's tracker. Without the return path the flow is
two-stage but not coarse-to-fine, and stage 1's Doppler ambiguity is never
resolved.

## 2. The four structural departures

### 2.1 No angle of arrival. The map is delay-Doppler, not range-AoA

The paper beamforms across an 8-element ULA (eq 4, 13, 31) and its fine
output is a range-AoA map (eq 33) thresholded in range-azimuth (eq 34). This
implemented branch consumes one receive channel. There is no spatial aperture,
so there is no AoA, at any SNR, by any processing. The available B210 channels
make a future multi-channel branch possible; they do not create angle data in
this one.

Consequences that must not be glossed:

- The fine map's second axis is Doppler. CA-CFAR and DBSCAN are axis-agnostic
  and transfer unchanged; the AoA estimator does not exist here.
- Cartesian position is not observable. See 2.4.
- The paper's headline comparison is AoA error CDFs across UEs. That
  comparison cannot be reproduced. What can be, and is, is the
  delay/range-rate accuracy of the two stages against commanded truth.

### 2.2 One receiver, so no fusion and no uplink reporting

Paper section III is a distributed layer: several UEs compute Gaussian state
parameters (eq 22), report them over Release-17 Small Data Transmission on
pre-configured PUSCH (eq 23), and the gNB fuses them by weighted least
squares (eq 24) before an EKF forecast (eq 27). None of that is reproducible
with one receiver, and handoff section 12 puts multi-node fusion out of
scope.

What survives is the *information flow*: a coarse estimate with an
uncertainty, handed to a stage that uses it as a gate. Here it is handed in
process over a GNU Radio message port rather than over a radio link. The
message payload carries everything a second node would need, so adding one
later changes nothing in these blocks.

### 2.3 Bistatic and one-way, so no factor of 4 pi

Eq 11 reads `v = lambda * dphi / (4 pi T_rep)`, which is the monostatic
two-way round trip. This channel injects `exp(-j 2 pi f_c tau)` on a one-way
excess path (handoff section 6), so a tap's phase is `-2 pi f_c tau` and

```
f_d = dphi / (2 pi T_rep)        excess range rate = -lambda * f_d
```

Using the paper's constant would halve every velocity estimate. The same
one-way convention is used everywhere in this codebase, including
`coarse.CoarseResult.range_rate_mps` and `detect.Detection.range_rate_mps`.

### 2.4 The tracker state is the observable pair, not Cartesian position

The paper's filter is six-state, `[x, y, z, vx, vy, vz]` (eq 18-21), and it
is observable there because each UE contributes range *and* AoA and several
UEs are fused. Here one bistatic receiver produces two numbers per dwell:
excess delay and residual Doppler. A 3-D position consistent with those lies
on the intersection of an iso-delay ellipsoid and an iso-Doppler cone, which
is a curve. Position is not observable.

`py/ntn_isac/track.py` therefore tracks `[excess_range_m, range_rate_mps]`
with the constant-velocity transition of eq 20 reduced to two dimensions and
`H = I2`. The measurement function is then exactly linear, so this is a
linear Kalman filter rather than an EKF - which is the same simplification
the paper reaches by its own route (eq 25: "the measurement Jacobian H
simplifies to an identity matrix, eliminating the need for complex analytical
derivatives"), just for a structural reason instead of a geometric one.

A six-state filter fed two numbers per update would report its own process
model back as a position, with a covariance that looks tight because nothing
ever contradicts it. That is the failure this refuses.

## 3. Stage 1: what changed and what it costs

Faithful to the paper: frequency-domain matched filtering against the known
SSB (eq 6), MTI by adjacent-repetition subtraction (eq 7), non-coherent range
profile (eq 8), Gaussian prediction gate (eq 9), parabolic sub-cell
refinement (eq 10), and velocity from the inter-repetition phase difference
(eq 12) rather than a Doppler FFT - the paper's reasoning for which
("standard Doppler FFT processing is unsuitable for the irregular and limited
SSB repetitions in 5G NR") applies here unchanged.

Adaptations:

**One beam, not a sweep.** The illuminator is a LEO satellite with a fixed
beam over the pass, so SSB index carries no angular sector. The "search radar
scanning a field of view" framing does not apply; the pulsed-illuminator
framing, which is what stage 1 actually uses, does.

**Matched filtering is a per-subcarrier product.** Eq 6 is written as an FFT,
a conjugate multiply and an IFFT. On an OFDM waveform the FFTs are already
done by the demodulator, so the matched filter is one multiply and one IFFT.
No time-domain correlation anywhere.

**PBCH payload is excluded.** `ref_mask` marks only PSS, SSS and PBCH-DMRS -
what a receiver can regenerate from NCellID. PBCH payload is occupied but not
known without decoding the MIB, and counting it as a reference would flatter
stage 1 with knowledge a UE does not have yet.

**The Hamming window is applied across the SSB span, not across the sparse
reference elements.** A window shapes an aperture, and the aperture is the
240-subcarrier block. Tapering by position-in-list would mean nothing.

### 3.1 One SSB per 20 ms is a 50 Hz PRF, and that is a hard ambiguity

The waveform carries one SSB in slot 0 with a period equal to the file
length, so the pulse-pair repetition interval is the 20 ms frame: PRF 50 Hz,
unambiguous only to +-25 Hz. `traj_meta.json` puts the drone's residual
Doppler between -36.7 and +77.1 Hz.

Measured, dwell 0 of a clean run: coarse Doppler `+18.51 Hz` against a truth
of `-31.49 Hz`. The error is one PRF to within **6 mHz**. There is no
processing fix at this PRF; the ambiguity is resolved by prediction, which is
what the paper's own forecast gating provides.

Two ways it gets resolved here, both measured:

| bootstrap route | dwells to converge | resulting Doppler error |
|---|---|---|
| stage 2's refine message | 1 | +-0.04 Hz |
| stage 1 alone, from the delay sequence | ~4 | +-0.05 Hz |

The second route matters because stage 2 is not always available. The
ambiguous Doppler carries no usable rate information, so the rate is measured
as zero with a variance wide enough to cover any alias - which makes its
Kalman gain negligible - and the filter recovers the rate from the *sequence*
of delay residuals through the cross-covariance instead.

This took two attempts. Both failures are worth knowing because both look
like working code:

- Using the gate's floored delay sigma (83 m, one cell) as the tracker's
  measurement sigma instead of the unfloored matched-filter bound (1.2 m at
  34 dB) stretches convergence from 5 dwells to 84. `sigma_delay_s` and
  `sigma_delay_meas_s` are now separate for this reason.
- Applying the static-clutter test to the *un-unwrapped* Doppler deadlocks
  the whole loop. -31.5 Hz aliases to +18.5 Hz, which is inside the +-20 Hz
  zero-Doppler notch, so the measurement is refused, so no rate is learned,
  so nothing ever unwraps it. The symptom was a permanent +50.00 Hz error and
  a track frozen at one hit. The test is now applied only once the track's
  rate sigma is inside a sixth of an alias interval.

### 3.2 MTI blind speeds come with the same PRF

Adjacent-repetition subtraction has response `2|sin(pi f_d T_rep)|`, which
nulls at every multiple of the PRF. The direct path sits near 0 Hz after
de-ramp and is cancelled - measured notch depth **18-21 dB** with the drone
present, **42 dB** without it - which is the point. But a target at 50 Hz is
cancelled just as hard, and the drone crosses exactly that: at t = 320 s its
residual Doppler is 0.1 Hz and MTI costs it 35 dB.

`coarse.mti_response_db` exists so a blind-speed epoch is reported rather
than mistaken for a lost target.

### 3.3 Comb-4 PBCH-DMRS puts grating lobes at 5 km

PBCH-DMRS is one subcarrier in four across the SSB, and a comb of spacing
4 x 15 kHz = 60 kHz puts delay-axis replicas at multiples of 1/60 kHz, which
is 4996 m. Measured, those replicas stand **26 dB** above the noise floor -
well above a -25 dB drone. All of them are far outside the 1400 m cyclic
prefix budget, so confining the peak search to reachable delays
(`coarse.valid_delays`) is sufficient. Without it a flat gate locks stage 1
onto a grating lobe of the direct path.

## 4. Stage 2: what changed

Faithful: the channel estimate, static clutter removal, range processing by
IFFT along subcarriers, CA-CFAR (eq 34-35), DBSCAN (eq 36) and cluster
centroids (eq 37).

### 4.1 The clutter null is in the Doppler domain, not adjacent-symbol MTI

Paper section IV-B says "static clutter removal using MTI filtering". Read as
an adjacent-symbol difference along the symbol axis - the obvious analogy
with stage 1 - that is a high-pass whose first null is at the symbol rate.
The symbol period here is 71.35 us, so the PRF is 14 kHz and the response at
30 Hz is **-37 dB**. It would cancel every target in this scenario along with
the clutter.

`detect.doppler_notch` nulls zero Doppler after the slow-time FFT instead,
sized against the dwell's own Doppler resolution. At a 100 ms dwell that is
10 Hz per cell, so a +-2 cell notch discards excess range rates within about
+-3 m/s of the direct path's. The configured clutter tap is at +0.5 Hz only at
the selected azimuth and epoch, so it falls inside the notch and produces no
track in the verification run. That is not generic: a stationary object 250 m
from the receiver can reach **25.2 Hz** at 2478 MHz as the satellite line of
sight rotates, outside a +-20 Hz notch. The zero-Doppler notch is therefore a
bench-specific approximation; an ephemeris-driven clutter-locus mask is the
software replacement.

Enable that replacement with `--clutter-locus` in
`gr/flowgraphs/twostage.py` or the corresponding Grid Sense parameter. For a
stationary ground object at receiver range `d` and line-of-sight angular rate
`omega`, it excludes `|f| <= (fc/c) d omega`. Excess delay does not uniquely
determine `d`, so the implementation does not pretend it does: it combines the
ground-geometry bound with an explicit maximum stationary-clutter range.
`tests/clutter_locus.py` sweeps 72 azimuths; 41.7% of 250 m scatterers survive
a +-20 Hz notch, while none can exceed the carrier-corrected 25.2 Hz envelope.

### 4.2 Pfa must be specified as an expected count, not per cell

The paper's `Pfa = 1e-3` is the right per-cell number for its range-azimuth
map. This delay-Doppler map is 2800 x 600 display bins, where 1e-3 budgets
1680 false alarms per dwell. Measured with that setting: **605 surviving
clusters**, the tracker spawned a track for each, association went quadratic
and a 100 ms dwell took 4.6 seconds.

`detect.pfa_for_count` converts a desired expected count per dwell into a
per-cell Pfa, dividing the cell count by `pad^2` because neighbouring bins of
an oversampled map are correlated and do not each get an independent throw.
Verified against exponential noise across three decades: realised Pfa within
8% of requested.

### 4.3 The parabolic peak replaces eq 37's centroid as the primary estimate

Both are computed and both are returned. Measured on a -25 dB drone at
638.3 m of true excess over eight consecutive dwells:

| estimator | mean error | rms |
|---|---|---|
| power-weighted centroid (eq 37) | +2.57 m | 2.61 m |
| three-point parabolic peak | -0.46 m | 0.46 m |

A centroid over the CFAR-surviving cells is biased whenever that cell set is
asymmetric, which next to the direct path's skirt it always is. 3.3 m is only
0.05 of a delay cell, but it is a bias and does not average away over a
track. `Detection.centroid_delay_s` keeps eq 37 for comparison.

### 4.4 New: the aperture has a 50 Hz Doppler comb, and it defeats the notch

Slot 0 of every frame carries the SSB and no PDSCH (handoff section 4), so
the sensing weight `W` changes structurally once per 20 ms frame. That is a
periodic amplitude modulation at the 50 Hz frame rate, and it convolves every
tap in the map with a 50 Hz-spaced Doppler comb.

Measured on a filled grid with a -25 dB drone at -31.9 Hz: sidebands at
+-50, +-100, +-150 and +-200 Hz, the strongest about **9 dB** below the tap,
**eight comb orders** surviving CFAR, each spawning its own track. Also
observed at twice the *slot* rate: a 13 dB spur at +1968.17 Hz, which is
2 x 1000 - 31.9.

findings.md section 4 already noted that per-slot PRB randomisation "spreads
the pedestal across the whole Doppler axis". This is the precise form of that
spreading: not a smooth pedestal, a comb at the frame rate.

Two mitigations, both in place:

- The Doppler search is confined to +-slot_rate/2 = +-500 Hz, inside which no
  sideband of an in-band tap can fall (`ofdm.slot_rate_hz`).
- `detect.suppress_comb` rejects detections at frame-rate multiples of a
  co-delay reference peak. Legitimate rather than a fudge because `W` is known
  exactly, so the comb positions are known exactly - the same information
  CLEAN would use. The reference is taken from the *un-notched* map, which
  matters: the zero-Doppler notch removes a static scatterer's main peak but
  not its sidebands, so a clutter tap at +0.5 Hz vanishes while its +-50 Hz
  harmonics survive and read as two moving targets.

Cost, stated: a genuine target within a delay bin of a strong scatterer *and*
within a Doppler bin of an exact 50 Hz multiple from it is discarded.

### 4.5 What CFAR does not cover, and the SNR floor that patches it

CFAR fixes the false alarm rate against *noise*. It does nothing about the
direct path's own point spread function, whose 2-D skirt on a filled aperture
still stands about 31 dB above the noise floor at the drone's delay and
produced 4-5 dB detections at the zero-Doppler notch edge. Those are neither
noise nor targets.

The direct-path fix is CLEAN deconvolution of the known PSF. It is implemented
in `py/ntn_isac/clean.py` and used by the OTA path and by the realistic-target
software campaign. `min_det_snr_db` remains a conservative floor for the
legacy two-stage verification path. The low-level +25.00 Hz feature previously
flagged here is resolved: section 15 of `findings.md` shows that a hard notch
clips residue into its first surviving bin, and the one-cell guard band removes
that confident false detection.

## 5. The Renyi resource gate, and a result against it

Paper section IV-A gates the fine stage on a Renyi entropy criterion over the
downlink time-frequency occupancy - a proxy for ambiguity-function quality.
`py/ntn_isac/aperture.py` implements it, and because findings.md section 4
already measured the thing the proxy stands in for, the proxy is checkable.
`tests/entropy_pedestal.py` is that check.

**The proxy works, once computed per symbol.** The pedestal of a randomly
thinned aperture is set by the variance of the thinning, giving
`pedestal_dBc ~ -3.0103 * H_2 + 10 log10(1 - f)`. Measured agreement on
random allocations: **within 4.0 dB**. Both terms fall as occupancy rises, so
the entropy is monotone and the gate does its job.

It has to be per symbol. Over a whole coherent processing interval the union
of 20 independently drawn PRB sets touches every subcarrier, so a 10%
allocation reads as a full aperture and the law predicts -inf. The median over
symbols is used rather than the mean, because 4 of 280 symbols carry the
240-subcarrier SSB instead of the 300-subcarrier PDSCH, and averaging pulled a
fully allocated waveform to f = 0.994 - which made the law predict -46.7 dBc
for an aperture that has no thinning pedestal at all.

**The proxy is blind to allocation pattern, and pattern is worth 13.5 dB.**

| occupancy | pattern | H_2 (bits) | fragmentation | measured pedestal |
|---|---|---|---|---|
| 10% | random | 5.17 | 1.00 | -12.6 dBc |
| 10% | contiguous | 5.17 | 0.00 | -20.6 dBc |
| 50% | random | 7.29 | 0.50 | -21.1 dBc |
| 50% | contiguous | 7.29 | 0.00 | -34.7 dBc |

Random and contiguous PRB sets at the same occupancy have *identical*
per-symbol entropy - the same active count - and pedestals 13.5 dB apart. A
gate built only on entropy admits both. `aperture.run_fragmentation` is the
cheap addition that separates them, and `aperture.gate` takes it as an
optional extra criterion so the paper's own gate is what runs by default.

That discriminator needed one fix worth recording: normalised per subcarrier
it reported 0.08 versus 0.01 for the two 50% allocations, a spread no
threshold can use, because NR schedules whole 12-subcarrier PRBs and even a
maximally scattered set has runs at least 12 long. Normalised by the shortest
observed run - the actual allocation granularity - the same two allocations
read 0.50 and 0.00.

## 6. Measured end to end

`tests/verify_twostage.py`, checkpoints V8-V15, all passing. Filled grid,
t = 500 s, 100 ms dwell, drone at ~640 m of excess and -31.5 Hz of residual
Doppler. This is the legacy -25 dB algorithm-validation case, not the
link-budget headline; the realistic-target campaign is in `findings.md`
section 18.

| quantity | stage 1 (SSB) | stage 2 (payload-aided grid) |
|---|---|---|
| delay error | -0.2 to -1.0 m | -0.46 m mean, 0.46 m rms |
| Doppler error | +-0.04 Hz after unwrap | +0.14 Hz |
| SNR | 33-38 dB | 13-14 dB over the CFAR threshold |
| aperture | 3.60 MHz (240 SC) | 4.5 MHz (300 SC) |
| detections per dwell | 1 peak | 1 after 35 comb rejections |
| latency per 100 ms dwell | 82 ms | ~400 ms |

Track after 19 dwells: `646.2 +- 2.3 m`, `+4.72 +- 0.20 m/s` against a truth
of `+4.72 m/s`.

Target absent (drone disabled): zero fine detections, zero confirmed tracks,
and a tentative track from one marginal coarse peak ages out after 4 dwells.

**The occupancy result reproduces in the flowgraph.** Same channel, same
epoch, same Es/N0, only the PDSCH allocation changed:

| bundle | occupancy | peak/noise | drone detected |
|---|---|---|---|
| `wf_occ100` | 100% | 89.2 dB | yes, 13-14 dB |
| `wf` | 50% random | 80.0 dB | **no** |

9.2 dB of processing gain, and the legacy -25 dB validation echo goes from
clearly detected to invisible - findings.md section 4, live. This is the
controlled aperture demonstration, not a claim that -25 dB represents a small
satellite-illuminated target. Handoff section 8 wants this as a draggable
control; it is currently a choice of bundle.

## 7. Performance, honestly

Per 100 ms dwell in Python: demodulate 80 ms, stage 1 chain 1.4 ms, build the
variant 120 ms, map 104 ms, detect and cluster 172 ms. Total about 480 ms,
which is 4.8x the dwell.

In the two-block flowgraph the 80 ms demodulation happens twice, once per
block, because each buffers its own dwell. That is a deliberate 17% overhead
bought for architectural honesty: the two stages are separately schedulable,
separately strideable, and stage 1 keeps running when stage 2 is gated off or
falls behind. Collapsing them into one block would save the 80 ms and lose
that.

This is the reference-implementation speed that findings.md section 7 already
established for the Python channel, and it has the same consequence: fine for
the offline campaign and for replay mode, both non-real-time. Two things
follow:

- **Stage 1 alone keeps up** at 82 ms per 100 ms dwell, of which 80 ms is the
  OFDM demodulation. That asymmetry is not an accident of this
  implementation, it is the architecture: the cheap stage runs always, the
  expensive one runs when it is worth it.
- With a second variant mapped and the link metrics computed - what the
  dashboard asks for - stage 2 costs about 690 ms per dwell.
- Stage 2 takes a `stride`, so a live graph processes one dwell in N *by
  decision* rather than dropping buffers at random. Verified on the GRC
  canvas at stride 4: 14 stage-1 dwells, 4 stage-2 dwells, 10 skipped and
  counted.

## 8. The dashboard

`gr/dashboard.py`, handoff section 8. A separate process fed over ZMQ, so a
display crash cannot kill the flowgraph - which is why the sinks in the graph
are PUB and not REQ. It can also launch the flowgraph itself, so a demo is one
command.

```sh
. ./env.sh
$NTN_PY gr/dashboard.py                      # launches the graph too
$NTN_PY gr/dashboard.py --attach             # subscribe to a running one
$NTN_PY gr/dashboard.py --mode replay --capture cap.cf32
```

Three sockets: stage 1 on 5555, stage 2 on 5556, commanded truth on 5557.
Payloads are UTF-8 JSON in a PMT u8vector - see findings.md section 15 on why
not a PMT symbol.

Both delay-Doppler maps come from the **same dwell**, computed by one
`grid_sense` block with `extra_variants`. Two maps from two runs, or from two
blocks with their own buffers, would not be a comparison. The channel estimate
is shared across variants, so the second map costs a map (~100 ms) rather
than a map plus a channel estimate.

The colour scale is shared and is in dB over each map's *own* noise floor.
Handoff section 7 requires each map be normalised by sqrt(sum W) and reported
against its own floor, so absolute magnitudes are not comparable between
variants but dB-over-own-noise is. Plotting absolute magnitude would show the
audience an 11 dB scaling artefact instead of a result.

### 8.1 The occupancy demo

`docs/img/dashboard-occ100.png` and `dashboard-occ050.png` are the two frames.
Same channel, same epoch, same Es/N0, only the PDSCH allocation differs:

| | 100% | 50% random |
|---|---|---|
| reference-only peak/noise | 69.2 dB | 58.4 dB |
| payload-aided peak/noise | 81.8 dB | 75.7 dB |
| detection rate | 1.00 (52/52 dwells) | **0.00 (0/48)** |
| delay error | +0.71 m | no detection |
| offered throughput | 2.07 Mb/s | 1.08 Mb/s |

At 100% the drone is an isolated blob and the truth marker sits on the
detection. At 50% the map is filled with the thinned-aperture pedestal and its
50 Hz comb ridges, and the target is simply gone - the truth marker sits on
empty map. Handoff section 8 predicted the payload-aided map would "collapse
back toward the reference-only one"; what actually happens is stronger, and
findings.md section 4 explains why: the pedestal, not the processing gain,
governs weak-echo visibility.

The communication side halves at the same time, which makes the
sensing-versus-throughput trade visible in one frame.

### 8.2 Why the occupancy control is a switch and not a slider

Section 8 asks for a drag from 100% to 10%. The physics requires the
*transmitted* waveform to change: the effect is dominated by the transmitted
aperture, so masking resource elements in the receiver instead would reproduce
the processing-gain change and hide the pedestal change - it would show the
wrong result convincingly. Each occupancy is a different waveform bundle, so
the buttons relaunch the flowgraph. It takes about two seconds.

### 8.3 What the cards do and do not report

**No BLER.** It needs LDPC decoding, forbidden in the live path by handoff
section 12. The card shows the uncoded symbol error rate against `tx_grid`,
labelled as such on the display. See findings.md section 17.

**No elevation.** Section 8 asks for it in the pass strip. `traj.csv` carries
ranges and gains only - that is the section 3 file interface - and
`traj_meta.json` has only the min and max over the pass. Per-dwell elevation
needs a `gen_trajectory.m` column, which is a change to a frozen contract and
was not made unilaterally. Range, range rate and Doppler are all present.

**Throughput is offered, not delivered**, computed from the allocation,
modulation and code rate. Without a decoder a delivered rate cannot be known.

**The blind velocity band is printed on the range-rate card.** The
zero-Doppler notch plus its guard band makes excess range rates within
+-4.5 m/s of the direct path's undetectable, and a non-detection inside it is
a documented limitation rather than a miss.

## 9. Not built

- Anything requiring a second receiver: AoA, the paper's WLS fusion, a
  Cartesian track.
- Log-magnitude peak interpolation, which would reduce the +-1 m sub-bin
  delay bias of findings.md section 15.
- Per-dwell elevation, which needs a trajectory-generator column.
- A screen recording of the demo. Handoff section 8 wants one by day 12.
- Sim and replay modes touch no radio. The GRCon paper's hardware results are
  conducted captures on two GPSDO-locked B200s, scored in replay mode.
