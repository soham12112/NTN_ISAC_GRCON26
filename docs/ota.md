# Over the air: two laptops, two radios, a real object between them

Everything before this ran conducted, or did not run on a radio at all. Handoff
section 2 fixed that deliberately: "Radio 1 transmits, radio 2 receives, joined
by SMA cable with about 40 dB of attenuation... Conducted, never radiated", and
section 1 went further - "There is no physical reflector anywhere in the setup,
and that is deliberate, because it gives exact truth."

This document is what changes when both of those are dropped. It is not a
porting exercise. One number reorganises the whole receive path.

## 1. The number

The waveform occupies 300 subcarriers at 15 kHz, so 4.5 MHz, and the delay
resolution is `c/B` = **66.6 m of bistatic excess path**. A physical object
between two radios has nothing like that much:

| baseline | object offset from the TX-RX line | excess path | delay cells |
|---|---|---|---|
| 5 m | 1 m | 0.39 m | 0.006 |
| 10 m | 2 m | 0.77 m | 0.012 |
| 30 m | 5 m | 1.62 m | 0.024 |
| 200 m | 20 m | 3.96 m | 0.059 |

The object is inside the direct path's own resolution cell, and widening the
bandwidth does not rescue it: 100 MHz still gives 3 m cells, and it would blow
both handoff section 12's 20 MS/s ceiling and the B210's USB budget. Getting
66 m of excess path needs a 200 m baseline with the target 81 m off axis, which
is a different project.

So on this indoor over-the-air bench, **delay is not an observable and Doppler
is the only one**. That is not true of the deployed satellite geometry, where a
distant target can have kilometres of excess path and delay is primary. The
indoor case is standard direct-path-limited passive radar, and it has three
consequences that shape everything below.

**Sensitivity is not the problem.** A 0.3 m² corner reflector on a 5 m baseline
returns an echo 18 dB below the direct path; a 0.3 m plate in forward scatter,
where the Babinet cross-section is `4πA²/λ²`, returns one 6 dB below it. These
are enormous. The difficulty is entirely separation.

**Cancelling the direct path becomes the whole game.** `findings.md` sections 4
and 11 both already asked for CLEAN, twice observing that CFAR fixes the false
alarm rate against noise and says nothing about deterministic structure. Over a
cable that was the largest remaining improvement. Here it is the floor of the
measurement. See section 3.

**The clock offset stops mattering for the physical target.** A carrier offset
between the two radios is common to the direct path and the target, so it
cancels in the excess Doppler. The synthetic target is a different story -
section 5.

## 2. Synchronisation, and why it is two scalars

`receiver.EphemerisTracker` supplies `tau0(t)` by *commanding* the pass, which
is right over a cable driven by one process and wrong across two laptops. But
this is not blind acquisition either: the receiver has `tx_iq.cf32` and
`traj.csv`, so the only unknowns are the clock difference and where the
transmitter started.

A B210 derives its RF local oscillator and its sample clock from the same TCXO,
so one fractional error `eps` sets both at once - the carrier lands at
`+eps*fc` and the waveform's apparent delay drifts at `-eps` per second. Write
it as a delay and the carrier offset follows from it:

```
tau_clk(t) = d0 - eps*t      =>   -fc * d(tau_clk)/dt = +eps*fc
```

which is the same invariant handoff section 6 imposes on the channel block.
Estimating a carrier offset and a timing drift independently would describe a
transmitter that cannot exist. `ota.OtaTracker` therefore estimates exactly
`(d0, eps)` on top of the commanded trajectory, and presents the same interface
the rest of the chain already consumes.

| | free-running TCXOs | shared 10 MHz/PPS reference |
|---|---|---|
| eps | up to 4e-6 | ~1e-11 |
| carrier offset at 2.478 GHz | up to 9.9 kHz | 0.025 Hz |
| delay drift | 31 samples/s | 8e-5 samples/s |

The current hardware is a pair of B210s. Each radio's two receive channels are
coherent because they share the same clock and local oscillator. A shared
reference between the two radios is not required for the short-dwell physical
target: carrier offset is common to the direct path and echo and cancels in
excess Doppler. Synchronisation remains required for the commanded emulated
target and for the long coherent dwells considered as future work; a cabled
10 MHz and PPS reference is more dependable indoors than GPS antennas without
sky view.

Acquisition removes the commanded pass first and only then searches. That is
not an optimisation, it is necessary: the pass reaches 54.9 kHz of Doppler at
2.478 GHz, and a raw correlation locks at t = 300 s, where the range rate
passes through zero, and fails at t = 500 s, where the demo actually runs.
Measured, the raw version reported a 4.7 dB correlation peak and never locked.
De-ramped, the residual is the clock offset alone, so the search is ±12 kHz.

## 3. Cancelling the direct path

Because the aperture `W` is known exactly - it is `occ_mask`, written by the
generator and shipped in the bundle - a tap's contribution to `Z` is known in
closed form up to one complex scalar. So this is a least-squares subtraction,
not the subspace projection that ECA and its variants use when the reference is
measured and noisy: no degrees of freedom are spent and nothing a target could
occupy is removed.

Measured (`tests/verify_ota.py` V19), a rotating target at zero excess path,
against a target-absent run on the same geometry:

| echo | Hann | Hann + CLEAN | Blackman-Harris | B-H + CLEAN |
|---|---|---|---|---|
| −20 dB | +4.9 dB | +70.7 dB | +7.4 dB | **+73.6 dB** |
| −40 dB | +0.6 dB | +52.1 dB | +0.8 dB | **+54.0 dB** |
| −55 dB | +0.0 dB | +37.1 dB | +0.0 dB | **+38.9 dB** |

Uncancelled, every row reads the same number whatever the echo, because what is
being measured is the direct path's own structure and not the target at all. A
−40 dB co-delay target is **indistinguishable from no target**. Cancelled, 15 dB
of echo costs 15 dB of detection SNR, which is the signature of actually
measuring the target. The direct path itself comes down 74 dB.

The slow-time window matters for a reason worth stating: Hann's first sidelobe
is −31 dB, and `findings.md` section 11 measured the direct path's own point
spread function standing ~31 dB above the noise floor at the target's delay.
That is the same number and not a coincidence. A 4-term Blackman-Harris trades
1.9x mainlobe width - Doppler resolution this path has to spare - for −92 dB.

## 4. The physical target

### Why a rotating one

At 2.478 GHz the excess-range rate a target needs to clear the zero-Doppler
notch is set by the dwell. At 100 ms the notch plus its guard blinds ±30 Hz:

| target | speed | peak Doppler | clears the notch? |
|---|---|---|---|
| person walking | 1.3 m/s | 21 Hz | **no** |
| hand or arm | 2.0 m/s | 33 Hz | marginal |
| pendulum reflector | 1.5 m/s | 25 Hz | **no** |
| blade tip, r = 0.25 m at 7 rev/s | 11.0 m/s | **180 Hz** | yes, easily |

A person walking is invisible by construction at this carrier. A rotating
scatterer is not, and it brings more than Doppler. Its phase is sinusoidal with
modulation index `beta = 2*pi*r/lambda`, so the echo is a Bessel comb of lines
spaced at the rotation rate with a sharp envelope edge at the tip-speed Doppler
`2*pi*r*f_rot/lambda`. Two numbers come out, giving the rotation rate and the
tip radius separately, and both are checkable against a tachometer. It carries
its own ground truth, and it repeats - which is what the emulated drone was for
and what a person walking past cannot supply.

### What to build

A metal blade or a foil-covered fan blade of radius 0.2-0.3 m on a
variable-speed motor, standing on or beside the TX-RX line. A drone propeller
on a bench motor is better still: it is the same signature the emulated drone
is supposed to have, without flying anything indoors.

### Choosing the rotation rate

Slot 0 of every frame carries the SSB and no PDSCH, so `W` changes once per
20 ms and every tap acquires a Doppler comb at multiples of 25 Hz about 9 dB
down (`findings.md` sections 11 and 15). That comb lands squarely in the band a
rotating target occupies. **Pick a rate whose harmonics miss it.** At 7 rev/s
the blade's lines fall on multiples of 7 Hz and do not coincide with the comb
below 175 Hz. A rate that divides 25 puts its entire signature on masked bins.

This is not hypothetical. An earlier version of the V19 check measured the
direct path's 100 Hz comb order and reported it as the blade: it read exactly
+50.0 Hz and did not move when the echo was swept over 50 dB.
`udoppler.choose_rotation_hz` scores candidate rates against the comb.

### Dwell length

The tip-speed edge and the rotation rate need different dwells, and the block
publishes them on different ports for that reason.

| dwell | Doppler cell | extent (true 91 Hz) | best rate (true 7 rev/s) |
|---|---|---|---|
| 100 ms | 5.0 Hz | 110 Hz | 26.25 — unresolved |
| 500 ms | 1.0 Hz | 105 Hz | 3.00 — unresolved |
| **1000 ms** | 0.5 Hz | **98.5 Hz** | **7.00** |

At 100 ms the cell is 5 Hz against a 7 Hz line spacing, so the comb is simply
not resolved and the best candidate is a harmonic. Note that an unresolved
estimate is confident and wrong - it does not look like a failure - which is
why the table is in the test output rather than an assertion.

## 5. What the two clocks cost the *emulated* target

`ssb_coarse` and `grid_sense` track the pass from ephemeris and have no
clock-offset estimator. Measured in simulation, 1 ppm of offset takes
`grid_sense` from 91.1 dB of peak-to-noise to 62.3 dB and its detections become
nonsense. `ota_sense` estimates the offset itself and is unaffected.

So the honest summary is: **free-running gets you the physical target,
GPS-locked gets you both.** And both is the demo - a commanded target at 640 m
whose answer is known exactly, and a real one at zero excess path, in the same
capture. The first validates the chain; the second proves it is not validating
itself.

Wiring `ota_sense`'s clock estimate into the other two blocks automatically is
the obvious follow-up and is not built. Today the path is manual: read the
`tau_bias` `ota_sense` converges on and enter it in their Delay bias field,
which is handoff section 9's V4 calibration constant.

## 6. Band, power and legality

2.0 GHz is licensed spectrum almost everywhere, so the over-the-air bundles are
generated at **2478 MHz**. The occupied bandwidth is 4.5 MHz at 7.68 MS/s, so
about 7.7 MHz has to be clear; Wi-Fi channels 1, 6 and 11 occupy 2401-2423,
2426-2448 and 2451-2473, leaving 3 MHz gaps between them and a 10 MHz gap above
channel 11. 2478 MHz puts the signal in 2474.2-2481.8 MHz, clear of 1/6/11 and
overlapping only channel 13, which is not used in the US, and BLE advertising
channel 39 at 2480 MHz. Survey the local band before trusting that.

Under FCC part 15.247 this is a digital modulation system - the 6 dB bandwidth
is far above the 500 kHz floor - so it is licence-free to 30 dBm conducted with
a 6 dBi antenna. A B210 delivers about 10 dBm, which is 20 dB of headroom.

`tx_iq` is scaled to RMS 0.1 with 10.6 dB of peak-to-average, so `--amp` above
about 2.5 clips the DAC. Clipping is a nonlinearity the receiver's reference
does not contain, so it does not look like distortion downstream - it raises
the floor of every sensing measurement and looks like a receiver bug.

On the receive side, set the gain so the direct path sits 15-20 dB below full
scale. The ADC has to hold the direct path and the target at once, and clipping
the strong one destroys the weak one.

The carrier also scales the pass Doppler by `fc/2.0e9`: peak Doppler goes from
44.3 kHz to **54.9 kHz** and the rate from 547 to 678 Hz/s. Handoff section 7's
±50 kHz acquisition grid is no longer wide enough, which is another reason
acquisition de-ramps the commanded pass first rather than searching blind.

## 7. Running it

Regenerate the bundles once, on both machines:

```matlab
addpath matlab
gen_ota_bundles                 % data/wf_ota, wf_ota050, traj_ota at 2478 MHz
```

Check the chain with no radios at all:

```sh
. ./env.sh
$NTN_PY tests/verify_ota.py                      # V16-V19
$NTN_PY gr/flowgraphs/ota_rx.py --wf data/wf_ota --traj data/traj_ota \
        --mode sim --seconds 4 --sim-eps-ppm 2.5 --blade-rot-hz 7
```

Then, transmit laptop first:

```sh
$NTN_PY gr/flowgraphs/ota_tx.py --wf data/wf_ota --traj data/traj_ota \
        --t-start 300 --clock internal --tx-gain 70
```

It prints the epoch it chose and the exact receive command line to paste on the
other laptop. Bring the link up at `--t-start 300`, which is closest approach,
where the range rate passes through zero. Move to `--t-start 500` once it
closes: that is where the emulated drone sits at 640 m of excess path with
−39.0 Hz of residual Doppler at 2478 MHz, clear of the MTI blind speed. That
emulated path additionally requires a shared external reference between the
B210s; free-running remains valid for the physical target only.

Do not compress the trajectory to make acquisition easier. Handoff section 6 is
explicit that it makes the Doppler rate unphysical and invalidates the NTN
claim, and a high-elevation epoch is already slow without cheating.

### Running from GNU Radio Companion

Two things bite here and both present as an unrelated error.

**GRC's Run button uses the interpreter GNU Radio was built against, not
`$NTN_PY`.** On macOS that is Homebrew's python3.14, which has `gnuradio` and
`numpy` but not `scipy`, `pyzmq` or `pyqtgraph` - so a flowgraph that runs
perfectly from the command line dies in GRC on `ModuleNotFoundError: No module
named 'scipy'`, raised from inside a block constructor. Install them once into
that interpreter:

```sh
/opt/homebrew/bin/python3.14 -m pip install --user --break-system-packages \
    scipy pyzmq pyqtgraph
```

`--break-system-packages` is the PEP 668 escape hatch and is safe here: it
installs to `~/.local/lib/python3.14/site-packages` and does not touch
Homebrew's own tree. The `.venv` that `env.sh` prefers stays as it is; this just
makes the same packages visible to the interpreter GRC launches.

**GRC runs a flowgraph from the directory of its .grc file**, not from wherever
GRC was started, so a relative `data/wf_ota` resolves to
`gr-ntnisac/examples/data/wf_ota`. The three flowgraphs therefore anchor their
path variables at `$NTN_ROOT`, which `env.sh` sets - and `env.sh` has to be
sourced anyway or GRC will not find the ntnisac blocks at all, so if the
flowgraph opened, the variable is there. Start GRC from a shell that has
sourced it:

```sh
. ./env.sh && gnuradio-companion gr-ntnisac/examples/ntnisac_ota_bench.grc
```

### The three flowgraphs

The same two graphs open in GNU Radio Companion as
`gr-ntnisac/examples/ntnisac_ota_tx.grc` and `ntnisac_ota_rx.grc`. Source
`env.sh` before opening them; their waveform and trajectory paths are anchored
to `$NTN_ROOT`. They are generated by `make_ota_grc.py` beside them; edit that
rather than the YAML.

## 8. 3GPP NTN channel models

`gen_ntn_channel.m` layers the clusters of a 3GPP TR 38.811 tapped-delay-line
profile onto an existing trajectory. MATLAB's 5G Toolbox ships NTN-TDL-A
through -D plus NTN-TDLA100 and NTN-TDLC5, and Satellite Communications Toolbox
has `p681LMSChannel` for the ITU-R P.681 land-mobile environments; both are
already installed here, which is why this is built on them rather than on
Sionna and OpenNTN. For a SISO link the angular information in a full CDL is
exactly the part that cannot be used - there is one receive antenna and no
aperture to beamform over - so a TDL is the right level of model. OpenNTN earns
its TensorFlow dependency if this ever goes multi-antenna, and the file contract
below is what it would target.

The import needs **no change to the channel block**, which is the point. The
obvious approach - a complex gain per tap varying with time - is precisely what
handoff section 6 forbids, because a delay and a Doppler specified
independently disagree. Instead each cluster is emitted as a path whose range
*ramps* at exactly the rate whose implied Doppler is the one wanted:

```
R_p(t) = R_0(t) + c*tau_p - f_d,p * (c/fc) * (t - t_ref)
```

Delay and Doppler then agree by construction and `traj.csv` carries it
unchanged. Fading within a cluster comes from splitting it into sub-rays with
Dopplers drawn from the Jakes spectrum, which is what a cluster is anyway.

```matlab
gen_ntn_channel('InDir','data/traj_ota','Profile','NTN-TDL-C');
```

```
NTN-TDL-C, delay spread 100 ns, mobile 0.83 m/s -> max Doppler 6.86 Hz
K factor of tap 0 is 10.2 dB, so its LOS is dropped and the
trajectory's own path 0 carries it instead
cluster 0: excess     0.0 m (0 ns),  -10.2 dB vs the LOS
cluster 1: excess   444.1 m (1481 ns),  -23.0 dB vs the LOS
3 base paths + 2 cluster rays = 5 paths
estimated channel throughput 12.2 MS/s against 7.68 required  (1.59x)
```

Three things to know.

**Tap 0 is Rician and its line of sight is dropped.** The trajectory's path 0
already is the line of sight; adding tap 0 whole would count it twice. Only the
diffuse part is emitted, at −K dB.

**Path count costs real time.** The channel runs about 20 MS/s at 3 paths and
8 interpolator taps, and the cost is roughly linear, so the budget is about
8 paths against a 7.68 MS/s requirement. `SubRays` is 1 by default for that
reason and the function prints its estimate and warns when it is exceeded -
NTN-TDL-D at `SubRays=2` is 9 paths and 6.8 MS/s, which does not clear.

**Clusters drift, because a constant Doppler is a constant range rate.** The
ramp is referenced to `RefTime` so a cluster sits at its nominal excess at the
epoch being transmitted; over a 10 s capture it moves 1.4 m, over the whole
600 s table it moves the UT's own displacement.

And a limit worth being blunt about: these clusters are **statistical and have
no ground truth**. They make a communication measurement credible over a
standards-defined NTN channel. They are not targets, and every sensing
checkpoint must keep scoring against the commanded scatterers of
`gen_trajectory`.

## 9. What is not built

- Feeding `ota_sense`'s clock estimate to `ssb_coarse` and `grid_sense`
  automatically, so the emulated-target path works free-running (section 5).
- A dashboard panel for the micro-Doppler spectrum. The block publishes the
  Doppler profile on `ota` and `micro`; `gr/dashboard.py` does not plot it yet.
- Anything on real RF. Every number in this document is simulated, by
  `py/ntn_isac/otasim.py`, and the first job on hardware is to find out which of
  them survive.
