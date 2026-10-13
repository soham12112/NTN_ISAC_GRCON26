# What this is, and why it is built this way

Written to be read start to finish by someone who has not seen the code. It
explains what the system does, the decisions behind it, where the ideas come
from, and the places where reality disagreed with the plan.

Companion documents: `docs/running.md` is how to run it, `docs/findings.md` is
the measurement log, `docs/ota.md` is the over-the-air detail, `docs/two-stage.md`
is the sensing flow, `docs/file-interface.md` is the file contract, and
`ntn-isac-handoff.md` is the original build brief.

---

## 1. The idea in one page

A satellite is going to transmit a 5G signal to the ground whether or not
anybody wants to use it for radar. The question this project asks is: **can the
same signal, unchanged, also tell you what is moving underneath it?**

The specific claim is narrower and more interesting than "5G can do radar".
A radar needs to know what was transmitted in order to recognise its echo. The
obvious reference is the pilot symbols — the parts of a 5G frame that are fixed
and known in advance. A receiver that successfully demodulates the data can
reconstruct the data symbols too, so the whole transmitted frame can serve as
the radar reference rather than only the pilots.

That reconstruction is established prior art, not this project's invention:
COFDM passive radar has rebuilt ideal references from decoded broadcasts since
at least Poullin (2005), and later 5G NR passive-radar work uses reconstructed
user data as well. The contribution tested here is different: a scheduled
network changes which time-frequency aperture is available from one dwell to
the next, so sensing quality becomes a function of the scheduler. Broadcast
passive radar cannot ask that question because its aperture is fixed.

That is the headline comparison:

- **reference-only** sensing, using just the pilots;
- **payload-aided** sensing, using the whole frame, reconstructed from the
  receiver's own decisions;
- **genie**, using the true transmitted frame from a file. This exists to
  prove the payload-aided path is not cheating.

If payload-aided is better, you get radar for free from a communication link —
no extra spectrum, no extra power, no reserved resources. And the amount better
turns out to depend on how busy the network is, which makes sensing quality a
function of the scheduler. That is the most surprising result in the project.

### Sensing is bistatic, and there is a reason

The satellite transmits, a receiver on the ground listens. The receiver sees the
direct path from the satellite, and it sees echoes of the same signal bouncing
off things. Measuring the delay and Doppler of those echoes relative to the
direct path is the measurement.

It is not monostatic radar — the transmitter and receiver are not co-located,
and nothing here transmits and listens at the same time. Handoff section 12 puts
monostatic and self-interference cancellation explicitly out of scope.

---

## 2. What physically exists

```
   MATLAB                GNU Radio, TX laptop        GNU Radio, RX laptop
   ┌──────────┐          ┌────────────────┐          ┌──────────────────┐
   │ waveform │  files   │ File Source    │   RF     │ USRP Source      │
   │ generator│ ───────► │      ↓         │ ──────►  │      ↓           │
   │          │          │ LEO Channel    │          │ sync, demodulate │
   │ trajectory          │      ↓         │          │      ↓           │
   │ generator│ ───────► │ USRP Sink      │          │ sensing          │
   └──────────┘          └────────────────┘          └──────────────────┘
```

- **MATLAB** writes the transmitted waveform and the satellite's flight path to
  files. It is not in the loop at run time.
- **The transmit laptop** plays the waveform file, passes it through a digital
  emulation of the satellite channel, and transmits it.
- **The receive laptop** receives, works out what the channel did, and produces
  the sensing result.

Between the two antennas there is a real physical object — a rotating metal
blade — which produces a real echo.

### The three targets

This is the part people find confusing, so it is worth being explicit. There are
three different kinds of thing being detected, and they serve different
purposes:

| target | where it is | truth | resolved in |
|---|---|---|---|
| static clutter | emulated, 250 m from the receiver | commanded, exact | delay |
| drone | emulated, crossing at 15 m/s | commanded, exact | delay + Doppler |
| rotating blade | **physically present** between the antennas | measured RPM | Doppler only |

The emulated ones exist because they give **exact** ground truth: the software
put them there, so the right answer is known to the last decimal. That is what
you validate a processing chain against. The physical one exists because a chain
that only ever sees synthetic targets has not been shown to work on anything.

Running both in one capture is the demonstration. One proves the chain is
correct; the other proves it is not merely self-consistent.

---

## 3. The waveform

A standards-compliant 5G NR downlink, generated by MATLAB's 5G Toolbox.

| | | why |
|---|---|---|
| subcarrier spacing | 15 kHz | FR1, the simplest numerology |
| bandwidth | 25 PRB = 300 subcarriers = **4.5 MHz** | keeps the whole chain real-time on one host over USB 3 |
| sample rate | 7.68 MS/s, 512-point FFT | follows from the above |
| modulation | QPSK, low code rate | see below |
| file length | exactly 20 ms, 153600 samples | one SSB period, so the file loops seamlessly |
| carrier | **2478 MHz** | licence-free ISM; see section 8 |

**Why QPSK and not something faster.** This is not laziness. The sensing step
divides the received signal by the receiver's estimate of what was transmitted.
QPSK has constant amplitude, so that division never amplifies noise. 16QAM has
inner constellation points with small amplitude, and dividing by a small number
is how you turn a good measurement into a bad one. Constant modulus also makes
the hard decisions reliable. Higher-order modulation is a result to measure
later, not a default.

**Two numbers that govern everything downstream:**

- The **cyclic prefix** is 4.69 µs, which is 1406 m of extra path length. An
  echo arriving later than that falls outside the guard interval and the
  receiver cannot see it at all. This is the sensing scenario's outer boundary.
- The **delay resolution** is `c/bandwidth` = `3e8/4.5e6` = **66.6 m**. Two
  echoes closer together than that are one echo. This number reorganises the
  entire over-the-air design — see section 7.

---

## 4. The satellite channel

A real low-Earth-orbit pass, propagated from a TLE with the Satellite
Communications Toolbox: 550 km circular orbit, 67.5° maximum elevation, 600 s
sampled at 100 Hz. At 2478 MHz that gives:

| | |
|---|---|
| slant range | 594 to 2170 km |
| range rate | ±6636 m/s |
| Doppler | **±54.9 kHz** |
| Doppler rate | 678 Hz/s |
| delay walk | 170 samples/s at 7.68 MS/s |

Those numbers are the whole reason this is hard, and they are what makes it a
*non-terrestrial* problem rather than a terrestrial one with longer cables.

### The one rule the channel obeys

Per path, per sample:

```
y_p(t) = a_p(t) · x(t − τ_p(t)) · exp(−j·2π·f_c·τ_p(t))
```

**Delay is the only thing injected. Doppler is never a parameter anywhere in
this codebase.** It falls out of the derivative of the carrier phase term, which
is exactly the Doppler implied by the delay changing.

This looks like a stylistic choice and is not. If you specify a delay and a
Doppler independently, you are describing something that cannot exist — an
object whose range is changing at one rate while its velocity says another — and
any receiver that jointly tracks delay and Doppler will see data that
contradicts itself and produce confident nonsense. Everything that wants to add
a Doppler to this system has to express it as a range that changes. That
constraint is load-bearing in three separate places later: the clock model
(section 6), the 3GPP channel import (section 9), and the rotating target
(section 7).

### Where it runs

The channel is applied **digitally on the transmit side, before the DAC**. So
what leaves the antenna already carries the pass's Doppler, Doppler rate and
delay walk. This is an emulation of a satellite, not a satellite.

It is implemented in C++ as a GNU Radio block. The Python version, which is the
reference implementation and is what the tests check against, runs at about
1 MS/s — seven times too slow for real time. The C++ version runs at 12 to
22 MS/s depending on interpolator length, so every setting clears the 7.68 MS/s
requirement.

---

## 5. How sensing works

Five steps, one code path, one flag for the variant.

1. **Track and de-ramp.** Remove the direct path's bulk Doppler *before*
   demodulating. At 678 Hz/s, a 100 ms coherent integration smears across many
   Doppler bins otherwise.
2. **Demodulate.** Strip the cyclic prefix, FFT, extract the 300 subcarriers.
   This gives a grid `Y[symbol, subcarrier]` of what was received.
3. **Estimate the channel and equalise**, using the pilots.
4. **Decide what was sent.** Slice to the nearest QPSK point. Combined with the
   known pilots, this is `X_est` — the receiver's estimate of the entire
   transmitted grid. Reconstructing a reference this way is prior art; measuring
   how its sensing aperture changes under an NR scheduler is this project's
   contribution.
5. **Correlate and transform.** Form `Z = Y · conj(X_est) · W`, where `W`
   selects which resource elements to use. Then inverse-FFT across subcarriers
   to get delay, and FFT across symbols to get Doppler. The result is a
   delay-Doppler map: a picture of everything that reflected the signal.

The three variants differ only in `W` and where `X_est` comes from. Not two
receivers — one, with a flag.

### Correcting the delay walk, which the original plan missed

The satellite's range changes by up to 6.6 km/s, which is 170 samples per second
of delay at 7.68 MS/s. Over a 100 ms integration the echo moves 17 samples,
which is ten resolution cells, and the delay axis smears into uselessness.

The fix is nearly free because a delay is a linear phase across subcarriers, so
after retiming each symbol by a whole number of samples the leftover fraction
comes off in the frequency domain. Measured gain at a 100 ms integration:
**+4.3 dB**. Without it the long integration simply does not work.

One subtlety that cost real time to find: the delay must be evaluated at the
*arrival* time, not the transmit time. The difference is the propagation delay
itself — up to 5 ms here — and at 170 samples/s that is most of a sample of
timing bias.

### The two-stage flow

The sensing runs as a coarse stage and a fine stage, following a published
framework (source in section 10).

- **Stage 1** matched-filters against the SSB, which the satellite transmits
  regardless of what the scheduler is doing. Cheap — 82 ms per 100 ms dwell — so
  it always keeps up. It maintains the track.
- **Stage 2** does the full-grid sensing above. Expensive, so it runs on a
  fraction of dwells, *by decision* rather than by dropping buffers at random.

They exchange messages both ways: stage 1 hands stage 2 a range gate, stage 2
hands back refined measurements. That loop is what makes it coarse-to-fine, and
it is what resolves an ambiguity stage 1 cannot resolve alone — one SSB per
20 ms is a 50 Hz repetition rate, unambiguous only to ±25 Hz, against a target
that reaches 77 Hz.

---

## 6. The two-laptop problem

Over a cable driven by one process, the receiver knows exactly when the
transmitter started and both ends share a clock. Across two laptops neither is
true.

### The clock difference is one number, not two

A B210 derives its radio frequency oscillator and its sample clock from the same
crystal. So a single fractional error `eps` sets both at once:

- the carrier arrives `eps × f_c` away from where it should be;
- the waveform's apparent delay drifts at `eps` seconds per second.

Written as a delay, `τ_clk(t) = d0 − eps·t`, the carrier offset follows from it
automatically — the same rule as section 4. Estimating a frequency offset and a
timing drift as independent quantities would describe a transmitter that cannot
exist. So the receiver estimates exactly two scalars: a constant offset `d0` and
one `eps`.

| | free-running | shared 10 MHz/PPS reference |
|---|---|---|
| `eps` | up to 4e-6 | ~1e-11 |
| carrier offset at 2478 MHz | 9.9 kHz | 0.025 Hz |
| delay drift | 31 samples/s | 8e-5 samples/s |

The original brief assumed no clock discipline and said to measure the
resulting error floor rather than eliminate it. The current B210 pair does not
need cross-radio synchronisation for a short-dwell physical target: the clock
offset is common to the direct path and echo, so it cancels in excess Doppler.
It still matters for the commanded emulated path because `Grid Sense` has no
clock estimator: at 1 ppm its peak-to-noise falls from 91 dB to 62 dB.
**Free-running gets the physical target; a shared reference gets both, and is
mandatory for long coherent dwells.**

### Acquisition is not blind, and must not be built as if it were

The receiver has the transmitted waveform file and the flight path file. So the
satellite's own 54.9 kHz of Doppler is *known* and is removed before searching;
what is left is the clock offset alone, bounded by about 10 kHz.

Skipping that appears to work and does not. At closest approach the pass's
Doppler passes through zero, so a naive correlation locks. At the epoch the demo
actually uses, the same code reported a 4.7 dB correlation peak and never
locked, because the signal was 40 kHz outside the search window.

---

## 7. The physical target, and the number that changed everything

### A real object is unresolvable in delay

The delay resolution is 66.6 m. An object one metre off a five-metre baseline
has a bistatic excess path of **0.39 m** — six thousandths of a resolution cell.
It is inside the direct path's own cell.

More bandwidth does not help: 100 MHz still gives 3 m cells, and would blow both
the 20 MS/s scope limit and the B210's USB budget. Getting a full cell of
separation needs a 200 m baseline with the target 81 m off axis. That is an
outdoor radar site, not a bench.

**So on an indoor bench, delay is not an observable. Doppler is the only one.**
That is not a property of the deployed satellite geometry: a target a kilometre
from the receiver can contribute kilometres of excess path, making delay its
primary observable. Indoors, the unresolved case is standard
direct-path-limited passive radar, and it has three consequences.

### Consequence 1: sensitivity was never the problem

A modest corner reflector on a five-metre baseline returns an echo only 18 dB
below the direct path. A flat plate in forward scatter — where an object sitting
*on* the line between the antennas has a very large effective cross-section —
returns one 6 dB below. These are enormous signals. Every difficulty is
separation, not detection.

### Consequence 2: the direct path is the noise floor

The direct path is overwhelmingly stronger than the echo and sits in the same
delay cell. Removing it is the entire measurement.

The method is **CLEAN**: find the strongest thing in the map, work out exactly
what it would have written into the data, subtract that, repeat. It works here
in a way it usually cannot, because the transmitted pattern is *known exactly* —
it is shipped in the bundle — so a path's contribution is known in closed form up
to one complex number. That makes it a least-squares subtraction rather than the
statistical projection passive radar normally has to use, and nothing a target
could have occupied is removed.

Measured, a rotating target at zero excess path against a target-absent control:

| echo | no cancellation | with CLEAN |
|---|---|---|
| −20 dB | +4.9 dB | **+73.6 dB** |
| −40 dB | +0.6 dB | **+54.0 dB** |
| −55 dB | +0.0 dB | **+38.9 dB** |

Without cancellation every row reads the same regardless of echo strength,
because what is being measured is the direct path's own structure and not the
target at all. A −40 dB target is **indistinguishable from no target**. With
cancellation, 15 dB of echo costs 15 dB of detection margin — which is what
actually measuring the target looks like.

The window used across the symbol axis matters for a related reason. The default
Hann window has a first sidelobe at −31 dB, and the direct path's own spread was
measured at about 31 dB above the noise floor at the target's position. Same
number, not a coincidence. Switching to a Blackman-Harris window trades some
Doppler resolution — which this path has to spare — for −92 dB.

### Consequence 3: the target has to be spinning

At 2478 MHz, whether a target clears the zero-Doppler blind zone is set by its
speed:

| target | speed | Doppler | clears ±30 Hz? |
|---|---|---|---|
| person walking | 1.3 m/s | 21 Hz | **no** |
| pendulum reflector | 1.5 m/s | 25 Hz | **no** |
| blade tip, 0.25 m at 7 rev/s | 11.0 m/s | **180 Hz** | yes |

A person walking past is invisible by construction. A rotating scatterer is not,
and it brings something better: its echo is a comb of spectral lines **spaced at
the rotation rate**, with a sharp edge at the tip-speed Doppler. Two numbers come
out, giving the rotation rate and the tip radius separately, and both can be
checked against a tachometer. It carries its own ground truth and it repeats —
which is what the emulated targets were for and what a person cannot supply.

Measured on the bench: **11.00 rev/s recovered against a true 11.00**, tip radius
27.1 cm against a true 25 cm, from an echo 45 dB below the direct path.

Two practical constraints:

- **Avoid rotation rates that divide 25.** The transmitted frame changes
  structure once per 20 ms, which paints a comb of spurious Doppler lines at
  multiples of 25 Hz across everything. A rate that divides 25 puts the target's
  entire signature on exactly those masked bins. 7 and 11 are good; 5 is not.
- **The rotation rate needs a one-second integration.** The tip-speed edge is
  readable from a 100 ms dwell, but 100 ms gives 5 Hz Doppler bins against a 7 Hz
  line spacing, so the comb is unresolved. Measured: at 1 s it comes out exactly;
  at 100 ms and 500 ms it returns a harmonic — **confidently**, which is worse
  than failing.

---

## 8. Frequency, power and legality

The original design used 2.0 GHz, which is licensed spectrum almost everywhere.
The over-the-air work moved to **2478 MHz**, in the 2.4 GHz ISM band. Wi-Fi
channels 1, 6 and 11 occupy 2401–2423, 2426–2448 and 2451–2473, leaving 3 MHz
gaps between them and a 10 MHz gap above channel 11; 2478 fits the required
7.7 MHz in that gap. Under FCC part 15.247 this is a digital modulation system
and is licence-free to 30 dBm with a 6 dBi antenna. A B210 gives about 10 dBm.

Moving the carrier up scales the pass Doppler by `f_c/2.0 GHz`: peak Doppler goes
from 44.3 to 54.9 kHz and the rate from 547 to 678 Hz/s. The original ±50 kHz
acquisition search is no longer wide enough — which is a second reason
acquisition removes the known pass first rather than searching blind.

---

## 9. 3GPP channel models

Everything above uses a *geometric* channel: paths at commanded ranges, with
exact truth. That is what sensing needs. It is not what a realistic satellite
link looks like, which is multipath, fading and blockage.

3GPP TR 38.811 defines statistical non-terrestrial channel models for that.
MATLAB's 5G Toolbox ships them as NTN-TDL-A through -D, and the Satellite
Communications Toolbox has the ITU-R P.681 land-mobile models. Both were already
available, which is why they are used here rather than Sionna and OpenNTN — and
because for a single-antenna link, the angular information in a fuller model is
exactly the part that cannot be used. OpenNTN earns its dependency if this ever
goes multi-antenna.

**The import needs no change to the channel block**, which is the point. The
obvious approach — a complex gain per tap varying over time — is precisely what
section 4 forbids. Instead each cluster is emitted as a path whose range *ramps*
at exactly the rate whose implied Doppler is the one wanted. Delay and Doppler
then agree by construction and the existing file format carries it unchanged.

Three things to know:

- Tap 0 of these profiles contains the direct ray, which the trajectory already
  has. Adding it whole would count it twice, so only its diffuse part is used.
- Path count costs real time. About 8 paths is the budget; the tool prints its
  estimate and warns when a setting exceeds it.
- **These clusters are statistical and have no ground truth.** They make a
  communication measurement credible over a standards-defined channel. They are
  not targets, and sensing must keep being scored against the commanded paths.

---

## 10. Sources

- **3GPP TR 38.811**, non-terrestrial network channel models. The source of the
  NTN-TDL profiles; **TR 38.821** for the scenario parameters.
- **W. Khawaja, A. M. Alam, A. C. Gurbuz, "A Two-Stage 5G NR ISAC Framework for
  Coarse-to-Fine UAV Detection and Tracking", IEEE DySPAN 2026.** The
  coarse-to-fine structure of section 5. `docs/two-stage.md` documents every
  place this implementation departs from it, and there are several — the
  paper's false-alarm setting, its clutter filter, its peak estimator and its
  velocity constant all needed changing for this hardware.
- **ITU-R P.681**, land mobile satellite propagation, via `p681LMSChannel`.
- **MATLAB 5G Toolbox and Satellite Communications Toolbox** for waveform and
  orbit generation.
- **Passive radar literature** for CLEAN and the ECA family of direct-signal
  cancellers. This implementation uses the exactly-known-reference form, which
  is available here and generally is not.
- **D. Poullin, "Passive detection using digital broadcasters (DAB, DVB) with
  COFDM modulation", IEE Proceedings — Radar, Sonar and Navigation, 2005,
  DOI 10.1049/ip-rsn:20045017.** Early experimental use of a demodulated,
  reconstructed ideal broadcast reference.
- **M. Wypich, R. Maksymiuk, T. P. Zieliński, "Experimental Evaluation of
  5G NR OFDM-Based Passive Radar Exploiting Reference, Control, and User
  Data", Sensors, 2026, DOI 10.3390/s26041317.** Experimental 5G NR evidence
  that reconstructed user data improves passive-radar detection, and that
  decision errors bound the gain.
- **`ntn-isac-handoff.md`**, the original build brief, in the repository root.
  Section numbers referenced throughout the code and docs are its.
- **OpenNTN** (ant-uni-bremen), TR 38.811 models for Sionna. Evaluated, not
  used — see section 9.

---

## 11. Decisions, and which ones were revised

Locked at the start and not revisited:

- Waveform generated offline to files, behind a fixed contract, so the generator
  can be swapped for a real base station later without anything downstream
  changing.
- The satellite channel applied digitally on the transmit side.
- The display a separate process fed over a socket, so a crashed display cannot
  take a capture with it.
- No error-correction decoding anywhere in the live receive path. It is not
  needed — the sensing reference comes from hard decisions, not decoded bits.

Revised in the face of measurement:

| decision | what changed | why |
|---|---|---|
| no GPS discipline | optional for short physical-target dwells; required for the emulated-target path and long coherent dwells | the physical echo and direct path share the clock error, but the commanded emulated target does not |
| conducted only, no physical reflector | a real rotating target added | a chain validated only on synthetic targets has not been shown to work |
| 2.0 GHz | 2478 MHz ISM | 2.0 GHz is licensed nearly everywhere |
| detection limited by noise | detection limited by the direct path | measured: a −40 dB co-delay echo is invisible at *any* noise level without cancellation |
| CLEAN as a future improvement | CLEAN as the critical path | same measurement |
| Hann window throughout | Blackman-Harris over the air | its sidelobe level *is* the detection floor when the target shares a delay cell |
| occupancy affects sensing via processing gain | occupancy affects it via the *aperture* | going from 50% to 100% buys 11 dB of gain but 42 dB of sidelobe floor — a far larger effect, and a different explanation |

---

## 12. What is not built

- Anything on real RF. Every number in these documents is from simulation, and
  the first job on hardware is to find out which survive.
- Feeding the clock estimate to the emulated-target stages, so they work
  free-running too.
- A display panel for the micro-Doppler spectrum. The data is published; the
  dashboard does not plot it yet.
- Replacing the MATLAB generator with a real base station. The file contract
  exists precisely to make that a drop-in, and it has not been done.
- Anything above 20 MS/s, monostatic operation, multi-receiver fusion, or
  control-channel decoding. All explicitly out of scope.

---

## 13. The habit that found the most bugs

Worth recording, because it generalises.

A delay-Doppler map is a remarkably forgiving consumer. Several things that were
simply wrong did not move it at all: a grid that silently lost 3% of its symbols
every dwell, an amplitude error of √2 on every data symbol. Both were invisible
to sensing and both appeared the instant anything measured the *data* instead —
as a 3.3% symbol error rate on a noiseless channel, and as a flat 29% error
vector magnitude.

The pattern in the over-the-air work was the same but worse, because the failures
produced **confident wrong numbers rather than errors**:

- a rotation rate read off the interpolator's own comb rather than the target's,
  reported as 4.00 rev/s against a true 7.00 — with the tip-speed edge still
  correct, so it looked right;
- a "target" that turned out to be the transmitted frame's own 50 Hz structure,
  which read as exactly +50.0 Hz and did not move when the echo was swept over
  50 dB;
- a canceller that reported 0.0 dB on every dwell because it was looking for the
  direct path where the tracker believed it was rather than where it was.

The defence that worked every time: **sweep the parameter the number should
depend on.** If the echo drops 20 dB and the measurement does not move, it is not
measuring the echo. That single check is why the target-absent control run is in
the test suite and why the detection tables in these documents are sweeps rather
than single points.
