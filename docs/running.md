# How to run it

Three ways to run this, in the order you should try them.

| | what it needs | what it shows |
|---|---|---|
| **Bench** | one laptop, no radio | the whole receive path, both kinds of target |
| **Loopback** | one laptop, one radio, a cable | that the radio path works before two of them do |
| **Conducted** | two laptops, two radios, a splitter | real hardware against exact commanded truth |
| **Over the air** | two laptops, two radios, antennas | the actual demo |

Do not skip to the last. Almost everything that can go wrong is visible in the
first, where it costs seconds to diagnose instead of an afternoon on a bench
with two machines.

The third rung has its own document, `docs/conducted.md`, because it is not a
degraded version of the fourth: it is the only configuration where real RF
hardware and exact commanded truth exist at once, so it is where the simulated
numbers in these documents get checked. Read it before cabling two radios
together - a bare TX-to-RX path damages a receive front end.

---

# 0. Setup, once per machine

## 0.1 What has to be installed

- **MATLAB** with the 5G Toolbox and the Satellite Communications Toolbox. Only
  needed to generate the waveform and trajectory files; nothing at run time
  touches MATLAB.
- **GNU Radio 3.10** with `gr-uhd`, `gr-zeromq` and `gr-qtgui`, and **UHD 4.x**.
- **Python packages** `numpy`, `scipy`, `pyzmq`, `pyqtgraph` — in the
  interpreter GNU Radio was built against. See 0.4, this is the single most
  common way to lose an hour.

## 0.2 Build the out-of-tree module

```sh
cd gr-ntnisac && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../install -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native"
make -j8 && make install
cd ../..
```

On macOS add `-DPYTHON_EXECUTABLE=/opt/homebrew/bin/python3.14`, or whichever
interpreter has `gnuradio` in it.

## 0.3 Source the environment

```sh
. ./env.sh
```

This sets `$NTN_PY` to the right interpreter, puts `py/` and the installed OOT
on `PYTHONPATH`, and sets `GRC_BLOCKS_PATH` so the `ntnisac` blocks appear in
GNU Radio Companion. **Every command below assumes you have done this**, and
the flowgraphs resolve their data paths against `$NTN_ROOT`, which it also
sets.

`env.sh` handles both hosts: the Linux box (GNU Radio 3.10.1.1 under
`/usr/bin/python3`, with ROS Humble scrubbed from `PYTHONPATH` because its
numpy conflicts) and macOS (GNU Radio 3.10.12.0 under Homebrew python3.14, with
a `.venv` for the packages Homebrew does not supply).

## 0.4 The interpreter trap

`$NTN_PY` is what the command-line scripts use. **GNU Radio Companion's Run
button is not `$NTN_PY`** — it uses the interpreter GNU Radio itself was built
against. On macOS that is Homebrew's python3.14, which ships `gnuradio` and
`numpy` but not `scipy`, so a flowgraph that runs perfectly from the terminal
dies in GRC with:

```
ModuleNotFoundError: No module named 'scipy'
```

raised from inside a block constructor, several frames below anything you
wrote. Install the packages into that interpreter once:

```sh
/opt/homebrew/bin/python3.14 -m pip install --user --break-system-packages \
    scipy pyzmq pyqtgraph
```

`--break-system-packages` is the PEP 668 escape hatch. It writes to
`~/.local/lib/python3.14/site-packages` and does not touch Homebrew's own tree.

## 0.5 Generate the data

In MATLAB, from the repository root:

```matlab
addpath matlab
gen_ota_bundles                 % data/wf_ota, wf_ota050, traj_ota at 2478 MHz
```

Then compile a channel scene, which builds the trajectory including the
rotating object:

```sh
$NTN_PY tools/build_scene.py scenes/bench.yml --plot
```

That writes `data/traj_bench`. Everything below uses those.

**The scene file is the channel.** `scenes/bench.yml` lists the orbit and every
path - static scatterers, crossing targets, rotating objects, pendulums,
arbitrary range histories, and 3GPP TR 38.811 multipath - and one command
compiles it. It needs no MATLAB: the orbit is propagated analytically and the
TR 38.811 tap tables are in `py/ntn_isac/scene.py`, cross-checked against
MATLAB's `nrTDLChannel` to the nanosecond. `scenes/bench_both.yml` is the same
scene with the emulated clutter and drone as well.

Override any field from the command line, so a sweep is a shell loop rather
than a directory of near-identical files:

```sh
for g in -20 -30 -40 -50; do
  $NTN_PY tools/build_scene.py scenes/bench.yml \
          --set paths.1.gain_db=$g --out data/traj_g$g
done
```

`--plot` writes a picture of every path's excess path, residual Doppler and
gain against pass time, with the cyclic-prefix budget and the zero-Doppler
notch marked - which is the quickest way to see whether a target you have just
described is actually observable.

## 0.6 Check it

```sh
$NTN_PY tests/check_bundle.py data/wf_ota   # the file-interface contract
$NTN_PY tests/verify_v1_v3.py               # the channel, V1-V3
$NTN_PY tests/verify_twostage.py            # the sensing flow, V8-V15
$NTN_PY tests/verify_ota.py                 # the OTA additions, V16-V19
$NTN_PY tests/clutter_locus.py               # M3 stationary-clutter envelope
$NTN_PY tests/link_budget.py                 # link SNR versus detector margin
```

All four should end in `OK`. If `verify_ota` fails and the others pass, the
problem is in the new code; if they all fail, it is the environment.

---

# 1. Bench — one laptop, no radio

This runs the entire over-the-air receive path with the radio replaced by a
simulation of everything downstream of the antenna connector. It uses the same
blocks the live flowgraphs use, so a number that comes out here is a number the
receive path produces.

## 1.1 On the GRC canvas

```sh
. ./env.sh
gnuradio-companion gr-ntnisac/examples/ntnisac_ota_bench.grc
```

```
File Source ─► LEO Channel ─► Es/N0 Noise ─► LEO Truth ─┬─► OTA Sense ──► Message Print "ota"
                                                         │            └─► Message Print "micro"
                                                         ├─► SSB Coarse ◄─┐
                                                         └─► Grid Sense ──┘──► Message Print "drone"
```

Press Run. Output, per 100 ms dwell and once per second:

```
[ota] t= 311.900  direct  88.4 dB -> cancelled  59.5 dB   eps +0.000 ppm  resid +0.00 sa  -0.02 Hz
      micro: edge +-155.0 Hz (18.8 m/s), strongest line +110.0 Hz at 48 dB, 22 lines
[micro] 1000 ms dwell: rotation 11.00 rev/s = 660 RPM  edge +-155.0 Hz  radius 27.1 cm
```

Reading it:

- **`direct ... -> cancelled`** — how far above the noise the direct path stood,
  and how much of it CLEAN removed. The second number is the one that matters:
  it is the floor the physical target has to clear.
- **`eps`** — the estimated clock difference between the two radios, in parts
  per million. Zero on the bench unless you ask for one.
- **`resid`** — how far the tracker is from the truth, in samples of delay and
  hertz of Doppler. It should settle near zero within a few dwells.
- **`micro`** — the rotating object. `edge` is the tip-speed Doppler, from which
  the tip speed follows. `rotation` needs the 1 s dwell, so it appears on the
  `[micro]` line only.

## 1.2 The live dashboard

Text is not the only output. For the over-the-air path:

```sh
$NTN_PY gr/ota_dashboard.py --launch bench
```

![the over-the-air dashboard](img/ota-dashboard.png)

`--launch bench` starts the simulated flowgraph itself; without it the
dashboard attaches to one already running, which is what you want over the air.
It is a separate process fed over a socket on purpose, so that a display crash
cannot take a capture with it.

The two maps share **one colour scale**, fixed by the uncancelled map's noise
floor. That is deliberate: a cancelled map normalised to its own median looks
improved even when nothing has moved, which is the easiest way to convince
yourself a canceller works when it does not.

`gr/dashboard.py` is the other display, for the emulated-target two-stage flow,
with its own delay-Doppler maps and the PDSCH occupancy demo.

## 1.3 From the command line

Same thing, more knobs, no GUI:

```sh
$NTN_PY gr/flowgraphs/ota_rx.py --wf data/wf_ota --traj data/traj_ota \
        --mode sim --seconds 4 --sim-eps-ppm 2.5 --blade-rot-hz 7
```

Or as a script that writes a figure instead of text:

```sh
$NTN_PY tools/ota_demo.py --blade-gain-db -40
```

Three panels: the map as the conducted chain would produce it, the same dwell
after CLEAN on the same colour scale, and the Doppler profile at the direct
path's delay with the recovered rotation harmonics marked.

## 1.4 Things worth trying

**Turn the echo down.** This is the experiment the whole over-the-air design
rests on:

```sh
$NTN_PY tools/add_bench_target.py --gain-db -45 --rot-hz 11
```

Re-run. `Grid Sense` still finds the emulated drone at 640 m, because that one
is resolved in delay. Nothing finds the rotating object without CLEAN, because
it is 0.006 of a delay cell from the direct path and what hides it is not
noise. Lowering `esn0_db` does **not** reproduce that, which is the point.

**Change the rotation rate.** Avoid multiples and divisors of 25: that is where
the aperture's own Doppler comb lives, and a rate that divides it puts the
target's entire signature on masked bins. 7 and 11 are good, 5 is not.

```sh
$NTN_PY tools/add_bench_target.py --rot-hz 7
$NTN_PY tools/add_bench_target.py --rot-hz 5     # prints a warning
```

**Put both targets in one capture.**

```sh
$NTN_PY tools/add_bench_target.py --keep-emulated --out data/traj_bench_all
```

then point the flowgraph's `traj_dir` at it. Note that the rotation-rate
estimate degrades when both are present — the emulated drone sits in the same
Doppler band — which is why the default excludes it. A demo wants both; a
measurement of one has to exclude the other.

**Add a 3GPP NTN channel.** In MATLAB:

```matlab
gen_ntn_channel('InDir','data/traj_ota','Profile','NTN-TDL-C');
```

writes `data/traj_ota_ntn` with TR 38.811 multipath clusters layered on.

---

# 2. Loopback — one laptop, one radio

Before two machines, prove one. A B210 is full duplex, so transmit and receive
can run in one process on one radio, which removes the clock offset entirely
because both ends share a TCXO.

Connect the radio's **TX/RX** port to its **RX2** port through **at least 40 dB
of attenuation**. Do not skip the attenuator: the transmitter delivers about
10 dBm and the receiver's front end will not enjoy it.

```sh
$NTN_PY gr/flowgraphs/loopback.py --wf data/wf_ota --traj data/traj_ota
```

What you are checking: that the radio streams at 7.68 MS/s without underruns or
overruns, that the receiver acquires, and that the delay bias settles to a
stable number. **Write that number down** — it is the fixed delay through DAC,
filters, cable, ADC and pipeline, it is a property of your hardware, and both
over-the-air flowgraphs want it as `tau_bias`.

---

# 3. Over the air — two laptops, two radios

## 3.1 Physical setup

- **Radios.** One B210 per laptop. Note the serial of each
  (`uhd_find_devices`); the flowgraphs pin them by serial, because two B210s on
  one bench and an unqualified device string is a coin toss.
- **Optional shared reference.** A cabled 10 MHz and PPS reference is needed
  for the commanded emulated target and long coherent dwells, not for the
  short-dwell physical-target path. See 3.2.
- **RF antennas.** Two 2.4 GHz antennas, a few metres apart, roughly facing
  each other.
- **The target.** A metal or foil-covered blade of radius 0.2–0.3 m on a
  variable-speed motor, on or beside the line between the antennas. A drone
  propeller on a bench motor is better still. Set it to something near 7 rev/s
  (420 RPM) and **measure the actual rate** with a tachometer or a phone
  slow-motion video — that is your ground truth.

Why a rotating target and not a person walking: at 2.478 GHz a walker reaches
about 21 Hz of Doppler and the zero-Doppler notch blinds ±30 Hz, so a person is
invisible by construction. A 0.25 m blade at 7 rev/s reaches 180 Hz. It also
carries its own truth — the line spacing gives the rotation rate and the
envelope edge gives the tip radius — and it repeats.

## 3.2 Clocks: read this before deciding

With a **shared 10 MHz and PPS reference**, the fractional difference between
the two radios is negligible and both devices share an epoch. The B210 pair
does not provide the two internal GPSDOs described by the earlier B200 setup;
use an external reference if this mode is required.

**Free-running**, it is up to 4 ppm — 9.9 kHz of carrier offset at 2.478 GHz
and 31 samples per second of delay drift.

That difference decides what you can measure:

| | free-running | shared reference |
|---|---|---|
| physical target (`OTA Sense`) | **works** | works |
| emulated drone (`Grid Sense`) | **does not** | works |

`OTA Sense` estimates the clock offset itself, and in any case a carrier offset
is common to the direct path and the echo and cancels in the excess Doppler.
`Grid Sense` and `SSB Coarse` track the pass from ephemeris and have no clock
estimator: measured in simulation, 1 ppm takes `Grid Sense` from 91 dB of
peak-to-noise to 62 dB and its detections become nonsense.

So: **free-running gets you the physical target; a shared reference gets you
both.** The shared reference also becomes mandatory once the coherent dwell is
extended beyond the short OTA measurement.

## 3.3 Frequency and power

The bundles are generated at **2478 MHz**. Wi-Fi channels 1, 6 and 11 occupy
2401–2423, 2426–2448 and 2451–2473, and 2478 puts the 7.7 MHz you need in the
gap above channel 11. Survey your local band before trusting that.

Under FCC part 15.247 this is a digital modulation system and is licence-free to
30 dBm with a 6 dBi antenna; a B210 gives about 10 dBm, so there is 20 dB of
headroom. Outside the US, check your own rules.

Two gain settings that matter:

- **`--amp` on the transmitter** scales the samples before the DAC. `tx_iq` is
  already at RMS 0.1 with 10.6 dB of peak-to-average, so above about 2.5 you
  clip. Clipping is a nonlinearity the receiver's reference does not contain,
  so it does not look like distortion — it raises the floor of every sensing
  measurement and looks like a receiver bug.
- **`--rx-gain` on the receiver** should put the direct path 15–20 dB below full
  scale. The ADC has to hold the direct path and the target at once, and
  clipping the strong one destroys the weak one.

## 3.4 Run it

**Transmit laptop first:**

```sh
. ./env.sh
$NTN_PY gr/flowgraphs/ota_tx.py --wf data/wf_ota --traj data/traj_ota \
        --t-start 300 --clock internal --tx-gain 70
```

It waits for GPS lock, picks a shared start time on a round second, and prints
the exact receive command line to paste on the other machine:

```
    On the receive laptop, run:
      $NTN_PY gr/flowgraphs/ota_rx.py --wf data/wf_ota --traj data/traj_ota \
          --t-start 300 --epoch 1789012340.0 --clock internal
```

**Receive laptop:** paste that, adding your measured delay bias and a capture
file if you want one:

```sh
$NTN_PY gr/flowgraphs/ota_rx.py --wf data/wf_ota --traj data/traj_ota \
        --t-start 300 --epoch 1789012340.0 --clock internal \
        --tau-bias 512 --rx-gain 40 --capture-out data/run1.cf32
```

`--epoch` and `--t-start` are the only two things that must match, and together
they say "trajectory time `t_start` happened at absolute time `epoch`". Without
them the receiver has to search a whole pass instead of a frame, and the
emulated target's commanded truth means nothing.

**Or on the canvas:**

```sh
gnuradio-companion gr-ntnisac/examples/ntnisac_ota_tx.grc   # TX laptop
gnuradio-companion gr-ntnisac/examples/ntnisac_ota_rx.grc   # RX laptop
```

Set `epoch` and `t_start` by hand in both, to the same values. The Python
drivers are easier for exactly this reason.

## 3.5 Which epoch of the pass

- **`--t-start 300`** is closest approach: the range rate passes through zero,
  so Doppler is near zero. Bring the link up here. Nothing is being avoided —
  it is a real part of a real pass.
- **`--t-start 500`** is where the emulated drone sits at 640 m of excess path
  with −39.0 Hz of residual Doppler at 2478 MHz, clear of the MTI blind speed.
  Move here once the link closes, because that is where the synthetic target
  has its best separation. This emulated path requires the radios to share an
  external reference; free-running is valid for the physical target only.

**Do not compress the trajectory** to make acquisition easier. It makes the
Doppler rate unphysical and invalidates the whole NTN claim, and a
high-elevation epoch is already slow without cheating.

## 3.6 The dashboard

Add `--zmq tcp://127.0.0.1:5555` to the receive command, then in another
terminal:

```sh
$NTN_PY gr/dashboard.py --attach
```

It is a separate process on purpose, so that a display crash cannot take the
flowgraph — and with it the capture — down with it.

---

# 4. When it does not work

| symptom | likely cause |
|---|---|
| `ModuleNotFoundError: scipy` from GRC only | 0.4 — wrong interpreter |
| `FileNotFoundError: data/wf_ota/config.json` | `env.sh` not sourced, so `$NTN_ROOT` is unset |
| ntnisac blocks missing in GRC | `env.sh` not sourced, or the OOT not installed |
| `NOT LOCKED`, acq peak below ~12 dB | no signal, wrong frequency, or gain far too low |
| locked but `resid` never settles | `--epoch` mismatched between the two laptops |
| `USRP set X S/s, not 7680000` | `--mcr` must be 30.72e6, not the B210 default |
| `cancelled 0.0 dB` every dwell | CLEAN found nothing where it looked; check the tracker locked first |
| `rotation` reported but wrong, edge correct | dwell too short, or the emulated targets are on |
| underruns / overruns (`U`, `O`) | USB 3 port, or another process competing |

A general rule from building this: the failures that cost the most time were the
ones that produced a **confident wrong number** rather than an error — a
rotation rate read off the interpolator's comb, a target-absent run measured as
a detection, a canceller that reported 0.0 dB. If a number looks plausible but
does not move when you change the thing it should depend on, it is measuring
something else. Sweep the parameter; if the answer does not follow, stop.
