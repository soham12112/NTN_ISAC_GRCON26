# NTN ISAC (GRCon 2026)

Hardware-in-the-loop **5G NR integrated sensing and communication** over an
emulated LEO channel, as GNU Radio blocks.

This repository is the **end-to-end ISAC path**: simulation on one laptop, then
the same blocks over the air on two USRP B210s. Calibration scripts, lab
handoffs, and capture logs are not included.

## What you get

- **`gr-ntnisac`** — GNU Radio out-of-tree module (C++ LEO channel + Python
  sensing blocks)
- **`py/ntn_isac`** — shared Python library (OFDM, tracking, two-stage sensing,
  OTA sync, CLEAN, micro-Doppler)
- **`gr/flowgraphs`** — command-line drivers for sim, loopback, TX, and RX
- **`gr-ntnisac/examples/*.grc`** — the same graphs on a GNU Radio Companion
  canvas
- **`matlab/`** — waveform and trajectory generators (run once; not in the
  live path)
- **`data/wf_ota`** and **`data/traj_ota`** — a sample 2478 MHz bundle so you
  can run without MATLAB on the first try

## Requirements

- GNU Radio 3.10 with `gr-uhd`, `gr-zeromq`, `gr-qtgui`, UHD 4.x
- Python packages in **the interpreter GNU Radio was built against**:
  `numpy`, `scipy`, `pyzmq`, `pyqtgraph`
- Optional: MATLAB R2023b+ with 5G Toolbox and Satellite Communications Toolbox
  (only to regenerate waveforms)
- Optional: two USRP B210s for over-the-air

## Build

```sh
cd gr-ntnisac && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../install -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native"
make -j8 && make install
cd ../..
. ./env.sh
```

On macOS, point CMake at the interpreter that has `gnuradio`, for example
`-DPYTHON_EXECUTABLE=/opt/homebrew/bin/python3.14`.

`env.sh` sets `$NTN_PY`, `$NTN_ROOT`, `PYTHONPATH`, and `GRC_BLOCKS_PATH`.
Source it in every terminal you use for this project, including before opening
GNU Radio Companion.

## Simulation (no radios)

Two-stage coarse-to-fine sensing (emulated clutter + drone):

```sh
. ./env.sh
$NTN_PY gr/flowgraphs/twostage.py --wf data/wf_ota --traj data/traj_ota \
    --seconds 2 --t-start 500
```

Over-the-air receive path, still fully simulated (direct-path CLEAN + rotating
target):

```sh
$NTN_PY gr/flowgraphs/ota_rx.py --wf data/wf_ota --traj data/traj_ota \
    --mode sim --seconds 4 --sim-eps-ppm 2.5 --blade-rot-hz 7
```

On a GRC canvas:

```sh
. ./env.sh
gnuradio-companion gr-ntnisac/examples/ntnisac_twostage.grc
gnuradio-companion gr-ntnisac/examples/ntnisac_ota_bench.grc
```

Dashboards:

```sh
$NTN_PY gr/dashboard.py --attach          # two-stage maps
$NTN_PY gr/ota_dashboard.py --launch bench
```

Contract and sensing checks (need the sample bundles above):

```sh
$NTN_PY tests/check_bundle.py data/wf_ota
$NTN_PY tests/verify_twostage.py
$NTN_PY tests/verify_ota.py
```

## Over the air

Read `docs/running.md` section 3 and `docs/ota.md` before transmitting.
Highlights:

- Bundles are at **2478 MHz** (2.4 GHz ISM). Survey the local band first.
- A B210 is about 10 dBm. Check your local rules; in the US this is typically
  FCC Part 15.247 at this bandwidth.
- Indoor delay resolution is ~67 m, so a physical object is **unresolved in
  delay**. The OTA path measures **Doppler** after cancelling the direct path.
  Use a rotating metal blade (~7 or 11 rev/s), not a person walking.

One-radio cable loopback first (TX/RX to RX2 through **≥40 dB** attenuation):

```sh
$NTN_PY gr/flowgraphs/loopback.py --wf data/wf_ota --traj data/traj_ota
```

Two laptops, TX first:

```sh
$NTN_PY gr/flowgraphs/ota_tx.py --wf data/wf_ota --traj data/traj_ota \
        --t-start 300 --clock internal --tx-gain 70
```

Paste the printed receive command on the other laptop. Same graphs as
`ntnisac_ota_tx.grc` / `ntnisac_ota_rx.grc`.

Free-running clocks are enough for the **physical** target. A shared 10 MHz/PPS
reference is required if you also want the **emulated** drone from `Grid Sense`.

## Regenerating data

Python, no MATLAB (orbit + scatterers from a scene file):

```sh
$NTN_PY tools/build_scene.py scenes/bench.yml --plot
```

MATLAB (waveform + 2478 MHz OTA bundles):

```matlab
addpath matlab
gen_ota_bundles                 % data/wf_ota, wf_ota050, traj_ota
```

## Layout

```
py/ntn_isac/     Python library (channel, receiver, sensing, OTA)
gr-ntnisac/      OOT: leo_channel, esn0_noise, leo_truth,
                 ssb_coarse, grid_sense, ota_sense, link_budget
gr/flowgraphs/   twostage, ota_tx, ota_rx, loopback, tx_usrp
matlab/          waveform and trajectory generators
scenes/          YAML channel scenes (compiled by tools/build_scene.py)
docs/            architecture, how to run, OTA, two-stage sensing
```

Start with `docs/architecture.md`, then `docs/running.md`.

## License

`gr-ntnisac` is GPL-3.0-or-later (GNU Radio OOT). The rest of this tree is
released under the same license so the module can be distributed together.
See `LICENSE`.
