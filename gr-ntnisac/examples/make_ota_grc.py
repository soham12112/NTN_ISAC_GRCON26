#!/usr/bin/env python3
"""Generate the two over-the-air GRC flowgraphs.

Written as a generator rather than by hand because a .grc file is a few hundred
lines of YAML in which every block carries a dozen parameters that GRC will
silently default if they are missing, and a flowgraph that opens but is subtly
misconfigured is worse than one that fails to open. Run it from the repository
root:

    python3 gr-ntnisac/examples/make_ota_grc.py

and it rewrites ntnisac_ota_tx.grc and ntnisac_ota_rx.grc beside itself.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def state(x, y, enabled=True):
    return dict(bus_sink=False, bus_source=False, bus_structure=None,
                coordinate=[x, y], rotation=0,
                state="enabled" if enabled else "disabled")


def block(name, bid, params, x, y, enabled=True):
    p = dict(affinity="", alias="", comment="")
    p.update(params)
    return dict(name=name, id=bid, parameters=p, states=state(x, y, enabled))


def var(name, value, comment, x, y):
    return block(name, "variable", dict(value=value, comment=comment), x, y)



def import_os(x, y):
    """An `import os` block, so path variables can resolve against $NTN_ROOT.

    GRC runs a flowgraph from the directory of its .grc file, not from wherever
    the operator started GRC, so a relative "data/wf_ota" resolves to
    gr-ntnisac/examples/data/wf_ota and the bundle is not found. Hardcoding an
    absolute path would work on exactly one machine, and the whole point of the
    two over-the-air graphs is that they run on two.

    $NTN_ROOT is set by env.sh, which has to be sourced anyway or GRC will not
    find the ntnisac blocks at all - so if the flowgraph opened, the variable
    is there.
    """
    return block("import_os", "import", dict(imports="import os"), x, y)


def path_var(name, rel, comment, x, y):
    """A path variable anchored at $NTN_ROOT rather than the working directory."""
    return var(name,
               f'os.path.join(os.environ.get("NTN_ROOT", "."), {rel})',
               comment, x, y)


def options(fid, title, desc, comment, window=(1400, 900)):
    return dict(
        parameters=dict(
            author="NTN ISAC prototype", catch_exceptions="True",
            category="[ntnisac]", cmake_opt="", comment=comment,
            copyright="", description=desc, gen_cmake="On",
            gen_linking="dynamic", generate_options="no_gui",
            hier_block_src_path=".:", id=fid, max_nouts="0",
            output_language="python", placement="(0,0)", qt_qss_theme="",
            realtime_scheduling="True", run="True",
            run_command="{python} -u {filename}", run_options="run",
            sizing_mode="fixed", thread_safe_setters="",
            title=title, window_size=str(window).replace(" ", "")),
        states=state(8, 8))


def usrp_common(is_source, nchan=1):
    """The parameters both USRP blocks need.

    clock_source0 and time_source0 follow clock_src. The current hardware is a
    B210 pair, not the earlier GPSDO-equipped B200 pair. Use "internal" for the
    short-dwell physical target; a shared external 10 MHz/PPS mode is still
    needed before the commanded emulated target or long coherent dwells can be
    trusted across two radios.

    sync is set to 'sync: unknown pps' and start_time to the shared epoch, so
    both radios begin streaming on the same edge. Without it the transmitter's
    t_start and the receiver's are related only by how the operator started two
    programs, and the emulated target's commanded truth is worth nothing.
    """
    # A GRC parameter value is a plain Python expression evaluated in the
    # flowgraph's namespace, so a variable is referenced by bare name. The
    # ${...} form belongs to a block's own .yml template and is not understood
    # here - it compiles to "invalid syntax" at generation time.
    # Several of these are interpolated straight into generated source, so an
    # empty value is a syntax error rather than a default. dev_addr and
    # dev_args are joined with commas, so both must evaluate to strings;
    # stream_chans is passed to eval(), so it must be a literal list. The
    # failure mode is a SyntaxError in the mako template with the whole
    # template printed as the message, which does not obviously point here.
    d = dict(type="fc32", otw="", stream_args="", stream_chans="[]",
             dev_addr="dev_serial", dev_args="''", sync="unknown pps",
             start_time="epoch", clock_rate="30.72e6", num_mboards="1",
             clock_source0="clock_src", time_source0="clock_src",
             sd_spec0="", nchan=str(nchan), samp_rate="samp_rate",
             # clock_rate above is a literal, not `mcr`: GRC interpolates it as
             # source text into a string literal, so a variable name would be
             # passed to UHD verbatim as "master_clock_rate=mcr".
             center_freq0="rf_freq", gain0="gain", gain_type0="default",
             ant0="antenna", bw0="0", lo_source0="", lo_export0="False")
    if is_source:
        d.update(rx_agc0="Default", dc_offs_enb0="", dc_offs0="0",
                 iq_imbal_enb0="", iq_imbal0="0")
    else:
        d.update(len_tag_name="")
    return d


# --------------------------------------------------------------------- TX

TX_COMMENT = """TX LAPTOP. Waveform file -> emulated LEO pass -> USRP.

Run the receive flowgraph on the other laptop with the SAME epoch and t_start.
The two variables at the top left are the only things that must match.

The LEO channel is applied digitally before the DAC (handoff section 2), so
what leaves the antenna already carries the pass's Doppler, Doppler rate and
delay walk. Nothing about the emulation changes going over the air.

Power: at 2478 MHz under FCC part 15.247 this is a digital modulation system,
licence-free to 30 dBm with a 6 dBi antenna. A B210 gives about 10 dBm here.
tx_iq is scaled to RMS 0.1 with 10.6 dB of peak-to-average, so amp above ~2.5
clips the DAC - and clipping is a nonlinearity the receiver's reference does
not contain, so it quietly ruins every sensing measurement downstream."""


def tx_flowgraph():
    blocks = [
        import_os(8, 96),
        var("samp_rate", "7.68e6", "From config.json. Do not change alone.",
            184, 12),
        var("mcr", "30.72e6",
            "Master clock rate. 32e6 gives a decimation of 4.1667 which UHD "
            "approximates silently, putting a fractional resampler in a chain "
            "whose whole point is delay accuracy. 30.72e6 gives exactly 4.",
            304, 12),
        var("rf_freq", "2.478e9",
            "2.4 GHz ISM, above Wi-Fi channel 11 (which tops out at 2473) and "
            "below the band edge. Survey the local band first.", 448, 12),
        path_var("wf_dir", '"data", "wf_ota"',
                 "Anchored at $NTN_ROOT, not the working directory: GRC runs a "
                 "flowgraph from its own folder.", 592, 12),
        path_var("traj_dir", '"data", "traj_ota"', "Trajectory directory.",
                 736, 12),
        var("t_start", "300.0",
            "Trajectory time of the first transmitted sample. 300 is closest "
            "approach, where the range rate passes through zero: the easy "
            "epoch to bring a link up on. 500 is where the emulated drone sits "
            "at 640 m of excess path and is the one the demo uses.", 880, 12),
        var("epoch", "0.0",
            "MUST MATCH THE RECEIVER. Absolute GPS second at which the first "
            "sample is emitted. 0 means start immediately, which gives up the "
            "shared time base - use gr/flowgraphs/ota_tx.py to pick one.",
            1056, 12),
        var("clock_src", '"internal"',
            'Set to "internal" to run free and measure the clock-offset floor '
            "instead of eliminating it - handoff section 2's framing, which "
            "stays available on purpose.", 1200, 12),
        var("dev_serial", '""',
            "Set this to the transmitting B210 serial. Two radios on one bench and an "
            "unqualified device string is a coin toss.", 184, 100),
        var("gain", "70.0", "TX gain, dB.", 344, 100),
        var("antenna", '"TX/RX"', "", 464, 100),
        var("amp", "1.0", "Scalar before the DAC. Above ~2.5 clips.", 584, 100),
        var("tau_bias", "40.0",
            "Constant delay that keeps the interpolator's read pointer "
            "causally behind its write pointer. The receiver must use the "
            "same number.", 712, 100),

        block("blocks_file_source_0", "blocks_file_source",
              dict(file="wf_dir + '/tx_iq.cf32'", type="complex",
                   vlen="1", repeat="True", begin_tag="pmt.PMT_NIL",
                   length="0", offset="0", maxoutbuf="0", minoutbuf="0",
                   comment="repeat=True: the file is exactly 20 ms, one SSB "
                           "period, so it loops at a frame boundary."),
              184, 268),
        block("ntnisac_leo_channel_0", "ntnisac_leo_channel",
              dict(traj_csv="traj_dir + '/traj.csv'", fs="samp_rate",
                   fc="rf_freq", r_nominal="-1.0", tau_bias="tau_bias",
                   delay_line="131072", interp_taps="16", t_start="t_start",
                   maxoutbuf="0", minoutbuf="0",
                   comment="The emulated pass. 16-tap sinc: measured 14.2 MS/s "
                           "against a 7.68 MS/s requirement, so it clears real "
                           "time with margin."),
              448, 244),
        block("blocks_multiply_const_vxx_0", "blocks_multiply_const_vxx",
              dict(const="amp", type="complex", vlen="1", maxoutbuf="0",
                   minoutbuf="0", comment=""),
              760, 276),
        block("uhd_usrp_sink_0", "uhd_usrp_sink",
              usrp_common(False), 944, 236),
    ]
    conns = [
        ["blocks_file_source_0", "0", "ntnisac_leo_channel_0", "0"],
        ["ntnisac_leo_channel_0", "0", "blocks_multiply_const_vxx_0", "0"],
        ["blocks_multiply_const_vxx_0", "0", "uhd_usrp_sink_0", "0"],
    ]
    return dict(options=options("ntnisac_ota_tx",
                                "NTN ISAC over the air - TX laptop",
                                "File -> LEO channel -> USRP, GPS-disciplined",
                                TX_COMMENT),
                blocks=blocks, connections=conns,
                metadata=dict(file_format=1))


# --------------------------------------------------------------------- RX

RX_COMMENT = """RX LAPTOP. USRP -> two sensing paths -> ZMQ -> dashboard.

epoch and t_start MUST MATCH the transmit flowgraph.

There are two targets and they need different processing, so there are two
paths on the same samples.

  OTA Sense        the PHYSICAL object between the radios. At 4.5 MHz the
                   delay cell is 66.6 m and an object a metre off a 5 m
                   baseline has 0.39 m of bistatic excess path - 0.006 of a
                   cell - so it is inside the direct path's own cell and no
                   reachable bandwidth separates them. It is measured in
                   Doppler alone, after CLEAN removes the direct path.
                   Measured: a -40 dB co-delay echo reads +0.8 dB against a
                   target-absent run without cancellation and +54.0 dB with it.

  SSB Coarse +     the EMULATED drone at 640 m of excess path, with exact
  Grid Sense       commanded truth. Unchanged from the conducted flowgraph.

Running both is the point: a target whose answer is known exactly validates
the chain, and a target that is really there proves it is not validating
itself.

One caveat. Grid Sense and SSB Coarse track the pass from ephemeris and have
no clock-offset estimator, so they need the two radios on a shared reference.
Measured in simulation, 1 ppm of offset takes Grid Sense's peak-to-noise from
91 dB to 62 dB and its detections become nonsense. OTA Sense estimates the
offset itself and is immune, because a common carrier error cancels between
the direct path and the target. So: free-running gets the physical target; an
external shared 10 MHz/PPS reference gets both and enables long dwells."""


def rx_flowgraph():
    us = usrp_common(True)
    blocks = [
        import_os(8, 180),
        var("samp_rate", "7.68e6", "From config.json.", 184, 12),
        var("mcr", "30.72e6", "Exactly 4x decimation at 7.68 MS/s.", 304, 12),
        var("rf_freq", "2.478e9", "Must match the transmitter.", 448, 12),
        path_var("wf_dir", '"data", "wf_ota"',
                 "The receiver needs the bundle too: ref_mask and tx_grid are what "
            "make payload-aided sensing possible, and the reference it "
            "reconstructs from them IS the reference channel. This flowgraph "
                 "currently consumes one of the B210's two coherent RX channels.",
                 592, 12),
        path_var("traj_dir", '"data", "traj_ota"',
                 "Same trajectory the TX is playing.", 736, 12),
        var("t_start", "300.0", "MUST MATCH THE TRANSMITTER.", 880, 12),
        var("epoch", "0.0", "MUST MATCH THE TRANSMITTER.", 1024, 12),
        var("clock_src", '"internal"',
            'See the flowgraph comment: "internal" still gives the physical '
            "target but not the emulated one.", 1168, 12),
        var("dev_serial", '""', "Set this to the receiving B210 serial.",
            184, 100),
        var("gain", "40.0", "RX gain, dB. Set it so the direct path sits "
            "15-20 dB below full scale: the ADC has to hold the direct path "
            "and the target at once, and clipping the strong one destroys the "
            "weak one.", 344, 100),
        var("antenna", '"RX2"', "", 512, 100),
        var("tau_bias", "40.0",
            "Must match the transmitter. On real RF the extra fixed delay "
            "through cable, filters and pipeline is measured once and added "
            "here - handoff section 9's V4 calibration constant. OTA Sense "
            "converges on it and reports it.", 624, 100),
        var("n_frames", "5", "5 frames = 100 ms sensing dwell.", 824, 100),
        var("micro_frames", "50",
            "1 s micro-Doppler dwell. It has to be this long: 100 ms gives "
            "5 Hz Doppler cells against a 7 rev/s blade's 7 Hz line spacing, "
            "and an unresolved rotation rate comes out confident and wrong. "
            "Measured, 1 s recovers 7.00 and 11.00 exactly.", 936, 100),
        var("variant", '"payload-aided"',
            'The research claim. "reference-only" is the comparison.',
            1136, 100),

        block("uhd_usrp_source_0", "uhd_usrp_source",
              dict(us, maxoutbuf="0", minoutbuf="0"), 184, 300),
        block("ntnisac_ota_sense_0", "ntnisac_ota_sense",
              dict(wf_dir="wf_dir", traj_dir="traj_dir", n_frames="n_frames",
                   t_start="t_start", tau_bias="tau_bias", variant="variant",
                   pad="2", stride="1", window='"blackmanharris"',
                   gps_locked="clock_src == 'gpsdo'", n_clean="2",
                   clean_delay_ns="200.0", clean_doppler_hz="15.0",
                   micro_frames="micro_frames", micro_every="10",
                   rot_min_hz="3.0", rot_max_hz="40.0", snr_db="10.0",
                   cfo_max_hz="12e3", publish_profile="True",
                   link_metrics="True",
                   comment="The physical target. Blackman-Harris across slow "
                           "time: Hann's -31 dB sidelobe is the same number "
                           "as the direct path's own skirt at the target's "
                           "Doppler, and here the target is in the same delay "
                           "cell so that skirt IS the detection floor."),
              560, 188),
        block("ntnisac_ssb_coarse_0", "ntnisac_ssb_coarse",
              dict(wf_dir="wf_dir", traj_dir="traj_dir", n_frames="n_frames",
                   t_start="t_start", tau_bias="tau_bias", pad="2",
                   stride="1", gate_widen="3.0", sigma_accel="3.0",
                   min_snr_db="12.0", exclude_delay_cells="1.0",
                   notch_cells="2.0",
                   comment="Stage 1 for the emulated target. Needs the "
                           "GPSDOs: it has no clock-offset estimator."),
              560, 508),
        block("ntnisac_grid_sense_0", "ntnisac_grid_sense",
              dict(wf_dir="wf_dir", traj_dir="traj_dir", n_frames="n_frames",
                   t_start="t_start", tau_bias="tau_bias", variant="variant",
                   pad="2", stride="2", n_false_alarms="1.0",
                   notch_cells="2.0", exclude_delay_cells="1.0",
                   min_det_snr_db="6.0", min_renyi_bits="0.0",
                   max_fragmentation="-1.0", suppress_sidebands="True",
                   publish_maps="True", extra_variants='"reference-only"',
                   link_metrics="True", map_doppler_hz="250.0",
                   clutter_locus="True", clutter_max_range_m="250.0",
                   clutter_guard_bins="1.0",
                   clean_components="1", clean_delay_cells="1.0",
                   clean_doppler_hz="20.0",
                   comment="Stage 2. Extra variant maps the SAME dwell, which "
                           "is the only way two maps are a comparison."),
              560, 700),
        block("blocks_file_sink_0", "blocks_file_sink",
              dict(file='"data/capture.cf32"', type="complex", vlen="1",
                   append="False", unbuffered="False",
                   comment="Raw capture for the offline campaign. Disable if "
                           "not wanted: 7.68 MS/s is 61 MB per second."),
              560, 96, enabled=False),
        block("zeromq_pub_msg_sink_0", "zeromq_pub_msg_sink",
              dict(address="tcp://127.0.0.1:5555", bind="True",
                   timeout="100", comment="Physical target, per dwell."),
              1112, 180),
        block("zeromq_pub_msg_sink_1", "zeromq_pub_msg_sink",
              dict(address="tcp://127.0.0.1:5556", bind="True",
                   timeout="100",
                   comment="Micro-Doppler (1 s) and emulated-target sensing."),
              1112, 300),
        block("zeromq_pub_msg_sink_2", "zeromq_pub_msg_sink",
              dict(address="tcp://127.0.0.1:5557", bind="True",
                   timeout="100", comment="Sync state, for a lock indicator."),
              1112, 420),
    ]
    conns = [
        ["uhd_usrp_source_0", "0", "ntnisac_ota_sense_0", "0"],
        ["uhd_usrp_source_0", "0", "ntnisac_ssb_coarse_0", "0"],
        ["uhd_usrp_source_0", "0", "ntnisac_grid_sense_0", "0"],
        ["uhd_usrp_source_0", "0", "blocks_file_sink_0", "0"],
        ["ntnisac_ota_sense_0", "ota", "zeromq_pub_msg_sink_0", "in"],
        ["ntnisac_ota_sense_0", "micro", "zeromq_pub_msg_sink_1", "in"],
        ["ntnisac_ota_sense_0", "state", "zeromq_pub_msg_sink_2", "in"],
        ["ntnisac_ssb_coarse_0", "coarse", "ntnisac_grid_sense_0", "coarse"],
        ["ntnisac_grid_sense_0", "refine", "ntnisac_ssb_coarse_0", "refine"],
        ["ntnisac_grid_sense_0", "sense", "zeromq_pub_msg_sink_1", "in"],
    ]
    return dict(options=options("ntnisac_ota_rx",
                                "NTN ISAC over the air - RX laptop",
                                "USRP -> physical + emulated sensing -> ZMQ",
                                RX_COMMENT, (1500, 1000)),
                blocks=blocks, connections=conns,
                metadata=dict(file_format=1))


# ------------------------------------------------------------------- bench

BENCH_COMMENT = """BENCH. The whole over-the-air receive path, with no radio.

Run this first. It needs nothing but the data/ directory, it runs on one
laptop, and it exercises the identical blocks the two over-the-air flowgraphs
use - so a number that comes out here is a number the receive path produces.

    File Source -> LEO Channel -> Es/N0 Noise -> LEO Truth -+-> OTA Sense
                                                            +-> SSB Coarse
                                                            +-> Grid Sense

The rotating object that would physically sit between the two radios is the
fourth path in data/traj_bench, injected by tools/add_bench_target.py. That is
not a special case: a rotating scatterer is a range history like any other and
the channel block already takes any number of paths, so it plays through the
same C++ block the transmit flowgraph uses. Regenerate it with

    $NTN_PY tools/add_bench_target.py --rot-hz 7 --gain-db -30

What to look at. Set esn0_db and blade echo wherever you like, then compare the
two numbers OTA Sense prints per dwell: the direct path's peak-to-noise, and
how far CLEAN took it down. Then watch the "micro" messages, which arrive once
per second because that is what it takes to resolve a 7 rev/s line spacing -
at the 100 ms sensing dwell the Doppler cell is 5 Hz and the rate comes out a
harmonic.

The instructive experiment is to open tools/add_bench_target.py's --gain-db
down to -40 and compare Grid Sense with OTA Sense. Grid Sense still finds the
emulated drone at 640 m, because it is resolved in delay. Nothing finds the
rotating object without CLEAN, because it is 0.006 of a delay cell from the
direct path and what hides it is not noise."""


def bench_flowgraph():
    blocks = [
        import_os(8, 96),
        var("samp_rate", "7.68e6", "From config.json.", 184, 12),
        path_var("wf_dir", '"data", "wf_ota"',
                 "Anchored at $NTN_ROOT, so it does not matter where GRC runs "
                 "the flowgraph from.", 304, 12),
        path_var("traj_dir", '"data", "traj_bench"',
                 "traj_ota plus a rotating target as a fourth path, made by "
                 "tools/add_bench_target.py. Use traj_bench_all to keep the "
                 "emulated clutter and drone as well.", 448, 12),
        var("t_start", "300.0",
            "Closest approach: the range rate passes through zero, so this is "
            "the benign epoch. Try 500 to put the emulated drone at 640 m of "
            "excess path, which is where it has the most separation.",
            664, 12),
        var("tau_bias", "40.0", "", 840, 12),
        var("n_frames", "5", "5 frames = 100 ms sensing dwell.", 944, 12),
        var("esn0_db", "45.0",
            "Es/N0 per occupied subcarrier. The point of the bench is that "
            "lowering this is NOT what hides the rotating target.", 1064, 12),
        var("variant", '"payload-aided"',
            'The research claim. Try "reference-only" for the comparison.',
            1240, 12),

        block("blocks_file_source_0", "blocks_file_source",
              dict(file="wf_dir + '/tx_iq.cf32'", type="complex", vlen="1",
                   repeat="True", begin_tag="pmt.PMT_NIL", length="0",
                   offset="0", maxoutbuf="0", minoutbuf="0",
                   comment="20 ms, one SSB period, so repeat loops at a frame "
                           "boundary."),
              152, 220),
        block("ntnisac_leo_channel_0", "ntnisac_leo_channel",
              dict(traj_csv="traj_dir + '/traj.csv'", fs="samp_rate",
                   fc="2.478e9", r_nominal="-1.0", tau_bias="tau_bias",
                   delay_line="131072", interp_taps="16", t_start="t_start",
                   maxoutbuf="0", minoutbuf="0",
                   comment="Four paths: direct, static clutter, the emulated "
                           "drone, and the rotating object."),
              384, 196),
        block("ntnisac_esn0_noise_0", "ntnisac_esn0_noise",
              dict(esn0_db="esn0_db", n_fft="512", n_occ="300",
                   signal_rms="-1.0", settle="768000", seed="0",
                   maxoutbuf="0", minoutbuf="0",
                   comment="Calibrated against the occupied subcarriers."),
              696, 228),
        block("ntnisac_leo_truth_0", "ntnisac_leo_truth",
              dict(traj_dir="traj_dir", fs="samp_rate", t_start="t_start",
                   period_ms="100.0", tau_bias="tau_bias", port="truth",
                   maxoutbuf="0", minoutbuf="0",
                   comment="Commanded truth on the sample clock, for scoring."),
              920, 228),
        block("ntnisac_ota_sense_0", "ntnisac_ota_sense",
              dict(wf_dir="wf_dir", traj_dir="traj_dir", n_frames="n_frames",
                   t_start="t_start", tau_bias="tau_bias", variant="variant",
                   pad="2", stride="1", window='"blackmanharris"',
                   gps_locked="False", n_clean="2", clean_delay_ns="200.0",
                   clean_doppler_hz="15.0", micro_frames="50",
                   micro_every="10", rot_min_hz="3.0", rot_max_hz="40.0",
                   snr_db="10.0", cfo_max_hz="12e3", publish_profile="True",
                   link_metrics="True",
                   comment="The rotating object. Sync, CLEAN, micro-Doppler."),
              1176, 100),
        block("ntnisac_ssb_coarse_0", "ntnisac_ssb_coarse",
              dict(wf_dir="wf_dir", traj_dir="traj_dir", n_frames="n_frames",
                   t_start="t_start", tau_bias="tau_bias", pad="2",
                   stride="1", gate_widen="3.0", sigma_accel="3.0",
                   min_snr_db="12.0", exclude_delay_cells="1.0",
                   notch_cells="2.0",
                   comment="Stage 1, the emulated drone."),
              1176, 396),
        block("ntnisac_grid_sense_0", "ntnisac_grid_sense",
              dict(wf_dir="wf_dir", traj_dir="traj_dir", n_frames="n_frames",
                   t_start="t_start", tau_bias="tau_bias", variant="variant",
                   pad="2", stride="2", n_false_alarms="1.0",
                   notch_cells="2.0", exclude_delay_cells="1.0",
                   min_det_snr_db="6.0", min_renyi_bits="0.0",
                   max_fragmentation="-1.0", suppress_sidebands="True",
                   publish_maps="False", extra_variants='""',
                   link_metrics="False", map_doppler_hz="250.0",
                   clutter_locus="True", clutter_max_range_m="250.0",
                   clutter_guard_bins="1.0",
                   clean_components="1", clean_delay_cells="1.0",
                   clean_doppler_hz="20.0",
                   comment="Stage 2, the emulated drone. Stride 2 because it "
                           "costs more than its dwell in Python."),
              1176, 620),
        block("ntnisac_msg_print_0", "ntnisac_msg_print",
              dict(tag='"ota"', verbose="False",
                   comment="NOT Message Debug. Payloads are JSON in a "
                           "u8vector, which Message Debug prints as several "
                           "thousand decimal bytes."),
              1560, 96),
        block("ntnisac_msg_print_1", "ntnisac_msg_print",
              dict(tag='"micro"', verbose="False",
                   comment="Rotation rate, once per second."),
              1560, 232),
        block("ntnisac_msg_print_2", "ntnisac_msg_print",
              dict(tag='"drone"', verbose="False",
                   comment="The emulated target, resolved in delay."),
              1560, 620),
    ]
    conns = [
        ["blocks_file_source_0", "0", "ntnisac_leo_channel_0", "0"],
        ["ntnisac_leo_channel_0", "0", "ntnisac_esn0_noise_0", "0"],
        ["ntnisac_esn0_noise_0", "0", "ntnisac_leo_truth_0", "0"],
        ["ntnisac_leo_truth_0", "0", "ntnisac_ota_sense_0", "0"],
        ["ntnisac_leo_truth_0", "0", "ntnisac_ssb_coarse_0", "0"],
        ["ntnisac_leo_truth_0", "0", "ntnisac_grid_sense_0", "0"],
        ["ntnisac_ssb_coarse_0", "coarse", "ntnisac_grid_sense_0", "coarse"],
        ["ntnisac_grid_sense_0", "refine", "ntnisac_ssb_coarse_0", "refine"],
        ["ntnisac_ota_sense_0", "ota", "ntnisac_msg_print_0", "in"],
        ["ntnisac_ota_sense_0", "micro", "ntnisac_msg_print_1", "in"],
        ["ntnisac_grid_sense_0", "sense", "ntnisac_msg_print_2", "in"],
    ]
    return dict(options=options("ntnisac_ota_bench",
                                "NTN ISAC over-the-air bench (no radio)",
                                "The OTA receive path, simulated end to end",
                                BENCH_COMMENT, (1800, 1000)),
                blocks=blocks, connections=conns,
                metadata=dict(file_format=1))


def write(name, doc):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False,
                       width=100)
    print("wrote", path)


if __name__ == "__main__":
    write("ntnisac_ota_bench.grc", bench_flowgraph())
    write("ntnisac_ota_tx.grc", tx_flowgraph())
    write("ntnisac_ota_rx.grc", rx_flowgraph())
