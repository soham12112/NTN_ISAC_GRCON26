"""Optional disciplined start, so two laptops agree on the pass epoch.

The conducted setup never needed this: one process drove both ends, so the
trajectory time of a sample was whatever the driver said it was. Over the air
the receiver has to know which trajectory time the transmitter was at when it
emitted the sample now arriving, and it has no back channel.

Three clock modes, and the choice is not cosmetic:

`internal`  each radio on its own TCXO. Device time is set to zero when the
            program starts, so `--epoch` is in *device* time and the two zeros
            are separated by however far apart the operator launched the two
            programs. `OtaTracker` absorbs that into its `d0`, so the
            physical-target path is unaffected; `ssb_coarse` and `grid_sense`
            read it as trajectory time and their commanded truth becomes
            meaningless. Run those two only under `gpsdo` or `external`.
`gpsdo`     what the original B200 setup's internal GPSDOs bought. The current
            B210 pair has none, so this path is available only when the radio
            actually reports a `gps_locked` sensor.
`external`  a cabled 10 MHz and 1 PPS into both radios, which is the mode the
            B210 pair actually has. See `lock_external` for the one host
            requirement it carries.

What a shared reference buys, either way:

- a common 10 MHz, so the fractional clock error eps between the radios goes
  from a few parts per million to around 1e-11. At 2.478 GHz that is 9.9 kHz
  of carrier offset and 31 samples per second of delay drift, down to 0.025 Hz
  and 8e-5 samples per second. The first has to be tracked; the second is
  below anything the chain can measure over a 600 s pass;
- a common 1 PPS and an absolute timebase, so "the pass starts at second T0"
  is a statement both machines can act on to within tens of nanoseconds.

The short-dwell physical-target path does not need cross-radio lock: carrier
offset is common to the direct path and echo and cancels in excess Doppler.
The commanded emulated-target path and long coherent dwells still require a
shared reference. Both sides print their lock state and the receiver records it
in the capture sidecar.
"""

from __future__ import annotations

import math
import time


def lock_gpsdo(usrp, timeout_s=120.0, require=True, quiet=False):
    """Switch to the GPSDO and wait for it. Returns a dict of the lock state.

    Two locks matter and they are not the same. `ref_locked` says the 10 MHz
    has pulled in, which is what fixes eps. `gps_locked` says the receiver has
    a position and therefore valid time, which is what makes `gps_time` usable
    as a shared epoch. The reference can lock without the GPS having a fix, and
    a setup that checks only the first will happily agree on the wrong second.
    """
    info = dict(requested="gpsdo", ref_locked=False, gps_locked=False,
                gps_time=None, sensors=[])
    try:
        info["sensors"] = list(usrp.get_mboard_sensor_names(0))
    except Exception as exc:                           # noqa: BLE001
        info["error"] = str(exc)
        if require:
            raise
        return info

    if "gps_locked" not in info["sensors"]:
        info["error"] = "no GPSDO on this radio"
        if require:
            raise RuntimeError(
                "GPSDO requested but this radio reports no gps_locked sensor; "
                "run with --clock internal to accept a free-running reference")
        return info

    usrp.set_clock_source("gpsdo", 0)
    usrp.set_time_source("gpsdo", 0)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            info["ref_locked"] = bool(
                usrp.get_mboard_sensor("ref_locked", 0).to_bool())
        except Exception:                              # noqa: BLE001
            info["ref_locked"] = False
        try:
            info["gps_locked"] = bool(
                usrp.get_mboard_sensor("gps_locked", 0).to_bool())
        except Exception:                              # noqa: BLE001
            info["gps_locked"] = False
        if info["ref_locked"] and info["gps_locked"]:
            break
        if not quiet:
            print(f"    waiting for GPSDO: ref_locked={info['ref_locked']} "
                  f"gps_locked={info['gps_locked']} "
                  f"({time.time()-t0:.0f}/{timeout_s:.0f} s)")
        time.sleep(2.0)

    if info["gps_locked"]:
        try:
            info["gps_time"] = int(
                usrp.get_mboard_sensor("gps_time", 0).to_int())
        except Exception:                              # noqa: BLE001
            pass
        # Set the device clock on the next edge, then let that edge pass
        # before anything reads the time back. Setting it and immediately
        # scheduling against it is the classic way to be one second out.
        from gnuradio import uhd
        usrp.set_time_next_pps(uhd.time_spec(float(info["gps_time"] + 1)))
        time.sleep(1.1)
    elif require:
        raise RuntimeError(
            f"GPSDO did not lock within {timeout_s:.0f} s "
            f"(ref_locked={info['ref_locked']}). Check the antenna has sky "
            f"view, or run with --clock internal.")
    return info


def lock_external(usrp, timeout_s=30.0, require=True, quiet=False):
    """Discipline the radio from a cabled 10 MHz and 1 PPS. Returns the state.

    This is the mode the B210 pair has, since neither board carries a GPSDO.
    Both radios take the same 10 MHz and the same PPS - from an OctoClock, a
    GPSDO with a split output, or any distribution amplifier - and eps between
    them drops to the reference's own stability.

    The PPS gives a common *edge* but not a common *number*, so both machines
    have to independently choose the same value to stamp on it. They do that
    from the host clock: the next PPS edge is labelled `floor(host time) + 1`.

    **That makes the two host clocks a dependency.** They only have to agree to
    better than +-0.5 s for both to pick the same integer second, which NTP
    beats by three orders of magnitude - but a laptop with NTP disabled, or one
    that has just woken from sleep, will silently be a whole second out, and a
    second of trajectory time is 7.5 km of commanded range. The chosen value is
    printed on both sides so it can be eyeballed; check it rather than assuming.
    """
    from gnuradio import uhd

    info = dict(requested="external", ref_locked=False, gps_locked=False,
                gps_time=None, pps_time=None, sensors=[])
    try:
        info["sensors"] = list(usrp.get_mboard_sensor_names(0))
        usrp.set_clock_source("external", 0)
        usrp.set_time_source("external", 0)
    except Exception as exc:                           # noqa: BLE001
        info["error"] = str(exc)
        if require:
            raise
        return info

    # ref_locked is the only lock that exists here. There is no GPS receiver in
    # this path, so gps_locked stays False and `disciplined` is what the rest of
    # the chain should test instead.
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            info["ref_locked"] = bool(
                usrp.get_mboard_sensor("ref_locked", 0).to_bool())
        except Exception:                              # noqa: BLE001
            info["ref_locked"] = False
        if info["ref_locked"]:
            break
        if not quiet:
            print(f"    waiting for the external 10 MHz: ref_locked=False "
                  f"({time.time()-t0:.0f}/{timeout_s:.0f} s)")
        time.sleep(1.0)

    if not info["ref_locked"]:
        msg = (f"external reference did not lock within {timeout_s:.0f} s. "
               f"Check the 10 MHz is present at REF IN and is 0 to +15 dBm, "
               f"or run with --clock internal and --no-emulated.")
        info["error"] = msg
        if require:
            raise RuntimeError(msg)
        return info

    # Stamp the next PPS edge with a number both machines compute alike. Then
    # let the edge pass before anything reads the time back - setting it and
    # immediately scheduling against it is the classic way to be one second out.
    info["pps_time"] = float(math.floor(time.time()) + 1)
    usrp.set_time_next_pps(uhd.time_spec(info["pps_time"]))
    time.sleep(1.1)
    if not quiet:
        print(f"    external reference locked, PPS edge stamped "
              f"{info['pps_time']:.0f} (host clock; must match the other "
              f"laptop's)")
    return info


def set_internal(usrp):
    usrp.set_clock_source("internal", 0)
    usrp.set_time_source("internal", 0)
    from gnuradio import uhd
    usrp.set_time_now(uhd.time_spec(0.0))
    return dict(requested="internal", ref_locked=False, gps_locked=False,
                gps_time=None)


def configure(usrp, mode, timeout_s=120.0, quiet=False):
    """Apply one of the three clock modes. `try` prefers a GPSDO and falls back.

    One dispatcher rather than a ternary in each driver, because there are now
    three modes and the fallback rule differs per mode.
    """
    if mode == "internal":
        return set_internal(usrp)
    if mode == "external":
        return lock_external(usrp, timeout_s, require=True, quiet=quiet)
    return lock_gpsdo(usrp, timeout_s, require=(mode == "gpsdo"), quiet=quiet)


def disciplined(info):
    """True when this radio shares a reference with the other one.

    This is the flag `ota_sense` wants: it says "eps is negligible, do not
    estimate it", and it is equally true of a locked GPSDO and a cabled 10 MHz.
    Testing `gps_locked` alone would silently put the external mode back into
    free-running estimation.
    """
    if info.get("requested") == "external":
        return bool(info.get("ref_locked"))
    return bool(info.get("gps_locked"))


def next_epoch(usrp, lead_s=5.0, align_s=10.0, gps_time=None):
    """Choose a shared start time: the next multiple of `align_s`, at least
    `lead_s` away.

    Both laptops run this and must get the same answer, so it is a function of
    absolute time only - no local state, no negotiation. `align_s` being a
    round number is what lets the operator start the two programs by hand a few
    seconds apart and still have them agree.
    """
    if gps_time is not None:
        now = float(gps_time)
    else:
        try:
            now = float(usrp.get_time_now().get_real_secs())
        except Exception:                              # noqa: BLE001
            now = time.time()
    import math
    return float(math.ceil((now + lead_s) / align_s) * align_s)


def describe(info, epoch, t_start):
    if info.get("requested") == "external":
        lock = ("external 10 MHz/PPS locked" if info.get("ref_locked")
                else "external reference NOT locked")
    else:
        lock = ("GPSDO locked" if info.get("gps_locked") else
                ("reference locked, no GPS fix" if info.get("ref_locked")
                 else "free running"))
    stamp = ""
    if info.get("gps_time"):
        stamp = f", gps_time {info['gps_time']}"
    elif info.get("pps_time"):
        stamp = f", pps_time {info['pps_time']:.0f}"
    return (f"    clock {info.get('requested')}: {lock}{stamp}"
            + f"\n    epoch {epoch:.1f} = pass t={t_start:.1f} s"
            + ("" if disciplined(info) else
               "  [free running: --epoch is device time, so the emulated "
               "target needs --no-emulated]"))
