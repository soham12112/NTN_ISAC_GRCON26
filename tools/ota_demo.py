#!/usr/bin/env python3
"""The whole over-the-air story, on one laptop, with no radio and no RF.

    . ./env.sh
    $NTN_PY tools/ota_demo.py

Everything downstream of the antenna connector, driven by `otasim.simulate`
instead of a USRP: the emulated LEO pass, a second laptop's clock offset, and a
rotating object sitting in the same delay cell as the direct path. It is the
same code the live flowgraph runs - `ota.OtaTracker`, `clean.clean`,
`udoppler.analyse` - so a number that comes out here is a number the receive
path produces, not a separate model of it.

What it is for is the one thing a delay-Doppler map cannot show on its own:
that over the air the direct path, not the noise, is what a real target hides
behind. Run it with `--blade-gain-db -40` and the target is invisible in the
uncancelled map and obvious in the cancelled one, at the same colour scale.

It writes a figure. Panels, left to right:

  1  the map as the conducted chain would produce it. The direct path and its
     structure dominate everything within tens of hertz of zero Doppler.
  2  the same dwell after CLEAN, on the SAME colour scale. If the scale moved
     between the two, the comparison would be meaningless.
  3  the Doppler profile at the direct path's delay, before and after, with the
     recovered rotation harmonics marked.

`--eps-ppm` is worth trying at 0 and at 2.5. It is the fractional difference
between two free-running B200 references, and it changes nothing about the
physical target - a carrier offset is common to the direct path and the echo
and cancels in the excess Doppler - while ruining the emulated one, which is
tracked from ephemeris with no clock estimator. That asymmetry is the argument
for putting GPS antennas on both radios.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from ntn_isac import clean, ota, otasim, receiver, sensing, udoppler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wf", default=os.path.join(ROOT, "data", "wf_ota"))
    p.add_argument("--traj", default=os.path.join(ROOT, "data", "traj_ota"))
    p.add_argument("--t-start", type=float, default=300.0, dest="t_start")
    p.add_argument("--seconds", type=float, default=1.3)
    p.add_argument("--n-frames", type=int, default=5, dest="n_frames",
                   help="frames per sensing dwell; 5 = 100 ms")
    p.add_argument("--blade-gain-db", type=float, default=-30.0,
                   dest="blade_gain_db",
                   help="echo-to-direct ratio of the rotating object. -999 "
                        "runs target-absent, which is the control")
    p.add_argument("--blade-radius-m", type=float, default=0.25,
                   dest="blade_radius_m")
    p.add_argument("--rot-hz", type=float, default=7.0, dest="rot_hz",
                   help="revolutions per second. Avoid multiples and divisors "
                        "of 25: that is where the aperture comb lives")
    p.add_argument("--eps-ppm", type=float, default=0.0, dest="eps_ppm",
                   help="clock difference between the two radios, ppm. "
                        "0 is GPS-disciplined, 2.5 is free-running")
    p.add_argument("--esn0-db", type=float, default=45.0, dest="esn0_db")
    p.add_argument("--window", default="blackmanharris",
                   choices=("hann", "blackmanharris"))
    p.add_argument("--emulated", action="store_true",
                   help="leave the emulated clutter and drone switched on. Off "
                        "by default: at t=300 the drone is two delay cells "
                        "from the direct path with ~20 Hz of residual Doppler, "
                        "which is the same cell and band the blade occupies")
    p.add_argument("--out", default=os.path.join(ROOT, "data", "ota_demo.png"))
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    blade = None
    if a.blade_gain_db > -900:
        blade = dict(excess_m=0.4, radius_m=a.blade_radius_m,
                     rot_hz=a.rot_hz, gain_db=a.blade_gain_db)

    print("=" * 74)
    print("NTN ISAC over-the-air chain, simulated end to end. No radio.")
    print("=" * 74)

    t0 = time.time()
    y, b, tr, cfg, truth = otasim.simulate(
        a.wf, a.traj, seconds=a.seconds, t_start=a.t_start,
        eps_ppm=a.eps_ppm, blade=blade, esn0_db=a.esn0_db,
        disable=() if a.emulated else (1, 2))
    fs, fc = b.fs, b.fc
    lam = 299792458.0 / fc
    n_frame = int(cfg["frame_len_samples"])
    n_dwell = a.n_frames * n_frame
    cell_m = 299792458.0 / (float(cfg["n_sc"]) * float(cfg["scs_khz"]) * 1e3)

    print(f"  carrier {fc/1e6:.3f} MHz, lambda {lam*100:.2f} cm, "
          f"{fs/1e6:.2f} MS/s, delay cell {cell_m:.1f} m")
    print(f"  pass t={a.t_start:.0f} s, Es/N0 {a.esn0_db:.0f} dB, "
          f"clock {a.eps_ppm:+.2f} ppm ({a.eps_ppm*1e-6*fc:+.0f} Hz)")
    if blade:
        edge = udoppler.predicted_extent_hz(fc, blade["radius_m"],
                                            blade["rot_hz"])
        print(f"  rotating object: r={blade['radius_m']} m at "
              f"{blade['rot_hz']:.1f} rev/s ({blade['rot_hz']*60:.0f} RPM), "
              f"echo {blade['gain_db']:.0f} dB")
        print(f"     excess path {blade['excess_m']} m = "
              f"{blade['excess_m']/cell_m:.4f} delay cells -> NOT resolvable "
              f"in delay")
        print(f"     predicted Doppler edge +-{edge:.1f} Hz, "
              f"lines every {blade['rot_hz']:.1f} Hz")
    else:
        print("  no physical object: this is the target-absent control")
    print(f"  emulated clutter and drone: "
          f"{'on' if a.emulated else 'off'}")
    print()

    # ---- the receive chain, exactly as ota_sense runs it --------------------
    trk = ota.OtaTracker(tr, fs, tau_bias_samples=truth["bias_samples"],
                         gps_locked=(a.eps_ppm == 0.0))
    t_acq = time.time()
    tt = a.t_start + np.arange(n_frame) / fs
    yd = receiver.deramp(y[:n_frame], tt, trk.tau_leo(tt), fc)
    acq = ota.acquire(yd, b.iq(), cfg, cfo_max_hz=12e3)
    pred = float(trk.tau0(np.array([a.t_start]))[0]) * fs
    err = (acq.lag_samples - pred + n_frame / 2.0) % n_frame - n_frame / 2.0
    trk.t_ref, trk.d0 = a.t_start, err / fs
    trk.eps = 0.0 if trk.gps_locked else acq.cfo_hz / fc
    print(f"[1] acquisition: lag {acq.lag_samples}, peak "
          f"{acq.peak_db:.1f} dB over the floor, carrier "
          f"{acq.cfo_hz:+.0f} Hz  [{1e3*(time.time()-t_acq):.0f} ms]")
    print(f"    seeded d0 {err:+.1f} samples, eps {trk.eps*1e6:+.3f} ppm")

    margin = int(0.006 * fs) + int(truth["bias_samples"])
    n_dw = int((y.size - margin) // n_dwell)
    rows, Zc_all, keep = [], [], None
    m_before = m_after = None

    for i in range(n_dw):
        t = a.t_start + i * n_dwell / fs
        seg = y[i * n_dwell:i * n_dwell + n_dwell + margin]
        if seg.size < n_dwell + margin:
            break
        Y, H, tb, st = otasim.demod_dwell(seg, b, trk, t, a.n_frames)
        d, f, _, _ = ota.residual_from_H(H, tb.occ_mask, cfg)
        trk.update(d, f, t)

        X, W = sensing.build_variant(Y, tb, sensing.PAYLOAD, H=H)
        Z = sensing.form_z(Y, X, W)
        w = float(W.sum())
        before = sensing.map_from_z(Z, w, cfg, 2, a.window)
        Zk, comps, maps = clean.clean(
            Z, W, cfg, n_components=2, pad=2, window=a.window,
            search=(before.peak_delay_s - 200e-9,
                    before.peak_delay_s + 200e-9),
            search_doppler=(before.peak_doppler_hz - 15.0,
                            before.peak_doppler_hz + 15.0))
        can = clean.cancellation_db(maps["before"], maps["after"],
                                    before.peak_delay_s,
                                    before.peak_doppler_hz)
        rows.append((t, d, f, trk.eps * 1e6, before.pk_nf_db,
                     can["cancelled_db"]))
        Zc_all.append(Zk)
        if m_before is None:
            m_before, m_after, w_sum = maps["before"], maps["after"], w

    print()
    print("[2] per dwell")
    print("      t        resid delay   resid fd     eps      direct   cancelled")
    for (t, d, f, e, pk, cdb) in rows:
        print(f"    {t:8.3f}   {d:+8.3f} sa  {f:+8.2f} Hz  {e:+7.3f} ppm  "
              f"{pk:6.1f} dB  {cdb:6.1f} dB")

    # ---- the long dwell, which is what resolves the rotation rate ----------
    Zc = np.concatenate(Zc_all, axis=0)
    dwell_s = Zc.shape[0] / int(cfg["n_sym_total"]) * n_frame / fs
    res = sensing.map_from_z(Zc, w_sum, cfg, 2, a.window)
    md = udoppler.analyse(res, cfg, res.delay_s[0], fc=fc,
                          rot_search_hz=(3.0, 40.0))
    print()
    print(f"[3] micro-Doppler over {dwell_s*1e3:.0f} ms "
          f"({Zc.shape[0]//int(cfg['n_sym_total'])} frames, Doppler cell "
          f"{res.doppler_hz[1]-res.doppler_hz[0]:.2f} Hz)")
    rot = ("unresolved" if not np.isfinite(md.rotation_hz)
           else f"{md.rotation_hz:.2f} rev/s = {md.rotation_hz*60:.0f} RPM")
    print(f"    rotation rate   {rot}")
    print(f"    Doppler edge    +-{md.extent_hz:.1f} Hz  "
          f"-> tip speed {md.tip_speed_mps:.2f} m/s")
    print(f"    implied radius  {md.radius_m*100:.1f} cm")
    print(f"    strongest line  {md.peak_hz:+.1f} Hz at "
          f"{md.peak_snr_db:.1f} dB over noise, {md.n_lines} lines")
    if blade:
        print(f"    truth           {blade['rot_hz']:.2f} rev/s, "
              f"{blade['radius_m']*100:.0f} cm, edge "
              f"+-{udoppler.predicted_extent_hz(fc, blade['radius_m'], blade['rot_hz']):.1f} Hz")
    print(f"\n  {time.time()-t0:.1f} s wall")

    if not a.no_plot:
        plot(m_before, m_after, res, md, cfg, blade, a, fc)


def plot(m_before, m_after, res_long, md, cfg, blade, a, fc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fd_lim = 260.0
    cell_m = 299792458.0 / (float(cfg["n_sc"]) * float(cfg["scs_khz"]) * 1e3)

    def panel(k, res, title):
        M, fd, ds = res.M, res.doppler_hz, res.delay_s
        j = int(np.argmin(np.abs(ds - m_before.peak_delay_s)))
        jj = slice(max(j - 12, 0), j + 13)
        ii = np.abs(fd) <= fd_lim
        db = 20 * np.log10(np.maximum(M[np.ix_(ii, range(jj.start, jj.stop))],
                                      1e-30) / max(m_before.noise, 1e-30))
        ext = [(jj.start - j) * (ds[1] - ds[0]) * 299792458.0,
               (jj.stop - j) * (ds[1] - ds[0]) * 299792458.0,
               fd[ii][0], fd[ii][-1]]
        im = ax[k].imshow(db, aspect="auto", origin="lower", extent=ext,
                          vmin=0, vmax=float(np.nanmax(
                              20 * np.log10(np.maximum(m_before.M, 1e-30)
                                            / max(m_before.noise, 1e-30)))),
                          cmap="viridis")
        ax[k].set_title(title, fontsize=10)
        ax[k].set_xlabel("excess path, m")
        ax[k].set_ylabel("Doppler, Hz")
        if blade:
            e = udoppler.predicted_extent_hz(fc, blade["radius_m"],
                                             blade["rot_hz"])
            for s in (-1, 1):
                ax[k].axhline(s * e, color="w", ls="--", lw=0.8, alpha=0.7)
            ax[k].text(ext[0] * 0.95, e + 8, "predicted tip-speed edge",
                       color="w", fontsize=7)
        plt.colorbar(im, ax=ax[k], label="dB over noise")

    panel(0, m_before, "1. Uncancelled, one 100 ms dwell\n"
                       "the direct path and its structure")
    panel(1, m_after, "2. After CLEAN, same dwell, same colour scale\n"
                      "what is left is the target")

    fd, lvl = udoppler.doppler_profile(res_long, res_long.delay_s[0])
    db = 20 * np.log10(np.maximum(lvl, 1e-30) / max(m_before.noise, 1e-30))
    fb, lb = udoppler.doppler_profile(m_before, m_before.peak_delay_s)
    dbb = 20 * np.log10(np.maximum(lb, 1e-30) / max(m_before.noise, 1e-30))
    sel = np.abs(fd) <= fd_lim
    selb = np.abs(fb) <= fd_lim
    ax[2].plot(fb[selb], dbb[selb], lw=0.9, color="0.6",
               label="uncancelled, 100 ms")
    ax[2].plot(fd[sel], db[sel], lw=1.0, color="C0",
               label=f"cancelled, {res_long.M.shape[0]//2//14*20/1000:.1f} s")
    if np.isfinite(md.rotation_hz):
        n = int(fd_lim / md.rotation_hz)
        for k in range(1, n + 1):
            for s in (-1, 1):
                ax[2].axvline(s * k * md.rotation_hz, color="C1", lw=0.5,
                              alpha=0.35)
        ax[2].plot([], [], color="C1", lw=0.5,
                   label=f"harmonics of {md.rotation_hz:.2f} rev/s")
    if blade:
        e = udoppler.predicted_extent_hz(fc, blade["radius_m"],
                                         blade["rot_hz"])
        for s in (-1, 1):
            ax[2].axvline(s * e, color="k", ls="--", lw=0.9)
        ax[2].plot([], [], color="k", ls="--", lw=0.9,
                   label=f"tip-speed edge, {e:.0f} Hz")
    ax[2].set_xlim(-fd_lim, fd_lim)
    ax[2].set_xlabel("Doppler, Hz")
    ax[2].set_ylabel("dB over noise")
    ax[2].set_title("3. Doppler at the direct path's delay\n"
                    "the target is here or nowhere", fontsize=10)
    ax[2].legend(fontsize=7, loc="upper right")
    ax[2].grid(alpha=0.25)

    sub = (f"{fc/1e6:.0f} MHz, delay cell {cell_m:.0f} m")
    if blade:
        sub += (f"  |  object at {blade['excess_m']} m excess = "
                f"{blade['excess_m']/cell_m:.3f} cells, echo "
                f"{blade['gain_db']:.0f} dB, {blade['rot_hz']:.0f} rev/s")
    else:
        sub += "  |  target absent (control)"
    sub += f"  |  clock {a.eps_ppm:+.1f} ppm"
    fig.suptitle(sub, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"  figure -> {a.out}")


if __name__ == "__main__":
    main()
