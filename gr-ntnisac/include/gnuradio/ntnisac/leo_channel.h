/* -*- c++ -*- */
/*
 * Copyright 2026 NTN ISAC prototype.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_NTNISAC_LEO_CHANNEL_H
#define INCLUDED_NTNISAC_LEO_CHANNEL_H

#include <gnuradio/ntnisac/api.h>
#include <gnuradio/sync_block.h>
#include <string>

namespace gr {
namespace ntnisac {

/*!
 * \brief Time-varying multipath channel for an emulated LEO pass.
 * \ingroup ntnisac
 *
 * Per path, per sample:
 *
 *     y_p(t) = a_p(t) * x(t - tau_p(t)) * exp(-j*2*pi*f_c*tau_p(t))
 *
 * summed over paths. Delay is the only thing injected. Doppler is not a
 * parameter: it falls out of the derivative of the carrier phase, which is
 * exactly the Doppler implied by the delay rate. Injecting the two
 * independently makes them disagree, and a receiver that jointly tracks delay
 * and Doppler then sees physically impossible data.
 *
 * The sample counter is absolute and never resets. The waveform file loops
 * every 20 ms; the pass does not.
 *
 * Ranges come from traj.csv, whose header is
 * t_s,R0_m,G0_db,R1_m,G1_db,... The number of paths is read from the header.
 * G is a plain echo-to-direct power ratio in dB, not a radar cross section,
 * and is held piecewise constant so that deliberate on/off edges (the
 * target-absent epochs) are not smeared by interpolation.
 */
class NTNISAC_API leo_channel : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<leo_channel> sptr;

    /*!
     * \param traj_csv          path to traj.csv
     * \param fs                sample rate, Hz
     * \param fc                carrier frequency, Hz
     * \param r_nominal         metres subtracted from every range before
     *                          forming tau. Negative means "use the minimum
     *                          range over the whole table", which is the
     *                          choice that keeps tau >= 0.
     * \param tau_bias_samples  constant delay keeping the interpolator's read
     *                          pointer causally behind the write pointer. The
     *                          receiver must use the same value.
     * \param delay_line        ring buffer length, must be a power of two
     * \param interp_taps       fractional-delay filter length; 4 selects the
     *                          cubic Farrow, even values >= 8 select a
     *                          Kaiser-windowed sinc bank
     * \param t_start           trajectory time of the first sample, seconds
     */
    static sptr make(const std::string& traj_csv,
                     double fs,
                     double fc,
                     double r_nominal = -1.0,
                     double tau_bias_samples = 16.0,
                     int delay_line = 131072,
                     int interp_taps = 8,
                     double t_start = 0.0);

    virtual int num_paths() const = 0;

    //! Switch a path off, for the target-absent runs that give false alarm rates.
    virtual void set_enable(int path, bool on) = 0;
    virtual bool enabled(int path) const = 0;

    //! Override the trajectory's G column for a path (echo-to-direct, dB).
    virtual void set_gain_db(int path, double gain_db) = 0;
    virtual void clear_gain_override(int path) = 0;

    //! Jump to a chosen point in the pass. Does not rescale time: compressing
    //! the trajectory makes the Doppler rate unphysical and invalidates the
    //! NTN claim, so there is deliberately no rate control.
    virtual void set_time(double t_s) = 0;
    virtual double t_now() const = 0;

    //! Commanded truth at the current time, for the dashboard and for scoring.
    virtual double tap_delay_samples(int path) const = 0;
    virtual double tap_doppler_hz(int path) const = 0;
    virtual double tap_excess_m(int path) const = 0;
    virtual double tap_gain_db(int path) const = 0;

    virtual double r_nominal() const = 0;
    virtual double tau_bias_samples() const = 0;
    virtual double max_delay_samples() const = 0;
    virtual double duration() const = 0;
};

} // namespace ntnisac
} // namespace gr

#endif /* INCLUDED_NTNISAC_LEO_CHANNEL_H */
