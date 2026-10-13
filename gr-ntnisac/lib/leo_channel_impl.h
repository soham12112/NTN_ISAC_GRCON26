/* -*- c++ -*- */
/*
 * Copyright 2026 NTN ISAC prototype.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_NTNISAC_LEO_CHANNEL_IMPL_H
#define INCLUDED_NTNISAC_LEO_CHANNEL_IMPL_H

#include <gnuradio/ntnisac/leo_channel.h>
#include <atomic>
#include <mutex>
#include <vector>

namespace gr {
namespace ntnisac {

/*!
 * Not-a-knot cubic spline on a uniformly sampled table.
 *
 * Matches scipy.interpolate.CubicSpline's default end condition, so the C++
 * block and the Python reference implementation agree; a different end
 * condition would make live and offline runs of the same pass disagree near
 * t = 0, which is exactly where the range rate is largest.
 */
class cubic_spline
{
public:
    void build(double x0, double dx, const std::vector<double>& y);
    double x0() const { return x0_; }
    double inv_dx() const { return inv_dx_; }
    double eval(double x) const;
    double deriv(double x) const;
    bool empty() const { return y_.empty(); }

private:
    inline size_t seg(double x, double& t) const;
    double x0_ = 0.0, dx_ = 1.0, inv_dx_ = 1.0;
    std::vector<double> y_, b_, c_, d_;
};

/*!
 * Polyphase fractional-delay interpolator.
 *
 * n_taps == 4 is the cubic Farrow of handoff section 6. Even n_taps >= 8 is a
 * Kaiser-windowed sinc bank, whose gain is flat across the fractional phase mu
 * where the Farrow's is not: cubic Lagrange droops 1.96 dB at this waveform's
 * band edge at mu = 0.5, and since the delay walks ~170 samples/s during a
 * pass, mu sweeps a sawtooth and that droop becomes amplitude modulation.
 * Measured cost of the Farrow on a weak target is only 0.8 dB, so 4 taps
 * remains a legitimate choice when throughput is tight.
 */
class frac_delay
{
public:
    void build(int n_taps, int n_phase = 1024, double cutoff = 0.45,
               double beta = 8.0);
    int n_taps() const { return n_taps_; }
    int k0() const { return k0_; }
    inline const float* taps(double mu) const;

private:
    int n_taps_ = 0, n_phase_ = 0, k0_ = 0;
    bool lagrange_ = false;
    std::vector<float> bank_;              // [n_phase][n_taps]
    mutable std::vector<float> scratch_;   // cubic Farrow evaluates per sample
};

struct path_state {
    cubic_spline range;                    // metres
    std::vector<double> g_db;              // piecewise constant, table order
    std::vector<double> g_amp;             // 10^(g/20), precomputed: std::pow
                                           // in the per-sample loop cost more
                                           // than the interpolator did
    double r_min = 0.0, r_max = 0.0;
};

class leo_channel_impl : public leo_channel
{
public:
    leo_channel_impl(const std::string& traj_csv, double fs, double fc,
                     double r_nominal, double tau_bias_samples, int delay_line,
                     int interp_taps, double t_start);
    ~leo_channel_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

    int num_paths() const override { return (int)paths_.size(); }
    void set_enable(int p, bool on) override;
    bool enabled(int p) const override;
    void set_gain_db(int p, double g) override;
    void clear_gain_override(int p) override;
    void set_time(double t) override;
    double t_now() const override;
    double tap_delay_samples(int p) const override;
    double tap_doppler_hz(int p) const override;
    double tap_excess_m(int p) const override;
    double tap_gain_db(int p) const override;
    double r_nominal() const override { return r_nominal_; }
    double tau_bias_samples() const override { return tau_bias_ * fs_; }
    double max_delay_samples() const override { return max_delay_; }
    double duration() const override { return t_lo_ + (n_rows_ - 1) * dt_tab_; }

private:
    void load_csv(const std::string& path);
    void process(const gr_complex* in, gr_complex* out, int n);
    inline double gain_amp(int p, double t) const;

    double fs_, fc_, r_nominal_, tau_bias_, t_start_;
    size_t mask_;
    size_t ring_len_;   // logical length; ring_ is longer by a mirror + slack
    int chunk_;
    std::vector<gr_complex> ring_;
    int64_t n_abs_ = 0;
    double max_delay_ = 0.0;

    std::vector<path_state> paths_;
    std::vector<uint8_t> enable_;
    std::vector<double> gain_override_;    // NaN == use the table
    double t_lo_ = 0.0, dt_tab_ = 0.0, inv_dt_tab_ = 0.0, inv_fs_ = 0.0;
    size_t n_rows_ = 0;

    frac_delay frac_;
    mutable std::mutex ctl_;

    // Control snapshots, reused so process() does not allocate.
    std::vector<uint8_t> en_snap_;
    std::vector<double> ov_snap_;
    size_t mirror_;   // logical indices below this are duplicated past the end
};

} // namespace ntnisac
} // namespace gr

#endif /* INCLUDED_NTNISAC_LEO_CHANNEL_IMPL_H */
