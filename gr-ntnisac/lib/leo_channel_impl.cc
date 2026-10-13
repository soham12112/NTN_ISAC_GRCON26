/* -*- c++ -*- */
/*
 * Copyright 2026 NTN ISAC prototype.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "leo_channel_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/logger.h>

#include <algorithm>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace ntnisac {

static constexpr double C_LIGHT = 299792458.0;
static constexpr double INV_C = 1.0 / 299792458.0;

// Samples between exact re-evaluations of tau. Inside a block tau is linear
// and the carrier phasor is a fixed rotation. Range acceleration peaks near
// 100 m/s^2 on this pass, so tau'' ~ 3.2e-7 s/s^2 and the linear-interpolation
// error over 64 samples is ~3e-18 s: 2e-11 samples of delay and 3.5e-8 rad of
// phase, both far below the float32 floor of the sample path itself.
static constexpr int PHASE_BLOCK = 64;
static constexpr double TWO_PI = 6.283185307179586476925286766559;

// ---------------------------------------------------------------- spline ---

void cubic_spline::build(double x0, double dx, const std::vector<double>& y)
{
    const size_t n = y.size();
    if (n < 2)
        throw std::runtime_error("cubic_spline: need at least two samples");
    x0_ = x0;
    dx_ = dx;
    inv_dx_ = 1.0 / dx;
    y_ = y;
    b_.assign(n, 0.0);
    c_.assign(n, 0.0);
    d_.assign(n, 0.0);

    const double h = dx, h2 = h * h;

    if (n < 4) { // too short for not-a-knot; straight lines
        for (size_t i = 0; i + 1 < n; ++i)
            b_[i] = (y_[i + 1] - y_[i]) / h;
        b_[n - 1] = b_[n - 2];
        return;
    }

    // sigma[i] = S''(x_i). Interior: sigma[i-1] + 4 sigma[i] + sigma[i+1] = r[i]
    // with r[i] = 6 (y[i-1] - 2 y[i] + y[i+1]) / h^2.
    //
    // Not-a-knot forces the third derivative to be continuous at x_1 and
    // x_{n-2}, i.e. sigma[0] - 2 sigma[1] + sigma[2] = 0 and likewise at the
    // far end. On a uniform grid, substituting the first of those into the
    // i = 1 equation collapses it to 6 sigma[1] = r[1], so sigma[1] and
    // sigma[n-2] are known outright and the remaining system is tridiagonal.
    std::vector<double> sig(n, 0.0);
    auto r = [&](size_t i) {
        return 6.0 * (y_[i - 1] - 2.0 * y_[i] + y_[i + 1]) / h2;
    };
    sig[1] = r(1) / 6.0;
    sig[n - 2] = r(n - 2) / 6.0;

    const size_t m = (n >= 6) ? n - 4 : 0; // unknowns sigma[2..n-3]
    if (m > 0) {
        std::vector<double> cp(m), dp(m);
        // Thomas algorithm, sub/super diagonals are 1, diagonal is 4.
        for (size_t k = 0; k < m; ++k) {
            const size_t i = k + 2;
            double rhs = r(i);
            if (k == 0)
                rhs -= sig[1];
            if (k == m - 1)
                rhs -= sig[n - 2];
            const double a = (k == 0) ? 0.0 : 1.0;
            const double den = 4.0 - a * (k == 0 ? 0.0 : cp[k - 1]);
            cp[k] = 1.0 / den;
            dp[k] = (rhs - a * (k == 0 ? 0.0 : dp[k - 1])) / den;
        }
        sig[m + 1] = dp[m - 1];
        for (size_t k = m - 1; k-- > 0;)
            sig[k + 2] = dp[k] - cp[k] * sig[k + 3];
    }
    sig[0] = 2.0 * sig[1] - sig[2];
    sig[n - 1] = 2.0 * sig[n - 2] - sig[n - 3];

    for (size_t i = 0; i + 1 < n; ++i) {
        b_[i] = (y_[i + 1] - y_[i]) / h - h * (2.0 * sig[i] + sig[i + 1]) / 6.0;
        c_[i] = 0.5 * sig[i];
        d_[i] = (sig[i + 1] - sig[i]) / (6.0 * h);
    }
    b_[n - 1] = b_[n - 2] + 2.0 * c_[n - 2] * h + 3.0 * d_[n - 2] * h * h;
    c_[n - 1] = c_[n - 2] + 3.0 * d_[n - 2] * h;
    d_[n - 1] = d_[n - 2];
}

inline size_t cubic_spline::seg(double x, double& t) const
{
    double f = (x - x0_) * inv_dx_;
    int64_t i = (int64_t)std::floor(f);
    const int64_t last = (int64_t)y_.size() - 2;
    if (i < 0)
        i = 0;
    if (i > last)
        i = last;              // extrapolate off the ends with the edge cubic
    t = x - (x0_ + (double)i * dx_);
    return (size_t)i;
}

double cubic_spline::eval(double x) const
{
    double t;
    const size_t i = seg(x, t);
    return y_[i] + t * (b_[i] + t * (c_[i] + t * d_[i]));
}

double cubic_spline::deriv(double x) const
{
    double t;
    const size_t i = seg(x, t);
    return b_[i] + t * (2.0 * c_[i] + t * 3.0 * d_[i]);
}

// ------------------------------------------------------------ frac_delay ---

static double bessel_i0(double x)
{
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 60; ++k) {
        t *= (x / (2.0 * k)) * (x / (2.0 * k));
        s += t;
        if (t < 1e-18 * s)
            break;
    }
    return s;
}

static inline double sinc_pi(double x)
{
    if (std::fabs(x) < 1e-12)
        return 1.0;
    const double a = M_PI * x;
    return std::sin(a) / a;
}

void frac_delay::build(int n_taps, int n_phase, double cutoff, double beta)
{
    // VOLK's vector kernels read past the end of the taps pointer, so every
    // tap buffer carries slack. Without it the last polyphase branch - the
    // one whose taps sit at the very end of the bank - hands VOLK a pointer
    // with nothing behind it. That fires for one sample in n_phase, which
    // makes it look like a layout-dependent heisenbug rather than an overrun.
    static const int PAD = 256;
    if (n_taps == 4) {
        lagrange_ = true;
        n_taps_ = 4;
        k0_ = 1;
        scratch_.assign(4 + PAD, 0.0f);
        return;
    }
    if (n_taps < 8 || (n_taps & 1))
        throw std::runtime_error(
            "leo_channel: interp_taps must be 4 (cubic Farrow) or an even value >= 8");
    lagrange_ = false;
    n_taps_ = n_taps;
    n_phase_ = n_phase;
    k0_ = n_taps / 2 - 1;
    bank_.assign((size_t)n_phase * n_taps + PAD, 0.0f);

    std::vector<double> win(n_taps);
    const double i0b = bessel_i0(beta);
    for (int t = 0; t < n_taps; ++t) {
        const double u = 2.0 * t / (double)(n_taps - 1) - 1.0;
        win[t] = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - u * u))) / i0b;
    }
    for (int p = 0; p < n_phase; ++p) {
        const double mu = (double)p / n_phase;
        double sum = 0.0;
        for (int t = 0; t < n_taps; ++t) {
            const double arg = mu + k0_ - t;
            const double h = 2.0 * cutoff * sinc_pi(2.0 * cutoff * arg) * win[t];
            bank_[(size_t)p * n_taps + t] = (float)h;
            sum += h;
        }
        for (int t = 0; t < n_taps; ++t) // exact DC gain per phase
            bank_[(size_t)p * n_taps + t] /= (float)sum;
    }
}

inline const float* frac_delay::taps(double mu) const
{
    if (lagrange_) {
        const double m1 = mu - 1.0, m2 = mu - 2.0, p1 = mu + 1.0;
        scratch_[0] = (float)(-mu * m1 * m2 / 6.0);
        scratch_[1] = (float)(p1 * m1 * m2 * 0.5);
        scratch_[2] = (float)(-p1 * mu * m2 * 0.5);
        scratch_[3] = (float)(p1 * mu * m1 / 6.0);
        return scratch_.data();
    }
    int p = (int)(mu * n_phase_);
    if (p < 0)
        p = 0;
    if (p >= n_phase_)
        p = n_phase_ - 1;
    return &bank_[(size_t)p * n_taps_];
}

// ---------------------------------------------------------------- block ---

leo_channel::sptr leo_channel::make(const std::string& traj_csv, double fs,
                                    double fc, double r_nominal,
                                    double tau_bias_samples, int delay_line,
                                    int interp_taps, double t_start)
{
    return gnuradio::make_block_sptr<leo_channel_impl>(
        traj_csv, fs, fc, r_nominal, tau_bias_samples, delay_line, interp_taps,
        t_start);
}

leo_channel_impl::leo_channel_impl(const std::string& traj_csv, double fs,
                                   double fc, double r_nominal,
                                   double tau_bias_samples, int delay_line,
                                   int interp_taps, double t_start)
    : gr::sync_block("leo_channel",
                     gr::io_signature::make(1, 1, sizeof(gr_complex)),
                     gr::io_signature::make(1, 1, sizeof(gr_complex))),
      fs_(fs), fc_(fc), tau_bias_(tau_bias_samples / fs), t_start_(t_start)
{
    if (delay_line <= 0 || (delay_line & (delay_line - 1)))
        throw std::runtime_error("leo_channel: delay_line must be a power of two");
    frac_.build(interp_taps);
    load_csv(traj_csv);

    double rmin = std::numeric_limits<double>::infinity();
    double rmax = -std::numeric_limits<double>::infinity();
    for (const auto& p : paths_) {
        rmin = std::min(rmin, p.r_min);
        rmax = std::max(rmax, p.r_max);
    }
    r_nominal_ = (r_nominal < 0.0) ? rmin : r_nominal;

    // Validate the entire table up front. A non-causal or over-long delay is a
    // configuration error, and finding it at construction beats throwing out of
    // work() with the flowgraph already running.
    const double d_lo = (rmin - r_nominal_) / C_LIGHT * fs_ + tau_bias_samples;
    const double d_hi = (rmax - r_nominal_) / C_LIGHT * fs_ + tau_bias_samples;
    if (d_lo < frac_.k0() + 2) {
        std::ostringstream o;
        o << "leo_channel: minimum delay " << d_lo << " samples is not causal for a "
          << frac_.n_taps() << "-tap window; raise tau_bias_samples to at least "
          << (frac_.k0() + 2 - (d_lo - tau_bias_samples)) << " or lower r_nominal";
        throw std::runtime_error(o.str());
    }
    if (d_hi > delay_line - frac_.n_taps()) {
        std::ostringstream o;
        o << "leo_channel: maximum delay " << d_hi << " samples exceeds the "
          << delay_line << "-sample delay line";
        throw std::runtime_error(o.str());
    }

    inv_fs_ = 1.0 / fs_;
    // The ring is allocated longer than its logical length. Samples whose
    // logical index falls in the first `mirror_` slots are written twice, once
    // at the index and once `delay_line` further on, so a tap window that
    // straddles the wrap is still contiguous in memory. That removes a branch
    // from the inner loop, and - the reason it exists - it gives VOLK's
    // vector kernels room to read past the nominal end of the window, which
    // they do for 16 taps and more and which segfaults on a bare allocation.
    mask_ = (size_t)delay_line - 1;
    ring_len_ = (size_t)delay_line;
    mirror_ = (size_t)(4 * frac_.n_taps()) + 512;
    ring_.assign(ring_len_ + 2 * mirror_, gr_complex(0.0f, 0.0f));
    chunk_ = delay_line / 2;
    set_max_noutput_items(chunk_);

    enable_.assign(paths_.size(), 1);
    gain_override_.assign(paths_.size(), std::numeric_limits<double>::quiet_NaN());

    {
        std::ostringstream o;
        o << "leo_channel: " << paths_.size() << " paths, tau " << d_lo << ".."
          << d_hi << " samples, r_nominal " << r_nominal_ << " m, "
          << frac_.n_taps() << "-tap interpolator, " << duration() << " s table";
        GR_LOG_INFO(d_logger, o.str());
    }
}

leo_channel_impl::~leo_channel_impl() {}

void leo_channel_impl::load_csv(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("leo_channel: cannot open " + path);

    std::string line;
    if (!std::getline(f, line))
        throw std::runtime_error("leo_channel: " + path + " is empty");

    std::vector<std::string> hdr;
    {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            hdr.push_back(tok);
    }
    if (hdr.empty() || hdr[0] != "t_s")
        throw std::runtime_error("leo_channel: " + path +
                                 " first column must be t_s");
    if ((hdr.size() - 1) % 2)
        throw std::runtime_error("leo_channel: " + path +
                                 " needs an R,G pair per path");
    const size_t np = (hdr.size() - 1) / 2;
    for (size_t p = 0; p < np; ++p) {
        const std::string wantR = "R" + std::to_string(p) + "_m";
        const std::string wantG = "G" + std::to_string(p) + "_db";
        if (hdr[1 + 2 * p] != wantR || hdr[2 + 2 * p] != wantG)
            throw std::runtime_error("leo_channel: " + path +
                                     " unexpected header for path " +
                                     std::to_string(p));
    }

    std::vector<double> t;
    std::vector<std::vector<double>> R(np), G(np);
    const size_t ncol = hdr.size();
    std::vector<double> row(ncol);
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        const char* s = line.c_str();
        char* end = nullptr;
        for (size_t c = 0; c < ncol; ++c) {
            row[c] = std::strtod(s, &end);
            if (end == s)
                throw std::runtime_error("leo_channel: malformed row in " + path);
            s = (*end == ',') ? end + 1 : end;
        }
        t.push_back(row[0]);
        for (size_t p = 0; p < np; ++p) {
            R[p].push_back(row[1 + 2 * p]);
            G[p].push_back(row[2 + 2 * p]);
        }
    }
    if (t.size() < 2)
        throw std::runtime_error("leo_channel: " + path + " has too few rows");

    // Uniform sampling is part of the trajectory contract (100 Hz); relying on
    // it turns the spline lookup into an O(1) index instead of a binary search.
    t_lo_ = t[0];
    n_rows_ = t.size();
    dt_tab_ = (t.back() - t.front()) / (double)(n_rows_ - 1);
    inv_dt_tab_ = 1.0 / dt_tab_;
    for (size_t i = 1; i < n_rows_; ++i) {
        const double d = t[i] - t[i - 1];
        if (std::fabs(d - dt_tab_) > 1e-6 * dt_tab_)
            throw std::runtime_error("leo_channel: " + path +
                                     " is not uniformly sampled");
    }

    paths_.resize(np);
    for (size_t p = 0; p < np; ++p) {
        paths_[p].range.build(t_lo_, dt_tab_, R[p]);
        paths_[p].g_db = G[p];
        paths_[p].g_amp.resize(G[p].size());
        for (size_t i = 0; i < G[p].size(); ++i)
            paths_[p].g_amp[i] =
                (G[p][i] <= -100.0) ? 0.0 : std::pow(10.0, G[p][i] / 20.0);
        paths_[p].r_min = *std::min_element(R[p].begin(), R[p].end());
        paths_[p].r_max = *std::max_element(R[p].begin(), R[p].end());
    }
}

inline double leo_channel_impl::gain_amp(int p, double t) const
{
    const double ov = gain_override_[p];
    if (!std::isnan(ov))
        return (ov <= -100.0) ? 0.0 : std::pow(10.0, ov / 20.0);
    int64_t i = (int64_t)((t - t_lo_) * inv_dt_tab_);
    if (i < 0)
        i = 0;
    if (i >= (int64_t)n_rows_)
        i = (int64_t)n_rows_ - 1;
    return paths_[p].g_amp[(size_t)i];   // "path absent" is stored as 0
}

void leo_channel_impl::process(const gr_complex* in, gr_complex* out, int n)
{
    const int64_t n0 = n_abs_;
    for (int i = 0; i < n; ++i) {
        const size_t w = (size_t)(n0 + i) & mask_;
        ring_[w] = in[i];
        if (w < mirror_)
            ring_[w + ring_len_] = in[i];
    }
    std::fill(out, out + n, gr_complex(0.0f, 0.0f));

    const int T = frac_.n_taps();
    const int k0 = frac_.k0();
    const double lo_causal = (double)(k0 + 2);
    const double hi_causal = (double)(ring_len_ - T);

    {
        std::lock_guard<std::mutex> lk(ctl_);
        en_snap_ = enable_;
        ov_snap_ = gain_override_;
    }

    for (size_t p = 0; p < paths_.size(); ++p) {
        if (!en_snap_[p])
            continue;
        const cubic_spline& sp = paths_[p].range;
        const double ov = ov_snap_[p];
        const bool have_ov = !std::isnan(ov);
        const double ov_amp =
            have_ov ? ((ov <= -100.0) ? 0.0 : std::pow(10.0, ov / 20.0)) : 0.0;
        if (have_ov && ov_amp == 0.0)
            continue;
        const double* gamp = paths_[p].g_amp.data();

        for (int blk = 0; blk < n; blk += PHASE_BLOCK) {
            const int m = std::min(PHASE_BLOCK, n - blk);

            const double ta = t_start_ + (double)(n0 + blk) * inv_fs_;
            const double tb = t_start_ + (double)(n0 + blk + m) * inv_fs_;

            // Gain is piecewise constant at the table rate (100 Hz), so it is
            // constant across a 64-sample block except at a table edge; taking
            // it at the block start is exact to within one table interval.
            double a;
            if (have_ov) {
                a = ov_amp;
            } else {
                int64_t gi = (int64_t)((ta - t_lo_) * inv_dt_tab_);
                if (gi < 0)
                    gi = 0;
                if (gi >= (int64_t)n_rows_)
                    gi = (int64_t)n_rows_ - 1;
                a = gamp[gi];
                if (a == 0.0)
                    continue;
            }

            const double tau_a = (sp.eval(ta) - r_nominal_) * INV_C + tau_bias_;
            const double tau_b = (sp.eval(tb) - r_nominal_) * INV_C + tau_bias_;
            const double dtau = (tau_b - tau_a) / (double)m;

            double d = tau_a * fs_;
            const double dd = dtau * fs_;

            // Exact phase at the block start, then a fixed rotation. The
            // per-step increment comes from the unwrapped tau difference, not
            // from differencing two reduced phases, so there is nothing to
            // unwrap and no drift: every block re-synchronises to sincos.
            double frac_cyc = fc_ * tau_a;
            frac_cyc -= std::floor(frac_cyc);
            double cs, sn;
            __builtin_sincos(-TWO_PI * frac_cyc, &sn, &cs);
            double zr = cs, zi = sn;
            const double dphi = -TWO_PI * fc_ * dtau;
            double wr, wi;
            __builtin_sincos(dphi, &wi, &wr);

            for (int i = blk; i < blk + m; ++i, d += dd) {
                double dc = d;
                if (dc < lo_causal)
                    dc = lo_causal;
                if (dc > hi_causal)
                    dc = hi_causal;
                if (dc > max_delay_)
                    max_delay_ = dc;

                const double pos = (double)(n0 + i) - dc;
                const int64_t base = (int64_t)std::floor(pos);
                const float* h = frac_.taps(pos - (double)base);

                // The tap window is contiguous in the ring except where it
                // straddles the wrap; the masked form defeats vectorisation.
                // Always contiguous, thanks to the mirrored tail, so this is
                // a straight dot product with no wrap branch.
                //
                // This was volk_32fc_32f_dot_prod_32fc. That kernel segfaults
                // in VOLK 2.5 when called per sample with num_points 16 or
                // more, and it is not an alignment or overrun problem on our
                // side: it still faults with the explicitly unaligned variant
                // and with 2 kB of slack behind every pointer it touches,
                // including the result. It is almost certainly not meant to be
                // called at this granularity. The scalar loop below is correct,
                // vectorises acceptably, and removes the dependency.
                const size_t ib = (size_t)(base - k0) & mask_;
                const float* rp = reinterpret_cast<const float*>(&ring_[ib]);
                float sr = 0.0f, si = 0.0f;
                for (int k = 0; k < T; ++k) {
                    sr += h[k] * rp[2 * k];
                    si += h[k] * rp[2 * k + 1];
                }
                const gr_complex acc(sr, si);

                const float ar = (float)(a * zr), ai = (float)(a * zi);
                out[i] += gr_complex(acc.real() * ar - acc.imag() * ai,
                                     acc.real() * ai + acc.imag() * ar);

                const double nzr = zr * wr - zi * wi;
                zi = zr * wi + zi * wr;
                zr = nzr;
            }
        }
    }
    n_abs_ = n0 + n;
}

int leo_channel_impl::work(int noutput_items,
                           gr_vector_const_void_star& input_items,
                           gr_vector_void_star& output_items)
{
    // A trajectory is a finite physical pass, not a spline to extrapolate
    // forever.  The waveform source may repeat indefinitely on a GRC canvas,
    // so terminate this stream when the sample clock reaches the last table
    // time.  Without this guard an edge cubic eventually grows beyond the
    // receive blocks' correctly table-sized propagation margin.
    const double now = t_start_ + (double)n_abs_ * inv_fs_;
    const double remain_s = duration() - now;
    if (remain_s <= 0.0)
        return gr::block::WORK_DONE;
    const int64_t remain_n = (int64_t)std::floor(remain_s * fs_);
    if (remain_n <= 0)
        return gr::block::WORK_DONE;
    noutput_items = (int)std::min<int64_t>(noutput_items, remain_n);

    const gr_complex* in = (const gr_complex*)input_items[0];
    gr_complex* out = (gr_complex*)output_items[0];
    int done = 0;
    while (done < noutput_items) {
        const int n = std::min(chunk_, noutput_items - done);
        process(in + done, out + done, n);
        done += n;
    }
    return noutput_items;
}

// ------------------------------------------------------------- controls ---

void leo_channel_impl::set_enable(int p, bool on)
{
    std::lock_guard<std::mutex> lk(ctl_);
    if (p >= 0 && p < (int)enable_.size())
        enable_[p] = on ? 1 : 0;
}

bool leo_channel_impl::enabled(int p) const
{
    std::lock_guard<std::mutex> lk(ctl_);
    return p >= 0 && p < (int)enable_.size() && enable_[p];
}

void leo_channel_impl::set_gain_db(int p, double g)
{
    std::lock_guard<std::mutex> lk(ctl_);
    if (p >= 0 && p < (int)gain_override_.size())
        gain_override_[p] = g;
}

void leo_channel_impl::clear_gain_override(int p)
{
    std::lock_guard<std::mutex> lk(ctl_);
    if (p >= 0 && p < (int)gain_override_.size())
        gain_override_[p] = std::numeric_limits<double>::quiet_NaN();
}

void leo_channel_impl::set_time(double t)
{
    std::lock_guard<std::mutex> lk(ctl_);
    t_start_ = t - (double)n_abs_ / fs_;
}

double leo_channel_impl::t_now() const
{
    std::lock_guard<std::mutex> lk(ctl_);
    return t_start_ + (double)n_abs_ / fs_;
}

double leo_channel_impl::tap_delay_samples(int p) const
{
    if (p < 0 || p >= (int)paths_.size())
        return 0.0;
    const double t = t_now();
    return ((paths_[p].range.eval(t) - r_nominal_) / C_LIGHT + tau_bias_) * fs_;
}

double leo_channel_impl::tap_doppler_hz(int p) const
{
    if (p < 0 || p >= (int)paths_.size())
        return 0.0;
    return -fc_ / C_LIGHT * paths_[p].range.deriv(t_now());
}

double leo_channel_impl::tap_excess_m(int p) const
{
    if (p < 0 || p >= (int)paths_.size() || paths_.empty())
        return 0.0;
    const double t = t_now();
    return paths_[p].range.eval(t) - paths_[0].range.eval(t);
}

double leo_channel_impl::tap_gain_db(int p) const
{
    if (p < 0 || p >= (int)paths_.size())
        return -200.0;
    const double a = gain_amp(p, t_now());
    return (a <= 0.0) ? -200.0 : 20.0 * std::log10(a);
}

} // namespace ntnisac
} // namespace gr
