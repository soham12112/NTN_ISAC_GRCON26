/*
 * Copyright 2026 NTN ISAC prototype.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/***********************************************************************************/
/* Hand-edited: the block exposes runtime controls that bindtool would not pick up. */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(leo_channel.h)                                             */
/* BINDTOOL_HEADER_FILE_HASH(f7e3759a31e91d41f070190603c0a231)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/ntnisac/leo_channel.h>
#include <leo_channel_pydoc.h>

void bind_leo_channel(py::module& m)
{
    using leo_channel = gr::ntnisac::leo_channel;

    py::class_<leo_channel,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<leo_channel>>(m, "leo_channel", D(leo_channel))

        .def(py::init(&leo_channel::make),
             py::arg("traj_csv"),
             py::arg("fs"),
             py::arg("fc"),
             py::arg("r_nominal") = -1.0,
             py::arg("tau_bias_samples") = 16.0,
             py::arg("delay_line") = 131072,
             py::arg("interp_taps") = 8,
             py::arg("t_start") = 0.0,
             D(leo_channel, make))

        .def("num_paths", &leo_channel::num_paths)
        .def("set_enable", &leo_channel::set_enable, py::arg("path"), py::arg("on"),
             "Switch a path off, for target-absent runs.")
        .def("enabled", &leo_channel::enabled, py::arg("path"))
        .def("set_gain_db", &leo_channel::set_gain_db, py::arg("path"),
             py::arg("gain_db"), "Override the trajectory G column, echo-to-direct dB.")
        .def("clear_gain_override", &leo_channel::clear_gain_override, py::arg("path"))
        .def("set_time", &leo_channel::set_time, py::arg("t_s"),
             "Jump to a chosen point in the pass.")
        .def("t_now", &leo_channel::t_now)
        .def("tap_delay_samples", &leo_channel::tap_delay_samples, py::arg("path"))
        .def("tap_doppler_hz", &leo_channel::tap_doppler_hz, py::arg("path"))
        .def("tap_excess_m", &leo_channel::tap_excess_m, py::arg("path"))
        .def("tap_gain_db", &leo_channel::tap_gain_db, py::arg("path"))
        .def("r_nominal", &leo_channel::r_nominal)
        .def("tau_bias_samples", &leo_channel::tau_bias_samples)
        .def("max_delay_samples", &leo_channel::max_delay_samples)
        .def("duration", &leo_channel::duration)
        ;
}
