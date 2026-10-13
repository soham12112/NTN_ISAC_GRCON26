#
# Copyright 2008,2009 Free Software Foundation, Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

# The presence of this file turns this directory into a Python package

'''
GNU Radio blocks for the NTN ISAC prototype.

Channel side, all usable live:

    leo_channel   C++: time-varying multipath for an emulated LEO pass
    esn0_noise    calibrated AWGN at a specified Es/N0
    leo_truth     commanded trajectory truth, published on the sample clock

Sensing side, the two-stage coarse-to-fine flow of the DySPAN 2026 paper
"A Two-Stage 5G NR ISAC Framework for Coarse-to-Fine UAV Detection and
Tracking", adapted to a single receive antenna - see docs/two-stage.md for
every deviation and why:

    ssb_coarse    stage 1: SSB matched filter, MTI, pulse-pair, tracker
    grid_sense    stage 2: gated delay-Doppler map, CFAR, clustering
    link_budget   message-only far-illuminator feasibility calculation

The two are joined by messages in both directions: ssb_coarse publishes a
Gaussian range gate on "coarse", grid_sense returns refined measurements on
"refine". That loop is what makes the flow coarse-to-fine rather than merely
two-stage, and it is what resolves stage 1's 50 Hz Doppler ambiguity.
'''
import os

# import pybind11 generated symbols into the ntnisac namespace
try:
    # this might fail if the module is python-only
    from .ntnisac_python import *
except ModuleNotFoundError:
    pass

# pure-python blocks
from .esn0_noise import esn0_noise
from .leo_truth import leo_truth
from .ssb_coarse import ssb_coarse
from .grid_sense import grid_sense
from .ota_sense import ota_sense
from .link_budget import link_budget
from .msg_print import msg_print
