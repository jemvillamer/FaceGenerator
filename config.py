# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Global configuration."""

import os

#----------------------------------------------------------------------------
# Paths.

result_dir = 'results'
data_dir = 'datasets'
# Anchored to the project folder (not the current directory), so launching
# from anywhere finds the same downloaded model and calibration files.
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
run_dir_ignore = ['results', 'datasets', 'cache']

#----------------------------------------------------------------------------
