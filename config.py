# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Global configuration."""

import os
import sys

#----------------------------------------------------------------------------
# Paths.

result_dir = 'results'
data_dir = 'datasets'

# True when running as the packaged desktop app (built by PyInstaller, see
# packaging/). Its install folder may be read-only (e.g. Program Files), so
# downloads and logs go to the user's local app data folder instead.
frozen = bool(getattr(sys, 'frozen', False))

if frozen:
    _writable_base = os.path.join(
        os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'), 'FaceGenerator')
    # Read-only data files shipped inside the app (calibration, optionally the model).
    bundled_data_dir = os.path.join(sys._MEIPASS, 'data')
else:
    # Anchored to the project folder (not the current directory), so launching
    # from anywhere finds the same downloaded model and calibration files.
    _writable_base = os.path.dirname(os.path.abspath(__file__))
    bundled_data_dir = None

# Downloaded model and boundary files.
cache_dir = os.path.join(_writable_base, 'cache')
# App log files (see log_setup.py).
log_dir = os.path.join(_writable_base, 'logs')


def data_file(name):
    """Where to read the data file `name` from: the copy shipped with the
    packaged app if it has one, otherwise cache_dir (where downloads go)."""
    if bundled_data_dir:
        shipped = os.path.join(bundled_data_dir, name)
        if os.path.isfile(shipped):
            return shipped
    return os.path.join(cache_dir, name)


run_dir_ignore = ['results', 'datasets', 'cache']

#----------------------------------------------------------------------------
