"""Logging for the app and the scripts.

configure() sets up:
  - the console: INFO and above, short "[module] message" lines;
  - a log file, logs/facegen.log: DEBUG and above with timestamps, rotated at
    1 MB with 3 backups (at most ~4 MB);
  - quieter TensorFlow (its C++ warnings and Python deprecation notices);
    TensorFlow errors still show. Call configure() before TensorFlow is imported;
  - a sys.excepthook that logs unexpected errors with their traceback and,
    if set_error_reporter() was given a callback, lets the UI tell the user.
    (Without a custom hook, PyQt5 aborts the app on an error raised in a slot,
    printing nothing.)

Standard library only; safe to call more than once.
"""
import logging
import logging.handlers
import os
import sys

import config

LOG_FILE = os.path.join(config.log_dir, 'facegen.log')

# Loggers of this project: DEBUG reaches the log file. Everything else
# (PIL, urllib3, ...) stays at INFO so the file isn't flooded.
APP_LOGGERS = ('app', 'latents', 'generator', 'model', 'engine', 'downloads',
               'components', 'log_setup', 'fit_age_direction', 'measure_age_response')

_configured = False
_error_reporter = None
log = logging.getLogger('log_setup')


class _ConsoleFormatter(logging.Formatter):
    """'[module] message', with the level named for warnings and errors."""
    _plain = logging.Formatter('[%(name)s] %(message)s')
    _flagged = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')

    def format(self, record):
        return (self._flagged if record.levelno >= logging.WARNING else self._plain).format(record)


def configure(log_file=True, log_path=None):
    """Set up logging once. log_file=False logs to the console only (scripts).
    log_path overrides the log file location (tests). Returns the log file
    path, or None when not logging to a file."""
    global _configured
    if _configured:
        return active_log_file()

    # Quiet TensorFlow. The C++ setting only works if set before TF is imported.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # hide INFO and WARNING, keep errors
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for name in APP_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(_ConsoleFormatter())
    root.addHandler(console)

    path = None
    if log_file:
        path = log_path or LOG_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        to_file = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
        to_file.setLevel(logging.DEBUG)
        to_file.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-7s [%(name)s] %(message)s'))
        root.addHandler(to_file)

    sys.excepthook = _log_uncaught
    _configured = True
    return path


def set_error_reporter(callback):
    """callback() is called after an unexpected error has been logged."""
    global _error_reporter
    _error_reporter = callback


def active_log_file():
    """Path of the log file being written, or None (e.g. not configured)."""
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return h.baseFilename
    return None


def details_hint():
    """Text for error messages pointing at the log file ('' if not logging to one)."""
    path = active_log_file()
    return f"\n\nDetails were saved to:\n{path}" if path else ""


def _log_uncaught(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logging.getLogger('app').critical('Unexpected error', exc_info=(exc_type, exc, tb))
    if _error_reporter is not None:
        try:
            _error_reporter()
        except Exception:
            logging.getLogger('app').exception('Could not show the error to the user')
