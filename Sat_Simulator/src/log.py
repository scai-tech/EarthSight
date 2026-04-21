"""Module-level logging utilities for the EarthSight satellite simulation.

Provides functions to create, write to, and manage a file-based logger
that records timestamped simulation events.  On import, the logging file
is cleared and a ``logging.Logger`` instance is configured.  The
``Log()`` function is the primary entry point for writing tab-delimited
log entries tagged with the current simulation time.  Helper functions
allow the simulator to update the current time, flush, and close the
log file.
"""

import logging
import os
import src.const as const
from src.utils import Time

def clear_logging_file():
    """Truncate the logging file to zero bytes.

    Called once at module import time to ensure each simulation run
    starts with a clean log file.  The file path is read from
    ``const.LOGGING_FILE``.  Creates parent directories if they do
    not already exist.
    """
    os.makedirs(os.path.dirname(const.LOGGING_FILE) or ".", exist_ok=True)
    with open(const.LOGGING_FILE, "w+") as out:
        out.write("")

def setup_logger():
    """Create and configure a file-based logger for simulation events.

    Sets up a ``logging.Logger`` named ``"custom_logger"`` at the INFO
    level with a ``FileHandler`` writing to ``const.LOGGING_FILE``.
    Messages are written with a bare ``%(message)s`` format so that the
    caller controls the exact layout.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("custom_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(const.LOGGING_FILE, encoding="ascii")
    formatter = logging.Formatter("%(message)s")  # Custom formatting
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

clear_logging_file()
loggingCurrentTime = Time()  # Will be updated by the simulator class
currentTimeStr = loggingCurrentTime.to_str()
logger = setup_logger()

def reconfigure(log_file_path):
    """Reconfigure the logger to write to a new file path.

    Closes and removes all existing handlers, updates ``const.LOGGING_FILE``,
    creates parent directories if needed, clears the new file, and attaches
    a fresh ``FileHandler``.

    Args:
        log_file_path: The new file path for the simulation log.
    """
    global logger
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    const.LOGGING_FILE = log_file_path
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    clear_logging_file()

    handler = logging.FileHandler(const.LOGGING_FILE, encoding="ascii")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def Log(description: str, *args) -> None:
    """Write a timestamped, tab-delimited log entry.

    Each entry is formatted as::

        <current_sim_time>\\t<description>\\t<arg1>\\t<arg2>\\t...

    Args:
        description: A short human-readable label for the event.
        *args: Additional values to include in the log line.  Each
            value is converted to a string via ``str()``.
    """
    global logger, loggingCurrentTime
    log_entry = f"{loggingCurrentTime.to_str()}\t{description}\t" + "\t".join(map(str, args))
    logger.info(log_entry)

def update_logging_file():
    """Flush all log handlers so that buffered entries are written to disk."""
    global logger
    for handler in logger.handlers:
        handler.flush()

def update_logging_time(time: Time):
    """Update the module-level simulation time used to tag log entries.

    A copy of the supplied Time is stored so that later mutations of
    the caller's object do not affect the logged timestamps.

    Args:
        time: The current simulation Time to record.
    """
    global loggingCurrentTime, currentTimeStr
    loggingCurrentTime = time.copy()
    currentTimeStr = loggingCurrentTime.to_str()

def get_logging_time():
    """Return a copy of the current simulation logging time.

    Returns:
        Time: A new Time instance equal to the current logging time.
    """
    global loggingCurrentTime
    return loggingCurrentTime.copy()

def get_logging_time_no_copy():
    """Return a direct reference to the current simulation logging time.

    Unlike ``get_logging_time()``, this does **not** copy the Time
    object, so callers must take care not to mutate it unintentionally.

    Returns:
        Time: The module-level Time instance used for logging timestamps.
    """
    global loggingCurrentTime
    return loggingCurrentTime

def close_logging_file():
    """Close all log file handlers and shut down the logging subsystem.

    Should be called once at the end of a simulation run to ensure all
    data is flushed and file handles are released.
    """
    global logger
    for handler in logger.handlers:
        handler.close()
    logging.shutdown()
