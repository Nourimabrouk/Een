"""
Logging Configuration for Unity Mathematics
==========================================

Handles Unicode characters and φ-harmonic logging properly on all platforms.
"""

import logging
import sys
import os
from typing import Optional


def setup_unity_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, use_unicode: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for Unity Mathematics.

    Args:
        level: Logging level
        log_file: Optional log file path
        use_unicode: Whether to use Unicode characters in logs
    """

    # Create logger
    logger = logging.getLogger("unity_mathematics")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    if use_unicode and sys.platform.startswith("win"):
        # Windows-safe format without Unicode characters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [Unity] %(message)s"
        )
    elif use_unicode:
        # Unix-like systems can handle Unicode
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [φ-Harmonic] %(message)s"
        )
    else:
        # Fallback format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [Unity] %(message)s"
        )

    # Console handler with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Set encoding for Windows
    if sys.platform.startswith("win"):
        console_handler.stream.reconfigure(encoding="utf-8")

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger with proper Unicode handling."""
    if name:
        return logging.getLogger(f"unity_mathematics.{name}")
    return logging.getLogger("unity_mathematics")


# Global logger instance
unity_logger = get_logger()


def log_unity_event(message: str, level: str = "info"):
    """Log a Unity Mathematics event with proper encoding."""
    log_func = getattr(unity_logger, level.lower(), unity_logger.info)
    log_func(message)


def log_phi_harmonic(message: str, level: str = "info"):
    """Log φ-harmonic events safely."""
    # Replace φ with phi for Windows compatibility
    if sys.platform.startswith("win"):
        message = message.replace("φ", "phi")
    log_unity_event(message, level)
