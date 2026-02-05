#!/usr/bin/env python3
"""
BBAC ICS Framework - Centralized Logger

Provides unified logging configuration for the entire framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'bbac',
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        log_to_console: Whether to log to console
        log_format: Optional custom log format
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with default BBAC configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Configure root logger for BBAC framework
def configure_root_logger(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None
):
    """
    Configure root logger for entire framework.
    
    Args:
        level: Logging level
        log_dir: Optional directory for log files
    """
    log_file = None
    if log_dir:
        log_dir = Path(log_dir)
        log_file = log_dir / 'bbac.log'
    
    setup_logger(
        name='',  # Root logger
        level=level,
        log_file=log_file,
        log_to_console=True
    )


__all__ = ['setup_logger', 'get_logger', 'configure_root_logger']