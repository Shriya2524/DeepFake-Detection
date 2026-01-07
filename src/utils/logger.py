"""
Logging utilities for deepfake detection system.

Provides consistent logging across all modules with file and console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Try to import colorlog for colored console output
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# Global logger registry
_loggers = {}


def setup_logger(
    name: str = "deepfake_detection",
    level: Union[str, int] = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    console: bool = True,
    file_logging: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (creates if doesn't exist)
        console: Whether to log to console
        file_logging: Whether to log to file
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if HAS_COLORLOG:
            # Colored console output
            color_format = "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s"
            console_formatter = colorlog.ColoredFormatter(
                color_format,
                datefmt=date_format,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Store in registry
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "deepfake_detection") -> logging.Logger:
    """
    Get a logger by name, creating if necessary.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create a basic logger if not set up
    return setup_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Example:
        class MyModel(LoggerMixin):
            def train(self):
                self.logger.info("Training started")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


class ProgressLogger:
    """
    Context manager for logging progress of long-running operations.
    
    Example:
        with ProgressLogger("Training", total=100) as plog:
            for i in range(100):
                # do work
                plog.update(i + 1)
    """
    
    def __init__(
        self,
        operation: str,
        total: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10
    ):
        """
        Initialize progress logger.
        
        Args:
            operation: Name of the operation being performed
            total: Total number of steps (optional)
            logger: Logger instance to use
            log_interval: Log progress every N percent
        """
        self.operation = operation
        self.total = total
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = 0
        self.start_time = None
    
    def __enter__(self):
        """Start the operation."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete the operation."""
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} (took {duration})")
        else:
            self.logger.error(f"Failed: {self.operation} after {duration}")
        return False
    
    def update(self, current: int, message: Optional[str] = None) -> None:
        """
        Update progress.
        
        Args:
            current: Current step number
            message: Optional message to log
        """
        self.current = current
        
        if self.total:
            percent = int((current / self.total) * 100)
            
            if percent >= self.last_logged_percent + self.log_interval:
                elapsed = datetime.now() - self.start_time
                msg = f"{self.operation}: {percent}% ({current}/{self.total})"
                if message:
                    msg += f" - {message}"
                msg += f" [elapsed: {elapsed}]"
                self.logger.info(msg)
                self.last_logged_percent = percent

