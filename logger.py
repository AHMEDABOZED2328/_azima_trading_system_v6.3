#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
AzImA Trading System v6.2 - Centralized Logging System
═══════════════════════════════════════════════════════════════════

Provides a centralized logging configuration for all modules.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File rotation (prevents huge log files)
- Console and file output
- Colored console output for better readability
- Separate logs for different components

Author: Ahmed (AzImA Team)
Date: February 2026
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import sys


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(
    name: str = "azima",
    log_dir: Path = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger
    
    Args:
        name: Logger name (e.g., 'azima', 'azima.training', 'azima.prediction')
        log_dir: Directory for log files (default: ./logs)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers (avoid duplicates)
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if file_output:
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Main log file
        log_file = log_dir / f"{name.replace('.', '_')}_{datetime.now():%Y%m%d}.log"
        
        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        module_name: Name of the module (e.g., 'training', 'prediction', 'feature_engineering')
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"azima.{module_name}")


# ═══════════════════════════════════════════════════════════════
# Pre-configured loggers for common use cases
# ═══════════════════════════════════════════════════════════════

def get_training_logger() -> logging.Logger:
    """Get logger for training scripts"""
    return get_module_logger("training")


def get_prediction_logger() -> logging.Logger:
    """Get logger for prediction scripts"""
    return get_module_logger("prediction")


def get_feature_logger() -> logging.Logger:
    """Get logger for feature engineering"""
    return get_module_logger("feature_engineering")


def get_data_logger() -> logging.Logger:
    """Get logger for data processing"""
    return get_module_logger("data")


# ═══════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Setup main logger
    logger = setup_logger("azima", level=logging.DEBUG)
    
    logger.debug("This is a DEBUG message")
    logger.info("✅ This is an INFO message")
    logger.warning("⚠️ This is a WARNING message")
    logger.error("❌ This is an ERROR message")
    logger.critical("🔥 This is a CRITICAL message")
    
    # Module-specific loggers
    train_logger = get_training_logger()
    train_logger.info("🚀 Training started")
    
    pred_logger = get_prediction_logger()
    pred_logger.info("🔮 Making predictions...")
    
    print("\n✅ Logging system test complete!")
    print(f"   Log files saved to: {Path(__file__).parent / 'logs'}")
