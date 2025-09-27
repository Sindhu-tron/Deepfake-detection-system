"""
Simple logging system
"""

import logging
import sys

class StructuredLogger:
    """Simple logger"""
    
    def __init__(self, name="deepfake_detection", log_level="INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message, **kwargs):
        self.logger.info(f"{message} {kwargs if kwargs else ''}")
    
    def error(self, message, **kwargs):
        self.logger.error(f"{message} {kwargs if kwargs else ''}")
    
    def warning(self, message, **kwargs):
        self.logger.warning(f"{message} {kwargs if kwargs else ''}")

def get_logger():
    """Get logger instance"""
    return StructuredLogger()