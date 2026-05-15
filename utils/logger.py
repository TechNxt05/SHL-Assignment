import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    """
    Structured JSON formatter for production-oriented observability.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        
        # Include extra context if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logger(name: str = "shl_recommender"):
    """
    Initialize a structured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Global logger instance
logger = setup_logger()

class RequestContext:
    """
    Utility to track request context for logging.
    """
    def __init__(self, request_id: str = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.start_time = time.time()
        
    def get_latency_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)
    
    def log(self, level: int, msg: str, stage: str = None, metadata: Dict[str, Any] = None):
        """Log with request context."""
        extra = {
            "request_id": self.request_id,
            "latency_ms": self.get_latency_ms(),
        }
        if stage:
            extra["stage"] = stage
        if metadata:
            extra["metadata"] = metadata
            
        logger.log(level, msg, extra=extra)
