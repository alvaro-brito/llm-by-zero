import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Configure main application logger
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Create formatters
default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# Main log file handler
main_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, "app.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
main_handler.setFormatter(detailed_formatter)
logger.addHandler(main_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(default_formatter)
logger.addHandler(console_handler)

def get_model_logger(model_id: int) -> logging.Logger:
    """Create or get a logger for a specific model."""
    logger_name = f"model_{model_id}"
    model_logger = logging.getLogger(logger_name)
    
    # Only add handlers if they don't exist
    if not model_logger.handlers:
        model_logger.setLevel(logging.INFO)
        
        # Create model-specific log file
        log_file = os.path.join(LOGS_DIR, f"model_{model_id}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        file_handler.setFormatter(detailed_formatter)
        model_logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(default_formatter)
        model_logger.addHandler(console_handler)
    
    return model_logger 