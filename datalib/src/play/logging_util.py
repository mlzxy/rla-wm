import logging

def get_logger(name: str = "datalib"):
    """Get a logger for the datalib package."""
    return logging.getLogger(name)
