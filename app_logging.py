"""
This module configures and provides application-wide logging functionality.
It sets up a standardized logging format and creates a logger instance that
can be imported and used throughout the application.

The logging configuration includes:
- Timestamp
- Log level
- Message content

This ensures consistent logging format across all components of the application.
"""

import logging

# Configure the root logger with a specific format and level
logging.basicConfig(
    level=logging.INFO,  # Set default logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format: timestamp - level - message
)

# Create a logger instance for this module
# __name__ ensures the logger is named after the current module
logger = logging.getLogger(__name__)