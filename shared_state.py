# shared_state.py

"""
This module provides thread-safe shared state management for the application.
It uses Python's threading.Lock to ensure thread safety when multiple components
access shared data simultaneously.

The module defines a SharedState class that serves as a central container for
application-wide data that needs to be accessed by multiple threads. This includes
metrics data, report parts, and visualization status.

Thread safety is achieved through the use of Lock objects that prevent multiple
threads from modifying the same data simultaneously.
"""

from threading import Lock

# Shared state for thread-safe data sharing
class SharedState:
    """
    Thread-safe shared state container for the application.
    
    This class provides a centralized location for storing and accessing data
    that needs to be shared between different components of the application.
    All data access is protected by locks to ensure thread safety.
    
    Attributes:
        metrics (dict): Stores processed metrics data from PDF analysis.
            This includes all the metrics extracted from the PDFs and their
            associated values, statuses, and trends.
            
        report_parts (dict): Stores different sections of the generated report.
            Each section (overview, metrics summary, etc.) is stored separately
            and can be accessed by different threads.
            
        lock (Lock): Thread lock for general operations.
            Used to protect access to metrics and report_parts.
            
        visualization_ready (bool): Flag indicating visualization status.
            Set to True when visualizations have been generated and are ready
            for use.
            
        viz_lock (Lock): Thread lock for visualization operations.
            Used specifically to protect visualization-related operations
            to prevent race conditions during chart generation.
    """
    def __init__(self):
        """
        Initialize the SharedState object with default values.
        
        Sets up the initial state with:
        - Empty metrics dictionary
        - Empty report_parts dictionary
        - New Lock objects for thread safety
        - visualization_ready flag set to False
        """
        self.metrics = None  # Will store the processed metrics data
        self.report_parts = {}  # Will store different sections of the report
        self.lock = Lock()  # Lock for general operations
        self.visualization_ready = False  # Flag for visualization status
        self.viz_lock = Lock()  # Lock specifically for visualization operations

# Create a single instance of SharedState to be used throughout the application
shared_state = SharedState()
