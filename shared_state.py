# shared_state.py

from threading import Lock

# Shared state for thread-safe data sharing
class SharedState:
    """
    Thread-safe shared state container for the application.
    
    Attributes:
        metrics (dict): Stores processed metrics data
        report_parts (dict): Stores different sections of the generated report
        lock (Lock): Thread lock for general operations
        visualization_ready (bool): Flag indicating visualization status
        viz_lock (Lock): Thread lock for visualization operations
    """
    def __init__(self):
        self.metrics = None
        self.report_parts = {}
        self.lock = Lock()
        self.visualization_ready = False
        self.viz_lock = Lock()

shared_state = SharedState()
