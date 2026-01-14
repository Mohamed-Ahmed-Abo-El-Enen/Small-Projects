import threading
from typing import Dict, Any


class SingletonMeta(type):
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def clear_instance(mcs, cls):
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]
                print(f"Cleared singleton instance of {cls.__name__}")