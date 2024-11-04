from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetricsCalculator(ABC):
    @abstractmethod
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        pass