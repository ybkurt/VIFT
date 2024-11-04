import random
from typing import Dict, Any
from .base_metrics_calculator import BaseMetricsCalculator

class RandomMetricsCalculator(BaseMetricsCalculator):
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        for seq_name in results.keys():
            metrics[f'{seq_name}_rmse'] = random.uniform(0, 1)
        return metrics