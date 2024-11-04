from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any

class BaseTester(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any], save_dir: str):
        pass