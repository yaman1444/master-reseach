from typing import Tuple, Any
from abc import ABC, abstractmethod

class ModelPort(ABC):
    @abstractmethod
    def build_model(self, img_size: Tuple[int, int], dropout_rate: float, l2_reg: float) -> Tuple[Any, Any]:
        """
        Retourne (model, base_model)
        """
        pass
