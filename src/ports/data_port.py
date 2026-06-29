from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

class DataPort(ABC):
    @abstractmethod
    def get_train_test_generators(self, data_dir: str, img_size: Tuple[int, int], batch_size: int) -> Tuple[Any, Any, Dict[int, float]]:
        """
        Retourne (train_generator, test_generator, class_weights)
        """
        pass
