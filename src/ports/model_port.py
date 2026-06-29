from typing import Tuple, Any, Callable, Optional
from abc import ABC, abstractmethod

class ModelPort(ABC):
    @abstractmethod
    def build_model(self, img_size: Tuple[int, int], dropout_rate: float, l2_reg: float) -> Tuple[Any, Any]:
        """
        Retourne (model, base_model)
        """
        pass
    
    @abstractmethod
    def get_preprocessing_fn(self) -> Optional[Callable]:
        """
        Retourne la fonction de prétraitement spécifique au modèle.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Retourne le nom du modèle (ex: 'DenseNet-121', 'EfficientNet-B0').
        """
        pass
