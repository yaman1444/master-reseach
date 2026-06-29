import tensorflow as tf
from pathlib import Path
from src.presentation.config import MODELS_DIR

class UNetAdapter:
    def __init__(self):
        self.model_path = MODELS_DIR / 'unet_segmentation.keras'
        self.model = None

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Le modèle U-Net est introuvable à {self.model_path}. "
                                    f"Veuillez entraîner U-Net d'abord ou vérifier le chemin.")
        print(f"Chargement du modèle U-Net depuis {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        return self.model

    def predict_mask(self, image_tensor):
        """
        Prédit le masque pour un batch d'images.
        image_tensor: shape (B, H, W, 3)
        Returns: masque binaire shape (B, H, W, 1)
        """
        if self.model is None:
            self.load_model()
            
        preds = self.model.predict(image_tensor, verbose=0)
        # Binarisation
        masks = (preds > 0.5).astype(image_tensor.dtype)
        return masks
