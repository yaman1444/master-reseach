import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Ajouter la racine du projet au PYTHONPATH pour pouvoir importer src
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.adapters.unet_adapter import UNetAdapter
from src.adapters.efficientnet_adapter import EfficientNetAdapter

def apply_clahe_fallback(img):
    """
    Applique CLAHE en fallback si la méthode du domaine échoue ou n'est pas importable.
    """
    if len(img.shape) == 3:
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final / 255.0
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply((img * 255).astype(np.uint8)) / 255.0

def test_cascade(original_img_path):
    print(f"Chargement de l'image : {original_img_path}")
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {original_img_path}")
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    print("Initialisation des adaptateurs de modèles...")
    # 1. Adaptateur U-Net (Respect de la Clean Architecture)
    unet = UNetAdapter()
    
    # 2. Adaptateur EfficientNet-B0
    effnet_adapter = EfficientNetAdapter()
    effnet_model, _ = effnet_adapter.build_model((224, 224), 0.3, 0.001)
    
    # -------------------------------------------
    # ETAPE 1 : Segmentation par U-Net
    # -------------------------------------------
    print("Segmentation par U-Net...")
    unet_input = cv2.resize(original_img, (256, 256)) / 255.0
    unet_input_tensor = np.expand_dims(unet_input, 0)
    
    try:
        mask_binary = unet.predict_mask(unet_input_tensor)[0]
    except Exception as e:
        print(f"Attention: {e}. Utilisation d'un masque simulé pour le test.")
        mask_binary = np.ones((256, 256, 1), dtype=np.float32)

    # -------------------------------------------
    # ETAPE 2 : Application du masque sur l'image d'origine
    # -------------------------------------------
    print("Application du masque et prétraitement (CLAHE)...")
    mask_resized = cv2.resize(mask_binary, (224, 224))
    img_224 = cv2.resize(original_img, (224, 224)) / 255.0
    
    # CLAHE
    try:
        from src.domain.preprocessing import apply_clahe
        img_clahe = apply_clahe(img_224)
    except ImportError:
        img_clahe = apply_clahe_fallback(img_224)
        
    # Multiplication par le masque
    masked_img = img_clahe * np.expand_dims(mask_resized, -1)

    # -------------------------------------------
    # ETAPE 3 : Classification par EfficientNet-B0
    # -------------------------------------------
    print("Classification par EfficientNet-B0...")
    pred_probs = effnet_model.predict(np.expand_dims(masked_img, 0), verbose=0)[0]
    predicted_class = np.argmax(pred_probs)
    
    classes = ['Normal', 'Bénin', 'Malin']
    print("\n--- RÉSULTATS DE LA CASCADE ---")
    print(f"Probabilités : {pred_probs}")
    print(f"Classe finale prédite par la cascade : {classes[predicted_class]}")
    
    return predicted_class

if __name__ == "__main__":
    dummy_path = os.path.join(project_root, "dummy_test_img.jpg")
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite(dummy_path, dummy_img)
    
    try:
        test_cascade(dummy_path)
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
