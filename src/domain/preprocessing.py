import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess_input

def apply_clahe(image):
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Retourne l'image en float32 dans la plage [0, 255].
    Améliore le contraste local pour les tissus denses (BI-RADS C/D).
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final.astype(np.float32)


def preprocess_for_densenet(image):
    """
    CLAHE + normalisation officielle DenseNet (mode 'torch').
    DenseNet ImageNet weights attend: [0,255] -> [0,1] -> (x - mean) / std
    La fonction preprocess_input de Keras gère tout cela automatiquement.
    """
    img = apply_clahe(image)
    return densenet_preprocess_input(img)


def preprocess_for_efficientnet(image):
    """
    CLAHE + normalisation officielle EfficientNet.
    EfficientNet preprocess_input attend [0, 255].
    """
    img = apply_clahe(image)
    return effnet_preprocess_input(img)
