import cv2
import numpy as np

def apply_clahe(image):
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Retourne l'image en float32 dans la plage [0, 255].
    La normalisation finale est laissée à la fonction appelante.
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
    CLAHE + normalisation [0, 1].
    DenseNet121 avec poids ImageNet attend des valeurs [0, 1].
    """
    img = apply_clahe(image)
    return img / 255.0


def preprocess_for_efficientnet(image):
    """
    CLAHE uniquement, retourne [0, 255].
    EfficientNetB0 Keras contient sa propre couche de normalisation interne.
    """
    return apply_clahe(image)
