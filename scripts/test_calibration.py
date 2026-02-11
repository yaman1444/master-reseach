"""
Test batch: Comparer prédictions standard vs calibrées
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
from pathlib import Path
from cbam import CBAM
from focal_loss import FocalLoss

def predict_image(model, img_path, threshold_config=None):
    """Prédire une image avec/sans seuil calibré"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_input = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array_input, verbose=0)[0]
    
    # Standard
    pred_std = np.argmax(predictions)
    conf_std = predictions[pred_std]
    
    # Calibré
    if threshold_config:
        threshold = threshold_config.get('optimal_threshold_malignant', 0.5)
        if predictions[1] >= threshold:  # malignant idx = 1
            pred_cal = 1
            conf_cal = predictions[1]
        else:
            pred_cal = np.argmax(predictions)
            conf_cal = predictions[pred_cal]
    else:
        pred_cal = pred_std
        conf_cal = conf_std
    
    return {
        'probs': predictions,
        'pred_std': pred_std,
        'conf_std': conf_std,
        'pred_cal': pred_cal,
        'conf_cal': conf_cal
    }

# Charger modèle
print("Chargement modèle...")
custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
model = tf.keras.models.load_model('./models/densenet121_improved.keras', 
                                   custom_objects=custom_objects, compile=False)

# Charger config seuils
with open('./results/densenet121_improved_thresholds.json', 'r') as f:
    threshold_config = json.load(f)

print(f"Seuil calibré malignant: {threshold_config['optimal_threshold_malignant']:.3f}\n")

# Test sur quelques images
test_images = [
    ('../datasets/test/grave/malignant (1).png', 'malignant'),
    ('../datasets/test/grave/malignant (2).png', 'malignant'),
    ('../datasets/test/debut/benign (1).png', 'benign'),
    ('../datasets/test/normal/normal (1).png', 'normal'),
]

class_names = ['benign', 'malignant', 'normal']

print("="*80)
print("COMPARAISON STANDARD vs CALIBRÉ")
print("="*80 + "\n")

for img_path, true_label in test_images:
    if not Path(img_path).exists():
        continue
    
    result = predict_image(model, img_path, threshold_config)
    
    print(f"Image: {Path(img_path).name}")
    print(f"  Vérité: {true_label}")
    print(f"  Probas: B={result['probs'][0]:.3f}, M={result['probs'][1]:.3f}, N={result['probs'][2]:.3f}")
    print(f"  Standard:  {class_names[result['pred_std']]} ({result['conf_std']:.3f})")
    print(f"  Calibré:   {class_names[result['pred_cal']]} ({result['conf_cal']:.3f})")
    
    # Changement?
    if result['pred_std'] != result['pred_cal']:
        print(f"  ⚠️  CHANGEMENT: {class_names[result['pred_std']]} → {class_names[result['pred_cal']]}")
    
    print()
