import os
import json
import numpy as np
import tensorflow as tf
from cbam import CBAM
from focal_loss import FocalLoss
from augmentation import apply_clahe

def load_and_preprocess_image(img_path, target_size=(320, 320), use_clahe=True):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    
    if use_clahe:
        img_array = apply_clahe(img_array)
        
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    print("🧪 Test d'Inférence Manuel - DenseNet121 Calibré")
    
    model_path = 'models/densenet121_final.keras'
    thresholds_path = 'results/densenet121_exp3_thresholds.json'
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        return

    # Load custom objects
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Load thresholds
    with open(thresholds_path, 'r') as f:
        threshold_data = json.load(f)
    
    thresholds = threshold_data['thresholds']
    class_names = ['debut', 'grave', 'normal']
    t_arr = np.array([thresholds[name] for name in class_names])
    
    test_samples = [
        ('../datasets_split/test/debut/benign (11).png', 'debut'),
        ('../datasets_split/test/debut/benign (130).png', 'debut'),
        ('../datasets_split/test/grave/malignant (10).png', 'grave'),
        ('../datasets_split/test/grave/malignant (17).png', 'grave'),
        ('../datasets_split/test/normal/normal (4).png', 'normal'),
        ('../datasets_split/test/normal/normal (31).png', 'normal')
    ]
    
    print(f"\n{'='*90}")
    print(f"{'IMAGE':<30} | {'REAL':<10} | {'PRED':<10} | {'RAW PROBS (D/G/N)':<30}")
    print(f"{'='*90}")
    
    for img_path, real_class in test_samples:
        if not os.path.exists(img_path):
            print(f"⚠️ Image introuvable : {img_path}")
            continue
            
        img_array = load_and_preprocess_image(img_path)
        
        # Raw prediction
        y_probs = model.predict(img_array, verbose=0)[0]
        
        # Calibrated prediction
        adjusted_probs = y_probs / t_arr
        pred_idx = np.argmax(adjusted_probs)
        pred_class = class_names[pred_idx]
        
        status = "✅" if pred_class == real_class else "❌"
        img_name = os.path.basename(img_path)
        probs_str = f"[{y_probs[0]:.3f}, {y_probs[1]:.3f}, {y_probs[2]:.3f}]"
        print(f"{img_name:<30} | {real_class:<10} | {pred_class:<10} | {probs_str:<30} {status}")

if __name__ == '__main__':
    main()
