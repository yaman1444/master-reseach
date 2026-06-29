import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cbam import CBAM
from focal_loss import FocalLoss
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("🛠️ Vérification Globale du Modèle - DenseNet121")
    
    model_path = 'models/densenet121_final.keras'
    thresholds_path = 'results/densenet121_exp3_thresholds.json'
    test_dir = '../datasets_split/test/'
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        return

    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # EXACT generator setup from train_advanced.py
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=(320, 320),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\n🔮 Prédiction sur le set complet...")
    y_probs = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    # 1. Standard Argmax
    y_pred_std = np.argmax(y_probs, axis=1)
    acc_std = accuracy_score(y_true, y_pred_std)
    print(f"\n📊 Résultat Standard (Argmax): {acc_std:.4f}")
    
    # 2. Calibrated
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            t_data = json.load(f)
        thresholds = t_data['thresholds']
        t_arr = np.array([thresholds[name] for name in class_names])
        
        adjusted_probs = y_probs / t_arr
        y_pred_calib = np.argmax(adjusted_probs, axis=1)
        acc_calib = accuracy_score(y_true, y_pred_calib)
        print(f"⚖️ Résultat Calibré: {acc_calib:.4f}")
        from sklearn.metrics import confusion_matrix
        print("\nConfusion Matrix (Calibrated):")
        cm = confusion_matrix(y_true, y_pred_calib)
        print(cm)
        
        # Check per-class accuracy
        for i, name in enumerate(class_names):
            class_acc = cm[i,i] / cm[i].sum()
            print(f"Accuracy for {name}: {class_acc:.2%}")

if __name__ == '__main__':
    main()
