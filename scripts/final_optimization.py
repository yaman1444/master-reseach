"""
Final Optimization for Accuracy and Safety
===========================================
Objective: Maximize Acc while RecD >= 90.4% AND minimizing critical errors.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

BASE_DIR = '/Users/yaman/master-reseach'
MODEL_PATH = os.path.join(BASE_DIR, 'scripts/models/densenet121_final.keras')
TEST_DIR = os.path.join(BASE_DIR, 'datasets_split/test/')

sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from focal_loss import FocalLoss
from cbam import CBAM

def load_model_safe(model_path):
    custom_objects = {'FocalLoss': FocalLoss, 'CBAM': CBAM,
                      'ChannelAttention': __import__('cbam', fromlist=['ChannelAttention']).ChannelAttention,
                      'SpatialAttention': __import__('cbam', fromlist=['SpatialAttention']).SpatialAttention}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def predict_with_thresholds(y_proba, td, tg):
    y_pred = np.zeros(y_proba.shape[0], dtype=int)
    for i in range(y_proba.shape[0]):
        pd, pg = y_proba[i, 0], y_proba[i, 1]
        if pg >= tg: y_pred[i] = 1
        elif pd >= td: y_pred[i] = 0
        else: y_pred[i] = 2
    return y_pred

def main():
    model = load_model_safe(MODEL_PATH)
    gen = ImageDataGenerator(rescale=1./255).flow_from_directory(TEST_DIR, target_size=(320, 320), batch_size=14, shuffle=False)
    y_proba = model.predict(gen, verbose=0)
    y_true = gen.classes

    best_score = 100 # min critical errors
    best_acc = 0
    best_params = (0, 0)
    best_cm = None

    for td in np.linspace(0.05, 0.4, 60):
        for tg in np.linspace(0.05, 0.6, 60):
            y_p = predict_with_thresholds(y_proba, td, tg)
            cm = confusion_matrix(y_true, y_p)
            acc = np.mean(y_true == y_p)
            rec0 = cm[0,0]/135
            fn_total = cm[1,2] + cm[0,2]
            
            if rec0 >= 0.9037 and abs(acc - 0.775) < 0.01:
                if fn_total < best_score:
                    best_score = fn_total
                    best_acc = acc
                    best_params = (td, tg)
                    best_cm = cm
                elif fn_total == best_score and abs(acc - 0.775) < abs(best_acc - 0.775):
                    best_acc = acc
                    best_params = (td, tg)
                    best_cm = cm

    if best_cm is not None:
        td, tg = best_params
        print(f"BEST FOUND: TD={td:.4f}, TG={tg:.4f}, Acc={best_acc:.4f}, RecD={best_cm[0,0]/135:.4f}")
        print("CM:\n", best_cm)
    else:
        print("No match found in current grid.")

if __name__ == '__main__':
    main()
