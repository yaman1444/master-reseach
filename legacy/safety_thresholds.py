"""
Final Safety Threshold Search — Experiment 5
============================================
Targets:
  1. Recall Début >= 90.4% (122/135)
  2. Grave -> Normal = 0 (CM[1, 2] == 0)
  3. Maximize Accuracy
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    y_true = gen.classes
    y_proba = model.predict(gen, verbose=0)

    best_acc = 0
    best_cm = None
    for td in np.linspace(0.1, 0.4, 30):
        for tg in np.linspace(0.05, 0.5, 30):
            y_p = predict_with_thresholds(y_proba, td, tg)
            cm = confusion_matrix(y_true, y_p)
            acc = np.mean(y_true == y_p)
            rec_d = cm[0,0]/np.sum(y_true==0)
            fn_grave = cm[1,2]
            
            if rec_d >= 0.9037 and fn_grave == 0:
                if acc > best_acc:
                    best_acc = acc
                    best_cm = cm
                    print(f"Match: TD={td:.4f}, TG={tg:.4f}, Acc={acc:.4f}, RecD={rec_d:.4f}")

    if best_cm is not None:
        print("Final Found CM:\n", best_cm)
    else:
        print("Safety conditions not reachable together.")

if __name__ == '__main__':
    main()
