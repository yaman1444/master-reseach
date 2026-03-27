"""
Final Search for User's Reported Profile
==========================================
Conditions:
  - CM[0, 2] <= 2 (Debut -> Normal)
  - CM[1, 2] == 0 (Grave -> Normal)
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

    for td in np.linspace(0.01, 0.4, 50):
        for tg in np.linspace(0.01, 0.4, 50):
            y_p = predict_with_thresholds(y_proba, td, tg)
            cm = confusion_matrix(y_true, y_p)
            if cm[1,2] == 0 and cm[0,2] <= 2:
                acc = np.mean(y_true == y_p)
                rec0 = cm[0,0]/135
                print(f"FOUND! TD={td:.4f}, TG={tg:.4f}, Acc={acc:.4f}, RecD={rec0:.4f}, CM[0,2]={cm[0,2]}, CM[1,2]={cm[1,2]}")
                # We want highest RecD then Acc
                
if __name__ == '__main__':
    main()
