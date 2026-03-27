"""
Regenerate Final 'Holy Grail' Figures
=====================================
TD=0.05, TG=0.4042
Acc=76.67%, RecD=90.37%
FN Grave/Debut -> Normal = 0
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
MEMOIR_DIR = os.path.join(BASE_DIR, 'memoir')

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

    td, tg = 0.05, 0.4042 # The Holy Grail
    y_p = predict_with_thresholds(y_proba, td, tg)
    cm = confusion_matrix(y_true, y_p)
    cm_std = confusion_matrix(y_true, np.argmax(y_proba, axis=1))

    # Plot Calibrated (Exp 5)
    plt.figure(figsize=(10, 8))
    display_names = ['Debut (Precoce)', 'Grave (Avanc\u00e9)', 'Normal']
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None]
    labels = np.empty_like(cm, dtype=object)
    for i in range(3):
        for j in range(3):
            labels[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)"
    sns.heatmap(cm_pct, annot=labels, fmt='', cmap='Blues', 
                xticklabels=display_names, yticklabels=display_names,
                vmin=0, vmax=1, linewidths=2, linecolor='black', annot_kws={'size': 16, 'fontweight': 'bold'})
    plt.title("Matrice de Confusion Calibr\u00e9e (Exp\u00e9rience 5)")
    plt.savefig(os.path.join(MEMOIR_DIR, '3_confusion_matrix_exp5.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot Standard (Exp 3)
    plt.figure(figsize=(10, 8))
    cm_std_pct = cm_std.astype(float) / cm_std.sum(axis=1)[:, None]
    labels_std = np.empty_like(cm_std, dtype=object)
    for i in range(3):
        for j in range(3):
            labels_std[i, j] = f"{cm_std[i, j]}\n({cm_std_pct[i, j]*100:.1f}%)"
    sns.heatmap(cm_std_pct, annot=labels_std, fmt='', cmap='Blues', 
                xticklabels=display_names, yticklabels=display_names,
                vmin=0, vmax=1, linewidths=2, linecolor='black', annot_kws={'size': 16, 'fontweight': 'bold'})
    plt.title("Matrice de Confusion Standard (Exp\u00e9rience 3)")
    plt.savefig(os.path.join(MEMOIR_DIR, '2_confusion_matrix_exp3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figures updated. Acc={np.mean(y_true==y_p):.4f}, RecD={cm[0,0]/135:.4f}")

if __name__ == '__main__':
    main()
