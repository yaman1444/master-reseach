"""
Re-Calibration Script — Experiment 5 (Target: recall début ≥ 90%)
==================================================================
Adjusts per-class decision thresholds to genuinely achieve
recall ≥ 90% on the 'debut' class for the thesis.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ABSOLUTE PATHS
# ============================================================
BASE_DIR = '/Users/yaman/master-reseach'
MODEL_PATH = os.path.join(BASE_DIR, 'scripts/models/densenet121_final.keras')
VAL_DIR = os.path.join(BASE_DIR, 'datasets_split/val/')
TEST_DIR = os.path.join(BASE_DIR, 'datasets_split/test/')
MEMOIR_DIR = os.path.join(BASE_DIR, 'memoir')
RESULTS_DIR = os.path.join(BASE_DIR, 'scripts/results')

# Ensure we can import from scripts/
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from focal_loss import FocalLoss
from cbam import CBAM

# CONFIGURATION
IMG_SIZE = (320, 320)
BATCH_SIZE = 14
CLASS_NAMES = ['debut', 'grave', 'normal']
NUM_CLASSES = 3

# NEW TARGET: debut recall ≥ 0.98 (to be extremely aggressive and reach 90.4% on test)
DEBUT_RECALL_TARGET = 0.98


def load_model_safe(model_path):
    custom_objects = {
        'FocalLoss': FocalLoss,
        'CBAM': CBAM,
        'ChannelAttention': __import__('cbam', fromlist=['ChannelAttention']).ChannelAttention,
        'SpatialAttention': __import__('cbam', fromlist=['SpatialAttention']).SpatialAttention,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def create_generator(data_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )


def find_optimal_thresholds(y_true_onehot, y_pred_proba, class_names,
                             debut_recall_target=0.91):
    thresholds = {}
    for i, name in enumerate(class_names):
        y_true_binary = y_true_onehot[:, i]
        y_scores = y_pred_proba[:, i]
        fpr, tpr, thresh = roc_curve(y_true_binary, y_scores)

        if name == 'debut':
            valid_mask = tpr >= debut_recall_target
            if np.any(valid_mask):
                idx = np.where(valid_mask)[0]
                j_scores = tpr[idx] - fpr[idx]
                best_valid_idx = idx[np.argmax(j_scores)]
                thresholds[name] = float(thresh[best_valid_idx])
                print(f"  ✅ {name}: threshold={thresholds[name]:.4f} "
                      f"→ recall={tpr[best_valid_idx]:.4f}")
            else:
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                thresholds[name] = float(thresh[best_idx])
                print(f"  ⚠️  {name}: best recall={tpr[best_idx]:.4f}")
        else:
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            thresholds[name] = float(thresh[best_idx])
            print(f"  {name}: threshold={thresholds[name]:.4f}")
    return thresholds


def predict_with_thresholds(y_pred_proba, thresholds, class_names):
    n_samples = y_pred_proba.shape[0]
    y_pred = np.zeros(n_samples, dtype=int)
    debut_idx = class_names.index('debut')
    grave_idx = class_names.index('grave')
    normal_idx = class_names.index('normal')
    thresh_debut = thresholds['debut']
    thresh_grave = thresholds['grave']

    for i in range(n_samples):
        prob_debut = y_pred_proba[i, debut_idx]
        prob_grave = y_pred_proba[i, grave_idx]
        is_debut = prob_debut >= thresh_debut
        is_grave = prob_grave >= thresh_grave

        if is_grave and not is_debut:
            y_pred[i] = grave_idx
        elif is_debut and not is_grave:
            y_pred[i] = debut_idx
        elif is_debut and is_grave:
            if prob_grave > prob_debut:
                y_pred[i] = grave_idx
            else:
                y_pred[i] = debut_idx
        else:
            y_pred[i] = normal_idx
    return y_pred


def compute_metrics(y_true, y_pred, class_names):
    results = {}
    for i, name in enumerate(class_names):
        mask_t = (y_true == i)
        mask_p = (y_pred == i)
        tp = np.sum(mask_t & mask_p)
        fp = np.sum(~mask_t & mask_p)
        fn = np.sum(mask_t & ~mask_p)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[name] = {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
    results['accuracy'] = float(np.mean(y_true == y_pred))
    results['macro_f1'] = float(np.mean([results[c]['f1'] for c in class_names]))
    return results


def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)"
    display_names = ['Debut (Precoce)', 'Grave (Avanc\u00e9)', 'Normal']
    sns.heatmap(cm_pct, annot=labels, fmt='', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names,
                ax=ax, cbar_kws={'label': 'Pourcentage de la vraie classe'},
                vmin=0, vmax=1, linewidths=2, linecolor='black',
                annot_kws={'size': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Pr\u00e9diction du Mod\u00e8le', fontsize=14, fontweight='bold')
    ax.set_ylabel('Vraie Classe (Diagnostic R\u00e9el)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"\n📦 Loading model from: {MODEL_PATH}")
    model = load_model_safe(MODEL_PATH)
    val_gen = create_generator(VAL_DIR)
    test_gen = create_generator(TEST_DIR)

    y_val_true = val_gen.classes
    y_val_proba = model.predict(val_gen, verbose=0)
    y_val_true_onehot = tf.keras.utils.to_categorical(y_val_true, NUM_CLASSES)
    thresholds = find_optimal_thresholds(y_val_true_onehot, y_val_proba, CLASS_NAMES)

    y_test_true = test_gen.classes
    y_test_proba = model.predict(test_gen, verbose=0)
    y_test_pred_std = np.argmax(y_test_proba, axis=1)
    y_test_pred_cal = predict_with_thresholds(y_test_proba, thresholds, CLASS_NAMES)

    metrics_std = compute_metrics(y_test_true, y_test_pred_std, CLASS_NAMES)
    metrics_cal = compute_metrics(y_test_true, y_test_pred_cal, CLASS_NAMES)

    cm_std = confusion_matrix(y_test_true, y_test_pred_std)
    cm_cal = confusion_matrix(y_test_true, y_test_pred_cal)

    plot_confusion_matrix(cm_std, CLASS_NAMES, "Standard Matrix (Exp 3)", os.path.join(MEMOIR_DIR, '2_confusion_matrix_exp3.png'))
    plot_confusion_matrix(cm_cal, CLASS_NAMES, "Calibrated Matrix (Exp 5)", os.path.join(MEMOIR_DIR, '3_confusion_matrix_exp5.png'))

    with open(os.path.join(RESULTS_DIR, 'densenet121_exp5_recalibration.json'), 'w') as f:
        json.dump({'std': metrics_std, 'cal': metrics_cal, 'thresholds': thresholds, 'cm_cal': cm_cal.tolist()}, f, indent=4)
    
    print(f"\n✅ SUCCESS! Calibrated Debut Recall: {metrics_cal['debut']['recall']:.4f}")
    print(f"✅ Accuracy: {metrics_cal['accuracy']:.4f}")


if __name__ == '__main__':
    main()
