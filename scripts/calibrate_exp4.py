"""
Post-Training Threshold Calibration — Experiment 4
====================================================
Optimizes per-class decision thresholds to maximize recall
on 'debut' (early detection) while maintaining acceptable precision.

Strategy:
  1. Load best model from Exp4
  2. Predict probabilities on validation set
  3. Find optimal threshold per class via ROC curves
  4. Apply priority: debut recall ≥ 88%
  5. Evaluate on test set with calibrated thresholds
  6. Compare standard vs calibrated metrics

Usage:
  python calibrate_exp4.py
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, precision_recall_curve

# Import custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focal_loss import FocalLoss
from cbam import CBAM

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = 'models/densenet121_final.keras'
VAL_DIR = '../datasets_split/val/'
TEST_DIR = '../datasets_split/test/'
IMG_SIZE = (320, 320)
BATCH_SIZE = 14
CLASS_NAMES = ['debut', 'grave', 'normal']
NUM_CLASSES = 3

# Target: debut recall ≥ 88%
DEBUT_RECALL_TARGET = 0.88


def load_model_safe(model_path):
    """Load model with custom objects."""
    custom_objects = {
        'FocalLoss': FocalLoss,
        'CBAM': CBAM,
        'ChannelAttention': __import__('cbam', fromlist=['ChannelAttention']).ChannelAttention,
        'SpatialAttention': __import__('cbam', fromlist=['SpatialAttention']).SpatialAttention,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def create_generator(data_dir):
    """Create a simple data generator (no augmentation)."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )


def find_optimal_thresholds(y_true_onehot, y_pred_proba, class_names,
                             debut_recall_target=0.88):
    """
    Find optimal thresholds per class.

    For 'debut': find threshold that achieves recall ≥ target
    For other classes: find threshold that maximizes F1 (Youden's J statistic)

    Returns:
        dict mapping class_name → optimal_threshold
    """
    thresholds = {}

    for i, name in enumerate(class_names):
        y_true_binary = y_true_onehot[:, i]
        y_scores = y_pred_proba[:, i]

        fpr, tpr, thresh = roc_curve(y_true_binary, y_scores)

        if name == 'debut':
            # Find lowest threshold that gives recall ≥ target
            # tpr = recall at each threshold
            valid_mask = tpr >= debut_recall_target
            if np.any(valid_mask):
                # Take the highest threshold (most precision) that still meets recall target
                idx = np.where(valid_mask)[0]
                # Among valid thresholds, pick the one with best J = tpr - fpr
                j_scores = tpr[idx] - fpr[idx]
                best_valid_idx = idx[np.argmax(j_scores)]
                thresholds[name] = float(thresh[best_valid_idx])
            else:
                # Can't reach target, use best J statistic
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                thresholds[name] = float(thresh[best_idx])
                print(f"  ⚠️  {name}: cannot reach recall={debut_recall_target:.0%}, "
                      f"best recall={tpr[best_idx]:.4f} at threshold={thresh[best_idx]:.4f}")
        else:
            # Maximize Youden's J = sensitivity + specificity - 1 = tpr - fpr
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            thresholds[name] = float(thresh[best_idx])

        print(f"  {name}: threshold={thresholds[name]:.4f}")

    return thresholds


def predict_with_thresholds(y_pred_proba, thresholds, class_names):
    """
    Apply per-class thresholds with clinical priority logic.

    Priority 1: 'debut' (early detection)
    Priority 2: 'grave' (severe)
    Priority 3: 'normal' (default if neither condition met)
    """
    n_samples = y_pred_proba.shape[0]
    y_pred = np.zeros(n_samples, dtype=int)

    # Indices: 0='debut', 1='grave', 2='normal'
    debut_idx = class_names.index('debut')
    grave_idx = class_names.index('grave')
    normal_idx = class_names.index('normal')

    thresh_debut = thresholds['debut']
    thresh_grave = thresholds['grave']

    for i in range(n_samples):
        prob_debut = y_pred_proba[i, debut_idx]
        prob_grave = y_pred_proba[i, grave_idx]
        prob_normal = y_pred_proba[i, normal_idx]

        is_debut = prob_debut >= thresh_debut
        is_grave = prob_grave >= thresh_grave

        if is_grave and not is_debut:
            y_pred[i] = grave_idx
        elif is_debut and not is_grave:
            y_pred[i] = debut_idx
        elif is_debut and is_grave:
            # Both exceed their threshold: pick the one with higher raw probability
            if prob_grave > prob_debut:
                y_pred[i] = grave_idx
            else:
                y_pred[i] = debut_idx
        else:
            # Neither exceeds threshold: default to 'normal'
            y_pred[i] = normal_idx

    return y_pred


def compute_metrics(y_true, y_pred, class_names):
    """Compute per-class and macro metrics."""
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

        results[name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'support': int(np.sum(mask_t))
        }

    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean([results[c]['f1'] for c in class_names]))

    results['accuracy'] = accuracy
    results['macro_f1'] = macro_f1

    return results


def print_comparison(standard, calibrated, class_names):
    """Side-by-side comparison of standard vs calibrated metrics."""
    print(f"\n{'='*75}")
    print(f"{'':20} {'--- STANDARD ---':>25} {'--- CALIBRATED ---':>25}")
    print(f"{'Class':20} {'Prec':>7} {'Recall':>7} {'F1':>7}  "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 75)

    for name in class_names:
        s = standard[name]
        c = calibrated[name]
        rec_delta = c['recall'] - s['recall']
        marker = " ⬆️" if rec_delta > 0.01 else (" ⬇️" if rec_delta < -0.01 else "")
        print(f"{name:20} {s['precision']:7.4f} {s['recall']:7.4f} {s['f1']:7.4f}  "
              f"{c['precision']:7.4f} {c['recall']:7.4f} {c['f1']:7.4f}{marker}")

    print("-" * 75)
    print(f"{'Accuracy':20} {'':>7} {'':>7} {standard['accuracy']:7.4f}  "
          f"{'':>7} {'':>7} {calibrated['accuracy']:7.4f}")
    print(f"{'Macro-F1':20} {'':>7} {'':>7} {standard['macro_f1']:7.4f}  "
          f"{'':>7} {'':>7} {calibrated['macro_f1']:7.4f}")
    print(f"{'='*75}")


def main():
    print("\n" + "=" * 70)
    print("🎯 POST-TRAINING THRESHOLD CALIBRATION — Experiment 4")
    print("=" * 70)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found: {MODEL_PATH}")
        print("   Run train_advanced.py first!")
        sys.exit(1)

    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = load_model_safe(MODEL_PATH)

    # Create generators
    print("\n📊 Loading data...")
    val_gen = create_generator(VAL_DIR)
    test_gen = create_generator(TEST_DIR)

    # ---- STEP 1: Get predictions on validation set ----
    print("\n🔍 STEP 1: Computing predictions on VALIDATION set...")
    val_gen.reset()
    y_val_true = val_gen.classes
    y_val_proba = model.predict(val_gen, verbose=0)
    y_val_pred_standard = np.argmax(y_val_proba, axis=1)

    # One-hot encode true labels
    y_val_true_onehot = tf.keras.utils.to_categorical(y_val_true, NUM_CLASSES)

    # ---- STEP 2: Find optimal thresholds ----
    print(f"\n🔧 STEP 2: Finding optimal thresholds (debut recall target: {DEBUT_RECALL_TARGET:.0%})...")
    thresholds = find_optimal_thresholds(
        y_val_true_onehot, y_val_proba, CLASS_NAMES,
        debut_recall_target=DEBUT_RECALL_TARGET
    )

    print(f"\n📋 Optimal thresholds:")
    for name, t in thresholds.items():
        default_note = " (default=0.333)" if abs(t - 1/3) > 0.05 else " ≈ default"
        print(f"   {name}: {t:.4f}{default_note}")

    # ---- STEP 3: Evaluate on VALIDATION set ----
    print("\n📊 STEP 3: Validation set comparison...")
    val_standard = compute_metrics(y_val_true, y_val_pred_standard, CLASS_NAMES)
    y_val_pred_calibrated = predict_with_thresholds(y_val_proba, thresholds, CLASS_NAMES)
    val_calibrated = compute_metrics(y_val_true, y_val_pred_calibrated, CLASS_NAMES)

    print_comparison(val_standard, val_calibrated, CLASS_NAMES)

    # ---- STEP 4: Evaluate on TEST set ----
    print("\n🧪 STEP 4: TEST set evaluation with calibrated thresholds...")
    test_gen.reset()
    y_test_true = test_gen.classes
    y_test_proba = model.predict(test_gen, verbose=0)
    y_test_pred_standard = np.argmax(y_test_proba, axis=1)
    y_test_pred_calibrated = predict_with_thresholds(y_test_proba, thresholds, CLASS_NAMES)

    test_standard = compute_metrics(y_test_true, y_test_pred_standard, CLASS_NAMES)
    test_calibrated = compute_metrics(y_test_true, y_test_pred_calibrated, CLASS_NAMES)

    print_comparison(test_standard, test_calibrated, CLASS_NAMES)

    # ---- STEP 5: Confusion matrix ----
    print("\n🔄 Confusion Matrix (Calibrated, Test Set):")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_test_true, y_test_pred_calibrated):
        cm[t, p] += 1
    print(f"{'':15} {'Pred:debut':>12} {'Pred:grave':>12} {'Pred:normal':>12}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  True:{name:8} {cm[i,0]:12d} {cm[i,1]:12d} {cm[i,2]:12d}")

    # ---- Save results ----
    save_data = {
        'thresholds': thresholds,
        'debut_recall_target': DEBUT_RECALL_TARGET,
        'validation_standard': val_standard,
        'validation_calibrated': val_calibrated,
        'test_standard': test_standard,
        'test_calibrated': test_calibrated,
        'confusion_matrix_calibrated': cm.tolist(),
    }

    os.makedirs('results', exist_ok=True)
    out_path = 'results/densenet121_exp4_calibration.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=4, default=str)
    print(f"\n💾 Results saved to: {out_path}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("✅ CALIBRATION COMPLETE!")
    debut_rec_cal = test_calibrated.get('debut', {}).get('recall', 0)
    debut_rec_std = test_standard.get('debut', {}).get('recall', 0)
    print(f"   Debut Recall: {debut_rec_std:.4f} → {debut_rec_cal:.4f} "
          f"({'✅ TARGET MET' if debut_rec_cal >= DEBUT_RECALL_TARGET else '❌ BELOW TARGET'})")
    print(f"   Macro-F1:     {test_standard['macro_f1']:.4f} → {test_calibrated['macro_f1']:.4f}")
    print(f"   Accuracy:     {test_standard['accuracy']:.4f} → {test_calibrated['accuracy']:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
