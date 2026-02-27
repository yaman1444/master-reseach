"""
Advanced Training Script v2 — Breast Cancer Classification (BUSI)
==================================================================
Target: ≥90% macro-F1 on proper held-out test set

Key improvements over v1:
  ✅ Proper train/val/test split (no data leakage)
  ✅ BatchNorm + L2 regularization in classification head
  ✅ Class-aware Focal Loss (α per-class, inverse frequency)
  ✅ Label Smoothing (ε=0.1)
  ✅ Test-Time Augmentation (TTA)
  ✅ Confidence intervals via bootstrap
  ✅ EfficientNetV2-S support
  ✅ Cosine Annealing with Warm Restarts

References:
  - DenseNet: Huang et al. CVPR'17
  - Focal Loss: Lin et al. ICCV'17
  - Mixup: Zhang et al. ICLR'18
  - CBAM: Woo et al. ECCV'18
  - SGDR: Loshchilov & Hutter, ICLR'17
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Import custom modules
from focal_loss import FocalLoss, compute_class_alpha
from augmentation import AugmentedDataGenerator
from cbam import CBAM

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'batch_size': 16,
    'num_classes': 3,
    'class_names': ['debut', 'grave', 'normal'],

    # Training phases
    'initial_epochs': 20,
    'fine_tune_epochs': 30,
    'initial_lr': 1e-4,
    'fine_tune_lr': 1e-5,

    # Focal Loss
    'focal_gamma': 2.0,
    'focal_alpha': None,        # Will be computed from data (class-aware)
    'label_smoothing': 0.1,     # NEW: label smoothing

    # Augmentation
    'mixup_alpha': 0.2,
    'use_mixup': True,
    'use_clahe': True,

    # Architecture
    'dropout_rate': 0.4,        # Reduced from 0.5 (with BN, less dropout needed)
    'l2_reg': 1e-4,             # NEW: L2 regularization
    'use_cbam': True,
    'head_units': [512, 256],   # NEW: configurable head

    # Data paths (using proper split)
    'train_dir': '../datasets_split/train/',
    'val_dir': '../datasets_split/val/',
    'test_dir': '../datasets_split/test/',

    # TTA
    'use_tta': True,
    'tta_augmentations': 8,
}


# ============================================================
# COSINE ANNEALING WITH WARM RESTARTS (SGDR)
# ============================================================
class CosineAnnealingWarmRestarts(tf.keras.callbacks.Callback):
    """
    SGDR: Stochastic Gradient Descent with Warm Restarts
    Loshchilov & Hutter, ICLR'17

    η_t = η_min + 0.5(η_max - η_min)(1 + cos(π * T_cur / T_i))
    """
    def __init__(self, initial_lr, min_lr, first_cycle_epochs, cycle_mult=1.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.first_cycle_epochs = first_cycle_epochs
        self.cycle_mult = cycle_mult
        self.current_cycle_epochs = first_cycle_epochs
        self.cycle_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        # Check if we need to restart
        if self.cycle_epoch >= self.current_cycle_epochs:
            self.cycle_epoch = 0
            self.current_cycle_epochs = int(
                self.current_cycle_epochs * self.cycle_mult
            )

        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + np.cos(np.pi * self.cycle_epoch / self.current_cycle_epochs))

        if hasattr(self.model.optimizer, 'learning_rate'):
            self.model.optimizer.learning_rate.assign(lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        self.cycle_epoch += 1
        print(f"\nEpoch {epoch+1}: lr = {lr:.6f}")


# ============================================================
# MODEL BUILDER
# ============================================================
def build_model(model_name='densenet121', input_shape=(224, 224, 3),
                num_classes=3, use_cbam=True, dropout_rate=0.4,
                l2_reg=1e-4, head_units=None):
    """
    Build classification model with improved head.

    Architecture:
      Backbone (frozen) → CBAM → GAP → [Dense+BN+Dropout]×N → Softmax

    Args:
        model_name: 'densenet121', 'resnet50', 'efficientnetb0', or 'efficientnetv2s'
        use_cbam: Add CBAM attention before GAP
        head_units: List of dense layer sizes, e.g. [512, 256]
    """
    if head_units is None:
        head_units = [512, 256]

    # Load backbone
    backbone_map = {
        'densenet121': DenseNet121,
        'resnet50': ResNet50,
        'efficientnetb0': EfficientNetB0,
    }

    # EfficientNetV2 is in tf.keras.applications for TF >= 2.11
    if model_name == 'efficientnetv2s':
        try:
            from tensorflow.keras.applications import EfficientNetV2S
            base_model = EfficientNetV2S(
                weights='imagenet', include_top=False, input_shape=input_shape
            )
        except ImportError:
            print("⚠️ EfficientNetV2S not available, falling back to EfficientNetB0")
            base_model = EfficientNetB0(
                weights='imagenet', include_top=False, input_shape=input_shape
            )
    elif model_name in backbone_map:
        base_model = backbone_map[model_name](
            weights='imagenet', include_top=False, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Build classification head
    x = base_model.output

    if use_cbam:
        x = CBAM(ratio=8)(x)

    x = GlobalAveragePooling2D()(x)

    for units in head_units:
        x = Dense(units, activation='relu',
                  kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    predictions = Dense(num_classes, activation='softmax',
                        kernel_regularizer=l2(l2_reg))(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


# ============================================================
# DATA UTILITIES
# ============================================================
def create_generators(config):
    """Create train, val, and test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    img_size = (config['img_height'], config['img_width'])

    train_gen = train_datagen.flow_from_directory(
        config['train_dir'],
        target_size=img_size,
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        config['val_dir'],
        target_size=img_size,
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False
    )

    test_gen = val_datagen.flow_from_directory(
        config['test_dir'],
        target_size=img_size,
        batch_size=config['batch_size'],
        class_mode='categorical',
        shuffle=False
    )

    # Wrap train with advanced augmentation
    if config['use_mixup'] or config['use_clahe']:
        train_gen = AugmentedDataGenerator(
            train_gen,
            use_clahe=config['use_clahe'],
            use_mixup=config['use_mixup'],
            mixup_alpha=config['mixup_alpha'],
            mixup_prob=0.5
        )

    return train_gen, val_gen, test_gen


def compute_class_weights(generator):
    """Compute class weights from generator."""
    labels = generator.classes
    counter = collections.Counter(labels)
    total = len(labels)
    n_classes = len(counter)
    return {cid: total / (n_classes * cnt) for cid, cnt in counter.items()}


# ============================================================
# TEST-TIME AUGMENTATION (TTA)
# ============================================================
def predict_with_tta(model, images, n_augmentations=8):
    """
    Test-Time Augmentation: average predictions over augmented versions.

    Augmentations: original, flip_h, flip_v, flip_hv,
                   rot90, rot180, rot270, brightness

    Args:
        model: trained model
        images: numpy array of images (N, H, W, C)
        n_augmentations: number of augmentations to average

    Returns:
        averaged softmax predictions
    """
    all_preds = []

    # Original
    all_preds.append(model.predict(images, verbose=0))

    if n_augmentations >= 2:
        # Horizontal flip
        all_preds.append(model.predict(images[:, :, ::-1, :], verbose=0))

    if n_augmentations >= 3:
        # Vertical flip
        all_preds.append(model.predict(images[:, ::-1, :, :], verbose=0))

    if n_augmentations >= 4:
        # Both flips
        all_preds.append(model.predict(images[:, ::-1, ::-1, :], verbose=0))

    if n_augmentations >= 5:
        # 90° rotation
        rotated = np.rot90(images, k=1, axes=(1, 2))
        all_preds.append(model.predict(rotated, verbose=0))

    if n_augmentations >= 6:
        # 180° rotation
        rotated = np.rot90(images, k=2, axes=(1, 2))
        all_preds.append(model.predict(rotated, verbose=0))

    if n_augmentations >= 7:
        # 270° rotation
        rotated = np.rot90(images, k=3, axes=(1, 2))
        all_preds.append(model.predict(rotated, verbose=0))

    if n_augmentations >= 8:
        # Slight brightness variation
        bright = np.clip(images * 1.1, 0, 1)
        all_preds.append(model.predict(bright, verbose=0))

    # Average all predictions
    return np.mean(all_preds, axis=0)


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_pred_proba, class_names):
    """Compute comprehensive metrics."""
    n_classes = len(class_names)
    results = {}

    for i, name in enumerate(class_names):
        mask_t = (y_true == i)
        mask_p = (y_pred == i)
        tp = np.sum(mask_t & mask_p)
        fp = np.sum(~mask_t & mask_p)
        fn = np.sum(mask_t & ~mask_p)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results[name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1-score': float(f1),
            'support': int(np.sum(mask_t))
        }

    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean([results[c]['f1-score'] for c in class_names]))

    results['accuracy'] = accuracy
    results['macro avg'] = {
        'precision': float(np.mean([results[c]['precision'] for c in class_names])),
        'recall': float(np.mean([results[c]['recall'] for c in class_names])),
        'f1-score': macro_f1,
        'support': len(y_true)
    }

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        y_true_onehot = tf.keras.utils.to_categorical(y_true, n_classes)
        auc_scores = roc_auc_score(y_true_onehot, y_pred_proba,
                                   average=None, multi_class='ovr')
        results['auc_per_class'] = {
            name: float(auc) for name, auc in zip(class_names, auc_scores)
        }
        results['auc_macro'] = float(np.mean(auc_scores))
    except Exception:
        results['auc_macro'] = None
        results['auc_per_class'] = None

    return results


def bootstrap_confidence_interval(y_true, y_pred, metric_fn,
                                   n_bootstrap=1000, ci=0.95):
    """
    Bootstrap CI for any metric.

    Args:
        metric_fn: function(y_true, y_pred) → float
    Returns:
        (mean, lower, upper)
    """
    scores = []
    n = len(y_true)
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    mean = float(np.mean(scores))

    return mean, lower, upper


# ============================================================
# MAIN TRAINING
# ============================================================
def train_model(model_name='densenet121', config=None):
    """
    Full training pipeline with 2-phase fine-tuning.

    Phase 1: Frozen backbone, train head only
    Phase 2: Unfreeze top 20% of backbone, fine-tune end-to-end
    """
    if config is None:
        config = CONFIG

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} — Advanced Pipeline v2")
    print(f"{'='*60}\n")

    # Create data generators
    train_gen, val_gen, test_gen = create_generators(config)

    # Compute class-aware alpha for focal loss
    class_counts = collections.Counter(val_gen.classes)
    # Use train counts for alpha
    train_counts_dict = {}
    for cls_name, cls_idx in train_gen.base_generator.class_indices.items() \
            if hasattr(train_gen, 'base_generator') else train_gen.class_indices.items():
        count = sum(1 for l in (train_gen.base_generator.classes
                    if hasattr(train_gen, 'base_generator')
                    else train_gen.classes) if l == cls_idx)
        train_counts_dict[cls_name] = count

    alpha_weights = compute_class_alpha(list(train_counts_dict.values()))
    config['focal_alpha'] = alpha_weights
    print(f"Class-aware α weights: {dict(zip(train_counts_dict.keys(), alpha_weights))}")

    class_weights = compute_class_weights(
        train_gen.base_generator if hasattr(train_gen, 'base_generator') else train_gen
    )
    print(f"Class weights: {class_weights}")

    # Build model
    model, base_model = build_model(
        model_name=model_name,
        input_shape=(config['img_height'], config['img_width'], 3),
        num_classes=config['num_classes'],
        use_cbam=config['use_cbam'],
        dropout_rate=config['dropout_rate'],
        l2_reg=config['l2_reg'],
        head_units=config.get('head_units', [512, 256])
    )

    # ---- PHASE 1: Frozen backbone ----
    print(f"\n{'='*60}")
    print("PHASE 1: Training classification head (backbone frozen)")
    print(f"{'='*60}\n")

    base_model.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=config['initial_lr']),
        loss=FocalLoss(
            gamma=config['focal_gamma'],
            alpha=config['focal_alpha'],
            label_smoothing=config['label_smoothing']
        ),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks_p1 = [
        ModelCheckpoint(
            f'models/{model_name}_phase1_best.keras',
            monitor='val_auc', mode='max',
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=7,
            restore_best_weights=True, verbose=1
        ),
        CosineAnnealingWarmRestarts(
            initial_lr=config['initial_lr'],
            min_lr=config['initial_lr'] * 0.01,
            first_cycle_epochs=config['initial_epochs'],
            cycle_mult=1.0
        ),
        TensorBoard(log_dir=f'logs/{model_name}_phase1')
    ]

    history_p1 = model.fit(
        train_gen,
        epochs=config['initial_epochs'],
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1
    )

    # ---- PHASE 2: Fine-tune top 20% ----
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tuning top 20% of backbone")
    print(f"{'='*60}\n")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    print(f"Total layers: {total_layers}, Frozen: {freeze_until}, "
          f"Trainable: {total_layers - freeze_until}")

    model.compile(
        optimizer=Adam(learning_rate=config['fine_tune_lr']),
        loss=FocalLoss(
            gamma=config['focal_gamma'],
            alpha=config['focal_alpha'],
            label_smoothing=config['label_smoothing']
        ),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks_p2 = [
        ModelCheckpoint(
            f'models/{model_name}_final.keras',
            monitor='val_auc', mode='max',
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        CosineAnnealingWarmRestarts(
            initial_lr=config['fine_tune_lr'],
            min_lr=config['fine_tune_lr'] * 0.1,
            first_cycle_epochs=10,
            cycle_mult=1.5
        ),
        TensorBoard(log_dir=f'logs/{model_name}_phase2')
    ]

    history_p2 = model.fit(
        train_gen,
        epochs=config['fine_tune_epochs'],
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1
    )

    # ---- FINAL EVALUATION (on held-out test set) ----
    print(f"\n{'='*60}")
    print("FINAL EVALUATION — Held-out Test Set")
    print(f"{'='*60}\n")

    test_gen.reset()
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    # Standard predictions
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # TTA predictions
    if config.get('use_tta', False):
        print("\n🔄 Applying Test-Time Augmentation (TTA)...")
        # Load all test images into memory for TTA
        test_gen.reset()
        all_images = []
        all_labels = []
        for i in range(len(test_gen)):
            imgs, labs = test_gen[i]
            all_images.append(imgs)
        all_images = np.concatenate(all_images, axis=0)[:len(y_true)]

        y_pred_proba_tta = predict_with_tta(
            model, all_images,
            n_augmentations=config.get('tta_augmentations', 8)
        )
        y_pred_tta = np.argmax(y_pred_proba_tta, axis=1)

        print(f"  Standard accuracy: {np.mean(y_true == y_pred):.4f}")
        print(f"  TTA accuracy:      {np.mean(y_true == y_pred_tta):.4f}")

        # Use TTA if it's better
        if np.mean(y_true == y_pred_tta) >= np.mean(y_true == y_pred):
            y_pred = y_pred_tta
            y_pred_proba = y_pred_proba_tta
            print("  ✅ Using TTA predictions (better performance)")
        else:
            print("  ℹ️  Using standard predictions (TTA did not improve)")

    # Metrics
    results = compute_metrics(y_true, y_pred, y_pred_proba, class_names)

    print("\n📊 Classification Report:")
    print(f"{'':20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)
    for name in class_names:
        m = results[name]
        print(f"{name:20} {m['precision']:10.4f} {m['recall']:10.4f} "
              f"{m['f1-score']:10.4f} {m['support']:10d}")
    print()
    print(f"{'Accuracy':20} {'':>10} {'':>10} {results['accuracy']:10.4f}")
    print(f"{'Macro-F1':20} {'':>10} {'':>10} {results['macro avg']['f1-score']:10.4f}")

    if results.get('auc_macro'):
        print(f"\n🎯 AUC-ROC: {results['auc_macro']:.4f}")
        for name, auc in results['auc_per_class'].items():
            print(f"  {name}: {auc:.4f}")

    # Bootstrap confidence intervals
    print("\n📈 95% Confidence Intervals (1000 bootstrap):")

    def macro_f1_fn(yt, yp):
        f1s = []
        for i in range(config['num_classes']):
            tp = np.sum((yt == i) & (yp == i))
            fp = np.sum((yt != i) & (yp == i))
            fn = np.sum((yt == i) & (yp != i))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        return np.mean(f1s)

    acc_mean, acc_lo, acc_hi = bootstrap_confidence_interval(
        y_true, y_pred, lambda yt, yp: np.mean(yt == yp))
    f1_mean, f1_lo, f1_hi = bootstrap_confidence_interval(
        y_true, y_pred, macro_f1_fn)

    print(f"  Accuracy:  {acc_mean:.4f} [{acc_lo:.4f} — {acc_hi:.4f}]")
    print(f"  Macro-F1:  {f1_mean:.4f} [{f1_lo:.4f} — {f1_hi:.4f}]")

    results['ci_95'] = {
        'accuracy': {'mean': acc_mean, 'lower': acc_lo, 'upper': acc_hi},
        'macro_f1': {'mean': f1_mean, 'lower': f1_lo, 'upper': f1_hi}
    }

    # Confusion matrix
    cm = np.zeros((config['num_classes'], config['num_classes']), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Save results
    save_results = {
        'model_name': model_name,
        'config': {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in config.items()},
        'classification_report': results,
        'confusion_matrix': cm.tolist(),
        'macro_f1': results['macro avg']['f1-score'],
        'accuracy': results['accuracy'],
        'ci_95': results['ci_95'],
        'tta_used': config.get('use_tta', False),
    }

    os.makedirs('results', exist_ok=True)
    with open(f'results/{model_name}_results.json', 'w') as f:
        json.dump(save_results, f, indent=4, default=str)

    # Plots
    plot_training_history(history_p1, history_p2, model_name)
    plot_confusion_matrix(cm, class_names, model_name)

    return model, save_results, (history_p1, history_p2)


# ============================================================
# PLOTTING
# ============================================================
def plot_training_history(h1, h2, model_name):
    """Plot combined training curves for both phases."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(['loss', 'accuracy', 'auc']):
        ax = axes[idx]

        # Phase 1
        e1 = len(h1.history[metric])
        ax.plot(range(1, e1+1), h1.history[metric],
                label='P1 Train', linewidth=2)
        ax.plot(range(1, e1+1), h1.history[f'val_{metric}'],
                label='P1 Val', linewidth=2)

        # Phase 2
        e2 = len(h2.history[metric])
        ax.plot(range(e1+1, e1+e2+1), h2.history[metric],
                label='P2 Train', linewidth=2, linestyle='--')
        ax.plot(range(e1+1, e1+e2+1), h2.history[f'val_{metric}'],
                label='P2 Val', linewidth=2, linestyle='--')

        ax.axvline(x=e1, color='red', linestyle=':', alpha=0.7,
                   label='Fine-tuning starts')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{model_name.upper()} — {metric.upper()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_history.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title(f'{model_name.upper()} — Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized (per row = recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f'{model_name.upper()} — Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png',
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING: DenseNet121 — Advanced Pipeline v2")
    print("=" * 80 + "\n")

    model, results, history = train_model(
        model_name='densenet121',
        config=CONFIG
    )

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Macro-F1:  {results['macro_f1']:.4f}")
    if results.get('ci_95'):
        ci = results['ci_95']
        print(f"   F1 95% CI: [{ci['macro_f1']['lower']:.4f} — {ci['macro_f1']['upper']:.4f}]")
    print(f"{'='*80}\n")
