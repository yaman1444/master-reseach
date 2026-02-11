"""
Script d'entraînement OPTIMISÉ avec corrections des problèmes identifiés
Objectif : >90% accuracy, convergence rapide, pas de stagnation
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from focal_loss import FocalLoss
from cbam import CBAM

# Seed
np.random.seed(42)
tf.random.set_seed(42)

# Configuration CORRIGÉE
CONFIG = {
    'data_dir': '../datasets',
    'img_size': (224, 224),
    'batch_size': 16,
    'initial_epochs': 20,
    'fine_tune_epochs': 30,
    'initial_lr': 1e-3,
    'fine_tune_lr': 1e-5,
    'dropout_rate': 0.4,
    'l2_reg': 5e-5,
}

def create_generators(data_dir, img_size, batch_size):
    """Générateurs avec augmentation équilibrée"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_dir = Path(data_dir) / 'train'
    test_dir = Path(data_dir) / 'test'
    
    # Calculer class weights
    class_counts = {}
    for class_name in ['debut', 'grave', 'normal']:
        class_dir = train_dir / class_name
        if class_dir.exists():
            class_counts[class_name] = len(list(class_dir.glob('*.png')))
    
    total = sum(class_counts.values())
    class_weights = {
        0: total / (3 * class_counts['debut']),
        1: total / (3 * class_counts['grave']),
        2: total / (3 * class_counts['normal'])
    }
    
    print(f"\n📊 Class weights:")
    print(f"   Benign:    {class_weights[0]:.3f}")
    print(f"   Malignant: {class_weights[1]:.3f}")
    print(f"   Normal:    {class_weights[2]:.3f}\n")
    
    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    test_gen = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, test_gen, class_weights

def build_model(img_size, dropout_rate, l2_reg):
    """Modèle DenseNet121 + CBAM"""
    
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, 3)
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = CBAM(reduction_ratio=16)(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

def train_model(config):
    """Entraînement en 2 phases"""
    
    print("="*80)
    print("ENTRAÎNEMENT DENSENET121 - VERSION OPTIMISÉE")
    print("="*80)
    
    # Données
    train_gen, test_gen, class_weights = create_generators(
        config['data_dir'],
        config['img_size'],
        config['batch_size']
    )
    
    # Modèle
    model, base_model = build_model(
        config['img_size'],
        config['dropout_rate'],
        config['l2_reg']
    )
    
    # PHASE 1: Entraîner la tête
    print("\n" + "="*80)
    print("PHASE 1: Training head (base frozen)")
    print("="*80 + "\n")
    
    model.compile(
        optimizer=keras.optimizers.Adam(config['initial_lr']),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    callbacks_p1 = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            mode='max',
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            './models/densenet121_phase1.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history1 = model.fit(
        train_gen,
        epochs=config['initial_epochs'],
        validation_data=test_gen,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1
    )
    
    # PHASE 2: Fine-tuning
    print("\n" + "="*80)
    print("PHASE 2: Fine-tuning (top 20% unfrozen)")
    print("="*80 + "\n")
    
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable = sum([1 for l in base_model.layers if l.trainable])
    print(f"Unfrozen layers: {trainable}/{total_layers}\n")
    
    model.compile(
        optimizer=keras.optimizers.Adam(config['fine_tune_lr']),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    callbacks_p2 = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            mode='max',
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            './models/densenet121_optimized.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_gen,
        epochs=config['fine_tune_epochs'],
        validation_data=test_gen,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1
    )
    
    # Évaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Métriques par classe
    metrics = {}
    for i, class_name in enumerate(['benign', 'malignant', 'normal']):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    print(f"📊 Final Results:")
    print(f"   Accuracy:  {test_acc:.4f}")
    print(f"   Macro-F1:  {macro_f1:.4f}\n")
    
    print("📈 Per-class metrics:")
    for class_name, m in metrics.items():
        print(f"   {class_name:10s}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
    
    # Sauvegarder
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'macro_f1': float(macro_f1),
        'metrics_by_class': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
        'config': config
    }
    
    with open('./results/densenet121_optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Graphiques
    plot_history(history1, history2)
    
    print(f"\n✅ Model saved: ./models/densenet121_optimized.keras")
    print(f"✅ Results saved: ./results/densenet121_optimized_results.json\n")
    print("="*80 + "\n")
    
    return model, results

def plot_history(history1, history2):
    """Plot training curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    # Accuracy
    axes[0].plot(epochs, acc, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
    axes[0].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning', linewidth=2)
    axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(epochs, loss, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    axes[1].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning', linewidth=2)
    axes[1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/densenet121_optimized_history.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plots saved: ./results/densenet121_optimized_history.png")

if __name__ == '__main__':
    Path('./models').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    
    model, results = train_model(CONFIG)
