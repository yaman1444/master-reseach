"""
Script d'entraînement optimisé pour améliorer les performances du modèle BUSI
Objectif : >90% accuracy et >88% macro-F1
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

# Seed pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# Configuration optimisée
CONFIG = {
    'data_dir': '../datasets',
    'img_size': (224, 224),
    'batch_size': 16,
    'initial_epochs': 25,
    'fine_tune_epochs': 40,
    'initial_lr': 1e-3,  # Augmenté pour convergence plus rapide
    'fine_tune_lr': 1e-5,  # Augmenté pour permettre l'apprentissage
    'dropout_rate': 0.4,  # Réduit pour éviter sous-apprentissage
    'l2_reg': 5e-5,  # Réduit pour plus de flexibilité
}

def create_balanced_generators(data_dir, img_size, batch_size):
    """Générateurs avec augmentation forte et équilibrage des classes"""
    
    # Augmentation TRÈS forte pour réduire overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Augmenté
        width_shift_range=0.3,  # Augmenté
        height_shift_range=0.3,  # Augmenté
        shear_range=0.3,  # Augmenté
        zoom_range=0.3,  # Augmenté
        horizontal_flip=True,
        vertical_flip=True,  # Ajouté
        brightness_range=[0.7, 1.3],  # Ajouté
        fill_mode='reflect'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Calculer les poids de classes pour équilibrage
    train_dir = Path(data_dir) / 'train'
    class_counts = {}
    for class_name in ['debut', 'grave', 'normal']:
        class_dir = train_dir / class_name
        if class_dir.exists():
            class_counts[class_name] = len(list(class_dir.glob('*.png')))
    
    total = sum(class_counts.values())
    class_weights = {
        0: total / (3 * class_counts['debut']),  # benign
        1: total / (3 * class_counts['grave']),  # malignant
        2: total / (3 * class_counts['normal'])  # normal
    }
    
    print(f"\n📊 Poids des classes (pour équilibrage):")
    print(f"   Benign (debut):    {class_weights[0]:.3f}")
    print(f"   Malignant (grave): {class_weights[1]:.3f}")
    print(f"   Normal:            {class_weights[2]:.3f}\n")
    
    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    test_gen = test_datagen.flow_from_directory(
        str(Path(data_dir) / 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, test_gen, class_weights

def build_improved_model(img_size, num_classes, dropout_rate, l2_reg):
    """Modèle avec régularisation renforcée"""
    
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, 3)
    )
    
    # Freeze base
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    
    # CBAM attention
    x = CBAM(reduction_ratio=16)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers avec régularisation forte
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

def train_improved_model(config):
    """Entraînement avec stratégie optimisée"""
    
    print("="*80)
    print("ENTRAÎNEMENT OPTIMISÉ DENSENET121 - DATASET BUSI")
    print("="*80)
    
    # 1. Préparer les données
    train_gen, test_gen, class_weights = create_balanced_generators(
        config['data_dir'],
        config['img_size'],
        config['batch_size']
    )
    
    # 2. Construire le modèle
    model, base_model = build_improved_model(
        config['img_size'],
        num_classes=3,
        dropout_rate=config['dropout_rate'],
        l2_reg=config['l2_reg']
    )
    
    # 3. Phase 1 : Entraîner la tête
    print("\n" + "="*80)
    print("PHASE 1 : Entraînement de la tête (base frozen)")
    print("="*80 + "\n")
    
    model.compile(
        optimizer=keras.optimizers.Adam(config['initial_lr']),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    callbacks_phase1 = [
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
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            './models/densenet121_phase1_best.keras',
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
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # 4. Phase 2 : Fine-tuning
    print("\n" + "="*80)
    print("PHASE 2 : Fine-tuning (top 20% unfrozen)")
    print("="*80 + "\n")
    
    # Unfreeze top 20% seulement
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.8)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"Layers dégelés : {total_layers - freeze_until}/{total_layers}\n")
    
    model.compile(
        optimizer=keras.optimizers.Adam(config['fine_tune_lr']),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            mode='max',
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            './models/densenet121_improved.keras',
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
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # 5. Évaluation finale
    print("\n" + "="*80)
    print("ÉVALUATION FINALE")
    print("="*80 + "\n")
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    
    # Prédictions pour métriques détaillées
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Calculer métriques par classe
    from collections import defaultdict
    metrics = defaultdict(dict)
    
    for i, class_name in enumerate(['benign', 'malignant', 'normal']):
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    print(f"📊 Résultats finaux:")
    print(f"   Accuracy:  {test_acc:.4f}")
    print(f"   Macro-F1:  {macro_f1:.4f}\n")
    
    print("📈 Métriques par classe:")
    for class_name, m in metrics.items():
        print(f"   {class_name:10s}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f}")
    
    # Sauvegarder résultats
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'macro_f1': float(macro_f1),
        'metrics_by_class': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
        'config': config
    }
    
    with open('./results/densenet121_improved_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Graphiques
    plot_training_history(history1, history2)
    
    print(f"\n✅ Modèle sauvegardé : ./models/densenet121_improved.keras")
    print(f"✅ Résultats sauvegardés : ./results/densenet121_improved_results.json\n")
    
    print("="*80 + "\n")
    
    return model, results

def plot_training_history(history1, history2):
    """Visualiser l'historique d'entraînement"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Combiner les historiques
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    # Accuracy
    axes[0].plot(epochs, acc, 'b-', label='Train')
    axes[0].plot(epochs, val_acc, 'r-', label='Validation')
    axes[0].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning start')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(epochs, loss, 'b-', label='Train')
    axes[1].plot(epochs, val_loss, 'r-', label='Validation')
    axes[1].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning start')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/densenet121_improved_history.png', dpi=150, bbox_inches='tight')
    print(f"✅ Graphiques sauvegardés : ./results/densenet121_improved_history.png")

if __name__ == '__main__':
    # Créer dossiers
    Path('./models').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    
    # Entraîner
    model, results = train_improved_model(CONFIG)
