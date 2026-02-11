"""
Ablation Study: Évaluation systématique de chaque composant
Tests: baseline, +augmentation, +dropout, +CBAM, +FocalLoss, +class_weight
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from focal_loss import FocalLoss
from cbam import CBAM

np.random.seed(42)
tf.random.set_seed(42)

def build_model(use_cbam=True, dropout_rate=0.5, img_size=(224, 224)):
    """Construire modèle avec options ablation"""
    
    base_model = DenseNet121(include_top=False, weights='imagenet', 
                             input_shape=(*img_size, 3))
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    
    if use_cbam:
        x = CBAM(reduction_ratio=16)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(3, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

def create_generators(train_dir, test_dir, use_augmentation=True, img_size=(224, 224), batch_size=16):
    """Créer générateurs avec/sans augmentation"""
    
    if use_augmentation:
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
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=True, seed=42
    )
    
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    
    # Class weights
    class_counts = {}
    for class_name in ['debut', 'grave', 'normal']:
        class_dir = Path(train_dir) / class_name
        if class_dir.exists():
            class_counts[class_name] = len(list(class_dir.glob('*.png')))
    
    total = sum(class_counts.values())
    class_weights = {
        0: total / (3 * class_counts['debut']),
        1: total / (3 * class_counts['grave']),
        2: total / (3 * class_counts['normal'])
    }
    
    return train_gen, test_gen, class_weights

def train_ablation_config(config_name, use_cbam, use_augmentation, dropout_rate, 
                         use_focal_loss, use_class_weight, train_dir, test_dir):
    """Entraîner une configuration d'ablation"""
    
    print(f"\n{'='*80}")
    print(f"ABLATION: {config_name}")
    print(f"{'='*80}\n")
    
    # Générateurs
    train_gen, test_gen, class_weights = create_generators(
        train_dir, test_dir, use_augmentation
    )
    
    # Modèle
    model = build_model(use_cbam, dropout_rate)
    
    # Loss
    loss = FocalLoss(gamma=2.0, alpha=0.25) if use_focal_loss else 'categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, 
                                     restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, 
                                         patience=3, mode='max', min_lr=1e-6)
    ]
    
    # Entraînement
    history = model.fit(
        train_gen,
        epochs=15,
        validation_data=test_gen,
        class_weight=class_weights if use_class_weight else None,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Métriques
    metrics = {}
    for i, class_name in enumerate(['benign', 'malignant', 'normal']):
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    accuracy = np.mean(y_pred == y_true)
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    print(f"\nRésultats {config_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()}
    }

def run_ablation_study(train_dir='../datasets/train/', test_dir='../datasets/test/'):
    """
    Étude d'ablation systématique
    
    Configurations:
    1. Baseline: Rien
    2. +Augmentation
    3. +Dropout
    4. +CBAM
    5. +FocalLoss
    6. +ClassWeight (Full)
    """
    
    configs = [
        ('baseline', False, False, 0.0, False, False),
        ('+augmentation', False, True, 0.0, False, False),
        ('+dropout', False, True, 0.5, False, False),
        ('+cbam', True, True, 0.5, False, False),
        ('+focal_loss', True, True, 0.5, True, False),
        ('+class_weight', True, True, 0.5, True, True),
    ]
    
    results = {}
    
    for config_name, use_cbam, use_aug, dropout, use_focal, use_cw in configs:
        try:
            result = train_ablation_config(
                config_name, use_cbam, use_aug, dropout, use_focal, use_cw,
                train_dir, test_dir
            )
            results[config_name] = result
        except Exception as e:
            print(f"❌ Erreur {config_name}: {e}")
            continue
    
    # Générer rapport
    generate_ablation_report(results)
    
    return results

def generate_ablation_report(results):
    """Générer rapport d'ablation"""
    
    rows = []
    for config_name, result in results.items():
        row = {
            'Configuration': config_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Macro-F1': f"{result['macro_f1']:.4f}",
            'Benign-F1': f"{result['metrics']['benign']['f1']:.4f}",
            'Malignant-F1': f"{result['metrics']['malignant']['f1']:.4f}",
            'Normal-F1': f"{result['metrics']['normal']['f1']:.4f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sauvegarder
    df.to_csv('results/ablation_densenet121.csv', index=False)
    
    print("\n" + "="*100)
    print("RÉSULTATS ABLATION STUDY")
    print("="*100 + "\n")
    print(df.to_string(index=False))
    
    # Markdown
    with open('results/ablation_densenet121.md', 'w') as f:
        f.write("# Ablation Study - DenseNet121\n\n")
        f.write("## Contribution de Chaque Composant\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Gains Incrémentaux\n\n")
        
        if 'baseline' in results:
            baseline_f1 = results['baseline']['macro_f1']
            for config_name, result in results.items():
                if config_name != 'baseline':
                    gain = (result['macro_f1'] - baseline_f1) * 100
                    f.write(f"- **{config_name}**: +{gain:.2f}% macro-F1\n")
    
    print(f"\n✅ Rapport sauvegardé: results/ablation_densenet121.md")
    
    # Graphique
    plot_ablation_results(results)

def plot_ablation_results(results):
    """Visualiser résultats ablation"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    configs = list(results.keys())
    macro_f1s = [results[c]['macro_f1'] for c in configs]
    
    # Gains incrémentaux
    baseline_f1 = macro_f1s[0] if configs else 0
    gains = [(f1 - baseline_f1) * 100 for f1 in macro_f1s]
    
    colors = ['gray' if i == 0 else 'green' if g > 0 else 'red' for i, g in enumerate(gains)]
    bars = ax.bar(configs, gains, alpha=0.8, color=colors)
    
    ax.set_ylabel('Macro-F1 Gain vs Baseline (%)', fontsize=12)
    ax.set_title('Ablation Study: Incremental Gains', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Labels
    for bar, gain in zip(bars, gains):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gain:+.2f}%', ha='center', 
               va='bottom' if gain >= 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/ablation_densenet121_plot.png', dpi=150, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: results/ablation_densenet121_plot.png")
    plt.close()

if __name__ == '__main__':
    
    Path('./results').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("ABLATION STUDY - DENSENET121")
    print("="*80 + "\n")
    
    results = run_ablation_study(
        train_dir='../datasets/train/',
        test_dir='../datasets/test/'
    )
    
    print("\n" + "="*80)
    print("✅ ABLATION STUDY TERMINÉE")
    print("="*80 + "\n")
