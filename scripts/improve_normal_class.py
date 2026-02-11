"""
Amélioration classe normal: Augmentation spécifique + analyse
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from focal_loss import FocalLoss
from cbam import CBAM

np.random.seed(42)
tf.random.set_seed(42)

def create_generators_normal_boost(train_dir, test_dir, img_size=(224, 224), batch_size=16):
    """Générateurs avec boost pour classe normal"""
    
    # Augmentation standard
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],  # Moins agressif pour normal
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=True, seed=42
    )
    
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    
    # Class weights avec boost normal
    class_counts = {}
    for class_name in ['debut', 'grave', 'normal']:
        class_dir = Path(train_dir) / class_name
        if class_dir.exists():
            class_counts[class_name] = len(list(class_dir.glob('*.png')))
    
    total = sum(class_counts.values())
    
    # Augmenter poids normal
    class_weights = {
        0: total / (3 * class_counts['debut']),
        1: total / (3 * class_counts['grave']),
        2: total / (2.5 * class_counts['normal'])  # Boost normal
    }
    
    print(f"Class weights (normal boosted):")
    print(f"  Benign:    {class_weights[0]:.3f}")
    print(f"  Malignant: {class_weights[1]:.3f}")
    print(f"  Normal:    {class_weights[2]:.3f} (boosted)")
    
    return train_gen, test_gen, class_weights

def build_model(img_size=(224, 224)):
    """Modèle DenseNet121 + CBAM"""
    base_model = DenseNet121(include_top=False, weights='imagenet', 
                             input_shape=(*img_size, 3))
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = CBAM(reduction_ratio=16)(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(3, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

def train_normal_boost(train_dir='../datasets/train/', test_dir='../datasets/test/', epochs=20):
    """Entraîner avec boost normal"""
    
    print("\n" + "="*80)
    print("ENTRAÎNEMENT AVEC BOOST CLASSE NORMAL")
    print("="*80 + "\n")
    
    # Générateurs
    train_gen, test_gen, class_weights = create_generators_normal_boost(train_dir, test_dir)
    
    # Modèle
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, 
                                     restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, 
                                         patience=4, mode='max', min_lr=1e-6),
        keras.callbacks.ModelCheckpoint('./models/densenet121_normal_boost.keras',
                                       monitor='val_accuracy', save_best_only=True, mode='max')
    ]
    
    # Entraînement
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Métriques par classe
    print("\n" + "="*80)
    print("RÉSULTATS AVEC BOOST NORMAL")
    print("="*80 + "\n")
    
    for i, class_name in enumerate(['benign', 'malignant', 'normal']):
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:10s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    accuracy = np.mean(y_pred == y_true)
    macro_f1 = np.mean([
        2 * (tp / (tp + fp) if (tp + fp) > 0 else 0) * (tp / (tp + fn) if (tp + fn) > 0 else 0) / 
        ((tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0))
        if ((tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0)) > 0 else 0
        for i in range(3)
        for mask_true in [(y_true == i)]
        for mask_pred in [(y_pred == i)]
        for tp in [np.sum(mask_true & mask_pred)]
        for fp in [np.sum(~mask_true & mask_pred)]
        for fn in [np.sum(mask_true & ~mask_pred)]
    ])
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    
    return model

if __name__ == '__main__':
    Path('./models').mkdir(exist_ok=True)
    train_normal_boost()
