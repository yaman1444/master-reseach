"""
K-Fold Cross-Validation pour validation robuste
Objectif: Moyenne ± écart-type de macro-F1 et AUC
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import sys

sys.path.append(str(Path(__file__).parent))
from focal_loss import FocalLoss
from cbam import CBAM

np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(data_dir):
    """Charger dataset complet pour k-fold"""
    
    from PIL import Image
    
    images = []
    labels = []
    
    class_map = {'debut': 0, 'grave': 1, 'normal': 2}
    
    for class_name, class_idx in class_map.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            continue
        
        for img_path in class_dir.glob('*.png'):
            try:
                img = Image.open(img_path).resize((224, 224))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_idx)
            except:
                continue
    
    return np.array(images), np.array(labels)

def build_model(img_size=(224, 224)):
    """Construire modèle DenseNet121 + CBAM"""
    
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

def run_kfold_validation(data_dir, n_splits=5, epochs=15):
    """
    K-fold cross-validation stratifiée
    
    Args:
        data_dir: Répertoire des données (train/)
        n_splits: Nombre de folds
        epochs: Epochs par fold
    
    Returns:
        Dict avec moyennes et écarts-types
    """
    
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
    print("="*80 + "\n")
    
    # Charger données
    print("Chargement du dataset...")
    X, y = load_dataset(data_dir)
    print(f"Dataset: {len(X)} images, {len(np.unique(y))} classes")
    
    # K-fold stratifié
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_splits}")
        print(f"{'='*80}\n")
        
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # One-hot encode
        y_train_cat = keras.utils.to_categorical(y_train, 3)
        y_val_cat = keras.utils.to_categorical(y_val, 3)
        
        # Class weights
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total / (3 * count) for i, count in enumerate(class_counts)}
        
        # Modèle
        model = build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
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
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=16,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Métriques
        accuracy = np.mean(y_pred == y_val)
        
        # F1 par classe
        f1_scores = []
        for i in range(3):
            mask_true = (y_val == i)
            mask_pred = (y_pred == i)
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores)
        
        # AUC
        try:
            auc = roc_auc_score(y_val_cat, y_pred_probs, average='macro', multi_class='ovr')
        except:
            auc = 0.0
        
        fold_results.append({
            'fold': fold_idx,
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'auc': float(auc),
            'f1_benign': float(f1_scores[0]),
            'f1_malignant': float(f1_scores[1]),
            'f1_normal': float(f1_scores[2])
        })
        
        print(f"\nFold {fold_idx} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Macro-F1:  {macro_f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
    
    # Statistiques globales
    print("\n" + "="*80)
    print("RÉSULTATS K-FOLD")
    print("="*80 + "\n")
    
    accuracies = [r['accuracy'] for r in fold_results]
    macro_f1s = [r['macro_f1'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    
    summary = {
        'n_folds': n_splits,
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'macro_f1_std': float(np.std(macro_f1s)),
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'fold_results': fold_results
    }
    
    print(f"Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Macro-F1:  {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    
    # Sauvegarder
    with open('./results/kfold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Résultats sauvegardés: ./results/kfold_summary.json")
    
    return summary

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation')
    parser.add_argument('--data_dir', type=str, default='../datasets/train')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=15)
    
    args = parser.parse_args()
    
    Path('./results').mkdir(exist_ok=True)
    
    summary = run_kfold_validation(args.data_dir, args.n_folds, args.epochs)
    
    print("\n" + "="*80)
    print("✅ K-FOLD VALIDATION TERMINÉE")
    print("="*80 + "\n")
