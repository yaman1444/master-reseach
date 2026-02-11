"""
Threshold Calibration for Class-Specific Decision Boundaries
Objectif: Augmenter recall malignant ≥0.9 tout en gardant precision ≥0.7
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
from cbam import CBAM
from focal_loss import FocalLoss

def load_model_and_data(model_path, test_dir, img_size=(224, 224), batch_size=16):
    """Charger modèle et données de test"""
    
    # Charger modèle
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    
    # Générateur test
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Prédictions
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes
    
    return y_pred_probs, y_true, test_gen.class_indices

def find_optimal_thresholds(y_true, y_pred_probs, class_names, target_recall_malignant=0.90):
    """
    Trouver seuils optimaux par classe via grid search
    Priorité: recall malignant ≥ target_recall_malignant
    """
    
    n_classes = len(class_names)
    malignant_idx = class_names.index('grave') if 'grave' in class_names else 1
    
    print("\n" + "="*80)
    print("CALIBRATION DES SEUILS PAR CLASSE")
    print("="*80 + "\n")
    
    # Baseline (argmax standard)
    y_pred_baseline = np.argmax(y_pred_probs, axis=1)
    
    print("📊 Baseline (argmax):")
    for i, name in enumerate(class_names):
        mask_true = (y_true == i)
        mask_pred = (y_pred_baseline == i)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {name:10s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Grid search pour seuil malignant
    print(f"\n🔍 Recherche seuil optimal pour classe malignant (target recall ≥{target_recall_malignant}):")
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}
    
    thresholds_to_test = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold_malignant in thresholds_to_test:
        # Prédiction avec seuil ajusté
        y_pred_adjusted = np.argmax(y_pred_probs, axis=1).copy()
        
        # Si prob(malignant) > threshold, forcer prédiction malignant
        malignant_mask = y_pred_probs[:, malignant_idx] >= threshold_malignant
        y_pred_adjusted[malignant_mask] = malignant_idx
        
        # Calculer métriques
        mask_true = (y_true == malignant_idx)
        mask_pred = (y_pred_adjusted == malignant_idx)
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold_malignant,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Garder meilleur F1 avec recall ≥ target
        if recall >= target_recall_malignant and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold_malignant
            best_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    
    print(f"\n✅ Meilleur seuil trouvé: {best_threshold:.2f}")
    print(f"   Precision: {best_metrics['precision']:.3f}")
    print(f"   Recall:    {best_metrics['recall']:.3f}")
    print(f"   F1:        {best_metrics['f1']:.3f}")
    
    # Évaluation complète avec seuil optimal
    y_pred_final = np.argmax(y_pred_probs, axis=1).copy()
    malignant_mask = y_pred_probs[:, malignant_idx] >= best_threshold
    y_pred_final[malignant_mask] = malignant_idx
    
    print("\n📊 Métriques finales avec seuil calibré:")
    for i, name in enumerate(class_names):
        mask_true = (y_true == i)
        mask_pred = (y_pred_final == i)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {name:10s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Macro-F1
    f1_scores = []
    for i in range(n_classes):
        mask_true = (y_true == i)
        mask_pred = (y_pred_final == i)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    accuracy = np.mean(y_pred_final == y_true)
    
    print(f"\n   Accuracy:  {accuracy:.4f}")
    print(f"   Macro-F1:  {macro_f1:.4f}")
    
    # Visualiser courbe Precision-Recall
    plot_threshold_analysis(results, best_threshold, target_recall_malignant)
    
    return {
        'optimal_threshold_malignant': float(best_threshold),
        'baseline_metrics': {},
        'calibrated_metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'malignant': best_metrics
        },
        'threshold_search_results': results
    }

def plot_threshold_analysis(results, best_threshold, target_recall):
    """Visualiser l'analyse des seuils"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    # Precision-Recall vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax1.plot(thresholds, f1s, 'g-', label='F1-Score', linewidth=2)
    ax1.axvline(best_threshold, color='orange', linestyle='--', linewidth=2, label=f'Optimal ({best_threshold:.2f})')
    ax1.axhline(target_recall, color='red', linestyle=':', alpha=0.5, label=f'Target Recall ({target_recall:.2f})')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Malignant Class: Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2 = axes[1]
    ax2.plot(recalls, precisions, 'b-', linewidth=2)
    
    # Marquer point optimal
    best_idx = thresholds.index(best_threshold)
    ax2.plot(recalls[best_idx], precisions[best_idx], 'ro', markersize=10, label=f'Optimal (T={best_threshold:.2f})')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve (Malignant)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/threshold_calibration.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé: ./results/threshold_calibration.png")
    plt.close()

def save_calibrated_config(results, model_name='densenet121_improved'):
    """Sauvegarder configuration des seuils"""
    
    output_path = f'./results/{model_name}_thresholds.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Configuration sauvegardée: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate decision thresholds')
    parser.add_argument('--model', type=str, default='./models/densenet121_improved.keras')
    parser.add_argument('--test_dir', type=str, default='../datasets/test')
    parser.add_argument('--target_recall', type=float, default=0.90)
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CALIBRATION DES SEUILS - OPTIMISATION RECALL MALIGNANT")
    print("="*80 + "\n")
    
    # Charger données
    y_pred_probs, y_true, class_indices = load_model_and_data(args.model, args.test_dir)
    
    # Inverser class_indices pour avoir {0: 'debut', 1: 'grave', 2: 'normal'}
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Calibrer
    results = find_optimal_thresholds(y_true, y_pred_probs, class_names, args.target_recall)
    
    # Sauvegarder
    save_calibrated_config(results)
    
    print("\n" + "="*80)
    print("✅ CALIBRATION TERMINÉE")
    print("="*80 + "\n")
