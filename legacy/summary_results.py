"""
Résumé rapide de tous les résultats disponibles
"""
import json
from pathlib import Path

results_dir = Path('./results')

print("\n" + "="*80)
print("RÉSUMÉ DES RÉSULTATS DISPONIBLES")
print("="*80 + "\n")

# Résultats modèles
result_files = list(results_dir.glob('*_results.json'))

if result_files:
    print("📊 MODÈLES ENTRAÎNÉS:\n")
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            model_name = result_file.stem.replace('_results', '')
            acc = data.get('test_accuracy', 0)
            f1 = data.get('macro_f1', 0)
            
            print(f"  {model_name}:")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Macro-F1:  {f1:.4f}")
            
            if 'metrics_by_class' in data:
                metrics = data['metrics_by_class']
                for cls, m in metrics.items():
                    print(f"      {cls:10s}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
            print()
        except:
            pass

# Calibration
threshold_file = results_dir / 'densenet121_improved_thresholds.json'
if threshold_file.exists():
    print("🎯 CALIBRATION DES SEUILS:\n")
    try:
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        
        threshold = data.get('optimal_threshold_malignant', 0)
        metrics = data.get('calibrated_metrics', {})
        
        print(f"  Seuil optimal malignant: {threshold:.3f}")
        print(f"  Accuracy calibrée:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Macro-F1 calibré:   {metrics.get('macro_f1', 0):.4f}")
        
        if 'malignant' in metrics:
            m = metrics['malignant']
            print(f"  Malignant: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
        print()
    except:
        pass

# K-fold
kfold_file = results_dir / 'kfold_summary.json'
if kfold_file.exists():
    print("📈 K-FOLD VALIDATION:\n")
    try:
        with open(kfold_file, 'r') as f:
            data = json.load(f)
        
        print(f"  Accuracy:  {data['accuracy_mean']:.4f} ± {data['accuracy_std']:.4f}")
        print(f"  Macro-F1:  {data['macro_f1_mean']:.4f} ± {data['macro_f1_std']:.4f}")
        print(f"  AUC:       {data['auc_mean']:.4f} ± {data['auc_std']:.4f}")
        print()
    except:
        pass

# Ablation
ablation_file = results_dir / 'ablation_densenet121.csv'
if ablation_file.exists():
    print("🔬 ABLATION STUDY:\n")
    try:
        import pandas as pd
        df = pd.read_csv(ablation_file)
        print(df.to_string(index=False))
        print()
    except:
        pass

print("="*80 + "\n")

# Fichiers disponibles
print("📁 FICHIERS GÉNÉRÉS:\n")
for f in sorted(results_dir.glob('*')):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} ({size_kb:6.1f} KB)")

print("\n" + "="*80 + "\n")
