"""
Script de comparaison des résultats entre différentes versions
"""
import json
from pathlib import Path
from tabulate import tabulate

def compare_results():
    """Compare les résultats de différents entraînements"""
    
    results_dir = Path('./results')
    
    # Fichiers de résultats à comparer
    result_files = [
        'densenet121_improved_results.json',
        'densenet121_optimized_results.json',
        'densenet121_results.json',  # Si existe
    ]
    
    all_results = []
    
    for filename in result_files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                model_name = filename.replace('_results.json', '').replace('densenet121_', '')
                
                row = {
                    'Model': model_name,
                    'Accuracy': f"{data['test_accuracy']:.4f}",
                    'Macro-F1': f"{data['macro_f1']:.4f}",
                    'Benign F1': f"{data['metrics_by_class']['benign']['f1']:.3f}",
                    'Malignant F1': f"{data['metrics_by_class']['malignant']['f1']:.3f}",
                    'Normal F1': f"{data['metrics_by_class']['normal']['f1']:.3f}",
                }
                
                all_results.append(row)
    
    if all_results:
        print("\n" + "="*80)
        print("COMPARAISON DES RÉSULTATS")
        print("="*80 + "\n")
        
        print(tabulate(all_results, headers='keys', tablefmt='grid'))
        
        print("\n" + "="*80)
        print("LÉGENDE")
        print("="*80)
        print("improved   : Version avec problèmes (LR trop faible, patience élevée)")
        print("optimized  : Version corrigée (LR augmenté, callbacks optimisés)")
        print("="*80 + "\n")
    else:
        print("❌ Aucun fichier de résultats trouvé dans ./results/")
        print("   Exécutez d'abord un entraînement.")

if __name__ == '__main__':
    try:
        compare_results()
    except ImportError:
        print("⚠️  Module 'tabulate' non installé")
        print("   Installation: pip install tabulate")
        print("\nAffichage simple:\n")
        
        # Fallback sans tabulate
        results_dir = Path('./results')
        for filename in ['densenet121_improved_results.json', 'densenet121_optimized_results.json']:
            filepath = results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    print(f"\n{filename}:")
                    print(f"  Accuracy: {data['test_accuracy']:.4f}")
                    print(f"  Macro-F1: {data['macro_f1']:.4f}")
