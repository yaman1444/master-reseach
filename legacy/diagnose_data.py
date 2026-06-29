"""
Script de diagnostic pour identifier les problèmes de données
"""
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset(data_dir):
    """Analyser la qualité et distribution du dataset"""
    
    print("="*80)
    print("DIAGNOSTIC DU DATASET BUSI")
    print("="*80 + "\n")
    
    train_dir = Path(data_dir) / 'train'
    test_dir = Path(data_dir) / 'test'
    
    # 1. Vérifier la structure
    print("📁 Structure des dossiers:")
    for split in ['train', 'test']:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            print(f"   ❌ {split}/ manquant!")
            continue
        
        print(f"\n   {split}/")
        for class_name in ['debut', 'grave', 'normal']:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.png')))
                print(f"      {class_name:10s}: {count:4d} images")
            else:
                print(f"      {class_name:10s}: ❌ MANQUANT")
    
    # 2. Analyser les images
    print("\n\n📊 Analyse des images:")
    
    all_sizes = []
    all_means = []
    all_stds = []
    
    for split in ['train', 'test']:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
            
        for class_name in ['debut', 'grave', 'normal']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in list(class_dir.glob('*.png'))[:10]:  # Sample 10 images
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    all_sizes.append(img_array.shape[:2])
                    all_means.append(img_array.mean())
                    all_stds.append(img_array.std())
                except Exception as e:
                    print(f"   ⚠️  Erreur lecture {img_path.name}: {e}")
    
    if all_sizes:
        size_counter = Counter([str(s) for s in all_sizes])
        print(f"\n   Tailles d'images (top 3):")
        for size, count in size_counter.most_common(3):
            print(f"      {size}: {count} images")
        
        print(f"\n   Statistiques pixel (échantillon):")
        print(f"      Moyenne: {np.mean(all_means):.2f} ± {np.std(all_means):.2f}")
        print(f"      Std:     {np.mean(all_stds):.2f} ± {np.std(all_stds):.2f}")
    
    # 3. Vérifier le déséquilibre
    print("\n\n⚖️  Déséquilibre des classes:")
    
    train_counts = {}
    for class_name in ['debut', 'grave', 'normal']:
        class_dir = train_dir / class_name
        if class_dir.exists():
            train_counts[class_name] = len(list(class_dir.glob('*.png')))
    
    if train_counts:
        total = sum(train_counts.values())
        max_count = max(train_counts.values())
        
        for class_name, count in train_counts.items():
            ratio = count / total * 100
            imbalance = max_count / count
            print(f"   {class_name:10s}: {count:4d} ({ratio:5.1f}%) - Imbalance ratio: {imbalance:.2f}x")
    
    # 4. Recommandations
    print("\n\n💡 Recommandations:")
    
    if train_counts:
        max_imbalance = max(train_counts.values()) / min(train_counts.values())
        if max_imbalance > 3:
            print("   ⚠️  DÉSÉQUILIBRE SÉVÈRE détecté (>3x)")
            print("      → Utiliser class_weights")
            print("      → Augmenter l'augmentation de données")
            print("      → Considérer SMOTE ou oversampling")
        
        if total < 1000:
            print("   ⚠️  DATASET PETIT (<1000 images)")
            print("      → Augmentation de données FORTE requise")
            print("      → Dropout élevé (0.5-0.6)")
            print("      → Fine-tuning progressif")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    analyze_dataset('../datasets')
