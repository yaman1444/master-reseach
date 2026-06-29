"""
Fusionner le dataset BUSI original avec le dataset RSNA pré-labellisé
Créer un dataset massif pour réentraînement
"""
import shutil
from pathlib import Path
from collections import defaultdict

def merge_datasets(busi_path, rsna_path, output_path):
    """
    Fusionner BUSI + RSNA labellisé
    
    Args:
        busi_path: datasets/train et datasets/test (BUSI original)
        rsna_path: datasets/rsna_labeled/train et test
        output_path: datasets/merged/train et test
    """
    
    print("="*80)
    print("FUSION DATASETS BUSI + RSNA")
    print("="*80 + "\n")
    
    busi_path = Path(busi_path)
    rsna_path = Path(rsna_path)
    output_path = Path(output_path)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    # Créer la structure de sortie
    for split in ['train', 'test']:
        for class_name in ['benign', 'malignant', 'normal']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # 1. Copier BUSI
    print("📦 Copie du dataset BUSI...")
    for split in ['train', 'test']:
        for class_name in ['benign', 'malignant', 'normal']:
            # Mapper les noms de classes BUSI
            busi_class = {
                'benign': 'debut',
                'malignant': 'grave',
                'normal': 'normal'
            }[class_name]
            
            source_folder = busi_path / split / busi_class
            dest_folder = output_path / split / class_name
            
            if source_folder.exists():
                for img_file in source_folder.glob('*.png'):
                    # Préfixer avec "busi_" pour traçabilité
                    new_name = f"busi_{img_file.name}"
                    shutil.copy2(img_file, dest_folder / new_name)
                    stats[split][class_name] += 1
    
    print(f"   ✅ BUSI copié: {sum(stats['train'].values())} train, {sum(stats['test'].values())} test\n")
    
    # 2. Copier RSNA labellisé
    print("📦 Copie du dataset RSNA labellisé...")
    rsna_stats = defaultdict(lambda: defaultdict(int))
    
    for split in ['train', 'test']:
        for class_name in ['benign', 'malignant', 'normal']:
            source_folder = rsna_path / split / class_name
            dest_folder = output_path / split / class_name
            
            if source_folder.exists():
                for img_file in source_folder.glob('*.png'):
                    # Préfixer avec "rsna_" pour traçabilité
                    new_name = f"rsna_{img_file.name}"
                    shutil.copy2(img_file, dest_folder / new_name)
                    rsna_stats[split][class_name] += 1
    
    print(f"   ✅ RSNA copié: {sum(rsna_stats['train'].values())} train, {sum(rsna_stats['test'].values())} test\n")
    
    # 3. Statistiques finales
    print("="*80)
    print("STATISTIQUES DATASET FUSIONNÉ")
    print("="*80 + "\n")
    
    total_train = sum(stats['train'].values()) + sum(rsna_stats['train'].values())
    total_test = sum(stats['test'].values()) + sum(rsna_stats['test'].values())
    
    print("📊 TRAIN:")
    for class_name in ['benign', 'malignant', 'normal']:
        busi_count = stats['train'][class_name]
        rsna_count = rsna_stats['train'][class_name]
        total = busi_count + rsna_count
        print(f"   {class_name:10s}: {total:5d} images (BUSI: {busi_count:4d} + RSNA: {rsna_count:4d})")
    print(f"   {'TOTAL':10s}: {total_train:5d} images\n")
    
    print("📊 TEST:")
    for class_name in ['benign', 'malignant', 'normal']:
        busi_count = stats['test'][class_name]
        rsna_count = rsna_stats['test'][class_name]
        total = busi_count + rsna_count
        print(f"   {class_name:10s}: {total:5d} images (BUSI: {busi_count:4d} + RSNA: {rsna_count:4d})")
    print(f"   {'TOTAL':10s}: {total_test:5d} images\n")
    
    print(f"🎯 DATASET TOTAL: {total_train + total_test} images")
    print(f"   Train: {total_train} ({total_train/(total_train+total_test)*100:.1f}%)")
    print(f"   Test:  {total_test} ({total_test/(total_train+total_test)*100:.1f}%)")
    
    print(f"\n📁 Dataset fusionné sauvegardé: {output_path}")
    
    print("\n" + "="*80)
    print("PROCHAINE ÉTAPE")
    print("="*80)
    print("\n🚀 RÉENTRAÎNER LE MODÈLE SUR LE DATASET MASSIF:")
    print(f"   cd scripts")
    print(f"   python train_advanced.py")
    print(f"\n   (Modifiez CONFIG['data_dir'] = '../datasets/merged' dans train_advanced.py)")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    # Chemins
    BUSI_PATH = Path('../datasets')  # Contient train/ et test/
    RSNA_PATH = Path('../datasets/rsna_labeled')
    OUTPUT_PATH = Path('../datasets/merged')
    
    # Vérifications
    if not BUSI_PATH.exists():
        print(f"❌ Dataset BUSI non trouvé: {BUSI_PATH}")
        exit(1)
    
    if not RSNA_PATH.exists():
        print(f"❌ Dataset RSNA labellisé non trouvé: {RSNA_PATH}")
        print("💡 Exécutez d'abord: python auto_label_rsna.py")
        exit(1)
    
    # Fusionner
    merge_datasets(BUSI_PATH, RSNA_PATH, OUTPUT_PATH)
