"""
Script pour analyser et préparer le dataset RSNA
Structure: rsna_bitirme/[numero]/[LCC, LMLO, RCC, RMLO]/*.png
"""
import os
import shutil
from pathlib import Path
import pandas as pd
from collections import defaultdict

def analyze_rsna_structure(rsna_path):
    """Analyser la structure du dataset RSNA"""
    print("\n" + "="*80)
    print("ANALYSE DU DATASET RSNA")
    print("="*80 + "\n")
    
    rsna_path = Path(rsna_path)
    
    if not rsna_path.exists():
        print(f"❌ Erreur: Le dossier {rsna_path} n'existe pas")
        return None
    
    # Statistiques
    stats = {
        'total_folders': 0,
        'total_images': 0,
        'by_view': defaultdict(int),
        'folders_list': []
    }
    
    # Parcourir les dossiers numérotés
    for folder in sorted(rsna_path.iterdir()):
        if folder.is_dir():
            stats['total_folders'] += 1
            folder_info = {
                'folder_name': folder.name,
                'images': defaultdict(list)
            }
            
            # Parcourir les sous-dossiers (LCC, LMLO, RCC, RMLO)
            for view_folder in folder.iterdir():
                if view_folder.is_dir():
                    view_name = view_folder.name
                    
                    # Compter les images
                    images = list(view_folder.glob('*.png')) + \
                            list(view_folder.glob('*.jpg')) + \
                            list(view_folder.glob('*.jpeg'))
                    
                    if images:
                        stats['by_view'][view_name] += len(images)
                        stats['total_images'] += len(images)
                        folder_info['images'][view_name] = [img.name for img in images]
            
            if folder_info['images']:
                stats['folders_list'].append(folder_info)
    
    # Afficher les statistiques
    print(f"📁 Dossiers patients: {stats['total_folders']}")
    print(f"🖼️  Total images: {stats['total_images']}")
    print(f"\n📊 Répartition par vue:")
    for view, count in sorted(stats['by_view'].items()):
        print(f"   {view}: {count} images")
    
    print(f"\n✅ Premiers dossiers analysés: {min(5, len(stats['folders_list']))}")
    for i, folder_info in enumerate(stats['folders_list'][:5]):
        print(f"\n   Dossier {folder_info['folder_name']}:")
        for view, images in folder_info['images'].items():
            print(f"      {view}: {len(images)} images")
    
    return stats

def check_for_labels(rsna_path):
    """Chercher un fichier CSV avec les labels"""
    print("\n" + "="*80)
    print("RECHERCHE DE FICHIER DE LABELS")
    print("="*80 + "\n")
    
    rsna_path = Path(rsna_path)
    parent_path = rsna_path.parent
    
    # Chercher des fichiers CSV
    csv_files = list(parent_path.glob('*.csv'))
    
    if csv_files:
        print(f"✅ Fichiers CSV trouvés:")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
            
            # Lire les premières lignes
            try:
                df = pd.read_csv(csv_file, nrows=5)
                print(f"     Colonnes: {list(df.columns)}")
                print(f"     Lignes: {len(pd.read_csv(csv_file))}")
            except Exception as e:
                print(f"     Erreur lecture: {e}")
        
        return csv_files
    else:
        print("❌ Aucun fichier CSV trouvé")
        print("⚠️  Sans labels, impossible de classifier les images")
        return None

def create_organized_dataset(rsna_path, output_path, labels_csv=None):
    """
    Organiser le dataset RSNA en structure train/test/benign/malignant/normal
    
    Args:
        rsna_path: Chemin vers rsna_bitirme/
        output_path: Chemin de sortie (ex: datasets/rsna_organized/)
        labels_csv: Fichier CSV avec les labels (optionnel)
    """
    print("\n" + "="*80)
    print("ORGANISATION DU DATASET")
    print("="*80 + "\n")
    
    rsna_path = Path(rsna_path)
    output_path = Path(output_path)
    
    # Créer la structure de sortie
    for split in ['train', 'test']:
        for category in ['benign', 'malignant', 'normal']:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
    
    if labels_csv:
        print("📋 Utilisation du fichier de labels...")
        # TODO: Implémenter avec le CSV
        print("⚠️  Fonction à implémenter avec votre fichier CSV spécifique")
    else:
        print("⚠️  Pas de labels - Organisation manuelle requise")
        print("\nOptions:")
        print("1. Fournir un fichier CSV avec les labels")
        print("2. Organiser manuellement les images")
        print("3. Utiliser un autre dataset avec labels")
    
    return output_path

def suggest_next_steps(stats):
    """Suggérer les prochaines étapes"""
    print("\n" + "="*80)
    print("PROCHAINES ÉTAPES RECOMMANDÉES")
    print("="*80 + "\n")
    
    print("1️⃣  TROUVER LES LABELS")
    print("   - Chercher un fichier CSV/Excel avec les diagnostics")
    print("   - Colonnes attendues: patient_id, diagnosis, cancer_type, etc.")
    print("   - Ou télécharger depuis Kaggle avec les métadonnées")
    
    print("\n2️⃣  NETTOYER LES IMAGES")
    print("   - Vérifier la qualité des images")
    print("   - Supprimer les images corrompues")
    print("   - Normaliser les tailles")
    
    print("\n3️⃣  ORGANISER PAR CLASSE")
    print("   - Créer: datasets/rsna_organized/train/[benign|malignant|normal]/")
    print("   - Copier les images selon les labels")
    print("   - Split 80/20 train/test")
    
    print("\n4️⃣  COMBINER AVEC DATASET ACTUEL")
    print("   - Fusionner avec datasets/train/ et datasets/test/")
    print("   - Augmenter la taille du dataset")
    print("   - Réentraîner le modèle")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    # Chemin vers le dataset RSNA - AJUSTEZ CE CHEMIN !
    import sys
    
    # Essayer plusieurs chemins possibles
    possible_paths = [
        '../datasets/rsna_bitirme',
        'datasets/rsna_bitirme',
        '../rsna_bitirme',
        'C:/Users/yaman/Desktop/moussokene_master_search/datasets/rsna_bitirme',
    ]
    
    rsna_path = None
    for path in possible_paths:
        if Path(path).exists():
            rsna_path = path
            print(f"✅ Dataset trouvé: {path}")
            break
    
    if not rsna_path:
        print("\n" + "="*80)
        print("❌ ERREUR: Dataset RSNA non trouvé")
        print("="*80)
        print("\nChemins testés:")
        for path in possible_paths:
            print(f"   ❌ {path}")
        print("\n💡 SOLUTION:")
        print("   1. Vérifiez où se trouve votre dossier rsna_bitirme")
        print("   2. Modifiez la ligne 'rsna_path = ...' dans ce script")
        print("   3. Ou exécutez: python prepare_rsna_dataset.py --path 'VOTRE_CHEMIN'")
        print("\n" + "="*80 + "\n")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PRÉPARATION DATASET RSNA POUR CLASSIFICATION CANCER DU SEIN")
    print("="*80)
    
    # 1. Analyser la structure
    stats = analyze_rsna_structure(rsna_path)
    
    if stats:
        # 2. Chercher les labels
        csv_files = check_for_labels(rsna_path)
        
        # 3. Suggérer les prochaines étapes
        suggest_next_steps(stats)
        
        print("\n💡 POUR CONTINUER:")
        print("   1. Localisez le fichier CSV avec les labels")
        print("   2. Exécutez: python prepare_rsna_dataset.py --csv labels.csv")
        print("   3. Ou contactez-moi avec le nom du fichier CSV")
    
    print("\n" + "="*80 + "\n")
