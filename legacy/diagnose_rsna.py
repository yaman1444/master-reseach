"""
Script de diagnostic approfondi pour comprendre la structure RSNA
"""
import os
from pathlib import Path
from collections import defaultdict

def deep_scan(rsna_path, max_folders=10):
    """Scanner en profondeur pour comprendre la structure réelle"""
    rsna_path = Path(rsna_path)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC APPROFONDI DU DATASET RSNA")
    print("="*80 + "\n")
    
    # 1. Lister les premiers dossiers
    folders = sorted([f for f in rsna_path.iterdir() if f.is_dir()])
    print(f"📁 Total dossiers racine: {len(folders)}\n")
    
    # 2. Examiner les premiers dossiers en détail
    print(f"🔍 Examen des {min(max_folders, len(folders))} premiers dossiers:\n")
    
    all_extensions = set()
    structure_examples = []
    
    for i, folder in enumerate(folders[:max_folders]):
        print(f"\n{'='*60}")
        print(f"Dossier {i+1}: {folder.name}")
        print('='*60)
        
        # Lister tout le contenu
        subfolders = [f for f in folder.iterdir() if f.is_dir()]
        files = [f for f in folder.iterdir() if f.is_file()]
        
        print(f"  📂 Sous-dossiers: {len(subfolders)}")
        if subfolders:
            for sf in subfolders[:5]:  # Montrer les 5 premiers
                print(f"     - {sf.name}")
                
                # Regarder dans les sous-dossiers
                sub_files = list(sf.iterdir())
                if sub_files:
                    print(f"       Contenu ({len(sub_files)} items):")
                    for item in sub_files[:3]:
                        if item.is_file():
                            all_extensions.add(item.suffix.lower())
                            print(f"         • {item.name} ({item.stat().st_size / 1024:.1f} KB)")
                        else:
                            print(f"         • {item.name}/ (dossier)")
        
        print(f"  📄 Fichiers directs: {len(files)}")
        if files:
            for f in files[:5]:
                all_extensions.add(f.suffix.lower())
                print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    # 3. Résumé des extensions trouvées
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ DES EXTENSIONS TROUVÉES")
    print('='*80)
    if all_extensions:
        print(f"Extensions: {', '.join(sorted(all_extensions))}")
    else:
        print("❌ Aucune extension de fichier trouvée !")
    
    # 4. Compter TOUS les fichiers (scan complet)
    print(f"\n{'='*80}")
    print("🔢 COMPTAGE COMPLET (peut prendre du temps...)")
    print('='*80)
    
    total_files = 0
    files_by_ext = defaultdict(int)
    
    for root, dirs, files in os.walk(rsna_path):
        for file in files:
            total_files += 1
            ext = Path(file).suffix.lower()
            files_by_ext[ext] += 1
    
    print(f"\n📊 Total fichiers trouvés: {total_files}")
    if files_by_ext:
        print("\nRépartition par extension:")
        for ext, count in sorted(files_by_ext.items(), key=lambda x: x[1], reverse=True):
            print(f"   {ext if ext else '(sans extension)'}: {count} fichiers")
    
    # 5. Chercher des fichiers CSV/labels
    print(f"\n{'='*80}")
    print("🔍 RECHERCHE DE FICHIERS LABELS")
    print('='*80)
    
    csv_files = list(rsna_path.parent.glob('*.csv')) + list(rsna_path.glob('*.csv'))
    txt_files = list(rsna_path.parent.glob('*.txt')) + list(rsna_path.glob('*.txt'))
    json_files = list(rsna_path.parent.glob('*.json')) + list(rsna_path.glob('*.json'))
    
    if csv_files:
        print(f"\n✅ Fichiers CSV trouvés:")
        for f in csv_files:
            print(f"   - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    if txt_files:
        print(f"\n📝 Fichiers TXT trouvés:")
        for f in txt_files[:10]:
            print(f"   - {f.name}")
    
    if json_files:
        print(f"\n📋 Fichiers JSON trouvés:")
        for f in json_files:
            print(f"   - {f.name}")
    
    if not (csv_files or txt_files or json_files):
        print("\n❌ Aucun fichier de métadonnées trouvé")
    
    print(f"\n{'='*80}\n")
    
    return {
        'total_files': total_files,
        'extensions': files_by_ext,
        'csv_files': csv_files,
        'txt_files': txt_files,
        'json_files': json_files
    }

if __name__ == '__main__':
    rsna_path = Path('../datasets/rsna_bitirme')
    
    if not rsna_path.exists():
        rsna_path = Path('datasets/rsna_bitirme')
    
    if not rsna_path.exists():
        print("❌ Dataset non trouvé !")
        print(f"Chemin testé: {rsna_path.absolute()}")
    else:
        print(f"✅ Dataset trouvé: {rsna_path.absolute()}\n")
        results = deep_scan(rsna_path, max_folders=10)
        
        print("\n💡 PROCHAINES ÉTAPES:")
        if results['total_files'] == 0:
            print("   ⚠️  Aucun fichier trouvé - vérifiez que le dataset est bien téléchargé")
        elif not any(ext in ['.png', '.jpg', '.jpeg', '.dcm'] for ext in results['extensions']):
            print("   ⚠️  Aucune image trouvée - format inhabituel ?")
        elif not results['csv_files']:
            print("   ⚠️  Pas de labels CSV - impossible de classifier automatiquement")
            print("   📌 Options:")
            print("      1. Chercher le CSV sur HuggingFace/Kaggle")
            print("      2. Utiliser un autre dataset avec labels")
            print("      3. Labelliser manuellement (long !)")
        else:
            print("   ✅ Dataset semble complet - analysez le CSV pour créer la structure")
