"""
Script pour télécharger les labels du dataset RSNA depuis HuggingFace
Dataset: https://huggingface.co/datasets/Alperennn/RSNA_BreastCanser
"""
import os
import requests
from pathlib import Path

def download_rsna_metadata():
    """Télécharger les métadonnées du dataset RSNA"""
    print("\n" + "="*80)
    print("TÉLÉCHARGEMENT DES LABELS RSNA DEPUIS HUGGINGFACE")
    print("="*80 + "\n")
    
    # URLs possibles pour les métadonnées
    urls = [
        "https://huggingface.co/datasets/Alperennn/RSNA_BreastCanser/raw/main/train.csv",
        "https://huggingface.co/datasets/Alperennn/RSNA_BreastCanser/raw/main/metadata.csv",
        "https://huggingface.co/datasets/Alperennn/RSNA_BreastCanser/raw/main/labels.csv",
        "https://huggingface.co/datasets/Alperennn/RSNA_BreastCanser/resolve/main/train.csv",
    ]
    
    output_dir = Path('../rsna_bitirme')
    output_dir.mkdir(exist_ok=True)
    
    for url in urls:
        print(f"🔍 Tentative: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = url.split('/')[-1]
                output_path = output_dir / filename
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Téléchargé: {output_path}")
                print(f"   Taille: {len(response.content)} bytes")
                return output_path
            else:
                print(f"   ❌ Erreur {response.status_code}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print("\n⚠️  Aucun fichier de métadonnées trouvé automatiquement")
    return None

def manual_instructions():
    """Instructions pour téléchargement manuel"""
    print("\n" + "="*80)
    print("TÉLÉCHARGEMENT MANUEL REQUIS")
    print("="*80 + "\n")
    
    print("📋 ÉTAPES À SUIVRE:")
    print("\n1️⃣  Aller sur HuggingFace:")
    print("   https://huggingface.co/datasets/Alperennn/RSNA_BreastCanser")
    
    print("\n2️⃣  Chercher le fichier de métadonnées:")
    print("   - Cliquer sur 'Files and versions'")
    print("   - Chercher: train.csv, metadata.csv, labels.csv, ou annotations.csv")
    
    print("\n3️⃣  Télécharger le fichier CSV:")
    print("   - Cliquer sur le fichier")
    print("   - Télécharger dans: rsna_bitirme/")
    
    print("\n4️⃣  Vérifier le contenu:")
    print("   - Ouvrir le CSV avec Excel/Notepad")
    print("   - Chercher les colonnes: patient_id, cancer, diagnosis, etc.")
    
    print("\n5️⃣  Me partager:")
    print("   - Le nom du fichier téléchargé")
    print("   - Les noms des colonnes importantes")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("RÉCUPÉRATION DES LABELS RSNA")
    print("="*80)
    
    # Essayer de télécharger automatiquement
    result = download_rsna_metadata()
    
    if not result:
        # Instructions manuelles
        manual_instructions()
        
        print("\n💡 ALTERNATIVE:")
        print("   Si vous avez déjà les images sans labels,")
        print("   je peux créer un script pour:")
        print("   1. Utiliser votre modèle actuel pour pré-labelliser")
        print("   2. Vous permettre de corriger manuellement")
        print("   3. Réentraîner avec les nouvelles images")
    
    print("\n" + "="*80 + "\n")
