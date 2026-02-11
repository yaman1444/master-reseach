"""
Pré-labellisation automatique du dataset RSNA avec le modèle DenseNet121 entraîné
Organise les 8403 images en benign/malignant/normal pour réentraînement
"""
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

# Importer les custom objects
sys.path.append(str(Path(__file__).parent))
from focal_loss import FocalLoss
from cbam import CBAM

# Configuration
CONFIG = {
    'model_path': './models/densenet121_final.keras',
    'rsna_path': '../datasets/rsna_bitirme',
    'output_path': '../datasets/rsna_labeled',
    'csv_output': '../datasets/rsna_predictions.csv',
    'img_size': (224, 224),
    'batch_size': 32,
    'confidence_threshold': 0.50,  # Seuil abaissé pour accepter plus de prédictions
}

CLASS_NAMES = ['benign', 'malignant', 'normal']

def load_model(model_path):
    """Charger le modèle avec custom objects"""
    print(f"\n📦 Chargement du modèle: {model_path}")
    
    custom_objects = {
        'FocalLoss': FocalLoss,
        'CBAM': CBAM,
        'focal_loss_fixed': FocalLoss(gamma=2.0, alpha=0.25)
    }
    
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print("✅ Modèle chargé avec succès\n")
    return model

def preprocess_image(img_path, img_size):
    """Prétraiter une image comme pendant l'entraînement"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Convertir en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionner
    img = cv2.resize(img, img_size)
    
    # Normaliser [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def predict_batch(model, image_paths, img_size, batch_size=32):
    """Prédire par batch pour optimiser la vitesse"""
    predictions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_indices = []
        
        for idx, path in enumerate(batch_paths):
            img = preprocess_image(path, img_size)
            if img is not None:
                batch_images.append(img)
                valid_indices.append(idx)
        
        if batch_images:
            batch_array = np.array(batch_images)
            batch_preds = model.predict(batch_array, verbose=0)
            
            # Remplir les prédictions
            pred_idx = 0
            for idx in range(len(batch_paths)):
                if idx in valid_indices:
                    predictions.append(batch_preds[pred_idx])
                    pred_idx += 1
                else:
                    predictions.append(None)
        else:
            predictions.extend([None] * len(batch_paths))
    
    return predictions

def collect_rsna_images(rsna_path):
    """Collecter toutes les images RSNA"""
    print("🔍 Collecte des images RSNA...")
    
    rsna_path = Path(rsna_path)
    image_paths = []
    
    for patient_folder in sorted(rsna_path.iterdir()):
        if patient_folder.is_dir():
            # Les images sont directement dans le dossier patient
            for img_file in patient_folder.glob('*.png'):
                image_paths.append(img_file)
    
    print(f"✅ {len(image_paths)} images trouvées\n")
    return image_paths

def auto_label_rsna(config):
    """Labelliser automatiquement le dataset RSNA"""
    
    print("="*80)
    print("PRÉ-LABELLISATION AUTOMATIQUE DATASET RSNA")
    print("="*80)
    
    # 1. Charger le modèle
    model = load_model(config['model_path'])
    
    # 2. Collecter les images
    image_paths = collect_rsna_images(config['rsna_path'])
    
    if not image_paths:
        print("❌ Aucune image trouvée !")
        return
    
    # 3. Créer la structure de sortie
    output_path = Path(config['output_path'])
    for split in ['train', 'test']:
        for class_name in CLASS_NAMES:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Dossier pour images à vérifier
    (output_path / 'to_verify').mkdir(parents=True, exist_ok=True)
    
    # 4. Prédire sur toutes les images
    print(f"🔮 Prédiction sur {len(image_paths)} images...")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Seuil confiance: {config['confidence_threshold']}\n")
    
    predictions = predict_batch(
        model, 
        image_paths, 
        config['img_size'], 
        config['batch_size']
    )
    
    # 5. Organiser les résultats
    results = []
    stats = {
        'benign': 0,
        'malignant': 0,
        'normal': 0,
        'to_verify': 0,
        'corrupted': 0
    }
    
    print("📊 Organisation des images...")
    for idx, (img_path, pred) in enumerate(zip(image_paths, predictions)):
        if idx % 500 == 0:
            print(f"   Progression: {idx}/{len(image_paths)} images traitées ({idx/len(image_paths)*100:.1f}%)")
        
        if pred is None:
            stats['corrupted'] += 1
            continue
        
        # Classe prédite et confiance
        class_idx = np.argmax(pred)
        confidence = pred[class_idx]
        predicted_class = CLASS_NAMES[class_idx]
        
        # Informations patient
        patient_id = img_path.parent.name
        view_type = img_path.stem  # ['LCC'], ['LMLO'], etc.
        
        # Décider où copier l'image
        if confidence >= config['confidence_threshold']:
            # Split 80/20 train/test
            split = 'train' if np.random.random() < 0.8 else 'test'
            dest_folder = output_path / split / predicted_class
            stats[predicted_class] += 1
        else:
            # Confiance faible -> à vérifier manuellement
            dest_folder = output_path / 'to_verify'
            stats['to_verify'] += 1
        
        # Nouveau nom de fichier
        new_filename = f"{patient_id}_{view_type}_{confidence:.3f}.png"
        dest_path = dest_folder / new_filename
        
        # Copier l'image
        shutil.copy2(img_path, dest_path)
        
        # Enregistrer les résultats
        results.append({
            'original_path': str(img_path),
            'patient_id': patient_id,
            'view_type': view_type,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'prob_benign': pred[0],
            'prob_malignant': pred[1],
            'prob_normal': pred[2],
            'needs_verification': confidence < config['confidence_threshold'],
            'new_path': str(dest_path)
        })
    
    # 6. Sauvegarder le CSV
    df = pd.DataFrame(results)
    df.to_csv(config['csv_output'], index=False)
    
    # 7. Afficher les statistiques
    print("\n" + "="*80)
    print("RÉSULTATS DE LA PRÉ-LABELLISATION")
    print("="*80 + "\n")
    
    print(f"📊 Répartition des prédictions:")
    print(f"   Benign:    {stats['benign']:4d} images ({stats['benign']/len(image_paths)*100:.1f}%)")
    print(f"   Malignant: {stats['malignant']:4d} images ({stats['malignant']/len(image_paths)*100:.1f}%)")
    print(f"   Normal:    {stats['normal']:4d} images ({stats['normal']/len(image_paths)*100:.1f}%)")
    print(f"   À vérifier: {stats['to_verify']:4d} images ({stats['to_verify']/len(image_paths)*100:.1f}%)")
    print(f"   Corrompues: {stats['corrupted']:4d} images")
    
    print(f"\n📁 Images organisées dans: {output_path}")
    print(f"📋 Prédictions sauvegardées: {config['csv_output']}")
    
    # Statistiques train/test
    train_count = sum(len(list((output_path / 'train' / c).glob('*.png'))) for c in CLASS_NAMES)
    test_count = sum(len(list((output_path / 'test' / c).glob('*.png'))) for c in CLASS_NAMES)
    
    print(f"\n📦 Split train/test:")
    print(f"   Train: {train_count} images ({train_count/(train_count+test_count)*100:.1f}%)")
    print(f"   Test:  {test_count} images ({test_count/(train_count+test_count)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("PROCHAINES ÉTAPES")
    print("="*80)
    print(f"\n1️⃣  VÉRIFIER LES PRÉDICTIONS DOUTEUSES ({stats['to_verify']} images)")
    print(f"   📂 Dossier: {output_path / 'to_verify'}")
    print(f"   💡 Déplacez manuellement vers train/[benign|malignant|normal]/")
    
    print(f"\n2️⃣  COMBINER AVEC DATASET BUSI ACTUEL")
    print(f"   python merge_datasets.py")
    
    print(f"\n3️⃣  RÉENTRAÎNER SUR DATASET MASSIF (~10K images)")
    print(f"   python train_advanced.py --data ../datasets/merged/")
    
    print("\n" + "="*80 + "\n")
    
    return df, stats

if __name__ == '__main__':
    # Vérifier que le modèle existe
    model_path = Path(CONFIG['model_path'])
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("💡 Entraînez d'abord le modèle avec: python train_advanced.py")
        sys.exit(1)
    
    # Vérifier que le dataset RSNA existe
    rsna_path = Path(CONFIG['rsna_path'])
    if not rsna_path.exists():
        print(f"❌ Dataset RSNA non trouvé: {rsna_path}")
        sys.exit(1)
    
    # Seed pour reproductibilité du split train/test
    np.random.seed(42)
    
    # Lancer la pré-labellisation
    df, stats = auto_label_rsna(CONFIG)
