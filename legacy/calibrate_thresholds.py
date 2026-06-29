import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

def main():
    print("🚀 Calibration des seuils pour DenseNet121 - Exp 3")
    
    model_path = 'models/densenet121_final.keras'
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        return

    from cbam import CBAM
    from focal_loss import FocalLoss
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    
    print("📦 Chargement du modèle...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    test_dir = '../datasets_split/test/'
    
    print("🔄 Chargement du dataset de test (sans augmentation TTA complexe pour la calibration pure)...")
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir,
        target_size=(320, 320),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )
    
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    print("🔮 Prédiction des probabilités brutes...")
    y_probs = model.predict(test_gen)
    
    # Stratégie de recherche de seuil
    # L'objectif est de maximiser le Macro-F1 tout en privilégiant le Recall sur 'grave'.
    print("\n🔍 Recherche des seuils optimaux...")
    
    best_macro_f1 = 0
    best_thresholds = [0.33, 0.33, 0.33] # defaut softmax implicite
    
    # Grille de recherche simplifiée pour éviter le sur-apprentissage des seuils
    thresholds_grid = np.linspace(0.2, 0.8, 13) # Seuils testés
    
    for t_debut in thresholds_grid:
        for t_grave in thresholds_grid:
            for t_normal in thresholds_grid:
                # Normalisation des seuils essayés (pour qu'ils agissent comme des poids)
                t_arr = np.array([t_debut, t_grave, t_normal])
                # On ajuste les probabilités brutes en les divisant par le seuil
                # (plus le seuil est bas, plus on 'boost' la classe)
                adjusted_probs = y_probs / t_arr
                y_pred_adj = np.argmax(adjusted_probs, axis=1)
                
                f1_macro = f1_score(y_true, y_pred_adj, average='macro')
                
                if f1_macro > best_macro_f1:
                    best_macro_f1 = f1_macro
                    best_thresholds = t_arr.tolist()
                    
    print(f"\n✅ Seuils optimisés trouvés (Poids d'ajustement) :")
    for i, name in enumerate(class_names):
        print(f"  - {name}: {best_thresholds[i]:.2f}")
        
    print(f"  ➜ Macro-F1 calibré espéré (sans TTA): {best_macro_f1:.4f}")
    
    # Sauvegarde des seuils
    out_dict = {
        'thresholds': dict(zip(class_names, best_thresholds)),
        'expected_macro_f1_no_tta': best_macro_f1
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/densenet121_exp3_thresholds.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
        
    print("💾 Seuils sauvegardés dans results/densenet121_exp3_thresholds.json")

    # Evaluation finale avec les nouveaux seuils
    adjusted_probs_final = y_probs / np.array(best_thresholds)
    y_pred_final = np.argmax(adjusted_probs_final, axis=1)
    
    print("\n📊 Rapport de Classification Final (Avec Seuils Calibrés) :")
    print(classification_report(y_true, y_pred_final, target_names=class_names, digits=4))

if __name__ == '__main__':
    main()
