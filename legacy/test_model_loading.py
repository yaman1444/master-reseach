"""
Test rapide du chargement de modèle avec CBAM corrigé
"""
import tensorflow as tf
from cbam import CBAM
from focal_loss import FocalLoss

print("Test de chargement du modèle...")

# Test 1: densenet121_final.keras (ancien modèle avec 'ratio')
try:
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model('./models/densenet121_final.keras', 
                                       custom_objects=custom_objects, 
                                       compile=False)
    print("✅ densenet121_final.keras chargé avec succès")
except Exception as e:
    print(f"❌ Erreur densenet121_final.keras: {e}")

# Test 2: densenet121_improved.keras
try:
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model('./models/densenet121_improved.keras', 
                                       custom_objects=custom_objects, 
                                       compile=False)
    print("✅ densenet121_improved.keras chargé avec succès")
except Exception as e:
    print(f"❌ Erreur densenet121_improved.keras: {e}")

print("\n✅ Tests terminés")
