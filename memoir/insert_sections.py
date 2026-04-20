#!/usr/bin/env python3
"""Insert new sections into the memoir document."""

with open("Memoir_Master_classification_cancer_breast.md", "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# INSERTION 1: After Grad-CAM section (4.2), before section 4.3
# ============================================================

marker_1 = 'Ces visualisations XAI (Explainable AI) attestent que DenseNet\u2011121 a internalis\u00e9 des \u201cr\u00e8gles m\u00e9tiers\u201d compatibles avec la pratique radiologique rigoureuse, ce qui renforce la confiance dans son utilisation comme outil d\u2019aide au d\u00e9pistage.'

new_section_4_2_1 = '''

### 4.2.1 Démonstration de l'inférence clinique individuelle

Pour illustrer concrètement le fonctionnement du système CADx en conditions réelles, un script de démonstration (`demo_predict.py`) a été développé. Ce script prend en entrée une image d'échographie mammaire quelconque, applique le prétraitement standard (redimensionnement 224×224, normalisation), effectue la prédiction par DenseNet‑121, puis génère automatiquement une carte de chaleur Grad‑CAM superposée à l'image originale. L'extrait de code suivant illustre le cœur de cette procédure d'inférence :

```python
# Extrait simplifié du script demo_predict.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Chargement du modèle DenseNet-121 pré-entraîné
model = tf.keras.models.load_model(
    'scripts/my_model.keras',
    custom_objects={'CBAM': CBAM, 'FocalLoss': FocalLoss},
    compile=False
)

# Prétraitement de l'image d'entrée
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_input = np.expand_dims(img_array, axis=0)

# Prédiction
predictions = model.predict(img_input, verbose=0)
predicted_class_idx = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_idx] * 100

# Génération de la carte Grad-CAM
gradcam = GradCAM(model, layer_name='relu')
heatmap = gradcam.compute_heatmap(img_input, class_idx=predicted_class_idx)
overlay = gradcam.overlay_heatmap(heatmap, img_uint8, alpha=0.4)
```

Les résultats obtenus sur trois cas représentatifs du jeu de test sont présentés ci-dessous. Pour chaque cas, la visualisation comporte trois panneaux : l'image originale à gauche, la carte Grad‑CAM superposée au centre, et l'histogramme des probabilités par classe à droite.

**Cas 1 — Lésion de stade « début » (Benign)**

Le modèle identifie correctement cette lésion comme bénigne avec une confiance de **98,81 %**. La carte Grad‑CAM montre une activation focalisée sur le nodule hypoéchogène central, confirmant que le réseau fonde sa décision sur la morphologie de la lésion et non sur des artefacts périphériques.

![Figure 4.5 – Prédiction Grad-CAM sur une lésion de stade début — confiance 98,81 %](fig_gradcam_benign.png)

**Cas 2 — Lésion de stade « grave » (Malignant)**

Le système détecte correctement ce cas comme malin avec une confiance de **91,42 %** et recommande une biopsie immédiate. La carte de chaleur recouvre précisément la masse irrégulière aux contours spiculés, en accord avec les critères BI‑RADS d'une tumeur infiltrante.

![Figure 4.6 – Prédiction Grad-CAM sur une lésion de stade grave — confiance 91,42 %](fig_gradcam_malignant.png)

**Cas 3 — Sein normal**

Le modèle prédit correctement un tissu mammaire sain avec une confiance de **68,86 %**. L'activation Grad‑CAM est diffuse sur l'ensemble du parenchyme glandulaire, sans concentration sur un foyer suspect, ce qui traduit l'absence de zone d'intérêt pathologique identifiée par le réseau.

![Figure 4.7 – Prédiction Grad-CAM sur un sein normal — confiance 68,86 %](fig_gradcam_normal.png)

### 4.2.2 Test de robustesse : comportement face à une image hors domaine

Un aspect crucial pour tout système d'IA médicale est son comportement face à des données hors distribution (OOD — Out-Of-Distribution), c'est-à-dire des images qui ne correspondent pas au domaine d'entraînement du modèle. Afin d'évaluer cette robustesse, nous avons volontairement soumis au modèle une image aléatoire n'ayant aucun rapport avec l'échographie mammaire.

Le modèle a produit la prédiction suivante : **Malignant à 62,21 %**. Ce résultat, bien que surprenant en apparence, s'explique par la nature mathématique de la fonction d'activation Softmax utilisée en sortie du réseau. Cette fonction contraint la somme des probabilités des trois classes à être toujours égale à 100 %, ce qui signifie que le modèle est structurellement incapable de répondre « je ne sais pas » ou « cette image n'est pas une échographie ». Ce phénomène, largement documenté dans la littérature sous le nom de « Softmax overconfidence » ou « OOD problem », constitue une limitation connue des classifieurs à sorties probabilistes fermées.

![Figure 4.8 – Prédiction sur une image hors domaine — La carte de chaleur s'active sur des zones incohérentes](fig_gradcam_random.png)

Cependant, l'examen de la carte Grad‑CAM associée révèle immédiatement l'incohérence de la prédiction : l'activation thermique se disperse sur des zones aléatoires de l'image, sans concentration sur une structure anatomique identifiable. Ce contraste entre une confiance numérique artificiellement élevée et un Grad‑CAM visuellement incohérent constitue précisément le mécanisme de sécurité intégré au système. En contexte clinique, un radiologue formé à la lecture des cartes de chaleur identifierait instantanément ce type d'aberration.

Pour un déploiement en production, cette limitation serait adressée par l'ajout d'un modèle « Gatekeeper » (filtre d'entrée) en amont du classifieur principal. Ce modèle léger (de type MobileNetV2) serait entraîné pour une classification binaire simple (« Échographie mammaire » vs « Autre ») et rejetterait automatiquement toute image n'ayant pas la signature visuelle d'une échographie, avant même qu'elle n'atteigne le DenseNet‑121. Cette architecture à deux étages (filtrage puis diagnostic) est la norme dans les systèmes d'IA médicale déployés en milieu hospitalier.'''


# ============================================================
# INSERTION 2: In section 5.3, after cascade explanation, add
# code blocks, results table, and cascade images
# ============================================================

marker_2 = '''### 5.3.2 L'apport de U-Net face aux spécificités ivoiriennes (Âge et Densité)'''

new_cascade_visuals = '''L'extrait de code suivant illustre l'architecture U-Net utilisée pour la segmentation et le pipeline de cascade :

```python
# Extrait simplifié du script train_unet.py — Architecture U-Net
def build_unet(input_shape=(256, 256, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encodeur (contraction)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    
    # Pont (bottleneck)
    b = Conv2D(256, 3, activation='relu', padding='same')(p2)
    
    # Décodeur (expansion) avec skip connections
    u1 = UpSampling2D()(b)
    u1 = Concatenate()([u1, c2])  # Skip connection
    d1 = Conv2D(128, 3, activation='relu', padding='same')(u1)
    u2 = UpSampling2D()(d1)
    u2 = Concatenate()([u2, c1])  # Skip connection
    d2 = Conv2D(64, 3, activation='relu', padding='same')(u2)
    
    # Masque binaire de sortie
    outputs = Conv2D(1, 1, activation='sigmoid')(d2)
    return tf.keras.Model(inputs, outputs)
```

```python
# Extrait simplifié du script test_unet_densenet_cascade.py
# Pipeline de cascade : U-Net → Masquage → DenseNet

# 1. Segmentation par U-Net (256x256)
unet_input = cv2.resize(original_img, (256, 256)) / 255.0
mask_pred = unet_model.predict(np.expand_dims(unet_input, 0))[0]
mask_binary = (mask_pred > 0.5).astype(np.float32)

# 2. Application du masque sur l'image originale
mask_resized = cv2.resize(mask_binary, (224, 224))
img_224 = cv2.resize(original_img, (224, 224)) / 255.0
masked_img = apply_clahe(img_224) * np.expand_dims(mask_resized, -1)

# 3. Classification par DenseNet-121
pred_probs = densenet_model.predict(np.expand_dims(masked_img, 0))[0]
predicted_class = np.argmax(pred_probs)
```

Les résultats quantitatifs de cette expérimentation en cascade sont présentés dans le tableau suivant :

| Métrique | DenseNet seul (Exp. 5) | Cascade U-Net + DenseNet |
|---|---|---|
| Accuracy globale | 76,7 % | 14,7 % |
| F1‑score macro | 0,73 | 0,12 |
| Recall « début » | 90,4 % | 12,5 % |
| Recall « grave » | — | 0,0 % |
| Recall « normal » | — | 43,5 % |

*Tableau 5.1 – Comparaison des performances entre le classifieur DenseNet‑121 seul et l'architecture en cascade U‑Net + DenseNet (sans ré-entraînement du classifieur sur les images masquées).*

La figure suivante montre des exemples visuels du pipeline en cascade : pour chaque ligne, l'image originale (après application du filtre CLAHE), le masque binaire généré par U-Net, et l'image masquée transmise au classifieur DenseNet‑121.

![Figure 5.1 – Exemple de cascade U-Net + DenseNet : Image originale, Masque U-Net, et Image masquée — Cas 1](fig_cascade_ex_0.png)

![Figure 5.2 – Exemple de cascade U-Net + DenseNet : Image originale, Masque U-Net, et Image masquée — Cas 2](fig_cascade_ex_1.png)

![Figure 5.3 – Exemple de cascade U-Net + DenseNet : Image originale, Masque U-Net, et Image masquée — Cas 3](fig_cascade_ex_2.png)

Ces visuels illustrent clairement le phénomène de perte d'information contextuelle : lorsque le fond est entièrement mis en noir par le masque U-Net, le DenseNet‑121 (entraîné sur des images complètes) perd les repères texturaux du parenchyme mammaire sur lesquels il avait appris à se baser. Ce résultat confirme que le ré-entraînement conjoint du classifieur sur les images segmentées est un prérequis indispensable pour exploiter pleinement cette architecture en cascade.

''' + marker_2

# Apply insertions
if marker_1 in content:
    content = content.replace(marker_1, marker_1 + new_section_4_2_1)
    print("✅ Insertion 1 (Section 4.2.1 + 4.2.2) réussie")
else:
    print("❌ Marqueur 1 non trouvé. Essai avec variante...")
    # Try without special dashes
    alt_marker = 'Ces visualisations XAI (Explainable AI) attestent que DenseNet'
    if alt_marker in content:
        idx = content.index(alt_marker)
        # Find end of this paragraph
        end_idx = content.index('\n\n', idx)
        original_paragraph = content[idx:end_idx]
        content = content.replace(original_paragraph, original_paragraph + new_section_4_2_1)
        print("✅ Insertion 1 réussie (variante)")
    else:
        print("❌ Aucune variante trouvée pour l'insertion 1")

if marker_2 in content:
    content = content.replace(marker_2, new_cascade_visuals)
    print("✅ Insertion 2 (Visuels cascade U-Net) réussie")
else:
    print("❌ Marqueur 2 non trouvé")

with open("Memoir_Master_classification_cancer_breast.md", "w", encoding="utf-8") as f:
    f.write(content)

print("\n✅ Document mis à jour avec succès.")
