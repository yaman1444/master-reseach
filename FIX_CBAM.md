# Fix CBAM Deserialization Error

## Problème
```
ValueError: Unrecognized keyword arguments passed to CBAM: {'ratio': 8}
```

## Cause
Le modèle `densenet121_final.keras` a été sauvegardé avec l'ancien paramètre `ratio` au lieu de `reduction_ratio`.

## Solution Appliquée

Le fichier `cbam.py` a été modifié pour accepter les deux paramètres :

```python
def __init__(self, reduction_ratio=8, ratio=None, kernel_size=7, **kwargs):
    # Support both 'ratio' (old) and 'reduction_ratio' (new)
    if ratio is not None:
        self.reduction_ratio = ratio
    else:
        self.reduction_ratio = reduction_ratio
```

## Test

```bash
cd scripts
python test_model_loading.py
```

## Utilisation

### Option 1: Utiliser densenet121_improved.keras (RECOMMANDÉ)

```bash
python demo_predict.py --image "path/to/image.png" \
                       --model models/densenet121_improved.keras
```

### Option 2: Utiliser densenet121_final.keras (avec fix CBAM)

```bash
python demo_predict.py --image "path/to/image.png" \
                       --model models/densenet121_final.keras
```

### Avec seuils calibrés

```bash
python demo_predict.py --image "path/to/image.png" \
                       --model models/densenet121_improved.keras \
                       --use_calibrated
```

## Résultats de Calibration

D'après votre exécution :

```
✅ Meilleur seuil trouvé: 0.30
   Precision: 0.735
   Recall:    0.907  ← Objectif atteint! (≥0.90)
   F1:        0.812

📊 Métriques finales avec seuil calibré:
   debut (benign)    : P=0.861, R=0.882, F1=0.872
   grave (malignant) : P=0.735, R=0.907, F1=0.812  ← Excellent!
   normal            : P=0.986, R=0.539, F1=0.697

   Accuracy:  0.8310
   Macro-F1:  0.7936
```

### Analyse

**Gains**:
- ✅ Recall malignant: 0.800 → 0.907 (+10.7%)
- ✅ Objectif ≥0.90 atteint!

**Trade-offs**:
- Precision malignant: 0.885 → 0.735 (-15%)
- Accuracy globale: 0.843 → 0.831 (-1.2%)
- Macro-F1: 0.823 → 0.794 (-2.9%)

**Interprétation Clinique**:
- Plus de faux positifs malignant (precision baisse)
- Mais BEAUCOUP moins de faux négatifs (recall monte)
- En contexte médical, c'est le bon trade-off!
- Mieux vaut un faux positif (biopsie inutile) qu'un faux négatif (cancer manqué)

## Recommandation

Pour la production/démo:
1. Utiliser `--use_calibrated` pour les cas suspects
2. Afficher les deux prédictions (standard + calibrée) pour comparaison
3. Laisser le médecin décider avec les deux informations

## Prochaines Étapes

1. ✅ Calibration terminée
2. ⏳ Tester demo_predict avec --use_calibrated
3. ⏳ Lancer ablation_study_v2.py
4. ⏳ Lancer kfold_validation.py
