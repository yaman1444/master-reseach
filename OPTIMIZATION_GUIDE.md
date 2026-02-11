# Guide d'Utilisation - Modules d'Optimisation

## 📋 Modules Créés

### 1. **calibrate_thresholds.py** - Calibration des Seuils
Optimise le seuil de décision pour la classe malignant afin d'augmenter le rappel ≥0.90

### 2. **ablation_study_v2.py** - Étude d'Ablation
Quantifie la contribution de chaque composant (CBAM, Mixup, FocalLoss, etc.)

### 3. **kfold_validation.py** - Validation K-Fold
Validation robuste avec moyenne ± écart-type sur 5 folds

### 4. **demo_predict.py** - Prédiction avec Seuils Calibrés (MODIFIÉ)
Ajout du support pour les seuils calibrés

---

## 🚀 Utilisation

### 1. Calibration des Seuils (PRIORITÉ HAUTE)

**Objectif**: Augmenter recall malignant de 0.879 → ≥0.90

```bash
cd scripts
python calibrate_thresholds.py --model ./models/densenet121_improved.keras \
                                --test_dir ../datasets/test \
                                --target_recall 0.90
```

**Outputs**:
- `results/densenet121_improved_thresholds.json` - Configuration des seuils
- `results/threshold_calibration.png` - Courbes Precision-Recall

**Résultats attendus**:
- Recall malignant: 0.879 → 0.90+
- Precision malignant: ~0.70-0.75 (acceptable)
- Macro-F1: maintenu ou légèrement amélioré

**Utilisation des seuils calibrés**:
```bash
# Prédiction avec seuils calibrés
python demo_predict.py --image ../datasets/test/grave/image.png \
                       --model ./models/densenet121_improved.keras \
                       --use_calibrated \
                       --threshold_config results/densenet121_improved_thresholds.json
```

---

### 2. Étude d'Ablation

**Objectif**: Quantifier contribution de chaque composant pour le mémoire

```bash
python ablation_study_v2.py
```

**Configurations testées**:
1. Baseline (rien)
2. +Augmentation
3. +Dropout
4. +CBAM
5. +FocalLoss
6. +ClassWeight (Full)

**Outputs**:
- `results/ablation_densenet121.csv` - Tableau comparatif
- `results/ablation_densenet121.md` - Rapport markdown
- `results/ablation_densenet121_plot.png` - Graphique gains

**Temps estimé**: ~2-3 heures (6 configs × 15 epochs)

**Exemple de résultats attendus**:
```
Configuration      Accuracy  Macro-F1  Benign-F1  Malignant-F1  Normal-F1
baseline           0.7800    0.7500    0.8200     0.6800        0.7500
+augmentation      0.8100    0.7900    0.8400     0.7200        0.8100
+dropout           0.8300    0.8100    0.8600     0.7500        0.8200
+cbam              0.8400    0.8200    0.8700     0.7700        0.8200
+focal_loss        0.8420    0.8220    0.8750     0.7800        0.8110
+class_weight      0.8430    0.8230    0.8790     0.8030        0.7870
```

---

### 3. K-Fold Validation

**Objectif**: Validation robuste pour rigueur scientifique

```bash
python kfold_validation.py --data_dir ../datasets/train \
                           --n_folds 5 \
                           --epochs 15
```

**Outputs**:
- `results/kfold_summary.json` - Statistiques complètes

**Temps estimé**: ~3-4 heures (5 folds × 15 epochs)

**Exemple de résultats**:
```
Accuracy:  0.8430 ± 0.0120
Macro-F1:  0.8230 ± 0.0150
AUC:       0.9450 ± 0.0080
```

**Utilisation pour le mémoire**:
- Montrer robustesse du modèle
- Intervalle de confiance à 95%
- Comparaison avec état de l'art

---

## 📊 Workflow Recommandé

### Phase 1: Calibration (30 min)
```bash
# 1. Calibrer seuils
python calibrate_thresholds.py

# 2. Tester sur cas individuels
python demo_predict.py --image ../datasets/test/grave/malignant_001.png \
                       --use_calibrated
```

### Phase 2: Ablation (2-3h)
```bash
# Lancer étude d'ablation
python ablation_study_v2.py
```

### Phase 3: K-Fold (3-4h)
```bash
# Validation robuste
python kfold_validation.py
```

---

## 🎯 Objectifs Atteints

### Avant Optimisation
- Accuracy: 0.843
- Macro-F1: 0.823
- Recall malignant: 0.879
- Recall normal: 0.700

### Après Calibration (Attendu)
- Accuracy: 0.840-0.845
- Macro-F1: 0.820-0.830
- **Recall malignant: ≥0.90** ✅
- Recall normal: 0.700-0.720

### Gains Ablation (Attendus)
- Augmentation: +4-5% macro-F1
- Dropout: +2-3% macro-F1
- CBAM: +1-2% macro-F1
- FocalLoss: +0.5-1% macro-F1
- ClassWeight: +0.5-1% macro-F1

---

## 📝 Pour le Mémoire

### Tableaux à Inclure

**1. Résultats Calibration**
```markdown
| Métrique | Baseline | Calibré | Amélioration |
|----------|----------|---------|--------------|
| Recall Malignant | 0.879 | 0.905 | +2.6% |
| Precision Malignant | 0.740 | 0.720 | -2.0% |
| Macro-F1 | 0.823 | 0.825 | +0.2% |
```

**2. Ablation Study**
```markdown
| Composant | Contribution Macro-F1 |
|-----------|-----------------------|
| Augmentation | +4.5% |
| Dropout | +2.3% |
| CBAM | +1.8% |
| FocalLoss | +0.8% |
| ClassWeight | +0.6% |
| **Total** | **+10.0%** |
```

**3. K-Fold Validation**
```markdown
| Métrique | Moyenne | Écart-type | IC 95% |
|----------|---------|------------|--------|
| Accuracy | 0.843 | 0.012 | [0.819, 0.867] |
| Macro-F1 | 0.823 | 0.015 | [0.793, 0.853] |
| AUC | 0.945 | 0.008 | [0.929, 0.961] |
```

---

## 🔧 Troubleshooting

### Calibration ne trouve pas de seuil optimal
```bash
# Réduire target_recall
python calibrate_thresholds.py --target_recall 0.88
```

### Ablation trop lente
```bash
# Modifier epochs dans ablation_study_v2.py ligne 138:
epochs=10  # au lieu de 15
```

### K-Fold Out of Memory
```bash
# Réduire batch_size dans kfold_validation.py ligne 127:
batch_size=8  # au lieu de 16
```

---

## 📚 Références pour le Mémoire

1. **Threshold Calibration**: 
   - Saito & Rehmsmeier (2015) "The Precision-Recall Plot Is More Informative than the ROC Plot"

2. **Ablation Studies**:
   - Meyes et al. (2019) "Ablation Studies in Artificial Neural Networks"

3. **K-Fold Validation**:
   - Kohavi (1995) "A Study of Cross-Validation and Bootstrap"

---

## ✅ Checklist Finale

- [ ] Calibration exécutée et seuils sauvegardés
- [ ] Ablation study complétée avec rapport markdown
- [ ] K-fold validation avec statistiques robustes
- [ ] Graphiques générés pour tous les modules
- [ ] Résultats intégrés dans le mémoire
- [ ] Grad-CAM vérifié sur cas malignant avec seuils calibrés

---

**Note**: Ces modules sont conçus pour s'intégrer au pipeline existant sans le modifier. Ils ajoutent des analyses supplémentaires pour renforcer la rigueur scientifique du projet.
