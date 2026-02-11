# 🚀 Guide de Démarrage Rapide - Version Corrigée

## ⚠️ Problème Identifié

Votre entraînement `train_improved.py` a rencontré plusieurs problèmes critiques:

1. **Performances faibles**: 84.3% accuracy (objectif: >96%)
2. **Stagnation**: Aucune amélioration pendant 47 epochs
3. **Temps extrême**: Un epoch a pris 28 heures!
4. **Hyperparamètres inadaptés**: Learning rate trop faible

## ✅ Solution: Nouveau Script Optimisé

### Fichiers Créés

```
scripts/
├── train_optimized.py      ← 🎯 UTILISER CELUI-CI
├── train_improved.py        ← Corrigé mais moins optimal
├── diagnose_data.py         ← Diagnostic du dataset
└── compare_results.py       ← Comparer les résultats

TROUBLESHOOTING.md           ← Documentation complète
```

---

## 📋 Étapes d'Exécution

### 1️⃣ Diagnostic (Optionnel mais Recommandé)

```bash
cd scripts
python diagnose_data.py
```

**Vérifie**:
- Structure des dossiers
- Nombre d'images par classe
- Déséquilibre des classes
- Qualité des images

### 2️⃣ Entraînement Optimisé

```bash
python train_optimized.py
```

**Temps estimé**: 40 minutes (vs 3-4 heures avant)

**Résultats attendus**:
- Accuracy: 90-93%
- Macro-F1: 88-91%
- Convergence stable

### 3️⃣ Comparaison des Résultats

```bash
python compare_results.py
```

Compare les performances entre les différentes versions.

---

## 🔧 Principales Corrections

### Hyperparamètres

| Paramètre | Avant | Après | Amélioration |
|-----------|-------|-------|--------------|
| Initial LR | 1e-4 | **1e-3** | 10x plus rapide |
| Fine-tune LR | 5e-6 | **1e-5** | 2x plus rapide |
| Dropout | 0.5 | **0.4** | Moins de régularisation |
| Unfreeze | 30% | **20%** | Plus stable |

### Callbacks

```python
# ❌ AVANT: Monitore val_loss, patience élevée
EarlyStopping(monitor='val_loss', patience=20)

# ✅ APRÈS: Monitore val_accuracy, patience réduite
EarlyStopping(monitor='val_accuracy', patience=8, mode='max')
```

### Epochs

```python
# ❌ AVANT: Trop d'epochs
initial_epochs: 30
fine_tune_epochs: 60

# ✅ APRÈS: Early stopping arrêtera avant
initial_epochs: 20
fine_tune_epochs: 30
```

---

## 📊 Comparaison des Versions

### train_improved.py (AVANT correction)
- ❌ Accuracy: 84.3%
- ❌ Macro-F1: 82.3%
- ❌ Temps: 3-4 heures
- ❌ Stagnation après epoch 1 de phase 2

### train_improved.py (APRÈS correction)
- ✅ Hyperparamètres corrigés
- ✅ Callbacks optimisés
- ✅ Temps réduit
- ⚠️ Toujours 2 phases séparées

### train_optimized.py (NOUVEAU - RECOMMANDÉ)
- ✅ Code simplifié et optimisé
- ✅ Meilleurs hyperparamètres
- ✅ Callbacks efficaces
- ✅ Convergence rapide
- ✅ Temps: ~40 minutes

---

## 🎯 Résultats Attendus

### Phase 1: Head Training
```
Epoch 1/20: val_accuracy: 0.78 → 0.82
Epoch 5/20: val_accuracy: 0.85 → 0.87
Epoch 10/20: val_accuracy: 0.87 → 0.88
Early stopping at epoch ~12-15
```

### Phase 2: Fine-Tuning
```
Epoch 1/30: val_accuracy: 0.88 → 0.89
Epoch 5/30: val_accuracy: 0.90 → 0.91
Epoch 10/30: val_accuracy: 0.91 → 0.92
Early stopping at epoch ~15-20
```

### Final
```
✅ Accuracy:  0.90-0.93
✅ Macro-F1:  0.88-0.91
✅ Temps total: ~40 minutes
```

---

## 🆘 Si Problèmes Persistent

### Accuracy < 85%

1. **Vérifier le dataset**:
   ```bash
   python diagnose_data.py
   ```

2. **Vérifier les dossiers**:
   ```
   datasets/
   ├── train/
   │   ├── debut/    (benign)
   │   ├── grave/    (malignant)
   │   └── normal/
   └── test/
       ├── debut/
       ├── grave/
       └── normal/
   ```

3. **Essayer sans Focal Loss**:
   ```python
   # Dans train_optimized.py, remplacer:
   loss=FocalLoss(gamma=2.0, alpha=0.25)
   # Par:
   loss='categorical_crossentropy'
   ```

### Out of Memory

```python
# Réduire batch_size dans CONFIG
'batch_size': 8,  # Au lieu de 16
```

### Training Trop Lent

```python
# Réduire taille d'image
'img_size': (128, 128),  # Au lieu de (224, 224)
```

---

## 📈 Prochaines Étapes

Une fois que `train_optimized.py` fonctionne bien:

1. **Visualisations**:
   ```bash
   python visualize_gradcam.py
   python visualize_all.py
   ```

2. **Comparaison multi-modèles**:
   ```bash
   python compare_models.py
   ```

3. **Ablation study**:
   ```bash
   python ablation_study.py
   ```

---

## 💡 Conseils

- ✅ Toujours exécuter `diagnose_data.py` en premier
- ✅ Surveiller les courbes d'entraînement
- ✅ Si val_accuracy stagne, arrêter manuellement (Ctrl+C)
- ✅ Comparer les résultats avec `compare_results.py`
- ❌ Ne pas augmenter patience au-delà de 10
- ❌ Ne pas réduire learning rate en dessous de 1e-6

---

## 📞 Support

Consultez `TROUBLESHOOTING.md` pour:
- Analyse détaillée des problèmes
- Solutions complètes
- Références scientifiques
- Checklist avant entraînement

---

**Bonne chance! 🚀**
