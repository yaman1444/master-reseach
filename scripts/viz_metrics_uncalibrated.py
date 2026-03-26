import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams.update({'font.size': 14})

json_path_uncalib = 'results/densenet121_results.json'

with open(json_path_uncalib, 'r') as f:
    data = json.load(f)

cm_uncalib = np.array(data['confusion_matrix'])
classes = ['Debut (Précoce)', 'Grave (Avancé)', 'Normal']

cm_normalized = cm_uncalib.astype('float') / cm_uncalib.sum(axis=1)[:, np.newaxis]

os.makedirs('results/presentation', exist_ok=True)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_normalized, annot=False, cmap="Blues", fmt=".2f",
            cbar=True, cbar_kws={'label': 'Pourcentage de la vraie classe'},
            xticklabels=classes, yticklabels=classes, linewidths=1, linecolor='black')

for i in range(cm_uncalib.shape[0]):
    for j in range(cm_uncalib.shape[1]):
        count = cm_uncalib[i, j]
        pct = cm_normalized[i, j] * 100
        text = f"{count}\n({pct:.1f}%)"
        color = "white" if cm_normalized[i, j] > 0.5 else "black"
        plt.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=color,
                 fontsize=14, fontweight='bold')

plt.title("Matrice de Confusion Standard (Expérience 3)\nSur le Test Set (Avant Calibration)", fontsize=16, pad=20, fontweight='bold')
plt.ylabel('Vraie Classe (Diagnostic Réel)', fontsize=14, fontweight='bold')
plt.xlabel('Prédiction du Modèle', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/presentation/2_confusion_matrix_exp3.png', dpi=300)
plt.close()

print("✅ Matrice de confusion non calibrée générée.")
