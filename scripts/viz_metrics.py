import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Activer le style Seaborn
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 14})

json_path = 'results/densenet121_exp4_calibration.json'

if not os.path.exists(json_path):
    print(f"❌ Error: {json_path} not found!")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

# Matrice de Confusion (Calibrée)
cm = np.array(data['confusion_matrix_calibrated'])
classes = ['Debut (Précoce)', 'Grave (Avancé)', 'Normal']

# Calcul pourcentages par VRAIE classe (lignes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

os.makedirs('results/presentation', exist_ok=True)

# 3. Graphique: Matrice de confusion avec annotations riches
plt.figure(figsize=(9, 7))
sns.heatmap(cm_normalized, annot=False, cmap="Blues", fmt=".2f",
            cbar=True, cbar_kws={'label': 'Pourcentage de la vraie classe'},
            xticklabels=classes, yticklabels=classes, linewidths=1, linecolor='black')

# Annotations manuelles avec Nombre (Pourcentage)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = cm_normalized[i, j] * 100
        text = f"{count}\n({pct:.1f}%)"
        
        # Couleur du texte adaptative
        color = "white" if cm_normalized[i, j] > 0.5 else "black"
        
        plt.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=color,
                 fontsize=14, fontweight='bold')

plt.title("Matrice de Confusion Calibrée (Priorité Clinique)\nExpérience 5 sur le Test Set", fontsize=16, pad=20, fontweight='bold')
plt.ylabel('Vraie Classe (Diagnostic Réel)', fontsize=14, fontweight='bold')
plt.xlabel('Prédiction du Modèle', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/presentation/3_confusion_matrix_exp5.png', dpi=300)
plt.close()

# 4. Graphique: Comparaison Recall Exp3 vs Exp5
exp3_recall = [71.85, 89.06, 78.05]
exp5_recall = [90.37, 64.06, 56.10]
metrics = classes

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, exp3_recall, width, label='Exp. 3 (Standard)', color='#7f8c8d')
rects2 = plt.bar(x + width/2, exp5_recall, width, label='Exp. 5 (Priorité Clinique)', color='#27ae60')

plt.ylabel('Score de Recall (%)', fontsize=14, fontweight='bold')
plt.title('Amélioration Majoure du Dépistage Précoce (Recall)', fontsize=16, fontweight='bold')
plt.xticks(x, metrics, fontsize=14)
plt.legend(fontsize=12)

# Ligne cible 90% pour debut
plt.axhline(y=90.0, color='r', linestyle='--', linewidth=2, label='Cible 90%')
plt.text(0-width/2, 92, 'Cible Clinique \n≥ 90%', color='r', fontsize=12, fontweight='bold')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('results/presentation/4_recall_comparison.png', dpi=300)
plt.close()

print("✅ Matrices et graphiques de recall générés dans 'results/presentation/'")
