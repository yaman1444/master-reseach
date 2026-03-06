import os
import matplotlib.pyplot as plt
import seaborn as sns

# Activer le style Seaborn pour des graphiques professionnels
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})

# Données du Split
datasets = ['Train', 'Validation', 'Test']
totals = [1104, 236, 240]

# Distribution par classe
# Classe 0: debut, Classe 1: grave, Classe 2: normal
classes = ['Debut (Précoce)', 'Grave (Avancé)', 'Normal']
train_counts = [624, 294, 186]
val_counts = [135, 60, 41]   # Approx basées sur le test split
test_counts = [135, 64, 41]

# Palette stylisée
colors = ['#3498db', '#e74c3c', '#2ecc71']

os.makedirs('results/presentation', exist_ok=True)

# 1. Graphique: Taille des Datasets
plt.figure(figsize=(10, 6))
bars = plt.bar(datasets, totals, color='#95a5a6', edgecolor='black', linewidth=1.5)
plt.title("Répartition Globale du Dataset (Total: 1580 images)", fontsize=18, pad=20)
plt.ylabel("Nombre d'images", fontsize=14)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/presentation/1_dataset_split.png', dpi=300)
plt.close()

# 2. Graphique: Distribution des classes (Train)
plt.figure(figsize=(10, 6))
bars = plt.bar(classes, train_counts, color=colors, edgecolor='black', linewidth=1.5)
plt.title("Déséquilibre des Classes (Set d'Entraînement)", fontsize=18, pad=20)
plt.ylabel("Nombre d'images", fontsize=14)

# Ajouter pourcentages
total_train = sum(train_counts)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / total_train) * 100
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{int(yval)} ({pct:.1f}%)", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/presentation/2_class_distribution_train.png', dpi=300)
plt.close()

print("✅ Graphiques du dataset générés dans 'results/presentation/1_dataset_split.png' et '2_class_distribution_train.png'")
