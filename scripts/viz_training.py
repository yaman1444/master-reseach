import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})

json_path = 'results/densenet121_results.json'

if not os.path.exists(json_path):
    print(f"❌ Error: {json_path} not found!")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

history = data.get('training_history', {})
if not history:
    print("❌ No history found in JSON!")
    exit(1)

# Extract metrics
epochs = range(1, len(history.get('loss', [])) + 1)
loss = history['loss']
val_loss = history['val_loss']
auc = history.get('auc', history.get('auc_1')) # Handles different metric naming
val_auc = history.get('val_auc', history.get('val_auc_1'))

os.makedirs('results/presentation', exist_ok=True)

# 5. Graphique: Courbes d'apprentissage (Loss)
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b-', label='Perte (Entraînement)', linewidth=2.5)
plt.plot(epochs, val_loss, 'r--', label='Perte (Validation)', linewidth=2.5)
plt.title('Courbes de Perte (DenseNet121 - Focal Loss)', fontsize=18, pad=20, fontweight='bold')
plt.xlabel('Époques', fontsize=14, fontweight='bold')
plt.ylabel('Focal Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('results/presentation/5_training_loss.png', dpi=300)
plt.close()

# 6. Graphique: Courbes d'apprentissage (AUC)
plt.figure(figsize=(10, 6))
plt.plot(epochs, auc, 'g-', label='AUC-ROC (Entraînement)', linewidth=2.5)
plt.plot(epochs, val_auc, 'm--', label='AUC-ROC (Validation)', linewidth=2.5)
plt.title("Évolution de l'AUC-ROC\n(Mesure de capacité de distinction)", fontsize=18, pad=20, fontweight='bold')
plt.xlabel('Époques', fontsize=14, fontweight='bold')
plt.ylabel('AUC', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)

# Highlight max val_auc
max_val_auc = max(val_auc)
max_epoch = val_auc.index(max_val_auc) + 1
plt.plot(max_epoch, max_val_auc, 'ko', markersize=10)
plt.annotate(f'Max AUC: {max_val_auc:.4f}',
             xy=(max_epoch, max_val_auc),
             xytext=(max_epoch - 5, max_val_auc - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9))

plt.tight_layout()
plt.savefig('results/presentation/6_training_auc.png', dpi=300)
plt.close()

print("✅ Courbes d'apprentissage générées dans 'results/presentation/'")
