import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14})

log_dir = 'logs/densenet121_phase2'

# We will collect 'epoch_loss' and 'epoch_auc' from train, and 'epoch_loss' and 'epoch_auc' from validation
train_acc = EventAccumulator(os.path.join(log_dir, 'train'))
train_acc.Reload()
val_acc = EventAccumulator(os.path.join(log_dir, 'validation'))
val_acc.Reload()

tags_train = train_acc.Tags()['scalars']
tags_val = val_acc.Tags()['scalars']

epochs = []
loss = []
val_loss = []
auc = []
val_auc = []

if 'epoch_loss' in tags_train:
    for event in train_acc.Scalars('epoch_loss'):
        epochs.append(event.step)
        loss.append(event.value)

if 'epoch_loss' in tags_val:
    for event in val_acc.Scalars('epoch_loss'):
        val_loss.append(event.value)

# Some models log as epoch_auc_1 or epoch_auc
auc_tag = 'epoch_auc' if 'epoch_auc' in tags_train else ('epoch_auc_1' if 'epoch_auc_1' in tags_train else None)
val_auc_tag = 'epoch_auc' if 'epoch_auc' in tags_val else ('epoch_auc_1' if 'epoch_auc_1' in tags_val else None)

if auc_tag:
    for event in train_acc.Scalars(auc_tag):
        auc.append(event.value)
if val_auc_tag:
    for event in val_acc.Scalars(val_auc_tag):
        val_auc.append(event.value)

os.makedirs('results/presentation', exist_ok=True)

if loss and val_loss:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Perte (Entraînement)', linewidth=2.5)
    plt.plot(epochs[:len(val_loss)], val_loss, 'r--', label='Perte (Validation)', linewidth=2.5)
    plt.title('Courbes de Perte (DenseNet121 - Focal Loss)', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Époques (Phase 2)', fontsize=14, fontweight='bold')
    plt.ylabel('Focal Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/presentation/5_training_loss.png', dpi=300)
    plt.close()
    print("✅ Courbes de perte (Loss) générées.")

if auc and val_auc:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, auc, 'g-', label='AUC-ROC (Entraînement)', linewidth=2.5)
    plt.plot(epochs[:len(val_auc)], val_auc, 'm--', label='AUC-ROC (Validation)', linewidth=2.5)
    plt.title("Évolution de l'AUC-ROC\n(Mesure de capacité de distinction)", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Époques (Phase 2)', fontsize=14, fontweight='bold')
    plt.ylabel('AUC', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    max_val_auc = max(val_auc)
    max_epoch = val_auc.index(max_val_auc) + epochs[0]  # Just getting the respective epoch approximation
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
    print("✅ Courbes d'AUC générées.")
