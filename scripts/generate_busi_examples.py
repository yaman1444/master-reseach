import cv2
import matplotlib.pyplot as plt
import os

# Chemins des dossiers contenant les images
base_dir = "/Users/yaman/master-reseach/datasets_split/test"
classes = ['normal', 'debut', 'grave']
titles = ['Normal', 'Suspect (Début/Bénin)', 'Malignant (Grave)']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, cls in enumerate(classes):
    class_dir = os.path.join(base_dir, cls)
    # Prendre la première image du dossier qui n'est PAS un masque
    img_name = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg')) and '_mask' not in f][0]
    img_path = os.path.join(class_dir, img_name)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(titles[i], fontsize=16, fontweight='bold')
    axes[i].axis('off')

plt.suptitle("Exemples d'échographies mammaires (Dataset BUSI)", fontsize=20, fontweight='bold', y=1.05)
plt.tight_layout()

out_path = "/Users/yaman/master-reseach/memoir/2_1_datasets_examples.jpg"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"✅ Image générée : {out_path}")
